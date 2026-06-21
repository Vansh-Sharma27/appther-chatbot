"""DynamoDB-backed state layer for the RAG chatbot.

Single-table design matching infra/dynamodb.tf:
  pk (HASH) + sk (RANGE)  →  type-prefixed keys
  expires_at (TTL)        →  auto-expire cache entries

Key prefixes:
  CACHE#<md5>    — answer cache
  FEEDBACK#<id>  — 👍/👎 feedback
  LEAD#<id>      — captured leads
  GAP#<id>       — content-gap log entries
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime

import boto3

logger = logging.getLogger(__name__)

# Default TTL for cached answers (24 hours).
_DEFAULT_CACHE_TTL = 86400


class AnswerCache:
    """LRU-like answer cache backed by DynamoDB with TTL expiry.

    Keys are derived from a hash of the (lowercased, stripped) question so
    identical questions produce cache hits regardless of whitespace/casing.
    """

    def __init__(self, table_name: str, client=None) -> None:
        self._table_name = table_name
        self._client = client or boto3.client("dynamodb", region_name="us-east-1")

    def _pk(self, question: str) -> str:
        normalized = question.lower().strip()
        digest = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()
        return f"CACHE#{digest}"

    def get(self, question: str) -> dict | None:
        """Return the cached RAGResult dict for *question*, or None."""
        pk = self._pk(question)
        resp = self._client.get_item(
            TableName=self._table_name,
            Key={"pk": {"S": pk}, "sk": {"S": pk}},
        )
        item = resp.get("Item")
        if not item or "data" not in item:
            return None
        return json.loads(item["data"]["S"])

    def set(self, question: str, result: dict, ttl_seconds: int = _DEFAULT_CACHE_TTL) -> None:
        """Store *result* for *question* with an expires_at TTL."""
        answer = result.get("answer")
        if not answer:
            raise ValueError("Cannot cache empty answer")
        pk = self._pk(question)
        expires_at = int(time.time()) + ttl_seconds
        self._client.put_item(
            TableName=self._table_name,
            Item={
                "pk": {"S": pk},
                "sk": {"S": pk},
                "type": {"S": "cache"},
                "data": {"S": json.dumps(result)},
                "expires_at": {"N": str(expires_at)},
            },
        )


class FeedbackStore:
    """Persist 👍/👎 feedback tied to the chunks that produced the answer."""

    def __init__(self, table_name: str, client=None) -> None:
        self._table_name = table_name
        self._client = client or boto3.client("dynamodb", region_name="us-east-1")

    def store(
        self,
        question: str,
        answer: str,
        thumbs_up: bool,
        chunks: list[dict],
        reason: str | None = None,
    ) -> None:
        """Store one feedback record."""
        feedback_id = str(uuid.uuid4())
        pk = f"FEEDBACK#{feedback_id}"
        item: dict = {
            "pk": {"S": pk},
            "sk": {"S": pk},
            "type": {"S": "feedback"},
            "question": {"S": question},
            "answer": {"S": answer},
            "thumbs_up": {"BOOL": thumbs_up},
            "chunks": {"S": json.dumps(chunks)},
            "created_at": {"S": datetime.now(UTC).isoformat()},
        }
        if reason:
            item["reason"] = {"S": reason}
        self._client.put_item(TableName=self._table_name, Item=item)


class LeadStore:
    """Capture leads from the /contact-us fallback flow."""

    def __init__(self, table_name: str, client=None) -> None:
        self._table_name = table_name
        self._client = client or boto3.client("dynamodb", region_name="us-east-1")

    def store(
        self,
        name: str,
        email: str,
        question: str,
        phone: str | None = None,
        message: str | None = None,
    ) -> None:
        """Store one lead record."""
        lead_id = str(uuid.uuid4())
        pk = f"LEAD#{lead_id}"
        item: dict = {
            "pk": {"S": pk},
            "sk": {"S": pk},
            "type": {"S": "lead"},
            "name": {"S": name},
            "email": {"S": email},
            "question": {"S": question},
            "created_at": {"S": datetime.now(UTC).isoformat()},
        }
        if phone:
            item["phone"] = {"S": phone}
        if message:
            item["message"] = {"S": message}
        self._client.put_item(TableName=self._table_name, Item=item)


class ContentGapLog:
    """Log unanswered questions so the content team can add coverage.

    Each entry records the original question (and optionally the rewritten
    query) so gaps are traceable.  Entries are queryable via recent().
    """

    def __init__(self, table_name: str, client=None) -> None:
        self._table_name = table_name
        self._client = client or boto3.client("dynamodb", region_name="us-east-1")

    def log(self, question: str, rewritten_query: str | None = None) -> None:
        """Record an unanswered question as a content gap."""
        gap_id = str(uuid.uuid4())
        pk = f"GAP#{gap_id}"
        item: dict = {
            "pk": {"S": pk},
            "sk": {"S": pk},
            "type": {"S": "gap"},
            "question": {"S": question},
            "created_at": {"S": datetime.now(UTC).isoformat()},
        }
        if rewritten_query:
            item["rewritten_query"] = {"S": rewritten_query}
        self._client.put_item(TableName=self._table_name, Item=item)

    def recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent content-gap entries, up to *limit*.

        NOTE: Uses a DynamoDB Scan with a FilterExpression, which reads every
        item in the table before applying the filter. This is acceptable at
        launch scale (a few thousand cache entries) but should be migrated to
        a GSI-query pattern once the table grows beyond ~10k items or the
        content-gap backlog becomes a frequently-used dashboard. See the GSI
        migration note in infra/variables.tf.
        """
        resp = self._client.scan(
            TableName=self._table_name,
            FilterExpression="begins_with(pk, :prefix)",
            ExpressionAttributeValues={":prefix": {"S": "GAP#"}},
            Limit=limit,
        )
        items = resp.get("Items", [])
        entries: list[dict] = []
        for item in items:
            entry: dict = {"question": item["question"]["S"]}
            if "rewritten_query" in item:
                entry["rewritten_query"] = item["rewritten_query"]["S"]
            if "created_at" in item:
                entry["created_at"] = item["created_at"]["S"]
            entries.append(entry)
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries[:limit]
