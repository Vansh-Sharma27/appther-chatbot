"""Crawl reconciliation, per-run reporting, and 404/410 de-index wiring.

After each crawl run this module:
  1. Reconciles every discovered URL against the actual fetch results and
     classifies each URL as fetched / redirected / failed_permanent /
     failed_transient / not_attempted.
  2. Emits a structured JSON report with per-run counts and a delta vs the
     previous run; raises a drop_alert flag when too many pages vanish.
  3. De-indexes URLs that returned permanent HTTP errors (404/410) from the
     LanceDB index so stale chunks are never served.
  4. Persists a failures.jsonl for targeted re-runs of only the failed URLs.

Public API
----------
  reconcile(discovered, fetch_results, robots_filtered_count, previous_report)
      → CrawlReport

  fetch_llms_txt_urls(client)
      → list[str]   (https:// links extracted from llms.txt / llms-full.txt)

  classify_crawl_cadence(url: DiscoveredURL)
      → Literal["weekly", "monthly", "yearly"]

  deindex_permanent_failures(report, index_uri, table_name)
      → list[str]   (de-indexed URLs)

  save_report(report, staging_dir)    → Path
  load_latest_report(staging_dir)     → CrawlReport | None
  save_failures(report, staging_dir)  → Path
  load_failures(staging_dir)          → list[dict]

  load_url_aliases(staging_dir)               → dict[str, str]
  update_url_aliases(report, staging_dir)     → dict[str, str]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx

from crawler.config import (
    DROP_ALERT_THRESHOLD,
    FAILURES_FILENAME,
    LANCE_TABLE_NAME,
    REPORTS_SUBDIR,
    STAGING_DIR,
    URL_ALIASES_FILENAME,
)
from crawler.models import DiscoveredURL, FetchResult
from crawler.urls import infer_page_type, normalize_url

if TYPE_CHECKING:
    from crawler.robots import RobotsChecker

logger = logging.getLogger(__name__)

UrlStatus = Literal[
    "fetched",
    "redirected",
    "failed_permanent",
    "failed_transient",
    "not_attempted",
]

LlmsGapReason = Literal[
    "robots_blocked",
    "fetch_failed",
    "not_attempted",
]


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class UrlRecord:
    """Per-URL outcome of a single crawl run."""

    url: str
    status: UrlStatus
    http_code: int = 0
    final_url: str | None = None
    error: str | None = None
    suggested_cadence: str | None = None
    # Set to "deindexed" when the URL's chunks are removed from LanceDB.
    action: str | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "http_code": self.http_code,
            "final_url": self.final_url,
            "error": self.error,
            "suggested_cadence": self.suggested_cadence,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UrlRecord:
        return cls(
            url=d["url"],
            status=d["status"],
            http_code=d.get("http_code", 0),
            final_url=d.get("final_url"),
            error=d.get("error"),
            suggested_cadence=d.get("suggested_cadence"),
            action=d.get("action"),
        )


@dataclass
class LlmsCoverageGap:
    """An llms.txt-referenced URL that was not successfully crawled this run.

    reason is one of:
      - "robots_blocked" : disallowed by robots.txt, so never fetched
      - "fetch_failed"   : attempted but returned a 4xx/5xx/network error
      - "not_attempted"  : present in the discovered set but absent from the
                           fetch results (e.g. an interrupted run)
    """

    url: str
    reason: LlmsGapReason

    def to_dict(self) -> dict:
        return {"url": self.url, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> LlmsCoverageGap:
        return cls(url=d["url"], reason=d["reason"])


@dataclass
class CrawlReport:
    """Aggregated metrics for one complete crawl run."""

    run_id: str
    # URL counts
    discovered: int
    robots_excluded: int
    fetched: int
    redirected: int
    failed_permanent: int
    failed_transient: int
    not_attempted: int
    deindexed: int
    # Delta vs previous run (None on first run)
    page_count_delta: int | None
    drop_alert: bool
    url_records: list[UrlRecord] = field(default_factory=list)
    # True when this run only re-fetched a subset of URLs (e.g. --targeted).
    # Partial runs never fire the drop alert and are never used as the
    # drop-alert baseline for a later run (see reconcile / load_latest_report).
    partial: bool = False
    # llms.txt-referenced URLs this run did not successfully crawl (H1).
    # Populated only on full runs when reconcile() is given the llms.txt URL
    # list; purely a coverage signal -- never affects drop_alert or exit code.
    llms_uncrawled: list[LlmsCoverageGap] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        """Pages that returned content this run (fetched + redirected)."""
        return self.fetched + self.redirected

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "discovered": self.discovered,
            "robots_excluded": self.robots_excluded,
            "fetched": self.fetched,
            "redirected": self.redirected,
            "failed_permanent": self.failed_permanent,
            "failed_transient": self.failed_transient,
            "not_attempted": self.not_attempted,
            "deindexed": self.deindexed,
            "page_count_delta": self.page_count_delta,
            "drop_alert": self.drop_alert,
            "partial": self.partial,
            "llms_uncrawled": [g.to_dict() for g in self.llms_uncrawled],
            "url_records": [r.to_dict() for r in self.url_records],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CrawlReport:
        records = [UrlRecord.from_dict(r) for r in d.get("url_records", [])]
        llms_gaps = [LlmsCoverageGap.from_dict(g) for g in d.get("llms_uncrawled", [])]
        return cls(
            run_id=d["run_id"],
            discovered=d["discovered"],
            robots_excluded=d.get("robots_excluded", 0),
            fetched=d["fetched"],
            redirected=d["redirected"],
            failed_permanent=d["failed_permanent"],
            failed_transient=d["failed_transient"],
            not_attempted=d["not_attempted"],
            deindexed=d.get("deindexed", 0),
            page_count_delta=d.get("page_count_delta"),
            drop_alert=d["drop_alert"],
            partial=d.get("partial", False),
            llms_uncrawled=llms_gaps,
            url_records=records,
        )


# ── Core reconciliation ───────────────────────────────────────────────────────


def reconcile(
    discovered: list[DiscoveredURL],
    fetch_results: list[FetchResult],
    robots_filtered_count: int = 0,
    previous_report: CrawlReport | None = None,
    partial: bool = False,
    llms_urls: list[str] | None = None,
    llms_blocked: list[str] | None = None,
) -> CrawlReport:
    """Build a CrawlReport by reconciling discovered URLs against fetch results.

    Each discovered URL is classified using FetchResult.status_code:
      - fetched          : HTTP 200 with html present, final_url == url
      - redirected       : HTTP 200 with html present, final_url != url
      - failed_permanent : 4xx (excluding 429) — should be de-indexed
      - failed_transient : 5xx / 429 / network error after all retries
      - not_attempted    : URL in discovered but absent from fetch_results
                           (e.g. the pipeline was interrupted)

    The drop_alert flag fires when the current ok_count (fetched + redirected)
    has fallen more than DROP_ALERT_THRESHOLD (5%) below the previous run's
    ok_count.  First-run reports never fire the alert.

    When ``partial`` is True the run only re-fetched a subset of URLs
    (e.g. a ``--targeted`` re-run of previous failures), so its ok_count is
    not comparable to a full crawl.  In that case the drop alert is
    suppressed and page_count_delta is left as None, and the resulting
    report is flagged ``partial`` so it is never selected as a later run's
    drop-alert baseline.

    When ``llms_urls`` is provided (the https:// links advertised by
    llms.txt / llms-full.txt), every referenced URL that was not
    successfully crawled this run is recorded in ``llms_uncrawled`` with a
    reason (``robots_blocked`` for URLs in ``llms_blocked``,
    ``fetch_failed`` for 4xx/5xx/network failures, ``not_attempted``
    otherwise).  This is purely a coverage signal: it never affects
    ``drop_alert``, ``page_count_delta``, or the process exit code.
    """
    run_id = datetime.now(UTC).isoformat()

    # Build a lookup from original requested URL → FetchResult.
    # fetch_results are keyed by FetchResult.url which equals the URL as queued
    # (tracking params already stripped by fetch_page before the request).
    fetch_by_url: dict[str, FetchResult] = {r.url: r for r in fetch_results}

    url_records: list[UrlRecord] = []
    counts: dict[str, int] = {
        "fetched": 0,
        "redirected": 0,
        "failed_permanent": 0,
        "failed_transient": 0,
        "not_attempted": 0,
    }

    for disc in discovered:
        cadence = classify_crawl_cadence(disc)
        result = fetch_by_url.get(disc.url)

        if result is None:
            counts["not_attempted"] += 1
            url_records.append(
                UrlRecord(
                    url=disc.url,
                    status="not_attempted",
                    suggested_cadence=cadence,
                )
            )
            continue

        status = _classify_fetch_result(result)
        counts[status] += 1
        url_records.append(
            UrlRecord(
                url=disc.url,
                status=status,
                http_code=result.status_code,
                final_url=result.final_url if result.final_url != disc.url else None,
                error=result.error,
                suggested_cadence=cadence,
            )
        )

    curr_ok = counts["fetched"] + counts["redirected"]
    if partial:
        # A partial run's ok_count is not comparable to a full crawl, so we
        # neither compute a delta nor fire the drop alert against the baseline.
        delta, alert = None, False
    else:
        delta, alert = _compute_delta_and_alert(curr_ok, previous_report)

    if alert:
        logger.warning(
            "DROP ALERT: ok pages dropped from %d to %d (>%.0f%% drop)",
            previous_report.ok_count if previous_report else 0,
            curr_ok,
            DROP_ALERT_THRESHOLD * 100,
        )

    llms_uncrawled = _llms_coverage_gaps(url_records, llms_urls, llms_blocked)

    report = CrawlReport(
        run_id=run_id,
        discovered=len(discovered),
        robots_excluded=robots_filtered_count,
        fetched=counts["fetched"],
        redirected=counts["redirected"],
        failed_permanent=counts["failed_permanent"],
        failed_transient=counts["failed_transient"],
        not_attempted=counts["not_attempted"],
        deindexed=0,
        page_count_delta=delta,
        drop_alert=alert,
        partial=partial,
        llms_uncrawled=llms_uncrawled,
        url_records=url_records,
    )

    _log_summary(report)
    return report


def _classify_fetch_result(result: FetchResult) -> UrlStatus:
    if result.is_transient_error:
        return "failed_transient"
    if result.is_permanent_error:
        return "failed_permanent"
    if result.ok:
        # HTTP 200 with content — check for redirect
        if result.final_url and result.final_url != result.url:
            return "redirected"
        return "fetched"
    # Any other non-200 non-error state (301/302 that was followed to a non-200
    # final page, or an unexpected code) → treat as transient if 5xx, else permanent.
    if 400 <= result.status_code < 500:
        return "failed_permanent"
    return "failed_transient"


def _compute_delta_and_alert(
    curr_ok: int,
    previous_report: CrawlReport | None,
) -> tuple[int | None, bool]:
    if previous_report is None:
        return None, False
    prev_ok = previous_report.ok_count
    delta = curr_ok - prev_ok
    if prev_ok == 0:
        return delta, False
    drop_ratio = (prev_ok - curr_ok) / prev_ok
    alert = drop_ratio > DROP_ALERT_THRESHOLD
    return delta, alert


def _llms_coverage_gaps(
    url_records: list[UrlRecord],
    llms_urls: list[str] | None,
    llms_blocked: list[str] | None,
) -> list[LlmsCoverageGap]:
    """Classify each llms.txt-referenced URL that was not successfully crawled.

    Compares the llms.txt URL list against this run's per-URL fetch outcomes
    using the same canonical key as the C3 de-index aliasing (lowercase
    scheme/host/path, tracking params stripped), so path-case and
    redirect-shift variants collapse correctly.
    """
    if not llms_urls:
        return []

    from crawler.normalize import canonicalize_url

    blocked_keys = {canonicalize_url(u) for u in (llms_blocked or [])}
    status_by_key: dict[str, str] = {}
    for rec in url_records:
        status_by_key[canonicalize_url(rec.url)] = rec.status

    gaps: list[LlmsCoverageGap] = []
    seen: set[str] = set()
    for raw in llms_urls:
        key = canonicalize_url(raw)
        if key in seen:
            continue
        status = status_by_key.get(key)
        if status in ("fetched", "redirected"):
            continue  # successfully crawled -- not a gap
        seen.add(key)
        if key in blocked_keys:
            reason: LlmsGapReason = "robots_blocked"
        elif status in ("failed_permanent", "failed_transient"):
            reason = "fetch_failed"
        else:
            reason = "not_attempted"
        gaps.append(LlmsCoverageGap(url=key, reason=reason))

    if gaps:
        logger.warning(
            "llms.txt coverage gap: %d referenced URL(s) not crawled this run",
            len(gaps),
        )
    return gaps


def _log_summary(report: CrawlReport) -> None:
    logger.info(
        "Reconciliation: discovered=%d robots_excluded=%d "
        "fetched=%d redirected=%d failed_permanent=%d failed_transient=%d "
        "not_attempted=%d | drop_alert=%s",
        report.discovered,
        report.robots_excluded,
        report.fetched,
        report.redirected,
        report.failed_permanent,
        report.failed_transient,
        report.not_attempted,
        report.drop_alert,
    )


# ── llms.txt URL extraction ───────────────────────────────────────────────────


def fetch_llms_txt_urls(client: httpx.Client | None = None) -> list[str]:
    """Return the https:// URLs referenced by llms.txt / llms-full.txt.

    These URLs are an additional cross-check in reconciliation: the caller merges
    them with the sitemap-discovered URL list so the report flags any
    llms.txt-referenced page that was not crawled.

    Delegates the fetch to the single shared crawler.llms.fetch_llms_txt (which
    uses http_client.create_client) so there is exactly one llms.txt fetcher with
    consistent headers/timeouts/pooling. Returns [] if the file cannot be fetched.
    """
    from crawler.llms import extract_llms_urls, fetch_llms_txt

    text, source_url = fetch_llms_txt(client=client)
    if not text:
        return []
    urls = extract_llms_urls(text)
    logger.info("fetch_llms_txt_urls: found %d unique URLs from %s", len(urls), source_url)
    return urls


def merge_llms_urls(
    discovered: list[DiscoveredURL],
    llms_urls: list[str],
    robots: RobotsChecker,
) -> tuple[list[DiscoveredURL], list[str]]:
    """Augment *discovered* with llms.txt-referenced pages the crawl missed.

    Returns ``(augmented, blocked)`` where:
      - ``augmented`` is *discovered* plus a ``DiscoveredURL(source="llms")``
        for every llms.txt URL not already present (compared on the C3
        canonical key) and allowed by robots.txt;
      - ``blocked`` lists the robots-disallowed llms.txt URLs -- these are
        never crawled but are surfaced in the coverage report via reconcile().
    """
    from crawler.normalize import canonicalize_url

    augmented = list(discovered)
    seen = {canonicalize_url(d.url) for d in discovered}
    blocked: list[str] = []
    for raw in llms_urls:
        key = canonicalize_url(raw)
        if key in seen:
            continue
        seen.add(key)
        norm = normalize_url(raw)
        if not robots.is_allowed(norm):
            blocked.append(norm)
            continue
        augmented.append(DiscoveredURL(url=norm, source="llms", page_type=infer_page_type(norm)))
    return augmented, blocked


# ── Cadence classification ────────────────────────────────────────────────────


def classify_crawl_cadence(url: DiscoveredURL) -> Literal["weekly", "monthly", "yearly"]:
    """Map changefreq / priority to a suggested re-crawl cadence.

    This is purely informational — it appears in the per-URL report entry
    and does not affect which URLs are crawled on any given run (all URLs
    are always attempted; the index's content_hash handles cost efficiency).
    """
    freq = (url.changefreq or "").lower()
    if freq in {"always", "hourly", "daily", "weekly"}:
        return "weekly"
    if freq == "monthly":
        return "monthly"
    if freq in {"yearly", "never"}:
        return "yearly"

    # Fall back to priority
    p = url.priority
    if p is not None:
        if p >= 0.7:
            return "weekly"
        if p >= 0.4:
            return "monthly"
        return "yearly"

    return "weekly"  # safe default


# ── De-index wiring ───────────────────────────────────────────────────────────


def deindex_permanent_failures(
    report: CrawlReport,
    index_uri: str,
    table_name: str = LANCE_TABLE_NAME,
    aliases: dict[str, str] | None = None,
    _delete_fn=None,
) -> list[str]:
    """Remove chunks for permanently-failed URLs (404/410) from the LanceDB index.

    Only ``failed_permanent`` URLs are de-indexed — transient failures are
    retried on the next run and must not lose their index entries.

    De-index keys MUST match the keys chunks were *written* under.  The index
    writes every chunk under ``canonicalize_url(fetch.final_url)`` — the path
    lower-cased and redirects followed (see
    ``crawler.normalize.build_normalized_doc``).  Deleting by the raw discovered
    URL instead silently removes 0 rows and leaves permanent orphans (C3).  This
    function reconstructs the stored key for each failed URL:

      * ``canonicalize_url(rec.url)`` lower-cases the path, matching pages that
        were indexed under a case-folded path (e.g. ``/company-DelhiNCR`` was
        stored as ``/company-delhincr``).
      * the optional *aliases* map (original-canonical → final-canonical,
        produced by :func:`update_url_aliases`) resolves redirect-shifted pages
        that were indexed under their redirect target but now 404 at the
        original address.  When no alias exists the base key is used unchanged,
        so behaviour degrades gracefully to the path-case fix.

    ``delete_chunks_for_urls`` stays a pure exact-match primitive; all key
    canonicalization happens here.

    Annotates each affected UrlRecord with action="deindexed" and updates
    report.deindexed in-place.

    Args:
        aliases: Optional redirect alias map from :func:`load_url_aliases`.
            Both keys and values are canonical URLs.  Defaults to no aliases.
        _delete_fn: Callable with the same signature as
            ``crawler.index.delete_chunks_for_urls``.  Injected in tests to
            avoid a real LanceDB/boto3 dependency; leave as None in production.

    Returns the list of unique canonical keys passed to the delete primitive.
    """
    if _delete_fn is None:
        from crawler.index import delete_chunks_for_urls as _delete_fn

    failed_records = [rec for rec in report.url_records if rec.status == "failed_permanent"]
    if not failed_records:
        return []

    # Imported lazily: crawler.normalize pulls in the extraction stack, which we
    # don't want to require when there is nothing to de-index.
    from crawler.normalize import canonicalize_url

    aliases = aliases or {}

    keys_to_deindex: list[str] = []
    for rec in failed_records:
        base_key = canonicalize_url(rec.url)
        keys_to_deindex.append(aliases.get(base_key, base_key))

    # Two failed URLs can canonicalize to the same stored key (e.g. mixed-case
    # duplicates); de-dupe while preserving order.
    unique_keys = list(dict.fromkeys(keys_to_deindex))

    deleted = _delete_fn(unique_keys, uri=index_uri, table_name=table_name)
    logger.info(
        "deindex_permanent_failures: removed %d chunks for %d failed URLs (%d unique keys)",
        deleted,
        len(failed_records),
        len(unique_keys),
    )

    # Annotate the de-indexed records and update the report counter.
    for rec in failed_records:
        rec.action = "deindexed"
    report.deindexed = len(failed_records)

    return unique_keys


def load_url_aliases(staging_dir: str = STAGING_DIR) -> dict[str, str]:
    """Load the persisted redirect alias map from <staging_dir>/url_aliases.json.

    The map sends a URL's *original* canonical form to the canonical form it was
    indexed under after following redirects.  It is consumed by
    :func:`deindex_permanent_failures` so a page that 404s at its original
    address can still be de-indexed under the redirect target it was stored as.

    Returns an empty dict when the file is missing or unreadable, so callers
    degrade gracefully to the path-case-only behaviour.
    """
    path = Path(staging_dir) / URL_ALIASES_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load URL alias map from %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("URL alias map at %s is not a JSON object; ignoring", path)
        return {}
    return {str(k): str(v) for k, v in data.items()}


def update_url_aliases(
    report: CrawlReport,
    staging_dir: str = STAGING_DIR,
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge this run's redirect mappings into the persisted alias map and save it.

    For every ``redirected`` URL in *report* we record
    ``canonicalize_url(original) → canonicalize_url(final_url)`` — exactly the
    key the index writes the page under (``canonicalize_url(fetch.final_url)``).
    The mapping is captured while the redirect still resolves (HTTP 200) so it
    is available on a later run if the original address starts returning
    404/410 and the page must be de-indexed.

    Existing mappings are preserved; a newer run wins on conflict.  Pass
    *existing* to reuse an already-loaded map instead of re-reading the file.

    Returns the merged alias map (also written to <staging_dir>/url_aliases.json).
    """
    from crawler.normalize import canonicalize_url

    aliases = dict(existing) if existing is not None else load_url_aliases(staging_dir)

    for rec in report.url_records:
        if rec.status != "redirected" or not rec.final_url:
            continue
        origin_key = canonicalize_url(rec.url)
        target_key = canonicalize_url(rec.final_url)
        if origin_key != target_key:
            aliases[origin_key] = target_key

    path = Path(staging_dir) / URL_ALIASES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved %d URL aliases to %s", len(aliases), path)
    return aliases


# ── Report persistence ────────────────────────────────────────────────────────


def save_report(report: CrawlReport, staging_dir: str = STAGING_DIR) -> Path:
    """Write the report as a timestamped JSON file under <staging_dir>/reports/.

    The run_id (ISO 8601) is used in the filename so reports sort chronologically.
    Returns the path written.
    """
    reports_dir = Path(staging_dir) / REPORTS_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize run_id for use in a filename (replace colons and dots)
    safe_id = report.run_id.replace(":", "-").replace("+", "").replace(".", "-")
    path = reports_dir / f"report_{safe_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved crawl report to %s", path)
    return path


def load_latest_report(staging_dir: str = STAGING_DIR) -> CrawlReport | None:
    """Return the most recent full (non-partial) CrawlReport, or None.

    Reports are sorted lexicographically by filename; since filenames embed the
    ISO-8601 run_id they sort chronologically.  Partial runs (e.g. --targeted)
    are skipped because their ok_count is not comparable to a full crawl and
    must not become the drop-alert baseline.  Unparseable reports are skipped.
    """
    reports_dir = Path(staging_dir) / REPORTS_SUBDIR
    if not reports_dir.exists():
        return None

    candidates = sorted(reports_dir.glob("report_*.json"))
    if not candidates:
        return None

    # Walk newest -> oldest and return the first full (non-partial) report.
    # Partial runs (e.g. --targeted) have a non-comparable ok_count, so they
    # must never be used as the drop-alert baseline.  Unparseable reports are
    # skipped so a single corrupt file cannot blind the baseline.
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report = CrawlReport.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load report from %s: %s", path, exc)
            continue
        if report.partial:
            logger.debug("Skipping partial report %s as baseline candidate", path)
            continue
        return report
    return None


# ── Failures persistence ──────────────────────────────────────────────────────


def save_failures(report: CrawlReport, staging_dir: str = STAGING_DIR) -> Path:
    """Persist URLs that failed (permanently or transiently) to <staging_dir>/failures.jsonl.

    The file is overwritten on every run so it always reflects the most recent
    set of failures.  The pipeline's --targeted flag reads this file to re-fetch
    only the failing URLs.
    """
    failures_path = Path(staging_dir) / FAILURES_FILENAME
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    failure_statuses = {"failed_permanent", "failed_transient", "not_attempted"}
    entries = [
        {
            "url": rec.url,
            "reason": rec.status,
            "status_code": rec.http_code,
            "error": rec.error,
        }
        for rec in report.url_records
        if rec.status in failure_statuses
    ]

    with failures_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    logger.info(
        "Saved %d failure entries to %s",
        len(entries),
        failures_path,
    )
    return failures_path


def load_failures(staging_dir: str = STAGING_DIR) -> list[dict]:
    """Load the failures list from the previous run.

    Returns a list of dicts with keys: url, reason, status_code, error.
    Returns an empty list if the file does not exist.
    """
    failures_path = Path(staging_dir) / FAILURES_FILENAME
    if not failures_path.exists():
        return []

    entries: list[dict] = []
    with failures_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed failures entry: %s", exc)
    return entries
