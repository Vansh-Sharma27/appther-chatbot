"""Query-time embedding wrapper.

Re-uses crawler.embed for the provider abstraction so the same model is used
at both ingest and query time. The only difference is input_type="query"
(instead of "document") as required by Voyage's asymmetric retrieval.

Public API:
    embed_query(text, provider) → list[float]
    get_query_provider(name, api_key) → EmbedProvider
"""

from __future__ import annotations

import os

from crawler.config import LANCE_TABLE_NAME, VOYAGE_EMBED_DIMS
from crawler.embed import EmbedProvider, embed_texts, get_provider
from crawler.index import read_index_meta


def get_query_provider(name: str = "voyage", api_key: str | None = None) -> EmbedProvider:
    """Return an embed provider configured for query-time use."""
    return get_provider(name, api_key=api_key)  # type: ignore[arg-type]


def embed_query(text: str, provider: EmbedProvider | None = None) -> list[float]:
    """Embed a single query string with input_type='query'."""
    p = provider or get_query_provider(api_key=os.environ.get("VOYAGE_API_KEY"))
    vectors = embed_texts([text], input_type="query", provider=p)
    return vectors[0]


def provider_from_index_meta(
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    api_key: str | None = None,
) -> EmbedProvider:
    """Read the index metadata and return a provider matching the ingest model.

    Raises ValueError if the index was built with an unknown/unsupported provider.
    """
    meta = read_index_meta(uri, table_name)
    provider_name = meta.get("provider", "voyage")
    dims = meta.get("dims", VOYAGE_EMBED_DIMS)

    p = get_query_provider(provider_name, api_key=api_key)
    if p.dims != dims:
        raise ValueError(
            f"Index was built with dims={dims} but provider {provider_name!r} "
            f"uses dims={p.dims}. Re-embed with matching dimensions."
        )
    return p
