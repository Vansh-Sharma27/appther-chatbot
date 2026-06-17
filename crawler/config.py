"""Central configuration for the crawler. Override via environment variables where noted."""

import os

# ── Site ──────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.appther.com"
SITEMAP_URL = "https://www.appther.com/sitemap.xml"
ROBOTS_URL = "https://www.appther.com/robots.txt"
LLMS_TXT_URL = "https://www.appther.com/llms.txt"
LLMS_FULL_URL = "https://www.appther.com/llms-full.txt"

# ── HTTP ──────────────────────────────────────────────────────────────────────
USER_AGENT = "AppTherChatbotCrawler/1.0 (+https://github.com/YOUR_ORG/appther-chatbot)"
REQUEST_TIMEOUT_SECONDS: float = 30.0

# Polite inter-request delay; overridden by robots.txt Crawl-delay when present.
DEFAULT_CRAWL_DELAY_SECONDS: float = 1.0

MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0  # wait = BACKOFF_BASE ** attempt (seconds)

# ── URL filtering ─────────────────────────────────────────────────────────────
# Tracking query params stripped from every URL before queuing or storing.
# Also strips anything matching the "utm_*" prefix even if not in this set.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "msclkid",
    }
)

# ── Page-type patterns (regex on URL path, first match wins) ──────────────────
# Tag is stored per-chunk for metadata filtering and display in citations.
PAGE_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"^/$", "home"),
    (r"^/faq(/|$)", "faq"),
    (r"^/case-study/", "case-study"),
    (r"^/blog/", "blog"),
    (r"^/services/", "service"),
    (r"^/industry/", "industry"),
    (r"^/industries/", "industry"),  # legacy path that 301s to /industry/
    (r"^/hire-", "hire"),
    (r"^/(privacy-policy|terms|cookie|legal)", "legal"),
    (r"^/(contact|about|team|company|careers)", "company"),
]

# ── BFS fallback limits ───────────────────────────────────────────────────────
BFS_MAX_DEPTH: int = 4
BFS_MAX_URLS: int = 200

# ── Staging ───────────────────────────────────────────────────────────────────
STAGING_DIR: str = os.getenv("STAGING_DIR", "staging")

# ── Normalization / dedup (Step 2) ──────────────────────────────────────────────
MINHASH_NUM_PERM: int = 128
NEAR_DUP_THRESHOLD: float = 0.85

# ── Chunking (Step 2) ────────────────────────────────────────────────────────────
CHUNK_MIN_TOKENS: int = 400
CHUNK_MAX_TOKENS: int = 600
CHUNK_OVERLAP_TOKENS: int = 65
# Approximate characters per token for size budgeting (no tokenizer dependency).
CHARS_PER_TOKEN: int = 4

# ── Embeddings (Step 3) ──────────────────────────────────────────────────────────
# Primary provider: Voyage AI
VOYAGE_EMBED_MODEL: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3.5")
VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
VOYAGE_EMBED_DIMS: int = 512  # Matryoshka truncation
# Vectors are STORED full-precision (float). The LanceDB vector column is float32
# and the Jina fallback (jina-embeddings-v3) cannot emit int8 at all, so storing
# provider int8 is intentionally unused. The ~4x (int8-equivalent) memory saving
# is delivered by LanceDB's scalar-quantization index instead -- see the "Vector
# index quantization" knobs below and crawler/index.py. This must stay "float":
# crawler.embed rejects any other value so int8 vectors can never be silently
# mis-stored in the float32 column.
VOYAGE_EMBED_DTYPE: str = "float"
VOYAGE_EMBED_BATCH_SIZE: int = 128  # max texts per API call
VOYAGE_INPUT_TYPE_DOC: str = "document"
VOYAGE_INPUT_TYPE_QUERY: str = "query"

# Standby provider: Jina AI
JINA_EMBED_MODEL: str = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3")
JINA_EMBED_DIMS: int = 512  # Matryoshka truncation to match Voyage
JINA_EMBED_BATCH_SIZE: int = 64
JINA_EMBED_URL: str = "https://api.jina.ai/v1/embeddings"
# Embedding API calls need a longer timeout than the 30s crawl default; used by
# the shared http_client.create_client() that JinaProvider now goes through.
JINA_EMBED_TIMEOUT_SECONDS: float = 120.0

# ── Vector store (Step 3) ────────────────────────────────────────────────────────
LANCE_TABLE_NAME: str = "chunks"
LANCE_JINA_TABLE_NAME: str = "chunks_jina"
# Index metadata key written into a sidecar JSON in the LanceDB directory
LANCE_META_FILENAME: str = ".index_meta.json"

# ── Vector index quantization (Step 3) ───────────────────────────────────────────
# The headline "int8 / ~4x storage" cost lever is realised HERE, not by changing
# the stored dtype: LanceDB applies scalar quantization (SQ) to the float vectors
# at index-build time (IVF_HNSW_SQ), giving an ~4x (int8-equivalent) reduction of
# the in-memory index with near-exact recall (recoverable via refine_factor at
# query time).
#
# An approximate (ANN) index is only worth building once the table is large enough
# that brute-force search is slow AND SQ training is viable. Below
# VECTOR_INDEX_MIN_ROWS we skip it entirely: exact (flat) search is faster and
# 100% accurate at launch scale (~3k chunks, a few MB), so quantization there
# would add complexity for no measurable gain. The lever pays off on the
# OpenSearch/100k+ graduation path.
VECTOR_INDEX_ENABLED: bool = True
VECTOR_INDEX_MIN_ROWS: int = 2048
VECTOR_INDEX_TYPE: str = "IVF_HNSW_SQ"
# Distance metric for the ANN index. "l2" matches the default flat-search metric so
# ranking is unchanged when the index turns on; for L2-normalised embeddings
# (Voyage/Jina here) l2 and cosine rank identically.
VECTOR_INDEX_METRIC: str = "l2"

# ── Verification (Step 4) ────────────────────────────────────────────────────────
# Alert when ok-page count drops more than this fraction vs the previous run.
DROP_ALERT_THRESHOLD: float = 0.05  # 5%
# Subdirectory under STAGING_DIR where per-run JSON reports are saved.
REPORTS_SUBDIR: str = "reports"
# Filename under STAGING_DIR that lists URLs failing the last fetch (for --targeted re-runs).
FAILURES_FILENAME: str = "failures.jsonl"
# Filename under STAGING_DIR mapping a URL's original canonical form to the
# canonical form it was indexed under after redirects were followed.  Lets
# de-index resolve redirect-shifted URLs that later return 404/410 at their
# original address (see crawler.verify.update_url_aliases / C3).
URL_ALIASES_FILENAME: str = "url_aliases.json"
