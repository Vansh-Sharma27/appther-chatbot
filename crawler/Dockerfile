FROM python:3.12-slim-bookworm AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY crawler/ ./crawler/

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -e ".[crawler,embed]" && \
    apt-get purge -y gcc && apt-get autoremove -y && \
    rm -rf /root/.cache/pip /var/lib/apt/lists/*

# ── Runtime image ──────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/crawler ./crawler

ENTRYPOINT ["python", "-m", "crawler.pipeline"]
CMD []
