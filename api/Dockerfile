FROM python:3.12-slim-bookworm AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY crawler/ ./crawler/
COPY api/ ./api/

RUN pip install --no-cache-dir -e ".[api,crawler,embed]" && \
    rm -rf /root/.cache/pip

# ── Runtime image ──────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

# AWS Lambda Web Adapter — translates Lambda runtime API to HTTP so uvicorn
# can serve requests with true response streaming (RESPONSE_STREAM mode).
# See: https://github.com/awslabs/aws-lambda-web-adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.0 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/api ./api
COPY --from=builder /app/crawler ./crawler

# Lambda Web Adapter expects the web server on this port
ENV PORT=8080
ENV READINESS_CHECK_PATH=/health

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
