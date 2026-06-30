# Appther RAG Chatbot

24/7 AI support chatbot for appther.com — serverless-first RAG pipeline on AWS.

**Architecture:** weekly crawler → Voyage embeddings → LanceDB-on-S3 → Lambda (hybrid retrieve + rerank) → Amazon Bedrock (Nova 2 Lite) → streaming React widget.  
**Cost:** ~$16–20/month post-free-tier, $0 idle.

## Repo layout

```
crawler/    ingestion pipeline (discovery, fetch, clean, chunk, embed, index)
api/        FastAPI RAG endpoint + streaming Lambda handler
            ├── main.py          FastAPI app, streaming /chat, /feedback, /lead, /health
            │                    Security: X-API-Key auth (optional), slowapi rate limiting,
            │                    CORS origin whitelist, CSP/HSTS/XFO security headers
            ├── state.py         DynamoDB-backed answer cache, feedback, leads, content-gap log
            └── rag/             RAG query core (rewrite, embed, retrieve, rerank, generate)
widget/     embeddable React chat widget (Step 7 — bundled as dist/widget.js, 44 kB)
infra/      Terraform (S3, DynamoDB, Lambda, CloudFront, WAF, ECR, Secrets)
eval/       golden Q&A set + RAGAS harness + jailbreak probes (Step 10)
plans/      implementation notes and ADRs
```

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.12 |
| Node | 20 |
| Terraform | ≥ 1.6 |
| Docker | any recent |
| AWS CLI | configured with `us-east-1` access |

## Provider API keys

Store real values via:
```bash
aws secretsmanager put-secret-value \
  --secret-id appther-chatbot/voyage-api-key \
  --secret-string '{"api_key":"<YOUR_KEY>"}'
```

| Secret path | Provider | Used for |
|---|---|---|
| `appther-chatbot/voyage-api-key` | [Voyage AI](https://www.voyageai.com) | Embeddings (`voyage-3.5`) + reranking (`rerank-2.5`) |
| `appther-chatbot/jina-api-key` | [Jina AI](https://jina.ai) | Fallback/standby embeddings (`jina-embeddings-v3`) |

> **LLM inference (Bedrock):** no API key needed — the Lambda IAM role is granted `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` directly. Enable model access for **Amazon Nova 2 Lite** and **NVIDIA Nemotron 3 Super 120B** in the Bedrock console (us-east-1) before deploying.

## Terraform (scaffold / local validation)

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # fill in bucket_suffix + email
terraform init -backend=false                  # CI validation (no AWS needed)
terraform validate
# terraform apply                              # requires AWS credentials
```

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[crawler,embed,api,dev]"
pre-commit install
pytest -q   # 462+ tests across crawler (Steps 1–4) and api (Steps 5–6)
```

### Running the API locally

```bash
# Requires VOYAGE_API_KEY and a built LanceDB index
export LANCE_INDEX_URI=./lance_index
export DYNAMODB_TABLE=appther-chatbot-main
# API key auth is disabled when API_AUTH_KEY is empty (dev mode).
# Set a key to enable:
# export API_AUTH_KEY=your-secret-key
# Rate limiting is set via RATE_LIMIT (default: 10/minute). Set to "1000/minute" to relax in dev:
# export RATE_LIMIT=1000/minute
uvicorn api.main:app --reload --port 8000
```

Then:

```bash
curl -N http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question": "What does Appther do?"}'
```

## Running the crawl pipeline

Full pipeline (discovers URLs, fetches, embeds, updates index, reconciles):

```bash
# Requires VOYAGE_API_KEY, JINA_API_KEY, and a LanceDB-accessible S3 URI
export LANCE_INDEX_URI=s3://your-bucket/lance_index
python -m crawler.pipeline
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Discovery only — no fetches, no I/O |
| `--skip-embed` | Fetch + verify without touching the index |
| `--targeted` | Re-fetch only URLs from the previous run's `failures.jsonl` |
| `--staging-dir DIR` | Override output directory (default: `staging/`) |
| `-v` | Enable DEBUG logging |

After each run a JSON report is written to `staging/reports/report_<timestamp>.json` and a `staging/failures.jsonl` lists any pages that failed.

## GitHub Actions cron

The weekly crawl (`0 2 * * 0` — Sundays 02:00 UTC) is defined in `.github/workflows/crawl.yml`. Set the following repository secrets before the first run:

| Secret | Purpose |
|---|---|
| `VOYAGE_API_KEY` | Voyage AI embeddings + rerank |
| `JINA_API_KEY` | Jina standby embeddings |
| `AWS_CRAWLER_ROLE_ARN` | IAM role ARN for OIDC-based S3 access (replaces static AWS keys) |
| `LANCE_INDEX_URI` | LanceDB S3 URI (e.g. `s3://bucket/lance_index`) |

The workflow uses GitHub OIDC to assume the crawler IAM role (`AWS_CRAWLER_ROLE_ARN`) instead of static AWS credentials. Before the first run:
1. Create the GitHub OIDC provider in your AWS account (one-time setup).
2. Deploy infra via Terraform to create the crawler IAM role with OIDC trust.
3. Set `AWS_CRAWLER_ROLE_ARN` to the role's ARN in GitHub secrets.

## Implementation steps

| Step | Status | Description |
|------|--------|-------------|
| 0 | ✅ | Project scaffold & infra foundation |
| 1 | ✅ | Crawler: discovery + fetch |
| 2 | ✅ | Clean, normalize, dedupe & chunk |
| 3 | ✅ | Embeddings & LanceDB index on S3 |
| 4 | ✅ | Crawl verification & refresh scheduling |
| 5 | ✅ | RAG query core (rewrite, retrieve, rerank, generate) |
| 6 | ✅ | **FastAPI Lambda endpoint + guardrails & state** |
| 7 | ✅ | Embeddable React chat widget |
| 8 | ⬜ | Deployment & edge security (CloudFront + WAF) |
| 9 | ⬜ | Observability & cost controls |
| 10 | ⬜ | Evaluation & hardening gate |
