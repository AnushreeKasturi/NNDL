# NNDL Legal Risk Classifier

This repository now includes a **production-oriented full application scaffold** for legal clause risk detection:

- **Backend API:** FastAPI + PostgreSQL + JWT auth
- **Async inference:** Redis + RQ worker
- **Frontend:** Next.js (TypeScript)
- **ML core:** Existing BERT / Legal-BERT / Longformer classifier pipeline
- **Deployment:** Docker Compose for cloud/on-prem parity

License: [MIT](./LICENSE)

## Architecture

```text
apps/
  api/
    app/
      main.py
      routes/          # auth, contracts, jobs, health
      services/        # queue + classifier runtime
      models.py        # users/contracts/inference_jobs
      schemas.py
      security.py
      tasks.py         # async job execution
    Dockerfile
    requirements.txt
  web/
    app/               # Next.js UI
    lib/api.ts
    Dockerfile
src/legal_risk_classifier/
  ...                  # training/evaluation/inference pipeline
docker-compose.yml
.env.example
```

## Quick start (full app)

1. Create env file:

```bash
cp .env.example .env
```

2. Start all services:

```bash
docker compose up --build
```

3. Open:

- Web: http://localhost:3000
- API docs: http://localhost:8000/docs

## API summary

- `POST /auth/register`
- `POST /auth/login`
- `POST /contracts` (JWT required)
- `GET /contracts` (JWT required)
- `POST /contracts/{contract_id}/analyze` (JWT required, async queue)
- `GET /jobs`
- `GET /jobs/{job_id}`

## A-Z working documentation

See [docs/a-z](./docs/a-z/README.md) for detailed flow-wise READMEs:
- web page flow
- API flow
- model inference flow
- queue/worker flow
- data/training flow
- deployment flow
- security/auth flow

## Training pipeline (model development)

The research/training tools are still available:

```bash
legal-risk-prepare --cuad_json /path/to/CUAD_v1.json --output_jsonl data/cuad_rows.jsonl
legal-risk-train --rows_jsonl data/cuad_rows.jsonl --model legal-bert --output_dir outputs/legal_bert
legal-risk-eval --model_dir outputs/legal_bert/best_model --rows_jsonl data/cuad_rows.jsonl
```

## Production notes

- Set a strong `JWT_SECRET_KEY` in `.env`.
- For real deployments, replace local volumes with managed Postgres/Redis.
- Add TLS termination and API rate limiting at ingress.
- Integrate object storage and encrypted-at-rest retention for contract documents.

## Environment variables and API keys

`cp .env.example .env` gives you all variables.

**Required for this codebase right now**
- `JWT_SECRET_KEY` (you generate this; no provider key to fetch)
- `DATABASE_URL` (your PostgreSQL connection)
- `REDIS_URL` (your Redis connection)

**Not required right now (optional)**
- `HUGGINGFACE_HUB_TOKEN` only if you use private/gated Hugging Face models
- `GROQ_API_KEY` only if you later add Groq-based LLM features
- `SENTRY_DSN` only if you add Sentry monitoring
