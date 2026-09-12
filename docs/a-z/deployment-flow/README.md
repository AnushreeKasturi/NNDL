# Deployment Flow (Cloud + On-Prem)

## Local parity stack
- `docker-compose.yml` starts:
  - `postgres`
  - `redis`
  - `api`
  - `worker`
  - `web`

## Cloud deployment pattern
- Split services into separate deploy units:
  - API container
  - Worker container
  - Managed PostgreSQL
  - Managed Redis
  - Web frontend
- Use same env contract (`.env.example`) across environments.

## On-prem deployment pattern
- Same containers can run in:
  - Docker Compose
  - Kubernetes
  - VM-based Docker runtime
- Keep DB and Redis inside private network segments.

## Production hardening
- Add ingress TLS and WAF/rate limiting.
- Use managed secrets for JWT and credentials.
- Add centralized logs, metrics, and alerting.
- Add encrypted storage policy for contract retention.

