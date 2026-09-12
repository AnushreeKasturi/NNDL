# API Flow

## FastAPI bootstrap
- App starts in `apps/api/app/main.py`.
- CORS is enabled and DB tables are initialized at startup.

## Auth endpoints
- `POST /auth/register`: creates user with bcrypt-hashed password.
- `POST /auth/login`: validates credentials and returns JWT token.

## Protected endpoints
- `get_current_user` decodes JWT and resolves the user.
- `POST /contracts`: saves contract text to PostgreSQL.
- `GET /contracts`: fetches current user contracts.
- `POST /contracts/{id}/analyze`: creates inference job and enqueues worker task.
- `GET /jobs`, `GET /jobs/{id}`: poll async analysis state and result.

## Data persistence
- SQLAlchemy models:
  - `users`
  - `contracts`
  - `inference_jobs`

