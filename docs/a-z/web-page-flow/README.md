# Web Page Flow

## 1. Entry
- User opens Next.js app (`apps/web/app/page.tsx`).
- UI renders auth, contract intake, contract list, jobs list, and latest result.

## 2. Authentication
- `Register` calls `POST /auth/register`.
- `Login` calls `POST /auth/login`.
- JWT token is stored in client state and attached as `Authorization: Bearer <token>`.

## 3. Contract submission
- User enters title + contract text.
- UI calls `POST /contracts`.
- Newly created contract is shown in Contracts section.

## 4. Analysis trigger
- User clicks `Analyze` on a contract.
- UI calls `POST /contracts/{contract_id}/analyze` with selected model.
- API returns queued job metadata.

## 5. Result retrieval
- User clicks `Refresh Jobs`.
- UI calls `GET /jobs`.
- On `succeeded`, UI prints JSON result payload.

