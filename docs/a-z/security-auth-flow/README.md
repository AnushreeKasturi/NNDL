# Security & Auth Flow

## Identity flow
- Passwords are hashed with bcrypt on registration.
- Login verifies password and issues signed JWT.
- Protected routes decode JWT and load current user.

## Required secrets
- `JWT_SECRET_KEY`: strong random secret for token signing.
- `DATABASE_URL`: DB credentialed connection string.
- `REDIS_URL`: queue backend connection string.

## Recommended controls
- Rotate JWT secrets through secret manager.
- Add token revocation/session list for enterprise posture.
- Add brute-force protection on login endpoints.
- Enforce HTTPS-only transport and secure cookie/session strategy if browser cookies are introduced.

