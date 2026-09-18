# Yutki-Samadhata: Local Authentication Architecture

## 1. Security Mandates & Principles

1. **Zero External Dependencies:**
   - No web servers, no REST/gRPC endpoints, no cloud authentication, no database engines (no PostgreSQL/MySQL/SQLite required).
   - Local, self-contained single-file state with atomic write semantics.

2. **Zero Plaintext Credentials:**
   - Passwords and recovery codes are **never** stored, cached, or emitted in plaintext anywhere on disk or in log streams.

3. **Modern Cryptographic Standards:**
   - Hashing Algorithm: **Argon2id** (RFC 9106 recommended profile: memory cost 19,456 KiB, time cost 2 iterations, parallelism 1 lane, random 16-byte cryptographically secure salt).
   - Random Generation: OS cryptographically secure random number generator (`rand::rngs::OsRng`).

---

## 2. Storage Layout & Data Schema

Auth state is stored in a local directory with restricted user permissions (e.g., `~/.yutki/auth_store.json` or `$YUTKI_HOME/auth_store.json`):

```json
{
  "version": 1,
  "users": {
    "engineer_admin": {
      "username": "engineer_admin",
      "password_hash": "$argon2id$v=19$m=19456,t=2,p=1$...",
      "recovery_code_hash": "$argon2id$v=19$m=19456,t=2,p=1$...",
      "created_at_utc": 1773835200,
      "last_login_utc": 1773838800
    }
  }
}
```

---

## 3. Workflows

### 3.1 Signup Flow
```mermaid
sequenceDiagram
    actor User
    participant CLI as Yutki CLI
    participant Auth as Auth Engine
    participant Store as Local Storage

    User->>CLI: Enter desired username & password
    CLI->>Auth: Validate password complexity
    Auth->>Auth: Generate 16-char base32 recovery code (OsRng)
    Auth->>Auth: Hash password with Argon2id
    Auth->>Auth: Hash recovery code with Argon2id
    Auth->>Store: Atomic persist {username, password_hash, recovery_code_hash}
    Store-->>CLI: Success
    CLI-->>User: Display recovery code in framed warning box ONCE
```

### 3.2 Login Flow
```mermaid
sequenceDiagram
    actor User
    participant CLI as Yutki CLI
    participant Auth as Auth Engine
    participant Store as Local Storage

    User->>CLI: Enter username & password
    CLI->>Auth: Authenticate(username, password)
    Auth->>Store: Load record for username
    Auth->>Auth: Verify password against Argon2id hash
    alt Match
        Auth-->>CLI: Ok(SessionToken)
        CLI-->>User: Enter Solver Dashboard
    else Mismatch / Not Found
        Auth-->>CLI: Err(InvalidCredentials)
        CLI-->>User: Display error message
    end
```

### 3.3 Forgot Password Flow
```mermaid
sequenceDiagram
    actor User
    participant CLI as Yutki CLI
    participant Auth as Auth Engine
    participant Store as Local Storage

    User->>CLI: Request Password Reset
    CLI->>User: Prompt for username & recovery code
    User->>CLI: Enter username & recovery code
    CLI->>Auth: VerifyRecoveryCode(username, recovery_code)
    Auth->>Store: Load record
    Auth->>Auth: Verify recovery code against Argon2id hash
    alt Valid Recovery Code
        Auth-->>CLI: Verified
        CLI->>User: Prompt for new password (with confirmation)
        User->>CLI: Enter new password
        CLI->>Auth: UpdatePassword(username, new_password)
        Auth->>Auth: Hash new password with Argon2id
        Auth->>Store: Atomic persist updated record
        Store-->>CLI: Success
        CLI-->>User: Password reset successful! Please log in.
    else Invalid Recovery Code
        Auth-->>CLI: Err(InvalidRecoveryCode)
        CLI-->>User: Verification failed!
    end
```

### 3.4 Logout Flow
- Destroys active in-memory session object.
- Returns terminal to the top-level main menu.
