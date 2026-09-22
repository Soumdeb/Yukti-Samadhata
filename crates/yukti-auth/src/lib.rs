//! Sovereign local authentication engine utilizing Argon2id password hashing
//! and cryptographically secure single-use recovery codes.
//! Zero network, zero external databases. Never stores plaintext passwords.

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum AuthError {
    #[error("User '{0}' already exists")]
    UserAlreadyExists(String),

    #[error("User '{0}' not found")]
    UserNotFound(String),

    #[error("Invalid password")]
    InvalidPassword,

    #[error("Invalid recovery code")]
    InvalidRecoveryCode,

    #[error("Local auth storage error: {0}")]
    StorageError(String),

    #[error("Corrupted local auth storage file: {0}")]
    CorruptedDatabase(String),

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("Password is too short (minimum 8 characters required)")]
    PasswordTooShort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecord {
    pub username: String,
    pub password_hash: String,
    pub recovery_code_hash: String,
    pub created_at_utc: u64,
    pub last_login_utc: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthDatabase {
    pub version: u32,
    pub users: HashMap<String, UserRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub username: String,
    pub token: String,
}

pub struct LocalAuthManager {
    storage_path: PathBuf,
}

impl LocalAuthManager {
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Self {
        Self {
            storage_path: storage_path.as_ref().to_path_buf(),
        }
    }

    pub fn default_storage_path() -> PathBuf {
        if let Some(app_data) = std::env::var_os("APPDATA") {
            PathBuf::from(app_data)
                .join("yukti-samadhata")
                .join("auth_store.json")
        } else if let Some(home) = dirs_home_dir() {
            home.join(".yukti").join("auth_store.json")
        } else {
            PathBuf::from("auth_store.json")
        }
    }

    pub fn load_db(&self) -> Result<AuthDatabase, AuthError> {
        if !self.storage_path.exists() {
            return Ok(AuthDatabase {
                version: 1,
                users: HashMap::new(),
            });
        }
        let file = File::open(&self.storage_path)
            .map_err(|e| AuthError::StorageError(format!("Failed to open auth database: {e}")))?;
        let reader = BufReader::new(file);
        let db: AuthDatabase = serde_json::from_reader(reader).map_err(|e| {
            AuthError::CorruptedDatabase(format!("Invalid JSON in auth database: {e}"))
        })?;
        Ok(db)
    }

    pub fn save_db(&self, db: &AuthDatabase) -> Result<(), AuthError> {
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                AuthError::StorageError(format!("Failed to create auth directory: {e}"))
            })?;
        }
        let json_data = serde_json::to_string_pretty(db)
            .map_err(|e| AuthError::StorageError(format!("Serialization error: {e}")))?;

        // Atomic write via temp file
        let temp_path = self.storage_path.with_extension("tmp");
        let mut temp_file = File::create(&temp_path).map_err(|e| {
            AuthError::StorageError(format!("Failed to create temp auth file: {e}"))
        })?;
        temp_file
            .write_all(json_data.as_bytes())
            .map_err(|e| AuthError::StorageError(format!("Failed to write temp auth file: {e}")))?;
        temp_file
            .sync_all()
            .map_err(|e| AuthError::StorageError(format!("Failed to sync temp auth file: {e}")))?;

        fs::rename(&temp_path, &self.storage_path)
            .map_err(|e| AuthError::StorageError(format!("Atomic rename failed: {e}")))?;
        Ok(())
    }

    /// Sign up a new user. Returns the one-time generated recovery code and an active session.
    pub fn signup(&self, username: &str, password: &str) -> Result<(String, Session), AuthError> {
        if password.len() < 8 {
            return Err(AuthError::PasswordTooShort);
        }

        let mut db = self.load_db()?;
        let clean_user = username.trim().to_lowercase();
        if clean_user.is_empty() {
            return Err(AuthError::StorageError("Username cannot be empty".into()));
        }
        if db.users.contains_key(&clean_user) {
            return Err(AuthError::UserAlreadyExists(clean_user));
        }

        let password_hash = hash_secret(password)?;
        let recovery_code = generate_recovery_code();
        let recovery_code_hash = hash_secret(&recovery_code)?;

        let now = current_timestamp();
        let record = UserRecord {
            username: clean_user.clone(),
            password_hash,
            recovery_code_hash,
            created_at_utc: now,
            last_login_utc: now,
        };

        db.users.insert(clean_user.clone(), record);
        self.save_db(&db)?;

        let session = Session {
            username: clean_user,
            token: generate_session_token(),
        };

        Ok((recovery_code, session))
    }

    /// Authenticate existing user with username and password.
    pub fn login(&self, username: &str, password: &str) -> Result<Session, AuthError> {
        let mut db = self.load_db()?;
        let clean_user = username.trim().to_lowercase();
        let record = db
            .users
            .get_mut(&clean_user)
            .ok_or_else(|| AuthError::UserNotFound(clean_user.clone()))?;

        if !verify_secret(password, &record.password_hash) {
            return Err(AuthError::InvalidPassword);
        }

        record.last_login_utc = current_timestamp();
        self.save_db(&db)?;

        Ok(Session {
            username: clean_user,
            token: generate_session_token(),
        })
    }

    /// Reset password using recovery code. Single-use: cycles recovery code and returns new code.
    pub fn reset_password(
        &self,
        username: &str,
        recovery_code: &str,
        new_password: &str,
    ) -> Result<String, AuthError> {
        if new_password.len() < 8 {
            return Err(AuthError::PasswordTooShort);
        }

        let mut db = self.load_db()?;
        let clean_user = username.trim().to_lowercase();
        let clean_code = recovery_code.trim().replace('-', "").to_uppercase();

        let record = db
            .users
            .get_mut(&clean_user)
            .ok_or_else(|| AuthError::UserNotFound(clean_user.clone()))?;

        // Verify formatted code or unformatted code
        let matches = verify_secret(&clean_code, &record.recovery_code_hash)
            || verify_secret(recovery_code.trim(), &record.recovery_code_hash);

        if !matches {
            return Err(AuthError::InvalidRecoveryCode);
        }

        let new_password_hash = hash_secret(new_password)?;
        record.password_hash = new_password_hash;
        // Cycle recovery code upon successful reset (single-use invariant)
        let new_recovery_code = generate_recovery_code();
        record.recovery_code_hash = hash_secret(&new_recovery_code)?;
        self.save_db(&db)?;

        Ok(new_recovery_code)
    }

    /// Invalidate session upon logout
    pub fn logout(&self, _session: Session) {
        // In-memory session is dropped; local store remains secure
    }
}

#[cfg(not(test))]
fn get_argon2_instance() -> Argon2<'static> {
    Argon2::default()
}

#[cfg(test)]
fn get_argon2_instance() -> Argon2<'static> {
    // Fast parameters for testing (RFC 9106 test profile: 1024 KiB, 1 iteration, 1 lane)
    use argon2::{Algorithm, Params, Version};
    let params = Params::new(1024, 1, 1, Some(32)).unwrap();
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

fn hash_secret(secret: &str) -> Result<String, AuthError> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = get_argon2_instance();
    let hash = argon2
        .hash_password(secret.as_bytes(), &salt)
        .map_err(|e| AuthError::CryptoError(e.to_string()))?;
    Ok(hash.to_string())
}

fn verify_secret(secret: &str, hash: &str) -> bool {
    if let Ok(parsed_hash) = PasswordHash::new(hash) {
        let argon2 = get_argon2_instance();
        argon2
            .verify_password(secret.as_bytes(), &parsed_hash)
            .is_ok()
    } else {
        false
    }
}

fn generate_recovery_code() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut rng = rand::thread_rng();
    let mut raw = String::with_capacity(16);
    for _ in 0..16 {
        let idx = rng.gen_range(0..CHARSET.len());
        raw.push(CHARSET[idx] as char);
    }
    // Format as XXXX-XXXX-XXXX-XXXX
    format!(
        "{}-{}-{}-{}",
        &raw[0..4],
        &raw[4..8],
        &raw[8..12],
        &raw[12..16]
    )
}

fn generate_session_token() -> String {
    let mut rng = rand::thread_rng();
    let token_bytes: [u8; 16] = rng.gen();
    token_bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn dirs_home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_temp_auth_manager() -> (LocalAuthManager, PathBuf) {
        let temp_dir =
            std::env::temp_dir().join(format!("yukti_test_auth_{}", rand::random::<u64>()));
        let db_path = temp_dir.join("auth_store.json");
        let auth = LocalAuthManager::new(&db_path);
        (auth, temp_dir)
    }

    #[test]
    fn test_signup() {
        let (auth, temp_dir) = create_temp_auth_manager();
        let (rec_code, session) = auth.signup("lead_engineer", "secure_password_99").unwrap();
        assert_eq!(session.username, "lead_engineer");
        assert_eq!(rec_code.len(), 19); // 16 chars + 3 hyphens
        assert!(!session.token.is_empty());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_duplicate_username() {
        let (auth, temp_dir) = create_temp_auth_manager();
        auth.signup("engineer_a", "password1234").unwrap();

        let err = auth.signup("engineer_a", "another_pass_567").unwrap_err();
        assert_eq!(err, AuthError::UserAlreadyExists("engineer_a".into()));

        // Case insensitivity check
        let err_upper = auth.signup("ENGINEER_A", "another_pass_567").unwrap_err();
        assert_eq!(err_upper, AuthError::UserAlreadyExists("engineer_a".into()));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_wrong_password() {
        let (auth, temp_dir) = create_temp_auth_manager();
        auth.signup("analyst", "correct_pass_123").unwrap();

        let err = auth.login("analyst", "wrong_password_999").unwrap_err();
        assert_eq!(err, AuthError::InvalidPassword);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_successful_login() {
        let (auth, temp_dir) = create_temp_auth_manager();
        auth.signup("solver_admin", "sovereign_secret_1").unwrap();

        let session = auth.login("solver_admin", "sovereign_secret_1").unwrap();
        assert_eq!(session.username, "solver_admin");
        assert!(!session.token.is_empty());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_wrong_recovery_code() {
        let (auth, temp_dir) = create_temp_auth_manager();
        let (_rec_code, _) = auth.signup("operator", "secret_pass_1").unwrap();

        let err = auth
            .reset_password("operator", "WRONG-CODE-0000-XXXX", "new_password_888")
            .unwrap_err();
        assert_eq!(err, AuthError::InvalidRecoveryCode);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_password_reset() {
        let (auth, temp_dir) = create_temp_auth_manager();
        let (rec_code, _) = auth.signup("operator2", "initial_password_1").unwrap();

        // Successfully reset password
        let new_rec_code = auth
            .reset_password("operator2", &rec_code, "brand_new_pass_999")
            .unwrap();
        assert_eq!(new_rec_code.len(), 19);
        assert_ne!(new_rec_code, rec_code); // Single-use cycle

        // Old password no longer authenticates
        let err = auth.login("operator2", "initial_password_1").unwrap_err();
        assert_eq!(err, AuthError::InvalidPassword);

        // Old recovery code no longer works (single-use enforcement)
        let rec_err = auth
            .reset_password("operator2", &rec_code, "another_pass_123")
            .unwrap_err();
        assert_eq!(rec_err, AuthError::InvalidRecoveryCode);

        // New password works
        let login_session = auth.login("operator2", "brand_new_pass_999").unwrap();
        assert_eq!(login_session.username, "operator2");

        // New recovery code works
        assert!(auth
            .reset_password("operator2", &new_rec_code, "final_password_777")
            .is_ok());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_corrupted_local_auth_file() {
        let (auth, temp_dir) = create_temp_auth_manager();
        if let Some(parent) = auth.storage_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        // Write invalid JSON into auth file
        fs::write(
            &auth.storage_path,
            b"{ this is corrupt binary data non-json }",
        )
        .unwrap();

        let err = auth.login("anyone", "some_password").unwrap_err();
        assert!(matches!(err, AuthError::CorruptedDatabase(_)));

        let err_signup = auth.signup("anyone", "some_password").unwrap_err();
        assert!(matches!(err_signup, AuthError::CorruptedDatabase(_)));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_short_password_rejection() {
        let (auth, temp_dir) = create_temp_auth_manager();
        let err = auth.signup("shorty", "short").unwrap_err();
        assert_eq!(err, AuthError::PasswordTooShort);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_no_plaintext_secrets_stored_on_disk() {
        let (auth, temp_dir) = create_temp_auth_manager();
        let plain_password = "UltraClassifiedPassword999!";
        let (rec_code, _) = auth.signup("security_officer", plain_password).unwrap();

        let raw_db = fs::read_to_string(&auth.storage_path).unwrap();

        // 1. Plaintext password must NEVER exist in stored database
        assert!(
            !raw_db.contains(plain_password),
            "Plaintext password found in auth storage!"
        );

        // 2. Plaintext recovery code must NEVER exist in stored database
        assert!(
            !raw_db.contains(&rec_code),
            "Plaintext recovery code found in auth storage!"
        );
        let stripped_code = rec_code.replace('-', "");
        assert!(
            !raw_db.contains(&stripped_code),
            "Plaintext stripped recovery code found in auth storage!"
        );

        // 3. Must use salted Argon2id hashes
        assert!(
            raw_db.contains("$argon2id$"),
            "Argon2id identifier missing from auth storage!"
        );

        let _ = fs::remove_dir_all(temp_dir);
    }
}
