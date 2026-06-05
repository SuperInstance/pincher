//! Capability tokens — minted, signed, and verified with blake3 keyed hash.
//!
//! For MVP, tokens use blake3's keyed-hash mode as a MAC instead of
//! Ed25519.  This is cryptographically sound and avoids adding extra
//! cryptographic dependency overhead.

use super::manifest::CapabilityManifest;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// A signed capability token that authorises a reflex to execute.
///
/// The token contains a [`CapabilityManifest`] declaring what the reflex
/// may do, plus issuance/expiry timestamps and a MAC signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    /// The capability manifest this token authorises.
    pub manifest: CapabilityManifest,
    /// When this token was minted.
    pub issued_at: DateTime<Utc>,
    /// When this token expires.
    pub expires_at: DateTime<Utc>,
    /// blake3 keyed-hash MAC over the serialised manifest + timestamps.
    pub signature: String,
}

impl CapabilityToken {
    /// Mint a new capability token for the given manifest.
    ///
    /// The token is valid for `ttl` from now.  The `secret` is the
    /// system-level key used for MAC computation.
    pub fn mint(manifest: CapabilityManifest, ttl: Duration, secret: &[u8]) -> Self {
        let issued_at = Utc::now();
        let expires_at = issued_at + ttl;
        let signature = compute_mac(&manifest, &issued_at, &expires_at, secret)
            .to_hex()
            .to_string();

        Self {
            manifest,
            issued_at,
            expires_at,
            signature,
        }
    }

    /// Verify the token's MAC and expiry.
    ///
    /// Returns `true` if the signature is valid **and** the token has not
    /// expired.
    pub fn verify(&self, secret: &[u8]) -> bool {
        // Check expiry first.
        if Utc::now() > self.expires_at {
            return false;
        }

        let expected = compute_mac(&self.manifest, &self.issued_at, &self.expires_at, secret);
        blake3::Hash::from_hex(&self.signature)
            .map(|h| h == expected)
            .unwrap_or(false)
    }

    /// Returns `true` if this token has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}

/// Compute a blake3 keyed-hash MAC over the token's content.
fn compute_mac(
    manifest: &CapabilityManifest,
    issued_at: &DateTime<Utc>,
    expires_at: &DateTime<Utc>,
    secret: &[u8],
) -> blake3::Hash {
    let key = blake3::derive_key("pincherOS/capability-token/v1", secret);
    let keyed = blake3::Hasher::new_keyed(&key);

    // We hash the serialised manifest + timestamps in a deterministic way.
    let manifest_json = serde_json::to_string(manifest).unwrap_or_default();
    let issued_str = issued_at.to_rfc3339();
    let expires_str = expires_at.to_rfc3339();

    let mut hasher = keyed;
    hasher.update(manifest_json.as_bytes());
    hasher.update(issued_str.as_bytes());
    hasher.update(expires_str.as_bytes());
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn mint_and_verify_round_trip() {
        let manifest = CapabilityManifest::empty(Uuid::new_v4());
        let secret = b"test-secret-key-1234567890123456";
        let token = CapabilityToken::mint(manifest, Duration::hours(1), secret);
        assert!(token.verify(secret));
    }

    #[test]
    fn wrong_secret_fails() {
        let manifest = CapabilityManifest::empty(Uuid::new_v4());
        let secret_a = b"secret-a-12345678901234567890";
        let secret_b = b"secret-b-12345678901234567890";
        let token = CapabilityToken::mint(manifest, Duration::hours(1), secret_a);
        assert!(!token.verify(secret_b));
    }
}
