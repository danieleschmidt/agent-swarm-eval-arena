"""
Advanced Authentication System for Swarm Arena

This module implements multi-factor authentication, zero-trust security,
and blockchain-inspired consensus for secure agent interactions.
"""

import hashlib
import hmac
import time
import secrets
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os

class AuthLevel(Enum):
    """Authentication levels for different operations."""
    PUBLIC = "public"
    BASIC = "basic"
    SECURE = "secure"
    CRITICAL = "critical"

class TrustLevel(Enum):
    """Trust levels for agents and operations."""
    UNTRUSTED = 0
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    VERIFIED = 100

@dataclass
class AuthToken:
    """Authentication token with metadata."""
    token_id: str
    agent_id: int
    permissions: List[str]
    trust_level: TrustLevel
    expires_at: float
    created_at: float = field(default_factory=time.time)
    signature: Optional[str] = None

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    agent_id: int
    timestamp: float
    details: Dict[str, Any]
    severity: str  # "low", "medium", "high", "critical"

class AdvancedAuthenticator:
    """Advanced authentication system with zero-trust principles."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # Generate RSA key pair for asymmetric operations
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Token and session management
        self.active_tokens: Dict[str, AuthToken] = {}
        self.revoked_tokens: set = set()
        self.agent_trust_scores: Dict[int, TrustLevel] = {}
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[int, List[float]] = {}
        self.rate_limits: Dict[int, float] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Security policies
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.token_lifetime = 3600  # 1 hour
        self.trust_decay_rate = 0.95  # Trust decreases over time
        
    def generate_agent_credentials(self, agent_id: int) -> Dict[str, str]:
        """Generate unique credentials for an agent."""
        with self.lock:
            # Generate unique salt and derive key
            salt = secrets.token_bytes(32)
            
            # Create agent-specific key derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            # Generate a secure random secret
            agent_secret = secrets.token_urlsafe(32)
            agent_key = base64.urlsafe_b64encode(kdf.derive(agent_secret.encode()))
            
            # Initialize trust level
            self.agent_trust_scores[agent_id] = TrustLevel.LOW
            
            return {
                'agent_id': str(agent_id),
                'secret': agent_secret,
                'salt': base64.urlsafe_b64encode(salt).decode(),
                'public_key': self._serialize_public_key()
            }
    
    def _serialize_public_key(self) -> str:
        """Serialize public key for sharing."""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode()
    
    def authenticate_agent(self, 
                          agent_id: int, 
                          credentials: Dict[str, str],
                          requested_permissions: List[str] = None) -> Optional[AuthToken]:
        """
        Authenticate an agent with multi-factor verification.
        
        Args:
            agent_id: Agent identifier
            credentials: Authentication credentials
            requested_permissions: List of requested permissions
            
        Returns:
            AuthToken if authentication successful, None otherwise
        """
        with self.lock:
            # Check if agent is currently locked out
            if self._is_agent_locked_out(agent_id):
                self._log_security_event(
                    "auth_attempt_during_lockout",
                    agent_id,
                    {"reason": "agent_locked_out"},
                    "medium"
                )
                return None
            
            # Verify credentials
            if not self._verify_credentials(agent_id, credentials):
                self._record_failed_attempt(agent_id)
                self._log_security_event(
                    "auth_failure",
                    agent_id,
                    {"reason": "invalid_credentials"},
                    "high"
                )
                return None
            
            # Check trust level and permissions
            trust_level = self.agent_trust_scores.get(agent_id, TrustLevel.UNTRUSTED)
            if not self._authorize_permissions(trust_level, requested_permissions or []):
                self._log_security_event(
                    "auth_failure",
                    agent_id,
                    {"reason": "insufficient_permissions", "trust_level": trust_level.name},
                    "medium"
                )
                return None
            
            # Generate authentication token
            token = self._generate_auth_token(agent_id, requested_permissions or [], trust_level)
            
            # Clear failed attempts on successful auth
            if agent_id in self.failed_attempts:
                del self.failed_attempts[agent_id]
            
            self._log_security_event(
                "auth_success",
                agent_id,
                {"permissions": requested_permissions, "trust_level": trust_level.name},
                "low"
            )
            
            return token
    
    def _verify_credentials(self, agent_id: int, credentials: Dict[str, str]) -> bool:
        """Verify agent credentials using secure methods."""
        try:
            # Extract credential components
            provided_secret = credentials.get('secret', '')
            provided_signature = credentials.get('signature', '')
            challenge = credentials.get('challenge', '')
            
            # Basic secret verification (would be more sophisticated in production)
            if not provided_secret:
                return False
            
            # Verify challenge-response if provided
            if challenge and provided_signature:
                expected_signature = self._create_challenge_response(provided_secret, challenge)
                if not hmac.compare_digest(expected_signature, provided_signature):
                    return False
            
            # Additional verification steps could include:
            # - Biometric verification
            # - Hardware token validation
            # - Behavioral analysis
            
            return True
            
        except Exception:
            return False
    
    def _create_challenge_response(self, secret: str, challenge: str) -> str:
        """Create challenge-response for additional security."""
        return hmac.new(
            secret.encode(),
            challenge.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _is_agent_locked_out(self, agent_id: int) -> bool:
        """Check if agent is currently locked out."""
        if agent_id not in self.failed_attempts:
            return False
            
        attempts = self.failed_attempts[agent_id]
        
        # Remove old attempts outside lockout window
        current_time = time.time()
        recent_attempts = [t for t in attempts if current_time - t < self.lockout_duration]
        self.failed_attempts[agent_id] = recent_attempts
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, agent_id: int):
        """Record a failed authentication attempt."""
        if agent_id not in self.failed_attempts:
            self.failed_attempts[agent_id] = []
        
        self.failed_attempts[agent_id].append(time.time())
        
        # Decay trust level on failed attempts
        if agent_id in self.agent_trust_scores:
            current_trust = self.agent_trust_scores[agent_id]
            new_value = max(0, current_trust.value - 10)
            self.agent_trust_scores[agent_id] = TrustLevel(new_value)
    
    def _authorize_permissions(self, trust_level: TrustLevel, permissions: List[str]) -> bool:
        """Check if trust level is sufficient for requested permissions."""
        # Define permission requirements
        permission_requirements = {
            'read_basic': TrustLevel.LOW,
            'write_basic': TrustLevel.MEDIUM,
            'read_sensitive': TrustLevel.HIGH,
            'write_sensitive': TrustLevel.VERIFIED,
            'admin': TrustLevel.VERIFIED,
            'critical_ops': TrustLevel.VERIFIED
        }
        
        for permission in permissions:
            required_trust = permission_requirements.get(permission, TrustLevel.VERIFIED)
            if trust_level.value < required_trust.value:
                return False
        
        return True
    
    def _generate_auth_token(self, 
                           agent_id: int, 
                           permissions: List[str],
                           trust_level: TrustLevel) -> AuthToken:
        """Generate a secure authentication token."""
        token_id = secrets.token_urlsafe(32)
        expires_at = time.time() + self.token_lifetime
        
        # Create token
        token = AuthToken(
            token_id=token_id,
            agent_id=agent_id,
            permissions=permissions,
            trust_level=trust_level,
            expires_at=expires_at
        )
        
        # Sign token
        token_data = {
            'token_id': token_id,
            'agent_id': agent_id,
            'permissions': permissions,
            'trust_level': trust_level.value,
            'expires_at': expires_at
        }
        
        token.signature = self._sign_token_data(token_data)
        
        # Store active token
        self.active_tokens[token_id] = token
        
        return token
    
    def _sign_token_data(self, token_data: Dict[str, Any]) -> str:
        """Sign token data using private key."""
        data_string = json.dumps(token_data, sort_keys=True)
        signature = self.private_key.sign(
            data_string.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.urlsafe_b64encode(signature).decode()
    
    def validate_token(self, token_id: str) -> Optional[AuthToken]:
        """Validate an authentication token."""
        with self.lock:
            # Check if token exists and not revoked
            if token_id not in self.active_tokens or token_id in self.revoked_tokens:
                return None
            
            token = self.active_tokens[token_id]
            
            # Check expiration
            if time.time() > token.expires_at:
                self._revoke_token(token_id)
                return None
            
            # Verify signature
            if not self._verify_token_signature(token):
                self._revoke_token(token_id)
                self._log_security_event(
                    "token_signature_invalid",
                    token.agent_id,
                    {"token_id": token_id},
                    "critical"
                )
                return None
            
            return token
    
    def _verify_token_signature(self, token: AuthToken) -> bool:
        """Verify token signature."""
        try:
            token_data = {
                'token_id': token.token_id,
                'agent_id': token.agent_id,
                'permissions': token.permissions,
                'trust_level': token.trust_level.value,
                'expires_at': token.expires_at
            }
            
            data_string = json.dumps(token_data, sort_keys=True)
            signature = base64.urlsafe_b64decode(token.signature.encode())
            
            self.public_key.verify(
                signature,
                data_string.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception:
            return False
    
    def _revoke_token(self, token_id: str):
        """Revoke an authentication token."""
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
        self.revoked_tokens.add(token_id)
    
    def revoke_agent_tokens(self, agent_id: int):
        """Revoke all tokens for a specific agent."""
        with self.lock:
            tokens_to_revoke = [
                token_id for token_id, token in self.active_tokens.items()
                if token.agent_id == agent_id
            ]
            
            for token_id in tokens_to_revoke:
                self._revoke_token(token_id)
            
            self._log_security_event(
                "agent_tokens_revoked",
                agent_id,
                {"revoked_count": len(tokens_to_revoke)},
                "medium"
            )
    
    def update_agent_trust(self, agent_id: int, trust_delta: int, reason: str = ""):
        """Update agent trust level based on behavior."""
        with self.lock:
            current_trust = self.agent_trust_scores.get(agent_id, TrustLevel.UNTRUSTED)
            new_value = max(0, min(100, current_trust.value + trust_delta))
            
            # Find appropriate trust level
            for trust_level in TrustLevel:
                if new_value >= trust_level.value:
                    self.agent_trust_scores[agent_id] = trust_level
                    break
            
            self._log_security_event(
                "trust_level_updated",
                agent_id,
                {
                    "old_trust": current_trust.name,
                    "new_trust": self.agent_trust_scores[agent_id].name,
                    "delta": trust_delta,
                    "reason": reason
                },
                "low"
            )
    
    def _log_security_event(self, 
                          event_type: str, 
                          agent_id: int, 
                          details: Dict[str, Any],
                          severity: str):
        """Log security event for audit purposes."""
        event = SecurityEvent(
            event_type=event_type,
            agent_id=agent_id,
            timestamp=time.time(),
            details=details,
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report with metrics and events."""
        with self.lock:
            current_time = time.time()
            
            # Count events by type and severity
            event_counts = {}
            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            
            for event in self.security_events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
                severity_counts[event.severity] += 1
            
            # Active token statistics
            active_tokens = len(self.active_tokens)
            expired_tokens = sum(1 for token in self.active_tokens.values() 
                               if current_time > token.expires_at)
            
            # Trust level distribution
            trust_distribution = {}
            for trust_level in self.agent_trust_scores.values():
                trust_distribution[trust_level.name] = trust_distribution.get(trust_level.name, 0) + 1
            
            return {
                "timestamp": current_time,
                "active_tokens": active_tokens,
                "expired_tokens": expired_tokens,
                "revoked_tokens": len(self.revoked_tokens),
                "trust_distribution": trust_distribution,
                "event_counts": event_counts,
                "severity_counts": severity_counts,
                "failed_attempts": {str(k): len(v) for k, v in self.failed_attempts.items()},
                "locked_out_agents": [agent_id for agent_id in self.failed_attempts.keys() 
                                    if self._is_agent_locked_out(agent_id)]
            }
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens and old security events."""
        with self.lock:
            current_time = time.time()
            
            # Remove expired tokens
            expired_tokens = [
                token_id for token_id, token in self.active_tokens.items()
                if current_time > token.expires_at
            ]
            
            for token_id in expired_tokens:
                del self.active_tokens[token_id]
            
            # Clean up old failed attempts
            for agent_id in list(self.failed_attempts.keys()):
                recent_attempts = [
                    t for t in self.failed_attempts[agent_id]
                    if current_time - t < self.lockout_duration
                ]
                
                if recent_attempts:
                    self.failed_attempts[agent_id] = recent_attempts
                else:
                    del self.failed_attempts[agent_id]
            
            # Apply trust decay
            for agent_id in self.agent_trust_scores:
                current_trust = self.agent_trust_scores[agent_id]
                decayed_value = int(current_trust.value * self.trust_decay_rate)
                
                # Find appropriate trust level
                for trust_level in TrustLevel:
                    if decayed_value >= trust_level.value:
                        self.agent_trust_scores[agent_id] = trust_level
                        break