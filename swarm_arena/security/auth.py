"""Authentication and authorization module for Swarm Arena."""

import secrets
import hashlib
import hmac
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import base64

from ..exceptions import SecurityError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class APIKey:
    """Represents an API key for authentication."""
    key_id: str
    hashed_key: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class SecureTokenManager:
    """Manages secure tokens and API keys."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self._salt = secrets.token_bytes(32)
        
    def create_api_key(self, permissions: List[str], expires_in_days: int = 30) -> str:
        """Create a new API key with specified permissions."""
        try:
            # Generate secure random key
            raw_key = secrets.token_urlsafe(32)
            key_id = secrets.token_hex(16)
            
            # Hash the key for storage
            hashed_key = self._hash_key(raw_key)
            
            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Store API key info
            api_key = APIKey(
                key_id=key_id,
                hashed_key=hashed_key,
                permissions=permissions.copy(),
                created_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            self.api_keys[key_id] = api_key
            
            logger.info(f"Created API key {key_id} with permissions: {permissions}")
            
            # Return the raw key (only time it's visible)
            return f"{key_id}.{raw_key}"
            
        except Exception as e:
            raise SecurityError(f"Failed to create API key: {str(e)}")
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return its info if valid."""
        try:
            # Parse key format: key_id.raw_key
            if "." not in key:
                return None
                
            key_id, raw_key = key.split(".", 1)
            
            # Check if key exists
            if key_id not in self.api_keys:
                logger.warning(f"Unknown API key ID: {key_id}")
                return None
            
            api_key = self.api_keys[key_id]
            
            # Check if key is active
            if not api_key.is_active:
                logger.warning(f"Inactive API key: {key_id}")
                return None
            
            # Check expiration
            if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                logger.warning(f"Expired API key: {key_id}")
                api_key.is_active = False
                return None
            
            # Verify key hash
            if not self._verify_key(raw_key, api_key.hashed_key):
                logger.warning(f"Invalid API key hash: {key_id}")
                return None
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = False
                logger.info(f"Revoked API key: {key_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to revoke API key {key_id}: {str(e)}")
            return False
    
    def cleanup_expired_keys(self) -> int:
        """Remove expired API keys and return count removed."""
        try:
            now = datetime.utcnow()
            expired_keys = []
            
            for key_id, api_key in self.api_keys.items():
                if api_key.expires_at and now > api_key.expires_at:
                    expired_keys.append(key_id)
            
            for key_id in expired_keys:
                del self.api_keys[key_id]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {str(e)}")
            return 0
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.pbkdf2_hex(
            raw_key.encode('utf-8'),
            self._salt,
            100000,  # iterations
            dklen=32
        )
    
    def _verify_key(self, raw_key: str, hashed_key: str) -> bool:
        """Verify a raw key against its hash."""
        try:
            computed_hash = self._hash_key(raw_key)
            return hmac.compare_digest(computed_hash, hashed_key)
        except Exception:
            return False


class PermissionChecker:
    """Handles permission checking for operations."""
    
    PERMISSIONS = {
        "simulation.run": "Run simulations",
        "simulation.create": "Create new simulations",
        "simulation.delete": "Delete simulations",
        "agents.modify": "Modify agent configurations",
        "data.read": "Read simulation data",
        "data.write": "Write simulation data",
        "admin.full": "Full administrative access",
        "monitoring.read": "Read monitoring data",
        "benchmarks.run": "Run benchmarks",
    }
    
    @classmethod
    def has_permission(cls, api_key: APIKey, required_permission: str) -> bool:
        """Check if an API key has the required permission."""
        try:
            # Admin permission grants everything
            if "admin.full" in api_key.permissions:
                return True
            
            # Check specific permission
            if required_permission in api_key.permissions:
                return True
            
            # Check wildcard permissions
            permission_parts = required_permission.split(".")
            for i in range(len(permission_parts)):
                wildcard = ".".join(permission_parts[:i+1]) + ".*"
                if wildcard in api_key.permissions:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check error: {str(e)}")
            return False
    
    @classmethod
    def get_permission_description(cls, permission: str) -> str:
        """Get human-readable description of a permission."""
        return cls.PERMISSIONS.get(permission, "Unknown permission")


class SecurityValidator:
    """Validates security configurations and inputs."""
    
    @staticmethod
    def validate_simulation_config(config: Dict[str, Any]) -> bool:
        """Validate simulation configuration for security issues."""
        try:
            # Check for reasonable agent limits
            num_agents = config.get("num_agents", 0)
            if num_agents > 10000:
                raise SecurityError(f"Too many agents requested: {num_agents} (max: 10000)")
            
            # Check arena size limits
            arena_size = config.get("arena_size", (1000, 1000))
            if isinstance(arena_size, (list, tuple)) and len(arena_size) == 2:
                if arena_size[0] > 50000 or arena_size[1] > 50000:
                    raise SecurityError(f"Arena too large: {arena_size} (max: 50000x50000)")
            
            # Check episode length
            episode_length = config.get("episode_length", 1000)
            if episode_length > 100000:
                raise SecurityError(f"Episode too long: {episode_length} (max: 100000)")
            
            # Validate file paths if present
            if "output_file" in config:
                SecurityValidator.validate_file_path(config["output_file"])
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"Security validation failed: {str(e)}")
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security issues."""
        try:
            # Check for path traversal attempts
            if ".." in file_path or file_path.startswith("/"):
                raise SecurityError(f"Invalid file path: {file_path}")
            
            # Check for executable extensions
            dangerous_extensions = [".exe", ".bat", ".sh", ".py", ".js"]
            if any(file_path.lower().endswith(ext) for ext in dangerous_extensions):
                raise SecurityError(f"Dangerous file extension: {file_path}")
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"File path validation failed: {str(e)}")
    
    @staticmethod
    def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks."""
        try:
            # Limit length
            if len(user_input) > max_length:
                user_input = user_input[:max_length]
            
            # Remove potentially dangerous characters
            dangerous_chars = ["<", ">", "&", "\"", "'", ";", "|", "&", "$"]
            for char in dangerous_chars:
                user_input = user_input.replace(char, "")
            
            return user_input.strip()
            
        except Exception as e:
            logger.error(f"Input sanitization failed: {str(e)}")
            return ""


class AuditLogger:
    """Logs security-relevant events for auditing."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        self.max_log_size = 10000
    
    def log_event(self, event_type: str, user_id: Optional[str], 
                  action: str, resource: str, success: bool, **kwargs) -> None:
        """Log a security event."""
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "success": success,
                "details": kwargs
            }
            
            self.audit_log.append(event)
            
            # Rotate log if too large
            if len(self.audit_log) > self.max_log_size:
                self.audit_log = self.audit_log[-self.max_log_size//2:]
            
            # Log to file logger as well
            logger.info(f"AUDIT: {event_type} - {action} on {resource} - {'SUCCESS' if success else 'FAILURE'}")
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self.audit_log[-limit:]
    
    def export_audit_log(self) -> str:
        """Export audit log as JSON string."""
        try:
            return json.dumps(self.audit_log, indent=2)
        except Exception as e:
            logger.error(f"Failed to export audit log: {str(e)}")
            return "[]"


# Global instances
token_manager = SecureTokenManager()
audit_logger = AuditLogger()


def require_permission(required_permission: str):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, you'd extract the API key from request context
            # For now, this is a placeholder for the pattern
            api_key = kwargs.get("api_key")
            if not api_key or not PermissionChecker.has_permission(api_key, required_permission):
                raise SecurityError(f"Permission denied: {required_permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator