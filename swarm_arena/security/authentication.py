"""Authentication and authorization for swarm arena."""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import jwt
import bcrypt


class UserRole(Enum):
    """User roles with different permissions."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    ARENA_CREATE = "arena_create"
    ARENA_DELETE = "arena_delete"
    ARENA_MODIFY = "arena_modify"
    ARENA_VIEW = "arena_view"
    EXPERIMENT_CREATE = "experiment_create"
    EXPERIMENT_DELETE = "experiment_delete"
    EXPERIMENT_VIEW = "experiment_view"
    SYSTEM_ADMIN = "system_admin"
    DATA_EXPORT = "data_export"


@dataclass
class User:
    """User account information."""
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    password_hash: Optional[str] = None


class AuthenticationManager:
    """Manages user authentication and authorization."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize authentication manager.
        
        Args:
            secret_key: Secret key for JWT tokens (generated if not provided)
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Default role permissions
        self.role_permissions = {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.RESEARCHER: [
                Permission.ARENA_CREATE,
                Permission.ARENA_MODIFY,
                Permission.ARENA_VIEW,
                Permission.EXPERIMENT_CREATE,
                Permission.EXPERIMENT_VIEW,
                Permission.DATA_EXPORT
            ],
            UserRole.VIEWER: [
                Permission.ARENA_VIEW,
                Permission.EXPERIMENT_VIEW
            ],
            UserRole.GUEST: [
                Permission.ARENA_VIEW
            ]
        }
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_password = secrets.token_urlsafe(16)
        
        self.create_user(
            username="admin",
            email="admin@swarm-arena.local",
            password=admin_password,
            role=UserRole.ADMIN
        )
        
        print(f"🔐 Default admin created: username='admin', password='{admin_password}'")
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, 
                   username: str,
                   email: str,
                   password: str,
                   role: UserRole = UserRole.GUEST) -> User:
        """Create a new user account.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password
            role: User role
            
        Returns:
            Created user object
        """
        if username in self.users:
            raise ValueError(f"Username '{username}' already exists")
        
        # Validate inputs
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if not email or '@' not in email:
            raise ValueError("Valid email required")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        # Create user
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions[role].copy(),
            created_at=time.time(),
            password_hash=self.hash_password(password)
        )
        
        self.users[username] = user
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        user = self.users.get(username)
        
        if not user or not user.is_active:
            return None
        
        if not user.password_hash or not self.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = time.time()
        
        # Create JWT token
        payload = {
            'username': username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'iat': time.time(),
            'exp': time.time() + 3600  # 1 hour expiry
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store session
        session_id = secrets.token_hex(16)
        self.sessions[session_id] = {
            'username': username,
            'token': token,
            'created_at': time.time()
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if user still exists and is active
            username = payload.get('username')
            user = self.users.get(username)
            
            if not user or not user.is_active:
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, token: str, permission: Permission) -> bool:
        """Check if token has specific permission.
        
        Args:
            token: JWT token
            permission: Permission to check
            
        Returns:
            True if permission granted
        """
        payload = self.verify_token(token)
        
        if not payload:
            return False
        
        permissions = payload.get('permissions', [])
        return permission.value in permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission.
        
        Args:
            permission: Required permission
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract token from kwargs or context
                token = kwargs.get('auth_token')
                
                if not token or not self.check_permission(token, permission):
                    raise PermissionError(f"Permission required: {permission.value}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def logout(self, token: str) -> bool:
        """Logout user by invalidating token.
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if logout successful
        """
        # Find and remove session
        for session_id, session in list(self.sessions.items()):
            if session.get('token') == token:
                del self.sessions[session_id]
                return True
        
        return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (without sensitive data).
        
        Args:
            username: Username
            
        Returns:
            User information dictionary
        """
        user = self.users.get(username)
        
        if not user:
            return None
        
        return {
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'created_at': user.created_at,
            'last_login': user.last_login,
            'is_active': user.is_active
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (admin only).
        
        Returns:
            List of user information
        """
        return [self.get_user_info(username) for username in self.users.keys()]
    
    def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Update user role and permissions.
        
        Args:
            username: Username
            new_role: New role
            
        Returns:
            True if update successful
        """
        user = self.users.get(username)
        
        if not user:
            return False
        
        user.role = new_role
        user.permissions = self.role_permissions[new_role].copy()
        
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account.
        
        Args:
            username: Username
            
        Returns:
            True if deactivation successful
        """
        user = self.users.get(username)
        
        if not user:
            return False
        
        user.is_active = False
        
        # Invalidate all user sessions
        for session_id, session in list(self.sessions.items()):
            if session.get('username') == username:
                del self.sessions[session_id]
        
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            # Check if session is older than 24 hours
            if current_time - session.get('created_at', 0) > 86400:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)


class APIKeyManager:
    """Manages API keys for programmatic access."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, 
                        username: str,
                        permissions: List[Permission],
                        expires_in: Optional[int] = None) -> str:
        """Generate API key for user.
        
        Args:
            username: Username
            permissions: List of permissions
            expires_in: Expiry time in seconds (None for no expiry)
            
        Returns:
            Generated API key
        """
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        
        key_data = {
            'username': username,
            'permissions': [p.value for p in permissions],
            'created_at': time.time(),
            'expires_at': time.time() + expires_in if expires_in else None,
            'is_active': True
        }
        
        self.api_keys[api_key] = key_data
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data.
        
        Args:
            api_key: API key to verify
            
        Returns:
            Key data if valid, None otherwise
        """
        key_data = self.api_keys.get(api_key)
        
        if not key_data or not key_data.get('is_active'):
            return None
        
        # Check expiry
        expires_at = key_data.get('expires_at')
        if expires_at and time.time() > expires_at:
            key_data['is_active'] = False
            return None
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revocation successful
        """
        key_data = self.api_keys.get(api_key)
        
        if not key_data:
            return False
        
        key_data['is_active'] = False
        return True


class SecurityAuditor:
    """Security auditing and logging."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    def log_event(self, 
                  event_type: str,
                  username: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  ip_address: Optional[str] = None) -> None:
        """Log security event.
        
        Args:
            event_type: Type of event
            username: Username involved
            details: Additional details
            ip_address: Client IP address
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'username': username,
            'details': details or {},
            'ip_address': ip_address
        }
        
        self.audit_log.append(event)
        
        # Keep only last 10000 events to prevent memory issues
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events.
        
        Args:
            limit: Maximum number of events
            
        Returns:
            List of recent events
        """
        return self.audit_log[-limit:]
    
    def get_failed_login_attempts(self, 
                                 username: Optional[str] = None,
                                 time_window: int = 3600) -> List[Dict[str, Any]]:
        """Get failed login attempts.
        
        Args:
            username: Filter by username (optional)
            time_window: Time window in seconds
            
        Returns:
            List of failed login attempts
        """
        current_time = time.time()
        failed_attempts = []
        
        for event in self.audit_log:
            if (event.get('event_type') == 'login_failed' and
                current_time - event.get('timestamp', 0) <= time_window):
                
                if username is None or event.get('username') == username:
                    failed_attempts.append(event)
        
        return failed_attempts