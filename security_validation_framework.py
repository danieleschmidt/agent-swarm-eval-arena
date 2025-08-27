#!/usr/bin/env python3
"""
Security validation framework with input sanitization, authentication, and vulnerability scanning.
Implements defense-in-depth security measures for production deployment.
"""

import sys
import os
import hashlib
import hmac
import secrets
import time
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robust_health_monitoring import SelfHealingArena


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'onload\s*=',                # Event handlers
        r'onclick\s*=',
        r'onerror\s*=',
        r'eval\s*\(',                 # Code execution
        r'exec\s*\(',
        r'system\s*\(',
        r'import\s+os',               # Python imports
        r'from\s+os\s+import',
        r'subprocess',
        r'__import__',
        r'\.\./',                     # Path traversal
        r'[;&|`]',                    # Command injection
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.DANGEROUS_PATTERNS]
        self.logger = logging.getLogger(__name__)
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Length check
        if len(value) > max_length:
            raise ValueError(f"Input too long: {len(value)} > {max_length}")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                self.logger.warning(f"Blocked dangerous pattern in input: {pattern.pattern}")
                raise ValueError("Input contains dangerous content")
        
        # Basic HTML encoding
        value = value.replace('&', '&amp;')
        value = value.replace('<', '&lt;')
        value = value.replace('>', '&gt;')
        value = value.replace('"', '&quot;')
        value = value.replace("'", '&#x27;')
        
        return value
    
    def sanitize_numeric(self, value: Union[int, float, str], 
                        min_val: float = float('-inf'), 
                        max_val: float = float('inf')) -> float:
        """Sanitize numeric input."""
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError("Input must be numeric")
        
        if not (min_val <= numeric_value <= max_val):
            raise ValueError(f"Numeric value out of range: {numeric_value}")
        
        return numeric_value
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            clean_key = self.sanitize_string(str(key), max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = self.sanitize_string(value)
            elif isinstance(value, (int, float)):
                clean_value = self.sanitize_numeric(value)
            elif isinstance(value, dict):
                clean_value = self.sanitize_config(value)  # Recursive
            elif isinstance(value, list):
                clean_value = [self.sanitize_string(str(item)) if isinstance(item, str)
                             else self.sanitize_numeric(item) if isinstance(item, (int, float))
                             else item for item in value]
            else:
                clean_value = value  # Allow other safe types
            
            sanitized[clean_key] = clean_value
        
        return sanitized


class AuthenticationManager:
    """Secure authentication and authorization system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA-256
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return key.hex(), salt.hex()
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash."""
        try:
            salt_bytes = bytes.fromhex(salt)
            expected_hash, _ = self.hash_password(password, salt_bytes)
            return hmac.compare_digest(expected_hash, hashed_password)
        except Exception:
            return False
    
    def generate_token(self, user_id: str, security_level: SecurityLevel = SecurityLevel.INTERNAL,
                      expires_in: int = 3600) -> str:
        """Generate secure authentication token."""
        token = secrets.token_urlsafe(32)
        
        self.active_tokens[token] = {
            'user_id': user_id,
            'security_level': security_level,
            'created_at': time.time(),
            'expires_at': time.time() + expires_in,
            'last_used': time.time()
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token."""
        if token not in self.active_tokens:
            return None
        
        token_info = self.active_tokens[token]
        
        # Check expiration
        if time.time() > token_info['expires_at']:
            del self.active_tokens[token]
            return None
        
        # Update last used time
        token_info['last_used'] = time.time()
        return token_info
    
    def revoke_token(self, token: str) -> bool:
        """Revoke authentication token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            return True
        return False
    
    def check_rate_limit(self, client_id: str, max_attempts: int = 5, 
                        window_seconds: int = 300) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        
        # Clean old attempts
        if client_id in self.failed_attempts:
            self.failed_attempts[client_id] = [
                attempt for attempt in self.failed_attempts[client_id]
                if current_time - attempt < window_seconds
            ]
        
        # Check current attempts
        attempts = len(self.failed_attempts.get(client_id, []))
        return attempts < max_attempts
    
    def record_failed_attempt(self, client_id: str) -> None:
        """Record failed authentication attempt."""
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
        
        self.failed_attempts[client_id].append(time.time())
        self.logger.warning(f"Failed authentication attempt from {client_id}")


class VulnerabilityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scan_results: List[Dict[str, Any]] = []
    
    def scan_configuration(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan configuration for security vulnerabilities."""
        vulnerabilities = []
        
        # Check for insecure default values
        insecure_defaults = {
            'debug': True,
            'allow_all_origins': True,
            'disable_authentication': True,
            'log_level': 'DEBUG'
        }
        
        for key, insecure_value in insecure_defaults.items():
            if config.get(key) == insecure_value:
                vulnerabilities.append({
                    'type': 'insecure_configuration',
                    'severity': ThreatLevel.MEDIUM,
                    'description': f"Insecure default value for '{key}'",
                    'recommendation': f"Change {key} to a secure value"
                })
        
        # Check for weak encryption settings
        if 'encryption' in config:
            encryption_config = config['encryption']
            if encryption_config.get('algorithm') in ['md5', 'sha1']:
                vulnerabilities.append({
                    'type': 'weak_cryptography',
                    'severity': ThreatLevel.HIGH,
                    'description': "Weak cryptographic algorithm detected",
                    'recommendation': "Use SHA-256 or stronger algorithms"
                })
        
        # Check for exposed secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']*["\']',
            r'secret\s*=\s*["\'][^"\']*["\']',
            r'api_key\s*=\s*["\'][^"\']*["\']'
        ]
        
        config_str = json.dumps(config, default=str)
        for pattern in secret_patterns:
            if re.search(pattern, config_str, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'exposed_secret',
                    'severity': ThreatLevel.HIGH,
                    'description': "Potential secret exposed in configuration",
                    'recommendation': "Use environment variables for secrets"
                })
        
        self.scan_results.extend(vulnerabilities)
        return vulnerabilities
    
    def scan_permissions(self, permissions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan permission configurations."""
        vulnerabilities = []
        
        # Check for overly permissive settings
        dangerous_permissions = ['*', 'all', 'admin', 'root']
        
        for permission, value in permissions.items():
            if value in dangerous_permissions:
                vulnerabilities.append({
                    'type': 'excessive_permissions',
                    'severity': ThreatLevel.HIGH,
                    'description': f"Overly permissive setting: {permission}={value}",
                    'recommendation': "Use principle of least privilege"
                })
        
        self.scan_results.extend(vulnerabilities)
        return vulnerabilities
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.scan_results:
            return {
                'status': 'secure',
                'vulnerabilities': [],
                'risk_level': ThreatLevel.LOW.value,
                'recommendations': []
            }
        
        # Count by severity
        severity_counts = {level.value: 0 for level in ThreatLevel}
        for vuln in self.scan_results:
            severity_counts[vuln['severity'].value] += 1
        
        # Determine overall risk level
        if severity_counts['critical'] > 0:
            risk_level = ThreatLevel.CRITICAL
        elif severity_counts['high'] > 0:
            risk_level = ThreatLevel.HIGH
        elif severity_counts['medium'] > 0:
            risk_level = ThreatLevel.MEDIUM
        else:
            risk_level = ThreatLevel.LOW
        
        return {
            'status': 'vulnerable' if risk_level != ThreatLevel.LOW else 'secure',
            'vulnerabilities': [
                {
                    'type': vuln['type'],
                    'severity': vuln['severity'].value,
                    'description': vuln['description'],
                    'recommendation': vuln['recommendation']
                }
                for vuln in self.scan_results
            ],
            'risk_level': risk_level.value,
            'severity_breakdown': severity_counts,
            'total_vulnerabilities': len(self.scan_results)
        }


class SecureArena(SelfHealingArena):
    """Arena with comprehensive security features."""
    
    def __init__(self, num_agents: int = 10, arena_size: tuple = (1000, 1000),
                 enable_security: bool = True):
        super().__init__(num_agents, arena_size)
        
        self.security_enabled = enable_security
        if self.security_enabled:
            self.input_sanitizer = InputSanitizer()
            self.auth_manager = AuthenticationManager()
            self.vulnerability_scanner = VulnerabilityScanner()
            
            # Setup security monitoring
            self._setup_security_monitoring()
        
        self.security_events: List[Dict[str, Any]] = []
    
    def _setup_security_monitoring(self) -> None:
        """Setup security-specific monitoring."""
        
        def check_authentication_health():
            """Monitor authentication system health."""
            # Check for too many active tokens
            active_tokens = len(self.auth_manager.active_tokens)
            if active_tokens > 100:
                return HealthStatus.DEGRADED
            
            # Check for rate limiting activity
            total_failed_attempts = sum(len(attempts) for attempts 
                                      in self.auth_manager.failed_attempts.values())
            if total_failed_attempts > 50:
                return HealthStatus.CRITICAL
            
            return HealthStatus.HEALTHY
        
        self.health_monitor.add_health_check('security_auth', check_authentication_health)
    
    def secure_configure(self, config: Dict[str, Any], 
                        auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Securely configure arena with validation."""
        if not self.security_enabled:
            return config
        
        # Authenticate if token provided
        if auth_token:
            token_info = self.auth_manager.validate_token(auth_token)
            if not token_info:
                raise PermissionError("Invalid authentication token")
            
            # Check authorization level
            if token_info['security_level'] not in [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET]:
                raise PermissionError("Insufficient permissions for configuration changes")
        
        # Sanitize configuration
        sanitized_config = self.input_sanitizer.sanitize_config(config)
        
        # Scan for vulnerabilities
        vulnerabilities = self.vulnerability_scanner.scan_configuration(sanitized_config)
        
        if vulnerabilities:
            high_risk_vulns = [v for v in vulnerabilities 
                             if v['severity'] in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
            
            if high_risk_vulns:
                self._log_security_event('configuration_blocked', {
                    'vulnerabilities': len(high_risk_vulns),
                    'config_keys': list(sanitized_config.keys())
                })
                raise ValueError(f"Configuration blocked due to {len(high_risk_vulns)} high-risk vulnerabilities")
        
        return sanitized_config
    
    def authenticate_user(self, username: str, password: str, 
                         client_id: str = "default") -> Optional[str]:
        """Authenticate user and return token."""
        if not self.security_enabled:
            return "mock_token"  # For demo purposes
        
        # Check rate limiting
        if not self.auth_manager.check_rate_limit(client_id):
            self._log_security_event('rate_limit_exceeded', {'client_id': client_id})
            raise PermissionError("Too many failed authentication attempts")
        
        # Simple demo authentication (in production, use proper user database)
        valid_users = {
            'admin': ('admin_password', SecurityLevel.SECRET),
            'operator': ('operator_pass', SecurityLevel.CONFIDENTIAL),
            'viewer': ('viewer_pass', SecurityLevel.INTERNAL)
        }
        
        if username in valid_users:
            expected_pass, security_level = valid_users[username]
            if password == expected_pass:  # In production, use hashed passwords
                token = self.auth_manager.generate_token(username, security_level)
                self._log_security_event('successful_authentication', {
                    'username': username,
                    'security_level': security_level.value
                })
                return token
        
        # Record failed attempt
        self.auth_manager.record_failed_attempt(client_id)
        self._log_security_event('failed_authentication', {
            'username': username,
            'client_id': client_id
        })
        return None
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        self.health_monitor.logger.warning(f"Security event: {event_type} - {details}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        if not self.security_enabled:
            return {'status': 'disabled', 'message': 'Security features disabled'}
        
        # Generate vulnerability report
        security_report = self.vulnerability_scanner.generate_security_report()
        
        # Authentication stats
        auth_stats = {
            'active_tokens': len(self.auth_manager.active_tokens),
            'failed_attempts_total': sum(len(attempts) for attempts 
                                       in self.auth_manager.failed_attempts.values()),
            'rate_limited_clients': len([client for client, attempts 
                                       in self.auth_manager.failed_attempts.items()
                                       if len(attempts) >= 5])
        }
        
        # Recent security events
        recent_events = [
            {
                'timestamp': time.strftime('%H:%M:%S', time.localtime(event['timestamp'])),
                'type': event['type'],
                'details': event['details']
            }
            for event in self.security_events[-10:]  # Last 10 events
        ]
        
        return {
            'security_status': security_report['status'],
            'risk_level': security_report['risk_level'],
            'vulnerabilities': security_report['total_vulnerabilities'],
            'authentication': auth_stats,
            'recent_events': recent_events,
            'security_report': security_report
        }


def print_security_dashboard(arena: SecureArena):
    """Print comprehensive security dashboard."""
    if not arena.security_enabled:
        print("\n🔒 Security: DISABLED")
        return
    
    dashboard = arena.get_security_dashboard()
    
    print(f"\n🔒 Security Dashboard")
    print("=" * 70)
    
    # Security status
    status_emoji = {
        'secure': '✅',
        'vulnerable': '🚨'
    }
    
    risk_emoji = {
        'low': '💚',
        'medium': '💛',
        'high': '🧡',
        'critical': '🔴'
    }
    
    status = dashboard['security_status']
    risk = dashboard['risk_level']
    
    print(f"{status_emoji.get(status, '❓')} Status: {status.upper()}")
    print(f"{risk_emoji.get(risk, '❓')} Risk Level: {risk.upper()}")
    print(f"🐛 Vulnerabilities: {dashboard['vulnerabilities']}")
    
    # Authentication stats
    auth = dashboard['authentication']
    print(f"\n🔐 Authentication:")
    print(f"  Active tokens: {auth['active_tokens']}")
    print(f"  Failed attempts: {auth['failed_attempts_total']}")
    print(f"  Rate limited clients: {auth['rate_limited_clients']}")
    
    # Recent security events
    if dashboard['recent_events']:
        print(f"\n🚨 Recent Security Events ({len(dashboard['recent_events'])}):")
        for event in dashboard['recent_events'][-3:]:  # Show last 3
            print(f"  {event['timestamp']} {event['type']}")
    
    # Vulnerability details
    if dashboard['vulnerabilities'] > 0:
        print(f"\n🐛 Vulnerability Details:")
        severity_counts = dashboard['security_report']['severity_breakdown']
        for severity, count in severity_counts.items():
            if count > 0:
                emoji = risk_emoji.get(severity, '❓')
                print(f"  {emoji} {severity.title()}: {count}")
    
    print("-" * 70)


def main():
    """Run security validation framework demonstration."""
    print("🏟️  Swarm Arena - Security Validation Framework Demo")
    print("=" * 80)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create secure arena
    print("Initializing secure arena with security features...")
    arena = SecureArena(num_agents=20, arena_size=(1000, 1000), enable_security=True)
    
    # Test authentication
    print("\n🔐 Testing Authentication System...")
    try:
        # Successful authentication
        token = arena.authenticate_user('admin', 'admin_password', 'demo_client')
        print(f"✅ Admin authenticated successfully. Token: {token[:16]}...")
        
        # Failed authentication
        try:
            arena.authenticate_user('hacker', 'wrong_password', 'bad_client')
        except PermissionError:
            pass
        print("❌ Invalid login attempt blocked")
        
    except Exception as e:
        print(f"Authentication error: {e}")
    
    # Test configuration security
    print("\n⚙️  Testing Configuration Security...")
    
    # Safe configuration
    safe_config = {
        'num_agents': 50,
        'arena_size': [1500, 1500],
        'debug': False
    }
    
    try:
        validated_config = arena.secure_configure(safe_config, token)
        print("✅ Safe configuration accepted")
    except Exception as e:
        print(f"❌ Safe configuration error: {e}")
    
    # Unsafe configuration
    unsafe_config = {
        'debug': True,  # Insecure default
        'allow_all_origins': True,  # Security risk
        'admin_password': 'password123'  # Exposed secret
    }
    
    try:
        arena.secure_configure(unsafe_config, token)
        print("❌ Unsafe configuration should have been blocked!")
    except ValueError as e:
        print(f"✅ Unsafe configuration blocked: {e}")
    
    # Test input sanitization
    print("\n🧹 Testing Input Sanitization...")
    
    sanitizer = InputSanitizer()
    
    # Test XSS attempt
    try:
        dangerous_input = '<script>alert("XSS")</script>'
        sanitizer.sanitize_string(dangerous_input)
        print("❌ XSS should have been blocked!")
    except ValueError:
        print("✅ XSS attempt blocked")
    
    # Test command injection
    try:
        dangerous_input = 'normal_input; rm -rf /'
        sanitizer.sanitize_string(dangerous_input)
        print("❌ Command injection should have been blocked!")
    except ValueError:
        print("✅ Command injection blocked")
    
    # Run simulation with security monitoring
    print("\n🏃 Running secure simulation...")
    
    for step in range(100):
        arena.step()
        
        if (step + 1) % 50 == 0:
            print_security_dashboard(arena)
    
    # Final security assessment
    print("\n📋 Final Security Assessment")
    print("=" * 80)
    
    final_dashboard = arena.get_security_dashboard()
    
    print(f"Overall Security Status: {final_dashboard['security_status'].upper()}")
    print(f"Risk Level: {final_dashboard['risk_level'].upper()}")
    print(f"Total Security Events: {len(arena.security_events)}")
    print(f"Active Authentication Tokens: {final_dashboard['authentication']['active_tokens']}")
    
    # Save security report
    report_filename = f"security_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(final_dashboard, f, indent=2, default=str)
    
    print(f"\n💾 Security report saved to: {report_filename}")
    
    # Cleanup
    arena.health_monitor.stop_monitoring()
    print("✅ Security validation framework demo completed successfully!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)