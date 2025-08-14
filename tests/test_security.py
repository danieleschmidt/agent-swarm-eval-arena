"""Security tests for Swarm Arena."""

import pytest
import tempfile
import json
from pathlib import Path

from swarm_arena.security.auth import (
    SecureTokenManager, PermissionChecker, SecurityValidator, AuditLogger
)
from swarm_arena.validation.input_validator import InputValidator
from swarm_arena.exceptions import SecurityError, ValidationError


class TestSecureTokenManager:
    """Test secure token management."""
    
    def test_create_api_key(self):
        """Test API key creation."""
        manager = SecureTokenManager()
        permissions = ["simulation.run", "data.read"]
        
        key = manager.create_api_key(permissions, expires_in_days=30)
        
        assert isinstance(key, str)
        assert "." in key  # Should contain key_id.raw_key format
        
        # Check that key was stored
        key_id = key.split(".")[0]
        assert key_id in manager.api_keys
        
        api_key_info = manager.api_keys[key_id]
        assert api_key_info.permissions == permissions
        assert api_key_info.is_active
    
    def test_validate_api_key(self):
        """Test API key validation."""
        manager = SecureTokenManager()
        permissions = ["simulation.run"]
        
        # Create and validate key
        key = manager.create_api_key(permissions)
        api_key_info = manager.validate_api_key(key)
        
        assert api_key_info is not None
        assert api_key_info.permissions == permissions
        assert api_key_info.is_active
    
    def test_invalid_api_key(self):
        """Test validation of invalid API key."""
        manager = SecureTokenManager()
        
        # Test completely invalid key
        assert manager.validate_api_key("invalid") is None
        assert manager.validate_api_key("invalid.key") is None
        
        # Test wrong format
        assert manager.validate_api_key("no_dot_separator") is None
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        manager = SecureTokenManager()
        key = manager.create_api_key(["simulation.run"])
        key_id = key.split(".")[0]
        
        # Revoke key
        assert manager.revoke_api_key(key_id) is True
        
        # Should no longer validate
        assert manager.validate_api_key(key) is None
        
        # Revoking non-existent key should return False
        assert manager.revoke_api_key("nonexistent") is False
    
    def test_cleanup_expired_keys(self):
        """Test cleanup of expired keys."""
        manager = SecureTokenManager()
        
        # Create expired key (0 days = expired immediately)
        key = manager.create_api_key(["test"], expires_in_days=0)
        key_id = key.split(".")[0]
        
        # Force expiration by setting past date
        import datetime
        manager.api_keys[key_id].expires_at = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        
        # Cleanup should remove 1 key
        removed_count = manager.cleanup_expired_keys()
        assert removed_count == 1
        assert key_id not in manager.api_keys


class TestPermissionChecker:
    """Test permission checking logic."""
    
    def test_has_permission_exact_match(self):
        """Test exact permission matching."""
        from swarm_arena.security.auth import APIKey
        import datetime
        
        api_key = APIKey(
            key_id="test",
            hashed_key="hash",
            permissions=["simulation.run", "data.read"],
            created_at=datetime.datetime.utcnow()
        )
        
        assert PermissionChecker.has_permission(api_key, "simulation.run")
        assert PermissionChecker.has_permission(api_key, "data.read")
        assert not PermissionChecker.has_permission(api_key, "admin.write")
    
    def test_has_permission_admin_access(self):
        """Test admin permission grants everything."""
        from swarm_arena.security.auth import APIKey
        import datetime
        
        api_key = APIKey(
            key_id="admin",
            hashed_key="hash",
            permissions=["admin.full"],
            created_at=datetime.datetime.utcnow()
        )
        
        assert PermissionChecker.has_permission(api_key, "simulation.run")
        assert PermissionChecker.has_permission(api_key, "data.write")
        assert PermissionChecker.has_permission(api_key, "anything.else")
    
    def test_has_permission_wildcard(self):
        """Test wildcard permission matching."""
        from swarm_arena.security.auth import APIKey
        import datetime
        
        api_key = APIKey(
            key_id="wildcard",
            hashed_key="hash",
            permissions=["simulation.*"],
            created_at=datetime.datetime.utcnow()
        )
        
        assert PermissionChecker.has_permission(api_key, "simulation.run")
        assert PermissionChecker.has_permission(api_key, "simulation.create")
        assert not PermissionChecker.has_permission(api_key, "data.read")


class TestSecurityValidator:
    """Test security validation functions."""
    
    def test_validate_simulation_config_valid(self):
        """Test validation of valid simulation config."""
        config = {
            "num_agents": 100,
            "arena_size": (1000, 1000),
            "episode_length": 1000
        }
        
        assert SecurityValidator.validate_simulation_config(config) is True
    
    def test_validate_simulation_config_too_many_agents(self):
        """Test rejection of too many agents."""
        config = {
            "num_agents": 20000,  # Too many
            "arena_size": (1000, 1000),
            "episode_length": 1000
        }
        
        with pytest.raises(SecurityError, match="Too many agents"):
            SecurityValidator.validate_simulation_config(config)
    
    def test_validate_simulation_config_arena_too_large(self):
        """Test rejection of arena that's too large."""
        config = {
            "num_agents": 100,
            "arena_size": (100000, 100000),  # Too large
            "episode_length": 1000
        }
        
        with pytest.raises(SecurityError, match="Arena too large"):
            SecurityValidator.validate_simulation_config(config)
    
    def test_validate_file_path_valid(self):
        """Test validation of safe file paths."""
        assert SecurityValidator.validate_file_path("results.json") is True
        assert SecurityValidator.validate_file_path("data/experiment.csv") is True
    
    def test_validate_file_path_path_traversal(self):
        """Test rejection of path traversal attempts."""
        with pytest.raises(SecurityError, match="Invalid file path"):
            SecurityValidator.validate_file_path("../../../etc/passwd")
        
        with pytest.raises(SecurityError, match="Invalid file path"):
            SecurityValidator.validate_file_path("/absolute/path")
    
    def test_validate_file_path_dangerous_extension(self):
        """Test rejection of dangerous file extensions."""
        with pytest.raises(SecurityError, match="Dangerous file extension"):
            SecurityValidator.validate_file_path("malware.exe")
        
        with pytest.raises(SecurityError, match="Dangerous file extension"):
            SecurityValidator.validate_file_path("script.sh")
    
    def test_sanitize_user_input(self):
        """Test user input sanitization."""
        # Normal input should pass through
        assert SecurityValidator.sanitize_user_input("normal text") == "normal text"
        
        # Dangerous characters should be removed
        dangerous_input = "<script>alert('xss')</script>"
        sanitized = SecurityValidator.sanitize_user_input(dangerous_input)
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "script" in sanitized  # Content preserved, tags removed
        
        # Long input should be truncated
        long_input = "a" * 2000
        sanitized = SecurityValidator.sanitize_user_input(long_input, max_length=1000)
        assert len(sanitized) == 1000


class TestInputValidator:
    """Test input validation system."""
    
    def test_validate_integer_field(self):
        """Test integer field validation."""
        validator = InputValidator()
        
        # Valid integers
        assert validator.validate_field("num_agents", 100) is True
        assert validator.validate_field("num_agents", 1) is True
        
        # Invalid integers
        with pytest.raises(ValidationError):
            validator.validate_field("num_agents", 0)  # Below minimum
        
        with pytest.raises(ValidationError):
            validator.validate_field("num_agents", 20000)  # Above maximum
        
        with pytest.raises(ValidationError):
            validator.validate_field("num_agents", "not_a_number")
    
    def test_validate_float_field(self):
        """Test float field validation."""
        validator = InputValidator()
        
        # Valid floats
        assert validator.validate_field("resource_spawn_rate", 0.5) is True
        assert validator.validate_field("resource_spawn_rate", 0.0) is True
        assert validator.validate_field("resource_spawn_rate", 1.0) is True
        
        # Invalid floats
        with pytest.raises(ValidationError):
            validator.validate_field("resource_spawn_rate", -0.1)  # Below minimum
        
        with pytest.raises(ValidationError):
            validator.validate_field("resource_spawn_rate", 1.5)  # Above maximum
    
    def test_validate_tuple_field(self):
        """Test tuple field validation."""
        validator = InputValidator()
        
        # Valid tuples
        assert validator.validate_field("arena_size", (1000, 1000)) is True
        assert validator.validate_field("arena_size", [800, 600]) is True  # List acceptable
        
        # Invalid tuples
        with pytest.raises(ValidationError):
            validator.validate_field("arena_size", (1000,))  # Wrong length
        
        with pytest.raises(ValidationError):
            validator.validate_field("arena_size", (5, 1000))  # Below minimum
        
        with pytest.raises(ValidationError):
            validator.validate_field("arena_size", (100000, 1000))  # Above maximum
    
    def test_validate_config_dict(self):
        """Test complete configuration validation."""
        validator = InputValidator()
        
        valid_config = {
            "num_agents": 100,
            "arena_size": (1000, 1000),
            "episode_length": 1000,
            "resource_spawn_rate": 0.1
        }
        
        validated = validator.validate_config(valid_config)
        assert validated["num_agents"] == 100
        assert validated["arena_size"] == (1000, 1000)
    
    def test_validate_file_upload(self):
        """Test file upload validation."""
        validator = InputValidator()
        
        # Valid JSON file
        json_content = json.dumps({"test": "data"}).encode('utf-8')
        assert validator.validate_file_upload("config.json", json_content) is True
        
        # File too large
        large_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        with pytest.raises(ValidationError, match="File too large"):
            validator.validate_file_upload("large.json", large_content)
        
        # Invalid JSON
        invalid_json = b"{ invalid json"
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validator.validate_file_upload("invalid.json", invalid_json)
        
        # Executable content
        exe_content = b"\x4d\x5a"  # PE executable header
        with pytest.raises(ValidationError, match="dangerous content"):
            validator.validate_file_upload("malware.txt", exe_content)


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def test_log_event(self):
        """Test logging security events."""
        logger = AuditLogger()
        
        logger.log_event(
            event_type="authentication",
            user_id="test_user",
            action="login",
            resource="system",
            success=True,
            ip_address="127.0.0.1"
        )
        
        events = logger.get_audit_log(limit=1)
        assert len(events) == 1
        
        event = events[0]
        assert event["event_type"] == "authentication"
        assert event["user_id"] == "test_user"
        assert event["action"] == "login"
        assert event["success"] is True
        assert event["details"]["ip_address"] == "127.0.0.1"
    
    def test_export_audit_log(self):
        """Test exporting audit log."""
        logger = AuditLogger()
        
        # Log some events
        for i in range(3):
            logger.log_event(
                event_type="test",
                user_id=f"user_{i}",
                action="test_action",
                resource="test_resource",
                success=True
            )
        
        # Export log
        exported = logger.export_audit_log()
        assert isinstance(exported, str)
        
        # Should be valid JSON
        parsed = json.loads(exported)
        assert len(parsed) == 3
        assert all(event["event_type"] == "test" for event in parsed)


class TestIntegratedSecurity:
    """Test integrated security features."""
    
    def test_end_to_end_security_flow(self):
        """Test complete security flow from API key to audit."""
        # Create token manager and audit logger
        token_manager = SecureTokenManager()
        audit_logger = AuditLogger()
        
        # Create API key
        permissions = ["simulation.run", "data.read"]
        api_key_str = token_manager.create_api_key(permissions)
        
        # Validate API key
        api_key = token_manager.validate_api_key(api_key_str)
        assert api_key is not None
        
        # Check permissions
        assert PermissionChecker.has_permission(api_key, "simulation.run")
        assert not PermissionChecker.has_permission(api_key, "admin.write")
        
        # Log security event
        audit_logger.log_event(
            event_type="authorization",
            user_id=api_key.key_id,
            action="permission_check",
            resource="simulation.run",
            success=True
        )
        
        # Verify audit log
        events = audit_logger.get_audit_log(limit=1)
        assert len(events) == 1
        assert events[0]["user_id"] == api_key.key_id
    
    def test_security_with_invalid_input(self):
        """Test security handling of various invalid inputs."""
        validator = InputValidator()
        
        # Test SQL injection patterns
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\x00\x01\x02",  # Null bytes
        ]
        
        for malicious_input in malicious_inputs:
            try:
                # Should not crash, even with malicious input
                validator.validate_field("agent_type", malicious_input)
            except ValidationError:
                # Expected to fail validation
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception with input '{malicious_input}': {e}")
    
    def test_rate_limiting_simulation(self):
        """Test that security measures don't impact normal operation."""
        token_manager = SecureTokenManager()
        validator = InputValidator()
        
        # Create many API keys rapidly (simulate high load)
        keys = []
        for i in range(50):
            key = token_manager.create_api_key([f"permission_{i}"])
            keys.append(key)
        
        # Validate all keys rapidly
        for key in keys:
            api_key = token_manager.validate_api_key(key)
            assert api_key is not None
        
        # Validate many configurations rapidly
        for i in range(100):
            config = {
                "num_agents": 10 + i,
                "arena_size": (1000 + i, 1000 + i),
                "episode_length": 500 + i
            }
            validated = validator.validate_config(config)
            assert validated["num_agents"] == 10 + i


if __name__ == "__main__":
    pytest.main([__file__])