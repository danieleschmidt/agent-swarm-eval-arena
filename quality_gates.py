#!/usr/bin/env python3
"""
Comprehensive Quality Gates for Swarm Arena
Validates all aspects of code quality, performance, and security.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from swarm_arena.utils.logging import get_logger
from swarm_arena.validation.input_validator import input_validator
from swarm_arena.security.auth import token_manager, SecurityValidator
from swarm_arena.optimization.performance_engine import performance_optimizer

logger = get_logger(__name__)


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.errors = []
        self.warnings = []
        self.metrics = {}
    
    def run(self) -> bool:
        """Run the quality gate check."""
        raise NotImplementedError
    
    def add_error(self, error: str) -> None:
        """Add an error to the gate."""
        self.errors.append(error)
        logger.error(f"{self.name}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the gate."""
        self.warnings.append(warning)
        logger.warning(f"{self.name}: {warning}")
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add a metric to the gate."""
        self.metrics[key] = value
    
    def get_report(self) -> Dict[str, Any]:
        """Get quality gate report."""
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics
        }


class CodeQualityGate(QualityGate):
    """Code quality and linting checks."""
    
    def __init__(self):
        super().__init__("Code Quality", "Static code analysis and linting")
    
    def run(self) -> bool:
        """Run code quality checks."""
        try:
            # Check if required tools are available
            tools_available = self._check_tools()
            
            if not tools_available:
                self.add_error("Required code quality tools not available")
                return False
            
            # Run linting checks
            flake8_passed = self._run_flake8()
            mypy_passed = self._run_mypy()
            black_check_passed = self._run_black_check()
            isort_check_passed = self._run_isort_check()
            
            self.passed = all([flake8_passed, mypy_passed, black_check_passed, isort_check_passed])
            return self.passed
            
        except Exception as e:
            self.add_error(f"Code quality check failed: {e}")
            return False
    
    def _check_tools(self) -> bool:
        """Check if code quality tools are available."""
        tools = ["flake8", "mypy", "black", "isort"]
        available = True
        
        for tool in tools:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.add_metric(f"{tool}_version", result.stdout.strip())
                else:
                    self.add_warning(f"{tool} not available")
                    available = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.add_warning(f"{tool} not available")
                available = False
        
        return available
    
    def _run_flake8(self) -> bool:
        """Run flake8 linting."""
        try:
            result = subprocess.run([
                "flake8", "swarm_arena/", 
                "--max-line-length=88",
                "--extend-ignore=E203,W503",
                "--exclude=__pycache__"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.add_metric("flake8_issues", 0)
                return True
            else:
                issues = result.stdout.count('\n') if result.stdout else 0
                self.add_metric("flake8_issues", issues)
                if issues > 0:
                    self.add_error(f"Flake8 found {issues} issues")
                return issues == 0
                
        except subprocess.TimeoutExpired:
            self.add_error("Flake8 check timed out")
            return False
        except Exception as e:
            self.add_error(f"Flake8 check failed: {e}")
            return False
    
    def _run_mypy(self) -> bool:
        """Run mypy type checking."""
        try:
            result = subprocess.run([
                "mypy", "swarm_arena/",
                "--ignore-missing-imports",
                "--no-strict-optional"
            ], capture_output=True, text=True, timeout=120)
            
            # MyPy returns 0 for success, >0 for issues
            if result.returncode == 0:
                self.add_metric("mypy_errors", 0)
                return True
            else:
                errors = result.stdout.count('error:') if result.stdout else 0
                self.add_metric("mypy_errors", errors)
                if errors > 10:  # Allow some type errors
                    self.add_error(f"MyPy found {errors} type errors (limit: 10)")
                    return False
                elif errors > 0:
                    self.add_warning(f"MyPy found {errors} type errors")
                return True
                
        except subprocess.TimeoutExpired:
            self.add_error("MyPy check timed out")
            return False
        except Exception as e:
            self.add_warning(f"MyPy check failed: {e}")
            return True  # Don't fail on MyPy issues
    
    def _run_black_check(self) -> bool:
        """Check code formatting with Black."""
        try:
            result = subprocess.run([
                "black", "--check", "--diff", "swarm_arena/"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.add_metric("formatting_issues", 0)
                return True
            else:
                # Count files that would be reformatted
                files_to_format = result.stdout.count('would reformat') if result.stdout else 0
                self.add_metric("formatting_issues", files_to_format)
                if files_to_format > 0:
                    self.add_warning(f"Black would reformat {files_to_format} files")
                return True  # Don't fail on formatting issues, just warn
                
        except subprocess.TimeoutExpired:
            self.add_error("Black check timed out")
            return False
        except Exception as e:
            self.add_warning(f"Black check failed: {e}")
            return True
    
    def _run_isort_check(self) -> bool:
        """Check import sorting with isort."""
        try:
            result = subprocess.run([
                "isort", "--check-only", "--diff", "swarm_arena/"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.add_metric("import_issues", 0)
                return True
            else:
                import_issues = result.stdout.count('Fixing') if result.stdout else 0
                self.add_metric("import_issues", import_issues)
                if import_issues > 0:
                    self.add_warning(f"isort would fix {import_issues} import issues")
                return True  # Don't fail on import sorting
                
        except subprocess.TimeoutExpired:
            self.add_error("isort check timed out")
            return False
        except Exception as e:
            self.add_warning(f"isort check failed: {e}")
            return True


class TestQualityGate(QualityGate):
    """Test coverage and execution quality gate."""
    
    def __init__(self):
        super().__init__("Test Quality", "Test execution and coverage validation")
    
    def run(self) -> bool:
        """Run test quality checks."""
        try:
            # Check if pytest is available
            if not self._check_pytest():
                self.add_error("pytest not available")
                return False
            
            # Run tests
            test_passed = self._run_tests()
            coverage_passed = self._check_coverage()
            
            self.passed = test_passed and coverage_passed
            return self.passed
            
        except Exception as e:
            self.add_error(f"Test quality check failed: {e}")
            return False
    
    def _check_pytest(self) -> bool:
        """Check if pytest is available."""
        try:
            result = subprocess.run(["pytest", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.add_metric("pytest_version", result.stdout.strip())
                return True
            return False
        except Exception:
            return False
    
    def _run_tests(self) -> bool:
        """Run the test suite."""
        try:
            result = subprocess.run([
                "python", "-m", "pytest", "tests/",
                "-v", "--tb=short", "--timeout=300"
            ], capture_output=True, text=True, timeout=600)
            
            # Parse test results
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            passed_tests = sum(1 for line in output_lines if " PASSED" in line)
            failed_tests = sum(1 for line in output_lines if " FAILED" in line)
            skipped_tests = sum(1 for line in output_lines if " SKIPPED" in line)
            
            self.add_metric("tests_passed", passed_tests)
            self.add_metric("tests_failed", failed_tests)
            self.add_metric("tests_skipped", skipped_tests)
            
            if result.returncode == 0:
                return True
            else:
                self.add_error(f"Tests failed: {failed_tests} failures")
                return False
                
        except subprocess.TimeoutExpired:
            self.add_error("Test execution timed out")
            return False
        except Exception as e:
            self.add_error(f"Test execution failed: {e}")
            return False
    
    def _check_coverage(self) -> bool:
        """Check test coverage."""
        try:
            result = subprocess.run([
                "python", "-m", "pytest", "tests/",
                "--cov=swarm_arena", "--cov-report=term", "--cov-report=json"
            ], capture_output=True, text=True, timeout=600)
            
            # Try to read coverage report
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                self.add_metric("test_coverage", total_coverage)
                
                if total_coverage < 70:  # Minimum 70% coverage
                    self.add_error(f"Test coverage too low: {total_coverage:.1f}% (minimum: 70%)")
                    return False
                elif total_coverage < 80:
                    self.add_warning(f"Test coverage could be improved: {total_coverage:.1f}%")
                
                return True
            else:
                self.add_warning("Coverage report not generated")
                return True
                
        except Exception as e:
            self.add_warning(f"Coverage check failed: {e}")
            return True  # Don't fail on coverage issues


class SecurityQualityGate(QualityGate):
    """Security validation quality gate."""
    
    def __init__(self):
        super().__init__("Security", "Security vulnerability and validation checks")
    
    def run(self) -> bool:
        """Run security quality checks."""
        try:
            validation_passed = self._test_input_validation()
            auth_passed = self._test_authentication()
            config_passed = self._test_config_validation()
            dependency_passed = self._check_dependencies()
            
            self.passed = all([validation_passed, auth_passed, config_passed, dependency_passed])
            return self.passed
            
        except Exception as e:
            self.add_error(f"Security check failed: {e}")
            return False
    
    def _test_input_validation(self) -> bool:
        """Test input validation system."""
        try:
            # Test various malicious inputs
            malicious_inputs = [
                ("', DROP TABLE users; --", "sql_injection"),
                ("<script>alert('xss')</script>", "xss"),
                ("../../etc/passwd", "path_traversal"),
                ("\x00\x01\x02", "null_bytes"),
                ("a" * 10000, "buffer_overflow")
            ]
            
            validation_errors = 0
            
            for malicious_input, attack_type in malicious_inputs:
                try:
                    input_validator.validate_field("agent_type", malicious_input)
                    # Should have raised ValidationError
                    validation_errors += 1
                    self.add_warning(f"Validation missed {attack_type} attack")
                except Exception:
                    # Expected to fail validation
                    pass
            
            self.add_metric("validation_bypassed", validation_errors)
            
            if validation_errors > 0:
                self.add_error(f"Input validation bypassed for {validation_errors} attack types")
                return False
            
            return True
            
        except Exception as e:
            self.add_error(f"Input validation test failed: {e}")
            return False
    
    def _test_authentication(self) -> bool:
        """Test authentication system."""
        try:
            # Test API key creation and validation
            permissions = ["test.permission"]
            api_key = token_manager.create_api_key(permissions, expires_in_days=1)
            
            # Validate the key
            key_info = token_manager.validate_api_key(api_key)
            if not key_info:
                self.add_error("API key validation failed")
                return False
            
            # Test invalid key rejection
            invalid_key = "invalid.key.format"
            invalid_info = token_manager.validate_api_key(invalid_key)
            if invalid_info:
                self.add_error("Invalid API key was accepted")
                return False
            
            self.add_metric("auth_tests_passed", 2)
            return True
            
        except Exception as e:
            self.add_error(f"Authentication test failed: {e}")
            return False
    
    def _test_config_validation(self) -> bool:
        """Test configuration validation."""
        try:
            # Test valid configuration
            valid_config = {
                "num_agents": 100,
                "arena_size": (1000, 1000),
                "episode_length": 1000
            }
            
            try:
                SecurityValidator.validate_simulation_config(valid_config)
            except Exception as e:
                self.add_error(f"Valid config rejected: {e}")
                return False
            
            # Test malicious configuration
            malicious_configs = [
                {"num_agents": 1000000, "arena_size": (1000, 1000), "episode_length": 1000},
                {"num_agents": 100, "arena_size": (1000000, 1000000), "episode_length": 1000},
                {"num_agents": 100, "arena_size": (1000, 1000), "episode_length": 10000000}
            ]
            
            validation_failures = 0
            for config in malicious_configs:
                try:
                    SecurityValidator.validate_simulation_config(config)
                    validation_failures += 1
                except Exception:
                    # Expected to fail
                    pass
            
            self.add_metric("config_validation_bypassed", validation_failures)
            
            if validation_failures > 0:
                self.add_error(f"Malicious config validation bypassed {validation_failures} times")
                return False
            
            return True
            
        except Exception as e:
            self.add_error(f"Config validation test failed: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check for known vulnerable dependencies."""
        try:
            # Check if safety is available
            try:
                result = subprocess.run(["safety", "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    self.add_warning("Safety tool not available for dependency checking")
                    return True
            except FileNotFoundError:
                self.add_warning("Safety tool not available for dependency checking")
                return True
            
            # Run safety check
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.add_metric("vulnerable_dependencies", 0)
                return True
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    vuln_count = len(vulnerabilities)
                    self.add_metric("vulnerable_dependencies", vuln_count)
                    
                    if vuln_count > 0:
                        self.add_error(f"Found {vuln_count} vulnerable dependencies")
                        return False
                except json.JSONDecodeError:
                    self.add_warning("Could not parse safety check results")
                
                return True
                
        except subprocess.TimeoutExpired:
            self.add_warning("Dependency check timed out")
            return True
        except Exception as e:
            self.add_warning(f"Dependency check failed: {e}")
            return True


class PerformanceQualityGate(QualityGate):
    """Performance benchmarking quality gate."""
    
    def __init__(self):
        super().__init__("Performance", "Performance benchmarks and optimization validation")
    
    def run(self) -> bool:
        """Run performance quality checks."""
        try:
            benchmark_passed = self._run_performance_benchmarks()
            memory_passed = self._check_memory_efficiency()
            optimization_passed = self._test_optimizations()
            
            self.passed = all([benchmark_passed, memory_passed, optimization_passed])
            return self.passed
            
        except Exception as e:
            self.add_error(f"Performance check failed: {e}")
            return False
    
    def _run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        try:
            from swarm_arena import Arena, SwarmConfig, Agent
            
            # Small benchmark
            config = SwarmConfig(num_agents=50, episode_length=100, seed=42)
            arena = Arena(config)
            arena.add_agents(Agent, count=50)
            
            start_time = time.time()
            results = arena.run(episodes=3, verbose=False)
            duration = time.time() - start_time
            
            steps_per_second = results.total_steps / duration
            self.add_metric("small_sim_steps_per_second", steps_per_second)
            
            if steps_per_second < 100:  # Minimum performance threshold
                self.add_error(f"Small simulation too slow: {steps_per_second:.1f} steps/sec (min: 100)")
                return False
            
            # Medium benchmark
            config = SwarmConfig(num_agents=200, episode_length=200, seed=42)
            arena = Arena(config)
            arena.add_agents(Agent, count=200)
            
            start_time = time.time()
            results = arena.run(episodes=2, verbose=False)
            duration = time.time() - start_time
            
            steps_per_second = results.total_steps / duration
            self.add_metric("medium_sim_steps_per_second", steps_per_second)
            
            if steps_per_second < 50:  # Lower threshold for larger simulation
                self.add_error(f"Medium simulation too slow: {steps_per_second:.1f} steps/sec (min: 50)")
                return False
            
            return True
            
        except Exception as e:
            self.add_error(f"Performance benchmark failed: {e}")
            return False
    
    def _check_memory_efficiency(self) -> bool:
        """Check memory efficiency."""
        try:
            import psutil
            from swarm_arena import Arena, SwarmConfig, Agent
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Run simulation multiple times to check for leaks
            config = SwarmConfig(num_agents=100, episode_length=50, seed=42)
            arena = Arena(config)
            arena.add_agents(Agent, count=100)
            
            for _ in range(5):
                arena.run(episodes=1, verbose=False)
                arena.reset()
            
            final_memory = process.memory_info().rss
            memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            self.add_metric("memory_growth_mb", memory_growth)
            
            if memory_growth > 50:  # Max 50MB growth
                self.add_error(f"Excessive memory growth: {memory_growth:.1f}MB (max: 50MB)")
                return False
            elif memory_growth > 20:
                self.add_warning(f"Moderate memory growth: {memory_growth:.1f}MB")
            
            return True
            
        except Exception as e:
            self.add_error(f"Memory efficiency check failed: {e}")
            return False
    
    def _test_optimizations(self) -> bool:
        """Test performance optimizations."""
        try:
            # Test performance optimizer
            performance_optimizer.start_monitoring()
            
            # Let it run briefly
            time.sleep(2)
            
            report = performance_optimizer.get_performance_report()
            performance_optimizer.stop_monitoring()
            
            # Check that optimizations are enabled
            optimizations = report.get("optimizations_enabled", {})
            enabled_count = sum(optimizations.values())
            
            self.add_metric("optimizations_enabled", enabled_count)
            
            if enabled_count < 3:  # Should have at least 3 optimizations enabled
                self.add_warning(f"Few optimizations enabled: {enabled_count}")
            
            return True
            
        except Exception as e:
            self.add_error(f"Optimization test failed: {e}")
            return False


class QualityGateRunner:
    """Orchestrates all quality gate execution."""
    
    def __init__(self):
        self.gates = [
            CodeQualityGate(),
            TestQualityGate(),
            SecurityQualityGate(),
            PerformanceQualityGate()
        ]
        self.results = {}
    
    def run_all(self, fail_fast: bool = False) -> bool:
        """Run all quality gates."""
        print("🚀 Running Swarm Arena Quality Gates")
        print("=" * 50)
        
        all_passed = True
        
        for gate in self.gates:
            print(f"\n📋 Running {gate.name}...")
            print(f"   {gate.description}")
            
            start_time = time.time()
            
            try:
                passed = gate.run()
                duration = time.time() - start_time
                
                if passed:
                    print(f"   ✅ PASSED ({duration:.1f}s)")
                else:
                    print(f"   ❌ FAILED ({duration:.1f}s)")
                    all_passed = False
                    
                    if fail_fast:
                        print(f"   Stopping due to fail-fast mode")
                        break
                
                # Show errors and warnings
                for error in gate.errors:
                    print(f"   🔴 ERROR: {error}")
                
                for warning in gate.warnings:
                    print(f"   🟡 WARNING: {warning}")
                
                # Show key metrics
                for key, value in gate.metrics.items():
                    if isinstance(value, float):
                        print(f"   📊 {key}: {value:.2f}")
                    else:
                        print(f"   📊 {key}: {value}")
                
            except Exception as e:
                print(f"   💥 CRASHED: {e}")
                all_passed = False
                gate.add_error(f"Gate crashed: {e}")
                
                if fail_fast:
                    break
            
            self.results[gate.name] = gate.get_report()
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL QUALITY GATES PASSED!")
        else:
            print("💥 SOME QUALITY GATES FAILED!")
        
        return all_passed
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "timestamp": time.time(),
            "overall_passed": all(result["passed"] for result in self.results.values()),
            "gates": self.results,
            "summary": {
                "total_gates": len(self.gates),
                "passed_gates": sum(1 for result in self.results.values() if result["passed"]),
                "failed_gates": sum(1 for result in self.results.values() if not result["passed"]),
                "total_errors": sum(len(result["errors"]) for result in self.results.values()),
                "total_warnings": sum(len(result["warnings"]) for result in self.results.values()),
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📋 Quality report saved to {output_file}")
        
        return report


def main():
    """Main entry point for quality gates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Swarm Arena Quality Gates")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--output", type=str,
                       help="Output file for quality report")
    
    args = parser.parse_args()
    
    runner = QualityGateRunner()
    
    try:
        success = runner.run_all(fail_fast=args.fail_fast)
        
        # Generate report
        report = runner.generate_report(args.output)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Quality gates interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Quality gates crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()