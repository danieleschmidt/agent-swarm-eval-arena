# Contributing to Agent Swarm Evaluation Arena

Thank you for your interest in contributing to the Agent Swarm Evaluation Arena! This document provides guidelines and information for contributors to help ensure a smooth collaboration process.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Standards](#documentation-standards)
6. [Submitting Changes](#submitting-changes)
7. [Issue Reporting](#issue-reporting)
8. [Community Guidelines](#community-guidelines)
9. [Recognition](#recognition)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.9 or higher
- Git version control
- Docker and Docker Compose
- Access to a development environment with at least 8GB RAM

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/agent-swarm-eval-arena.git
   cd agent-swarm-eval-arena
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   pytest tests/
   python -m swarm_arena --version
   ```

5. **Set Up Development Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your development settings
   ```

### Understanding the Codebase

The project follows a modular architecture:

```
swarm_arena/
├── core/           # Core arena logic and interfaces
├── distributed/    # Ray-based distributed execution
├── environments/   # Built-in environment implementations  
├── monitoring/     # Telemetry and observability
├── analysis/       # Data analysis and visualization
├── benchmarks/     # Standard benchmarks and metrics
└── utils/          # Shared utilities and helpers
```

Key entry points:
- `swarm_arena/core/arena.py` - Main Arena class
- `swarm_arena/core/agent.py` - Agent base classes
- `swarm_arena/distributed/ray_arena.py` - Distributed execution

## Development Workflow

### Branch Naming Convention

Use descriptive branch names following this pattern:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

Examples:
- `feature/communication-protocols`
- `fix/memory-leak-in-telemetry`
- `docs/api-reference-update`

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(monitoring): add real-time fairness metrics

Add comprehensive fairness monitoring including Gini coefficient,
envy-freeness, and proportional fairness calculations with 
configurable update intervals.

Closes #123
```

```
fix(distributed): resolve memory leak in worker cleanup

Workers weren't properly releasing GPU memory after episode completion.
Added explicit cleanup calls and memory monitoring.
```

### Development Process

1. **Check for Existing Issues**: Look for related issues before starting work
2. **Create an Issue**: For new features, create an issue to discuss the approach
3. **Create a Branch**: Branch from `main` using the naming convention
4. **Develop and Test**: Write code following our guidelines, include tests
5. **Run Quality Checks**: Ensure all linting, formatting, and tests pass
6. **Update Documentation**: Add/update relevant documentation
7. **Submit Pull Request**: Follow our PR template and guidelines

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length**: 88 characters (Black formatter default)
- **Imports**: Use absolute imports, group according to isort configuration
- **Docstrings**: Use Google-style docstrings for all public functions/classes
- **Type Hints**: Required for all public APIs and recommended elsewhere

### Formatting Tools

The project uses automated formatting tools:

```bash
# Format code
black swarm_arena/ tests/

# Sort imports  
isort swarm_arena/ tests/

# Check style compliance
flake8 swarm_arena/ tests/

# Type checking
mypy swarm_arena/
```

### Code Structure Guidelines

**Classes and Functions:**
```python
class AgentManager:
    """Manages the lifecycle of agents in the arena.
    
    This class handles agent creation, initialization, and cleanup,
    ensuring proper resource management and state consistency.
    
    Args:
        config: Configuration object with agent settings
        resource_limits: Optional resource constraints per agent
        
    Raises:
        ConfigurationError: When agent configuration is invalid
    """
    
    def __init__(self, config: ArenaConfig, resource_limits: Optional[Dict] = None):
        self._agents: Dict[str, Agent] = {}
        self._config = config
        self._resource_limits = resource_limits or {}
        
    def create_agent(self, agent_id: str, agent_class: Type[Agent]) -> Agent:
        """Create and register a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_class: Agent class to instantiate
            
        Returns:
            The created agent instance
            
        Raises:
            DuplicateAgentError: When agent_id already exists
        """
        if agent_id in self._agents:
            raise DuplicateAgentError(f"Agent {agent_id} already exists")
            
        agent = agent_class(agent_id, self._config)
        self._agents[agent_id] = agent
        return agent
```

**Error Handling:**
```python
# Use specific exception types
class SwarmArenaError(Exception):
    """Base exception for all swarm arena errors."""
    pass

class ConfigurationError(SwarmArenaError):
    """Raised when configuration is invalid."""
    pass

# Handle errors gracefully
try:
    result = arena.step(actions)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error in arena step: {e}")
    raise SwarmArenaError(f"Arena step failed: {e}") from e
```

### Performance Guidelines

**Memory Management:**
- Use generators for large datasets
- Implement proper cleanup in `__del__` methods
- Monitor memory usage in long-running processes

**Computational Efficiency:**
- Vectorize operations using NumPy/PyTorch
- Use multiprocessing/Ray for CPU-bound tasks
- Profile performance-critical code paths

**Example:**
```python
# Good: Vectorized operation
def update_positions(self, agents: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Update agent positions using vectorized operations."""
    return agents + velocities * self.dt

# Avoid: Loop-based operations for large arrays
def update_positions_slow(self, agents: List[Agent]) -> None:
    """Slow position update using loops."""
    for agent in agents:  # Avoid for large agent counts
        agent.position += agent.velocity * self.dt
```

## Testing Requirements

### Test Structure

All code must include comprehensive tests:

```bash
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Cross-component integration tests
├── performance/    # Performance and scalability tests
└── fixtures/       # Shared test data and utilities
```

### Test Requirements

- **Unit Tests**: All public functions/methods must have unit tests
- **Integration Tests**: Complex workflows must have integration tests
- **Performance Tests**: Performance-critical code needs benchmark tests
- **Coverage**: Minimum 90% code coverage required

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from swarm_arena.core.arena import Arena
from swarm_arena.core.config import ArenaConfig


class TestArena:
    """Test suite for Arena class."""
    
    @pytest.fixture
    def arena_config(self):
        """Provide a standard arena configuration for tests."""
        return ArenaConfig(
            num_agents=100,
            arena_size=(1000, 1000),
            episode_length=1000
        )
    
    @pytest.fixture
    def arena(self, arena_config):
        """Provide a configured arena instance."""
        return Arena(arena_config)
    
    def test_arena_initialization(self, arena):
        """Test that arena initializes correctly."""
        assert arena.num_agents == 100
        assert arena.arena_size == (1000, 1000)
        assert not arena.is_running
    
    def test_arena_start_stop(self, arena):
        """Test arena lifecycle management."""
        arena.start()
        assert arena.is_running
        
        arena.stop()
        assert not arena.is_running
    
    @patch('swarm_arena.core.arena.logger')
    def test_arena_error_handling(self, mock_logger, arena):
        """Test that arena handles errors gracefully."""
        with pytest.raises(ConfigurationError):
            arena.add_agents(None, count=-1)
        
        mock_logger.error.assert_called_once()
    
    @pytest.mark.performance
    def test_arena_performance(self, arena):
        """Test arena performance meets requirements."""
        import time
        
        start_time = time.time()
        results = arena.run(episodes=10)
        duration = time.time() - start_time
        
        # Should complete 10 episodes in under 10 seconds
        assert duration < 10.0
        assert len(results) == 10
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swarm_arena --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m performance  # Run performance tests only

# Run tests in parallel
pytest -n auto
```

## Documentation Standards

### Docstring Requirements

All public APIs must have comprehensive docstrings:

```python
def evaluate_fairness(self, trajectories: List[Trajectory], 
                     metrics: List[str] = None) -> FairnessReport:
    """Evaluate fairness metrics for agent trajectories.
    
    Computes various fairness indicators including Gini coefficient,
    envy-freeness, and proportional fairness based on resource
    allocation patterns in the provided trajectories.
    
    Args:
        trajectories: List of agent trajectories to analyze
        metrics: Specific fairness metrics to compute. If None,
            computes all available metrics. Options include:
            - 'gini': Gini coefficient for resource distribution
            - 'envy': Envy-freeness indicator
            - 'proportional': Proportional fairness measure
    
    Returns:
        FairnessReport containing computed metrics, statistical
        significance tests, and visualization data.
    
    Raises:
        ValueError: When trajectories list is empty
        MetricError: When requested metric is not available
        
    Example:
        >>> arena = Arena(config)
        >>> results = arena.run(episodes=10)
        >>> report = arena.evaluate_fairness(results.trajectories)
        >>> print(f"Gini coefficient: {report.gini_coefficient:.3f}")
        0.234
        
    Note:
        This method requires trajectories with resource allocation
        data. Trajectories without resource information will be
        excluded from fairness calculations.
    """
```

### README and Documentation Files

- **README.md**: Keep examples current with API changes
- **API Documentation**: Auto-generated from docstrings
- **Tutorials**: Step-by-step guides for common use cases
- **Architecture Docs**: High-level system design documentation

### Documentation Updates

When making changes:
1. Update relevant docstrings
2. Add/update examples in README if API changes
3. Update architecture documentation for structural changes
4. Add tutorial sections for new features

## Submitting Changes

### Pull Request Process

1. **Pre-submission Checklist:**
   - [ ] All tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Performance impact assessed
   - [ ] Breaking changes documented

2. **Pull Request Template:**
   Use the provided PR template and include:
   - Clear description of changes
   - Motivation and context
   - Testing methodology
   - Performance impact
   - Breaking changes (if any)

3. **Review Process:**
   - All PRs require at least 2 reviews
   - Performance-critical changes require performance team review
   - Breaking changes require maintainer approval
   - Documentation changes require technical writer review

### Pull Request Guidelines

**Title Format:**
```
<type>(scope): Brief description of changes

feat(monitoring): add real-time fairness metrics
fix(distributed): resolve memory leak in worker cleanup
docs(api): update agent configuration examples
```

**Description Template:**
```markdown
## Summary
Brief description of the changes and their purpose.

## Motivation
Why are these changes necessary? What problem do they solve?

## Changes
- List of specific changes made
- Include any API modifications
- Mention performance improvements/regressions

## Testing
- Describe testing approach
- Include performance test results
- List any manual testing performed

## Breaking Changes
- Document any breaking changes
- Provide migration guide for users

## Checklist
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Performance impact assessed
- [ ] Breaking changes documented
- [ ] Changelog updated
```

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, dependency versions
- **Reproduction Steps**: Minimal example to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Additional Context**: Logs, screenshots, related issues

### Feature Requests

Use the feature request template and include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives Considered**: Other approaches you've considered
- **Additional Context**: References, related work

### Performance Issues

For performance-related issues:

- **Performance Metrics**: Before/after measurements
- **Environment**: Hardware specs, system configuration
- **Workload**: Specific scenario causing performance issues
- **Profiling Data**: CPU/memory profiles if available

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **GitHub Discussions**: General questions, research discussions
- **Discord**: Real-time community chat and support
- **Mailing List**: Announcements and major updates

### Getting Help

1. **Check Documentation**: Search existing docs and examples
2. **Search Issues**: Look for similar problems/solutions
3. **Ask in Discussions**: For general questions
4. **Create an Issue**: For bugs or specific feature requests

### Code Review Etiquette

**As a Reviewer:**
- Be constructive and specific in feedback
- Explain the reasoning behind suggestions
- Acknowledge good practices and improvements
- Focus on code quality, not personal preferences

**As a Contributor:**
- Be responsive to feedback
- Ask questions if feedback is unclear
- Make requested changes promptly
- Thank reviewers for their time

## Recognition

### Contributor Recognition

We recognize contributions through:

- **Contributors File**: All contributors listed in CONTRIBUTORS.md
- **Release Notes**: Significant contributions highlighted
- **Annual Report**: Community awards and recognition
- **Conference Talks**: Opportunity to present work at conferences

### Types of Contributions

We value all types of contributions:

- **Code**: New features, bug fixes, performance improvements
- **Documentation**: Tutorials, API docs, examples
- **Testing**: Test cases, performance benchmarks, bug reports
- **Community**: Helping others, organizing events, outreach
- **Research**: Papers, datasets, evaluation metrics

### Mentorship Program

We offer mentorship for new contributors:

- **Pair Programming**: Work directly with experienced contributors
- **Guided Issues**: Specially labeled issues for newcomers
- **Code Review**: Detailed feedback to help you learn
- **Research Guidance**: Support for research-focused contributions

## Advanced Contributing

### Research Contributions

For research-focused contributions:

1. **Literature Review**: Understand existing work
2. **Methodology**: Clearly document experimental approach
3. **Reproducibility**: Provide complete reproduction instructions
4. **Validation**: Include statistical significance testing
5. **Publication**: Consider submitting to relevant venues

### Performance Contributions

For performance improvements:

1. **Baseline Measurements**: Establish current performance
2. **Profiling**: Identify bottlenecks using profilers
3. **Optimization**: Implement targeted improvements
4. **Benchmarking**: Demonstrate improvements quantitatively
5. **Documentation**: Update performance guidelines

### Integration Contributions

For new integrations:

1. **Interface Design**: Follow existing plugin patterns
2. **Error Handling**: Robust error handling and recovery
3. **Documentation**: Complete integration guide
4. **Testing**: Comprehensive integration tests
5. **Examples**: Working examples and tutorials

## Questions?

If you have questions about contributing, please:

1. Check this contributing guide
2. Search existing GitHub discussions
3. Ask in our Discord community
4. Open a GitHub discussion for complex questions
5. Contact maintainers directly for sensitive issues

Thank you for contributing to Agent Swarm Evaluation Arena! Your contributions help advance multi-agent AI research for the entire community.