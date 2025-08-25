"""
Project Analyzer: Intelligent analysis of project structure,
dependencies, and characteristics for autonomous decision making.
"""

import os
import ast
import json
import subprocess
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import re
from dataclasses import dataclass, field

@dataclass
class ProjectAnalysis:
    """Results of project analysis."""
    project_type: str
    language: str
    framework: str
    dependencies: List[str] = field(default_factory=list)
    file_structure: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, int] = field(default_factory=dict)
    quality_indicators: Dict[str, Any] = field(default_factory=dict)
    research_indicators: Dict[str, Any] = field(default_factory=dict)
    security_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_indicators: Dict[str, Any] = field(default_factory=dict)

class ProjectAnalyzer:
    """Intelligent project analysis engine."""
    
    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        self.framework_indicators = {
            'python': {
                'fastapi': ['fastapi', 'uvicorn'],
                'flask': ['flask'],
                'django': ['django'], 
                'pytorch': ['torch', 'pytorch'],
                'tensorflow': ['tensorflow'],
                'pandas': ['pandas'],
                'numpy': ['numpy'],
                'ray': ['ray']
            },
            'javascript': {
                'react': ['react', '@react'],
                'vue': ['vue'],
                'express': ['express'],
                'next': ['next']
            }
        }
    
    async def analyze_project(self, project_root: str) -> ProjectAnalysis:
        """Perform comprehensive project analysis."""
        
        project_path = Path(project_root)
        
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_root}")
        
        analysis = ProjectAnalysis(
            project_type="unknown",
            language="unknown", 
            framework="unknown"
        )
        
        # Basic file analysis
        await self._analyze_file_structure(project_path, analysis)
        
        # Language detection
        await self._detect_language(project_path, analysis)
        
        # Dependency analysis
        await self._analyze_dependencies(project_path, analysis)
        
        # Framework detection
        await self._detect_framework(analysis)
        
        # Project type classification
        await self._classify_project_type(project_path, analysis)
        
        # Complexity analysis
        await self._analyze_complexity(project_path, analysis)
        
        # Quality indicators
        await self._analyze_quality_indicators(project_path, analysis)
        
        # Research indicators
        await self._analyze_research_indicators(project_path, analysis)
        
        # Security analysis
        await self._analyze_security(project_path, analysis)
        
        # Performance indicators
        await self._analyze_performance_indicators(project_path, analysis)
        
        return analysis
    
    async def _analyze_file_structure(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze project file structure."""
        
        structure = {}
        file_counts = {}
        total_files = 0
        
        for root, dirs, files in os.walk(project_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'build', 'dist']]
            
            rel_root = os.path.relpath(root, project_path)
            if rel_root == '.':
                rel_root = ''
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                total_files += 1
                file_path = os.path.join(rel_root, file)
                
                # Count by extension
                ext = os.path.splitext(file)[1].lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1
                
                # Build structure
                if rel_root not in structure:
                    structure[rel_root] = []
                structure[rel_root].append(file)
        
        analysis.file_structure = {
            'structure': structure,
            'file_counts': file_counts,
            'total_files': total_files
        }
    
    async def _detect_language(self, project_path: Path, analysis: ProjectAnalysis):
        """Detect primary programming language."""
        
        file_counts = analysis.file_structure.get('file_counts', {})
        
        # Weight by significance
        language_scores = {}
        
        for ext, count in file_counts.items():
            if ext in self.supported_languages:
                lang = self.supported_languages[ext]
                
                # Weight certain files more heavily
                weight = 1
                if ext == '.py':
                    weight = 2  # Python files get higher weight
                elif ext in ['.js', '.ts']:
                    weight = 1.5
                
                language_scores[lang] = language_scores.get(lang, 0) + (count * weight)
        
        if language_scores:
            analysis.language = max(language_scores, key=language_scores.get)
        else:
            analysis.language = 'unknown'
    
    async def _analyze_dependencies(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze project dependencies."""
        
        dependencies = []
        
        # Python dependencies
        if analysis.language == 'python':
            dependencies.extend(await self._get_python_dependencies(project_path))
        
        # JavaScript dependencies  
        elif analysis.language in ['javascript', 'typescript']:
            dependencies.extend(await self._get_js_dependencies(project_path))
        
        analysis.dependencies = dependencies
    
    async def _get_python_dependencies(self, project_path: Path) -> List[str]:
        """Get Python dependencies from various files."""
        
        dependencies = []
        
        # From requirements.txt
        req_file = project_path / 'requirements.txt'
        if req_file.exists():
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name
                        dep = re.split(r'[>=<!=]', line)[0].strip()
                        dependencies.append(dep)
        
        # From pyproject.toml
        pyproject_file = project_path / 'pyproject.toml'
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                    
                deps = data.get('project', {}).get('dependencies', [])
                for dep in deps:
                    dep_name = re.split(r'[>=<!=]', dep)[0].strip()
                    dependencies.append(dep_name)
            except ImportError:
                # Fallback to simple parsing
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                    # Simple regex to find dependencies
                    matches = re.findall(r'"([^">=<!=]+)', content)
                    dependencies.extend(matches)
        
        # From setup.py (basic parsing)
        setup_file = project_path / 'setup.py'
        if setup_file.exists():
            try:
                with open(setup_file, 'r') as f:
                    content = f.read()
                    # Look for install_requires
                    matches = re.findall(r'install_requires=\[(.*?)\]', content, re.DOTALL)
                    if matches:
                        deps_str = matches[0]
                        deps = re.findall(r'"([^">=<!=]+)', deps_str)
                        dependencies.extend(deps)
            except Exception:
                pass
        
        return dependencies
    
    async def _get_js_dependencies(self, project_path: Path) -> List[str]:
        """Get JavaScript dependencies from package.json."""
        
        dependencies = []
        
        package_file = project_path / 'package.json'
        if package_file.exists():
            try:
                with open(package_file, 'r') as f:
                    data = json.load(f)
                    
                # Regular dependencies
                deps = data.get('dependencies', {})
                dependencies.extend(deps.keys())
                
                # Dev dependencies
                dev_deps = data.get('devDependencies', {})
                dependencies.extend(dev_deps.keys())
                
            except json.JSONDecodeError:
                pass
        
        return dependencies
    
    async def _detect_framework(self, analysis: ProjectAnalysis):
        """Detect framework based on dependencies."""
        
        language = analysis.language
        dependencies = analysis.dependencies
        
        if language in self.framework_indicators:
            framework_map = self.framework_indicators[language]
            
            # Score frameworks by dependency matches
            framework_scores = {}
            
            for framework, indicators in framework_map.items():
                score = 0
                for indicator in indicators:
                    if any(indicator in dep for dep in dependencies):
                        score += 1
                
                if score > 0:
                    framework_scores[framework] = score
            
            if framework_scores:
                analysis.framework = max(framework_scores, key=framework_scores.get)
            else:
                analysis.framework = 'custom'
        else:
            analysis.framework = 'unknown'
    
    async def _classify_project_type(self, project_path: Path, analysis: ProjectAnalysis):
        """Classify project type based on analysis."""
        
        dependencies = analysis.dependencies
        structure = analysis.file_structure.get('structure', {})
        
        # API service indicators
        api_indicators = ['fastapi', 'flask', 'django', 'express', 'gin']
        if any(dep in dependencies for dep in api_indicators):
            analysis.project_type = 'api_service'
            return
        
        # CLI tool indicators  
        cli_indicators = ['click', 'argparse', 'commander', 'yargs']
        if any(dep in dependencies for dep in cli_indicators) or 'cli.py' in str(structure):
            analysis.project_type = 'cli_tool'
            return
        
        # ML/Research indicators
        ml_indicators = ['torch', 'tensorflow', 'scikit-learn', 'pandas', 'numpy', 'research']
        if any(dep in dependencies for dep in ml_indicators):
            # Check for research-specific patterns
            research_files = ['experiment.py', 'benchmark.py', 'analysis.py']
            if any(file in str(structure) for file in research_files):
                analysis.project_type = 'research_platform'
            else:
                analysis.project_type = 'machine_learning'
            return
        
        # Web app indicators
        web_indicators = ['react', 'vue', 'angular', 'next', 'express']
        if any(dep in dependencies for dep in web_indicators):
            analysis.project_type = 'web_application'
            return
        
        # Library indicators
        lib_indicators = ['setup.py', '__init__.py', 'pyproject.toml']
        if any(indicator in str(structure) for indicator in lib_indicators):
            analysis.project_type = 'library'
            return
        
        # Default
        analysis.project_type = 'general'
    
    async def _analyze_complexity(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze code complexity metrics."""
        
        metrics = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'class_count': 0,
            'max_cyclomatic_complexity': 0,
            'average_function_length': 0
        }
        
        if analysis.language == 'python':
            await self._analyze_python_complexity(project_path, metrics)
        
        analysis.complexity_metrics = metrics
    
    async def _analyze_python_complexity(self, project_path: Path, metrics: Dict[str, int]):
        """Analyze Python code complexity."""
        
        function_lengths = []
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.split('\n')
                            
                            metrics['total_lines'] += len(lines)
                            
                            # Count code vs comment lines
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    metrics['code_lines'] += 1
                                elif line.startswith('#'):
                                    metrics['comment_lines'] += 1
                        
                        # AST analysis
                        try:
                            tree = ast.parse(content)
                            
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef):
                                    metrics['function_count'] += 1
                                    # Estimate function length
                                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                                        func_length = node.end_lineno - node.lineno
                                        function_lengths.append(func_length)
                                
                                elif isinstance(node, ast.ClassDef):
                                    metrics['class_count'] += 1
                        
                        except SyntaxError:
                            pass  # Skip files with syntax errors
                            
                    except (UnicodeDecodeError, FileNotFoundError):
                        pass  # Skip problematic files
        
        if function_lengths:
            metrics['average_function_length'] = int(sum(function_lengths) / len(function_lengths))
    
    async def _analyze_quality_indicators(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze code quality indicators."""
        
        indicators = {
            'has_tests': False,
            'has_ci_cd': False,
            'has_documentation': False,
            'has_type_hints': False,
            'has_linting_config': False,
            'test_coverage_estimate': 0.0,
            'documentation_completeness': 0.0
        }
        
        structure = analysis.file_structure.get('structure', {})
        all_files = []
        for files in structure.values():
            all_files.extend(files)
        
        # Check for tests
        test_indicators = ['test_', 'tests/', '_test.py', 'spec.py']
        indicators['has_tests'] = any(
            any(indicator in str(item) for indicator in test_indicators)
            for item in all_files
        )
        
        # Check for CI/CD
        ci_indicators = ['.github/', '.gitlab-ci.yml', 'Jenkinsfile', '.circleci/']
        indicators['has_ci_cd'] = any(
            indicator in str(structure) for indicator in ci_indicators
        )
        
        # Check for documentation
        doc_indicators = ['README.md', 'docs/', '*.md']
        indicators['has_documentation'] = any(
            'README' in file or 'docs' in str(structure) or file.endswith('.md')
            for file in all_files
        )
        
        # Check for linting config
        lint_indicators = ['.flake8', '.pylintrc', '.eslintrc', 'pyproject.toml']
        indicators['has_linting_config'] = any(
            indicator in str(all_files) for indicator in lint_indicators
        )
        
        analysis.quality_indicators = indicators
    
    async def _analyze_research_indicators(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze research-specific indicators."""
        
        indicators = {
            'has_experiments': False,
            'has_benchmarks': False,
            'has_publications': False,
            'has_datasets': False,
            'research_domains': [],
            'novelty_indicators': []
        }
        
        structure = analysis.file_structure.get('structure', {})
        all_files = []
        for files in structure.values():
            all_files.extend(files)
        
        # Research file patterns
        research_patterns = {
            'experiments': ['experiment', 'exp_', 'trial'],
            'benchmarks': ['benchmark', 'bench_', 'performance'],
            'publications': ['paper', 'publication', 'manuscript'],
            'datasets': ['data/', 'dataset', 'corpus']
        }
        
        for category, patterns in research_patterns.items():
            for pattern in patterns:
                if any(pattern in str(item).lower() for item in all_files):
                    indicators[f'has_{category}'] = True
                    break
        
        # Identify research domains based on dependencies and file names
        ml_deps = ['torch', 'tensorflow', 'scikit-learn']
        nlp_deps = ['transformers', 'nltk', 'spacy']
        vision_deps = ['opencv', 'PIL', 'torchvision']
        
        if any(dep in analysis.dependencies for dep in ml_deps):
            indicators['research_domains'].append('machine_learning')
        if any(dep in analysis.dependencies for dep in nlp_deps):
            indicators['research_domains'].append('natural_language_processing')
        if any(dep in analysis.dependencies for dep in vision_deps):
            indicators['research_domains'].append('computer_vision')
        
        # Look for novel algorithms
        novel_indicators = ['breakthrough', 'novel', 'new_', 'advanced_']
        for indicator in novel_indicators:
            if any(indicator in str(item).lower() for item in all_files):
                indicators['novelty_indicators'].append(indicator)
        
        analysis.research_indicators = indicators
    
    async def _analyze_security(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze security aspects."""
        
        security_analysis = {
            'has_security_config': False,
            'potential_vulnerabilities': [],
            'security_dependencies': [],
            'secrets_detected': False
        }
        
        # Check for security configurations
        security_configs = ['security.py', '.env.example', 'secrets/', 'config/security']
        structure = analysis.file_structure.get('structure', {})
        
        for config in security_configs:
            if any(config in str(item) for item in structure.values()):
                security_analysis['has_security_config'] = True
                break
        
        # Check for security-related dependencies
        security_deps = ['cryptography', 'pycryptodome', 'jwt', 'oauth', 'ssl']
        for dep in analysis.dependencies:
            if any(sec_dep in dep.lower() for sec_dep in security_deps):
                security_analysis['security_dependencies'].append(dep)
        
        analysis.security_analysis = security_analysis
    
    async def _analyze_performance_indicators(self, project_path: Path, analysis: ProjectAnalysis):
        """Analyze performance indicators."""
        
        indicators = {
            'has_caching': False,
            'has_async': False,
            'has_profiling': False,
            'performance_dependencies': [],
            'estimated_complexity': 'O(n)'
        }
        
        # Check for performance-related dependencies
        perf_deps = ['redis', 'memcached', 'asyncio', 'aiohttp', 'celery', 'ray']
        for dep in analysis.dependencies:
            if any(perf_dep in dep.lower() for perf_dep in perf_deps):
                indicators['performance_dependencies'].append(dep)
        
        # Check for async patterns
        if 'asyncio' in analysis.dependencies or 'aiohttp' in analysis.dependencies:
            indicators['has_async'] = True
        
        # Check for caching
        cache_deps = ['redis', 'memcached', 'cache']
        if any(dep in analysis.dependencies for dep in cache_deps):
            indicators['has_caching'] = True
        
        analysis.performance_indicators = indicators