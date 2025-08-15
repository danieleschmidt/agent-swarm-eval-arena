"""
Publication-Ready Research Framework for Multi-Agent Reinforcement Learning

This module provides comprehensive tools for conducting and documenting research
experiments that meet academic publication standards, including:
- Reproducible experimental design
- Statistical significance testing
- Baseline comparisons
- Peer-review ready documentation
- Automated result visualization
"""

import json
import os
import time
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from datetime import datetime

from ..core.arena import Arena, SimulationResults
from ..core.config import SwarmConfig
from ..core.agent import BaseAgent
from .experiment import ExperimentConfig


@dataclass
class ResearchHypothesis:
    """Define a research hypothesis with measurable success criteria."""
    
    hypothesis_id: str
    title: str
    description: str
    null_hypothesis: str
    alternative_hypothesis: str
    success_metrics: List[str]
    statistical_test: str = "mann_whitney_u"  # Default non-parametric test
    significance_level: float = 0.05
    effect_size_threshold: float = 0.2  # Cohen's d threshold
    expected_direction: str = "two_tailed"  # "greater", "less", "two_tailed"


@dataclass
class BaselineConfiguration:
    """Configuration for baseline approaches."""
    
    name: str
    description: str
    agent_class: type
    parameters: Dict[str, Any] = field(default_factory=dict)
    reference_paper: Optional[str] = None
    implementation_notes: str = ""


@dataclass
class ExperimentalResults:
    """Comprehensive results from a research experiment."""
    
    experiment_id: str
    hypothesis: ResearchHypothesis
    baseline_results: Dict[str, Dict[str, float]]
    novel_results: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    reproducibility_data: Dict[str, Any]
    runtime_stats: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_statistically_significant(self, metric: str) -> bool:
        """Check if results are statistically significant for given metric."""
        if metric not in self.statistical_tests:
            return False
        return self.statistical_tests[metric]["p_value"] < self.hypothesis.significance_level
    
    def get_effect_size(self, metric: str) -> float:
        """Get effect size for given metric."""
        return self.effect_sizes.get(metric, 0.0)
    
    def is_practically_significant(self, metric: str) -> bool:
        """Check if effect size meets practical significance threshold."""
        effect_size = abs(self.get_effect_size(metric))
        return effect_size >= self.hypothesis.effect_size_threshold


class PublicationFramework:
    """Framework for conducting publication-ready research experiments."""
    
    def __init__(self, project_name: str, output_dir: str = "research_output"):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "experiments").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "papers").mkdir(exist_ok=True)
        
        self.experiments: List[ExperimentalResults] = []
        self.baselines: Dict[str, BaselineConfiguration] = {}
        
    def register_baseline(self, baseline: BaselineConfiguration) -> None:
        """Register a baseline approach for comparison."""
        self.baselines[baseline.name] = baseline
        print(f"✅ Registered baseline: {baseline.name}")
    
    def design_experiment(
        self,
        hypothesis: ResearchHypothesis,
        config: SwarmConfig,
        num_runs: int = 30,
        num_episodes: int = 100
    ) -> ExperimentConfig:
        """Design a statistically sound experiment."""
        
        # Calculate required sample size for desired statistical power
        if num_runs < 30:
            print(f"⚠️  Warning: {num_runs} runs may not provide sufficient statistical power.")
            print("   Recommend at least 30 runs for reliable results.")
        
        exp_config = ExperimentConfig(
            name=f"{self.project_name}_{hypothesis.hypothesis_id}",
            description=f"Testing hypothesis: {hypothesis.title}",
            seed=42,  # For reproducibility
            parameters={
                "num_runs": num_runs,
                "num_episodes": num_episodes,
                "config": asdict(config)
            }
        )
        
        return exp_config
    
    def run_baseline_comparison(
        self,
        hypothesis: ResearchHypothesis,
        novel_agent_class: type,
        config: SwarmConfig,
        num_runs: int = 30,
        num_episodes: int = 100,
        baseline_names: Optional[List[str]] = None
    ) -> ExperimentalResults:
        """Run complete baseline comparison experiment."""
        
        print(f"🧪 Starting research experiment: {hypothesis.title}")
        start_time = time.time()
        
        # Use all baselines if none specified
        if baseline_names is None:
            baseline_names = list(self.baselines.keys())
        
        if not baseline_names:
            raise ValueError("No baselines available. Register baselines first.")
        
        # Run baseline experiments
        baseline_results = {}
        for baseline_name in baseline_names:
            print(f"📊 Running baseline: {baseline_name}")
            baseline = self.baselines[baseline_name]
            results = self._run_multiple_experiments(
                baseline.agent_class, config, num_runs, num_episodes, baseline.parameters
            )
            baseline_results[baseline_name] = results
        
        # Run novel approach experiments
        print(f"🚀 Running novel approach...")
        novel_results = self._run_multiple_experiments(
            novel_agent_class, config, num_runs, num_episodes
        )
        
        # Perform statistical analysis
        statistical_tests = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        for metric in hypothesis.success_metrics:
            # Compare novel approach with each baseline
            for baseline_name in baseline_names:
                baseline_metric_values = [
                    run_results[metric] for run_results in baseline_results[baseline_name]
                ]
                novel_metric_values = [
                    run_results[metric] for run_results in novel_results
                ]
                
                # Statistical test
                test_key = f"{baseline_name}_vs_novel_{metric}"
                statistical_tests[test_key] = self._perform_statistical_test(
                    baseline_metric_values, novel_metric_values, hypothesis.statistical_test
                )
                
                # Effect size (Cohen's d)
                effect_sizes[test_key] = self._calculate_cohens_d(
                    baseline_metric_values, novel_metric_values
                )
                
                # Confidence interval for difference
                confidence_intervals[test_key] = self._bootstrap_confidence_interval(
                    baseline_metric_values, novel_metric_values
                )
        
        # Create experimental results
        experiment_id = f"{hypothesis.hypothesis_id}_{int(time.time())}"
        results = ExperimentalResults(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            baseline_results={name: self._summarize_results(baseline_results[name]) 
                            for name in baseline_names},
            novel_results=self._summarize_results(novel_results),
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            reproducibility_data=self._capture_reproducibility_data(),
            runtime_stats={"total_runtime_seconds": time.time() - start_time}
        )
        
        # Save results
        self._save_experiment_results(results)
        self.experiments.append(results)
        
        # Generate summary
        self._print_experiment_summary(results)
        
        return results
    
    def _run_multiple_experiments(
        self,
        agent_class: type,
        config: SwarmConfig,
        num_runs: int,
        num_episodes: int,
        agent_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, float]]:
        """Run multiple independent experiments with different seeds."""
        
        results = []
        for run in range(num_runs):
            # Set unique seed for each run
            run_config = SwarmConfig(**{**asdict(config), 'seed': 42 + run})
            
            # Create arena and add agents
            arena = Arena(run_config)
            if agent_params:
                # Pass parameters to agent if supported
                try:
                    agents = [agent_class(**agent_params) for _ in range(config.num_agents)]
                except TypeError:
                    agents = [agent_class() for _ in range(config.num_agents)]
            else:
                agents = [agent_class() for _ in range(config.num_agents)]
            
            arena.agents = agents
            
            # Run simulation
            simulation_results = arena.run(num_episodes, verbose=False)
            
            # Extract metrics
            run_results = {
                "mean_reward": simulation_results.mean_reward,
                "fairness_index": simulation_results.fairness_index or 0.0,
                "total_steps": simulation_results.total_steps,
                "episode_length": simulation_results.episode_length
            }
            
            # Add custom metrics if available
            if hasattr(simulation_results, 'environment_stats'):
                for key, value in simulation_results.environment_stats.items():
                    if isinstance(value, (int, float)):
                        run_results[f"env_{key}"] = value
            
            results.append(run_results)
        
        return results
    
    def _perform_statistical_test(
        self,
        baseline_data: List[float],
        novel_data: List[float],
        test_type: str
    ) -> Dict[str, Any]:
        """Perform appropriate statistical test."""
        
        if test_type == "mann_whitney_u":
            statistic, p_value = stats.mannwhitneyu(
                novel_data, baseline_data, alternative='two-sided'
            )
            test_name = "Mann-Whitney U"
        elif test_type == "t_test":
            statistic, p_value = stats.ttest_ind(novel_data, baseline_data)
            test_name = "Independent t-test"
        elif test_type == "wilcoxon":
            # Paired test - requires same number of samples
            min_len = min(len(baseline_data), len(novel_data))
            statistic, p_value = stats.wilcoxon(
                novel_data[:min_len], baseline_data[:min_len]
            )
            test_name = "Wilcoxon signed-rank"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            "test_name": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "baseline_mean": np.mean(baseline_data),
            "baseline_std": np.std(baseline_data),
            "novel_mean": np.mean(novel_data),
            "novel_std": np.std(novel_data),
            "baseline_n": len(baseline_data),
            "novel_n": len(novel_data)
        }
    
    def _calculate_cohens_d(self, baseline_data: List[float], novel_data: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(baseline_data), len(novel_data)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(baseline_data, ddof=1) + 
             (n2 - 1) * np.var(novel_data, ddof=1)) / (n1 + n2 - 2)
        )
        
        # Cohen's d
        d = (np.mean(novel_data) - np.mean(baseline_data)) / pooled_std
        return float(d)
    
    def _bootstrap_confidence_interval(
        self,
        baseline_data: List[float],
        novel_data: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference in means."""
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            baseline_sample = np.random.choice(baseline_data, size=len(baseline_data), replace=True)
            novel_sample = np.random.choice(novel_data, size=len(novel_data), replace=True)
            diff = np.mean(novel_sample) - np.mean(baseline_sample)
            bootstrap_diffs.append(diff)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return (float(lower), float(upper))
    
    def _summarize_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Summarize multiple runs into mean and std."""
        if not results:
            return {}
        
        summary = {}
        for metric in results[0].keys():
            values = [run[metric] for run in results]
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)
        
        return summary
    
    def _capture_reproducibility_data(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        
        reproducibility_data = {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform
        }
        
        # Try to get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=Path(__file__).parent.parent.parent,
                universal_newlines=True
            ).strip()
            reproducibility_data["git_commit"] = git_commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            reproducibility_data["git_commit"] = "unknown"
        
        return reproducibility_data
    
    def _save_experiment_results(self, results: ExperimentalResults) -> None:
        """Save experiment results to file."""
        
        # Save as JSON (for human reading)
        json_path = self.output_dir / "experiments" / f"{results.experiment_id}.json"
        with open(json_path, 'w') as f:
            # Convert dataclass to dict, handling datetime objects
            results_dict = asdict(results)
            results_dict["timestamp"] = results.timestamp.isoformat()
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save as pickle (for programmatic access)
        pickle_path = self.output_dir / "data" / f"{results.experiment_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Results saved to {json_path}")
    
    def _print_experiment_summary(self, results: ExperimentalResults) -> None:
        """Print a comprehensive summary of experiment results."""
        
        print("\n" + "=" * 80)
        print(f"📋 EXPERIMENT SUMMARY: {results.hypothesis.title}")
        print("=" * 80)
        
        print(f"\n🎯 Hypothesis: {results.hypothesis.description}")
        print(f"📊 Metrics Tested: {', '.join(results.hypothesis.success_metrics)}")
        print(f"⏱️  Runtime: {results.runtime_stats['total_runtime_seconds']:.1f} seconds")
        
        print("\n📈 STATISTICAL RESULTS:")
        significant_results = []
        
        for test_key, test_result in results.statistical_tests.items():
            baseline_name = test_key.split('_vs_')[0]
            metric = test_key.split('_vs_novel_')[1]
            
            is_significant = test_result["p_value"] < results.hypothesis.significance_level
            effect_size = results.effect_sizes[test_key]
            
            print(f"\n  {metric.upper()} ({baseline_name} vs Novel):")
            print(f"    • {test_result['test_name']}: p = {test_result['p_value']:.4f}")
            print(f"    • Effect size (Cohen's d): {effect_size:.3f}")
            print(f"    • Baseline: {test_result['baseline_mean']:.3f} ± {test_result['baseline_std']:.3f}")
            print(f"    • Novel: {test_result['novel_mean']:.3f} ± {test_result['novel_std']:.3f}")
            
            if is_significant:
                print(f"    ✅ STATISTICALLY SIGNIFICANT (p < {results.hypothesis.significance_level})")
                significant_results.append((metric, baseline_name))
            else:
                print(f"    ❌ Not statistically significant")
            
            if abs(effect_size) >= results.hypothesis.effect_size_threshold:
                print(f"    ✅ PRACTICALLY SIGNIFICANT (|d| ≥ {results.hypothesis.effect_size_threshold})")
            else:
                print(f"    ⚠️  Small effect size")
        
        print(f"\n🏆 HYPOTHESIS RESULT:")
        if significant_results:
            print(f"    ✅ SUPPORTED - Found {len(significant_results)} significant improvements")
            for metric, baseline in significant_results:
                print(f"       • {metric} vs {baseline}")
        else:
            print(f"    ❌ NOT SUPPORTED - No statistically significant improvements found")
        
        print("\n" + "=" * 80)
    
    def generate_publication_figures(
        self,
        experiment_id: str,
        figure_style: str = "publication"
    ) -> List[str]:
        """Generate publication-ready figures for an experiment."""
        
        # Find experiment
        experiment = None
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                experiment = exp
                break
        
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Set publication style
        if figure_style == "publication":
            plt.style.use('seaborn-v0_8-paper')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'figure.titlesize': 16,
                'figure.dpi': 300
            })
        
        figure_files = []
        
        # 1. Performance comparison bar plot
        fig_file = self._create_performance_comparison_figure(experiment)
        figure_files.append(fig_file)
        
        # 2. Effect size visualization
        fig_file = self._create_effect_size_figure(experiment)
        figure_files.append(fig_file)
        
        # 3. Statistical significance heatmap
        fig_file = self._create_significance_heatmap(experiment)
        figure_files.append(fig_file)
        
        print(f"📊 Generated {len(figure_files)} publication figures")
        return figure_files
    
    def _create_performance_comparison_figure(self, experiment: ExperimentalResults) -> str:
        """Create performance comparison bar plot."""
        
        # Prepare data
        metrics = experiment.hypothesis.success_metrics
        baseline_names = list(experiment.baseline_results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Collect data
            approaches = baseline_names + ["Novel Approach"]
            means = []
            stds = []
            
            for baseline_name in baseline_names:
                means.append(experiment.baseline_results[baseline_name][f"{metric}_mean"])
                stds.append(experiment.baseline_results[baseline_name][f"{metric}_std"])
            
            means.append(experiment.novel_results[f"{metric}_mean"])
            stds.append(experiment.novel_results[f"{metric}_std"])
            
            # Create bar plot
            bars = ax.bar(approaches, means, yerr=stds, capsize=5, 
                         color=['lightblue'] * len(baseline_names) + ['red'])
            
            # Highlight novel approach
            bars[-1].set_color('darkred')
            bars[-1].set_alpha(0.8)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
            ax.tick_params(axis='x', rotation=45)
            
            # Add significance stars
            for j, baseline_name in enumerate(baseline_names):
                test_key = f"{baseline_name}_vs_novel_{metric}"
                if test_key in experiment.statistical_tests:
                    p_value = experiment.statistical_tests[test_key]["p_value"]
                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    
                    if stars:
                        max_height = max(means[j] + stds[j], means[-1] + stds[-1])
                        ax.text(j, max_height * 1.1, stars, ha='center', fontsize=14)
        
        plt.tight_layout()
        fig_path = self.output_dir / "figures" / f"{experiment.experiment_id}_performance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_path)
    
    def _create_effect_size_figure(self, experiment: ExperimentalResults) -> str:
        """Create effect size visualization."""
        
        # Prepare data
        effect_data = []
        for test_key, effect_size in experiment.effect_sizes.items():
            baseline_name = test_key.split('_vs_')[0]
            metric = test_key.split('_vs_novel_')[1]
            
            # Get confidence interval
            ci_lower, ci_upper = experiment.confidence_intervals[test_key]
            
            effect_data.append({
                'Baseline': baseline_name,
                'Metric': metric,
                'Effect Size': effect_size,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper
            })
        
        df = pd.DataFrame(effect_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Forest plot
        y_positions = range(len(df))
        colors = ['red' if abs(es) >= experiment.hypothesis.effect_size_threshold else 'gray' 
                 for es in df['Effect Size']]
        
        # Plot effect sizes with confidence intervals
        ax.errorbar(df['Effect Size'], y_positions, 
                   xerr=[df['Effect Size'] - df['CI Lower'], 
                         df['CI Upper'] - df['Effect Size']], 
                   fmt='o', capsize=5, color='black')
        
        # Color points by significance
        for i, (_, row) in enumerate(df.iterrows()):
            ax.scatter(row['Effect Size'], i, 
                      c=colors[i], s=100, alpha=0.7)
        
        # Add vertical lines for reference
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(experiment.hypothesis.effect_size_threshold, 
                  color='green', linestyle='--', alpha=0.5, 
                  label=f'Practical significance threshold')
        ax.axvline(-experiment.hypothesis.effect_size_threshold, 
                  color='green', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_ylabel("Comparison")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{row['Metric']} vs {row['Baseline']}" 
                           for _, row in df.iterrows()])
        ax.set_title("Effect Sizes with 95% Confidence Intervals")
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / "figures" / f"{experiment.experiment_id}_effect_sizes.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_path)
    
    def _create_significance_heatmap(self, experiment: ExperimentalResults) -> str:
        """Create statistical significance heatmap."""
        
        # Prepare data matrix
        metrics = experiment.hypothesis.success_metrics
        baselines = list(experiment.baseline_results.keys())
        
        # Create p-value matrix
        p_matrix = np.ones((len(baselines), len(metrics)))
        effect_matrix = np.zeros((len(baselines), len(metrics)))
        
        for i, baseline in enumerate(baselines):
            for j, metric in enumerate(metrics):
                test_key = f"{baseline}_vs_novel_{metric}"
                if test_key in experiment.statistical_tests:
                    p_matrix[i, j] = experiment.statistical_tests[test_key]["p_value"]
                    effect_matrix[i, j] = experiment.effect_sizes[test_key]
        
        # Create significance categories
        sig_matrix = np.where(p_matrix < 0.001, 3,
                     np.where(p_matrix < 0.01, 2,
                     np.where(p_matrix < 0.05, 1, 0)))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # P-value heatmap
        sns.heatmap(sig_matrix, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=baselines,
                   annot=p_matrix, fmt='.3f',
                   cmap='RdYlBu_r', vmin=0, vmax=3,
                   cbar_kws={'label': 'Significance Level'},
                   ax=ax1)
        ax1.set_title('Statistical Significance\n(p-values)')
        
        # Effect size heatmap
        sns.heatmap(effect_matrix,
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=baselines,
                   annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0,
                   cbar_kws={'label': "Effect Size (Cohen's d)"},
                   ax=ax2)
        ax2.set_title('Effect Sizes')
        
        plt.tight_layout()
        fig_path = self.output_dir / "figures" / f"{experiment.experiment_id}_significance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_path)
    
    def generate_latex_table(self, experiment_id: str) -> str:
        """Generate LaTeX table for publication."""
        
        # Find experiment
        experiment = None
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                experiment = exp
                break
        
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create LaTeX table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Performance Comparison Results}\n"
        latex += f"\\label{{tab:{experiment.experiment_id}}}\n"
        latex += "\\begin{tabular}{l" + "c" * len(experiment.hypothesis.success_metrics) + "}\n"
        latex += "\\toprule\n"
        
        # Header
        header = "Approach"
        for metric in experiment.hypothesis.success_metrics:
            header += f" & {metric.replace('_', ' ').title()}"
        latex += header + " \\\\\n"
        latex += "\\midrule\n"
        
        # Baseline rows
        for baseline_name in experiment.baseline_results.keys():
            row = baseline_name
            for metric in experiment.hypothesis.success_metrics:
                mean = experiment.baseline_results[baseline_name][f"{metric}_mean"]
                std = experiment.baseline_results[baseline_name][f"{metric}_std"]
                row += f" & ${mean:.3f} \\pm {std:.3f}$"
            latex += row + " \\\\\n"
        
        # Novel approach row
        row = "\\textbf{Novel Approach}"
        for metric in experiment.hypothesis.success_metrics:
            mean = experiment.novel_results[f"{metric}_mean"]
            std = experiment.novel_results[f"{metric}_std"]
            
            # Check if significantly better than any baseline
            is_significant = False
            for baseline_name in experiment.baseline_results.keys():
                test_key = f"{baseline_name}_vs_novel_{metric}"
                if (test_key in experiment.statistical_tests and 
                    experiment.statistical_tests[test_key]["p_value"] < experiment.hypothesis.significance_level):
                    is_significant = True
                    break
            
            if is_significant:
                row += f" & $\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
            else:
                row += f" & ${mean:.3f} \\pm {std:.3f}$"
        
        latex += row + " \\\\\n"
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        latex_path = self.output_dir / "papers" / f"{experiment.experiment_id}_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex)
        
        print(f"📄 LaTeX table saved to {latex_path}")
        return str(latex_path)
    
    def export_experiment_data(self, experiment_id: str, format: str = "csv") -> str:
        """Export raw experimental data for sharing."""
        
        # Find experiment
        experiment = None
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                experiment = exp
                break
        
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load raw data from pickle
        pickle_path = self.output_dir / "data" / f"{experiment.experiment_id}.pkl"
        
        if format == "csv":
            # Create summary CSV
            summary_data = []
            
            # Add baseline results
            for baseline_name, results in experiment.baseline_results.items():
                for metric, value in results.items():
                    if metric.endswith("_mean"):
                        base_metric = metric[:-5]
                        summary_data.append({
                            "approach": baseline_name,
                            "metric": base_metric,
                            "mean": value,
                            "std": results[f"{base_metric}_std"],
                            "min": results[f"{base_metric}_min"],
                            "max": results[f"{base_metric}_max"]
                        })
            
            # Add novel results
            for metric, value in experiment.novel_results.items():
                if metric.endswith("_mean"):
                    base_metric = metric[:-5]
                    summary_data.append({
                        "approach": "Novel Approach",
                        "metric": base_metric,
                        "mean": value,
                        "std": experiment.novel_results[f"{base_metric}_std"],
                        "min": experiment.novel_results[f"{base_metric}_min"],
                        "max": experiment.novel_results[f"{base_metric}_max"]
                    })
            
            df = pd.DataFrame(summary_data)
            export_path = self.output_dir / "data" / f"{experiment.experiment_id}_summary.csv"
            df.to_csv(export_path, index=False)
        
        elif format == "json":
            export_path = self.output_dir / "data" / f"{experiment.experiment_id}_full.json"
            with open(export_path, 'w') as f:
                results_dict = asdict(experiment)
                results_dict["timestamp"] = experiment.timestamp.isoformat()
                json.dump(results_dict, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"💾 Experiment data exported to {export_path}")
        return str(export_path)


# Example usage and research templates
class ResearchTemplates:
    """Pre-defined research templates for common MARL scenarios."""
    
    @staticmethod
    def emergent_cooperation_hypothesis() -> ResearchHypothesis:
        """Research template for emergent cooperation studies."""
        return ResearchHypothesis(
            hypothesis_id="emergent_cooperation",
            title="Novel Communication Protocol Enhances Emergent Cooperation",
            description="Testing whether our novel communication protocol leads to significantly better cooperative behavior compared to baseline approaches.",
            null_hypothesis="The novel communication protocol does not improve cooperation metrics compared to baselines.",
            alternative_hypothesis="The novel communication protocol significantly improves cooperation metrics compared to baselines.",
            success_metrics=["mean_reward", "fairness_index"],
            statistical_test="mann_whitney_u",
            significance_level=0.05,
            effect_size_threshold=0.3
        )
    
    @staticmethod
    def scalability_hypothesis() -> ResearchHypothesis:
        """Research template for scalability studies."""
        return ResearchHypothesis(
            hypothesis_id="scalability_performance",
            title="Algorithm Maintains Performance at Scale",
            description="Testing whether our algorithm maintains performance advantages as the number of agents increases.",
            null_hypothesis="Performance advantage diminishes significantly with increased agent count.",
            alternative_hypothesis="Performance advantage is maintained across different agent population sizes.",
            success_metrics=["mean_reward", "episode_length"],
            statistical_test="mann_whitney_u",
            significance_level=0.05,
            effect_size_threshold=0.2
        )
    
    @staticmethod
    def create_standard_baselines() -> List[BaselineConfiguration]:
        """Create standard baseline configurations."""
        from ..core.agent import RandomAgent, CooperativeAgent, CompetitiveAgent
        
        return [
            BaselineConfiguration(
                name="random_baseline",
                description="Random action selection baseline",
                agent_class=RandomAgent,
                reference_paper="Standard random baseline",
                implementation_notes="Pure random action selection for comparison"
            ),
            BaselineConfiguration(
                name="cooperative_baseline",
                description="Simple cooperative strategy",
                agent_class=CooperativeAgent,
                reference_paper="Cooperative multi-agent systems baseline",
                implementation_notes="Agents move towards centroid of nearby agents"
            ),
            BaselineConfiguration(
                name="competitive_baseline",
                description="Simple competitive strategy",
                agent_class=CompetitiveAgent,
                reference_paper="Competitive multi-agent systems baseline",
                implementation_notes="Agents move towards nearest resource"
            )
        ]