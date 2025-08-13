"""Fairness analysis and metrics for multi-agent systems."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FairnessMetrics:
    """Container for fairness measurement results."""
    
    gini_coefficient: float
    envy_freeness: float
    proportional_fairness: float
    max_min_fairness: float
    jain_fairness_index: float
    atkinson_index: float
    welfare: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "gini_coefficient": self.gini_coefficient,
            "envy_freeness": self.envy_freeness,
            "proportional_fairness": self.proportional_fairness,
            "max_min_fairness": self.max_min_fairness,
            "jain_fairness_index": self.jain_fairness_index,
            "atkinson_index": self.atkinson_index,
            "welfare": self.welfare
        }


class FairnessAnalyzer:
    """Analyzer for fairness metrics in multi-agent resource allocation."""
    
    def __init__(self, epsilon: float = 0.5):
        """Initialize fairness analyzer.
        
        Args:
            epsilon: Inequality aversion parameter for Atkinson index
        """
        self.epsilon = epsilon
        self.history: List[FairnessMetrics] = []
    
    def analyze_allocation(self, 
                          allocations: Dict[int, float],
                          contributions: Optional[Dict[int, float]] = None,
                          needs: Optional[Dict[int, float]] = None) -> FairnessMetrics:
        """Analyze fairness of resource allocation.
        
        Args:
            allocations: Dict mapping agent_id to allocated resources
            contributions: Dict mapping agent_id to contribution/effort
            needs: Dict mapping agent_id to needs/requirements
            
        Returns:
            FairnessMetrics object with computed fairness measures
        """
        if not allocations:
            return self._empty_metrics()
        
        agent_ids = list(allocations.keys())
        allocation_values = np.array([allocations[aid] for aid in agent_ids])
        
        # Set defaults for optional parameters
        if contributions is None:
            contributions = {aid: 1.0 for aid in agent_ids}
        if needs is None:
            needs = {aid: 1.0 for aid in agent_ids}
        
        contribution_values = np.array([contributions.get(aid, 1.0) for aid in agent_ids])
        need_values = np.array([needs.get(aid, 1.0) for aid in agent_ids])
        
        # Calculate fairness metrics
        metrics = FairnessMetrics(
            gini_coefficient=self._calculate_gini_coefficient(allocation_values),
            envy_freeness=self._calculate_envy_freeness(allocations),
            proportional_fairness=self._calculate_proportional_fairness(
                allocation_values, contribution_values
            ),
            max_min_fairness=self._calculate_max_min_fairness(allocation_values),
            jain_fairness_index=self._calculate_jain_fairness_index(allocation_values),
            atkinson_index=self._calculate_atkinson_index(allocation_values),
            welfare=self._calculate_welfare(allocation_values)
        )
        
        self.history.append(metrics)
        return metrics
    
    def analyze_trajectory_fairness(self, 
                                   episode_rewards: Dict[int, List[float]]) -> Dict[str, Any]:
        """Analyze fairness over an entire episode trajectory.
        
        Args:
            episode_rewards: Dict mapping agent_id to list of rewards over time
            
        Returns:
            Dictionary with temporal fairness analysis
        """
        if not episode_rewards:
            return {}
        
        agent_ids = list(episode_rewards.keys())
        max_steps = max(len(rewards) for rewards in episode_rewards.values())
        
        # Calculate fairness at each timestep
        temporal_metrics = []
        
        for step in range(max_steps):
            step_allocations = {}
            
            for agent_id in agent_ids:
                if step < len(episode_rewards[agent_id]):
                    step_allocations[agent_id] = episode_rewards[agent_id][step]
                else:
                    step_allocations[agent_id] = 0.0
            
            if any(v != 0 for v in step_allocations.values()):
                metrics = self.analyze_allocation(step_allocations)
                temporal_metrics.append(metrics)
        
        # Calculate summary statistics
        if temporal_metrics:
            avg_gini = np.mean([m.gini_coefficient for m in temporal_metrics])
            std_gini = np.std([m.gini_coefficient for m in temporal_metrics])
            
            avg_envy = np.mean([m.envy_freeness for m in temporal_metrics])
            trend_gini = self._calculate_trend([m.gini_coefficient for m in temporal_metrics])
        else:
            avg_gini = std_gini = avg_envy = trend_gini = 0.0
        
        return {
            "temporal_metrics": [m.to_dict() for m in temporal_metrics],
            "summary": {
                "average_gini_coefficient": avg_gini,
                "gini_coefficient_std": std_gini,
                "average_envy_freeness": avg_envy,
                "gini_trend": trend_gini,
                "fairness_improvement": trend_gini < 0  # Decreasing Gini is improvement
            }
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement.
        
        Lower values (closer to 0) indicate more equal distribution.
        """
        if len(values) == 0 or np.all(values == 0):
            return 0.0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(values)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))
    
    def _calculate_envy_freeness(self, allocations: Dict[int, float]) -> float:
        """Calculate envy-freeness metric.
        
        Returns fraction of agents that don't envy others' allocations.
        """
        if len(allocations) <= 1:
            return 1.0
        
        agent_ids = list(allocations.keys())
        envy_free_count = 0
        
        for agent1 in agent_ids:
            agent1_allocation = allocations[agent1]
            
            # Check if agent1 envies any other agent
            envies_someone = False
            for agent2 in agent_ids:
                if agent1 != agent2:
                    agent2_allocation = allocations[agent2]
                    
                    # Simple envy: agent1 envies agent2 if agent2 got more
                    if agent2_allocation > agent1_allocation:
                        envies_someone = True
                        break
            
            if not envies_someone:
                envy_free_count += 1
        
        return envy_free_count / len(agent_ids)
    
    def _calculate_proportional_fairness(self, 
                                       allocations: np.ndarray, 
                                       contributions: np.ndarray) -> float:
        """Calculate proportional fairness based on contributions.
        
        Measures how well allocations match contributions.
        """
        if len(allocations) == 0 or np.sum(contributions) == 0:
            return 1.0
        
        # Normalize allocations and contributions
        norm_allocations = allocations / np.sum(allocations) if np.sum(allocations) > 0 else allocations
        norm_contributions = contributions / np.sum(contributions)
        
        # Calculate proportional fairness as 1 - MSE
        mse = np.mean((norm_allocations - norm_contributions) ** 2)
        return max(0.0, 1.0 - mse)
    
    def _calculate_max_min_fairness(self, values: np.ndarray) -> float:
        """Calculate max-min fairness (ratio of min to max).
        
        Higher values indicate more fairness.
        """
        if len(values) == 0:
            return 1.0
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == 0:
            return 1.0
        
        return min_val / max_val
    
    def _calculate_jain_fairness_index(self, values: np.ndarray) -> float:
        """Calculate Jain's fairness index.
        
        Values closer to 1 indicate more fairness.
        """
        if len(values) == 0:
            return 1.0
        
        sum_values = np.sum(values)
        sum_squares = np.sum(values ** 2)
        
        if sum_squares == 0:
            return 1.0
        
        jain_index = (sum_values ** 2) / (len(values) * sum_squares)
        return min(1.0, jain_index)
    
    def _calculate_atkinson_index(self, values: np.ndarray) -> float:
        """Calculate Atkinson inequality index.
        
        Lower values indicate more equality.
        """
        if len(values) == 0 or np.any(values <= 0):
            return 0.0
        
        mean_value = np.mean(values)
        
        if mean_value == 0:
            return 0.0
        
        # Calculate equally distributed equivalent income
        if self.epsilon == 1:
            ede = np.exp(np.mean(np.log(values)))
        else:
            ede = (np.mean(values ** (1 - self.epsilon))) ** (1 / (1 - self.epsilon))
        
        atkinson = 1 - (ede / mean_value)
        return max(0.0, min(1.0, atkinson))
    
    def _calculate_welfare(self, values: np.ndarray) -> float:
        """Calculate total welfare (sum of all allocations)."""
        return float(np.sum(values))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _empty_metrics(self) -> FairnessMetrics:
        """Return empty/default fairness metrics."""
        return FairnessMetrics(
            gini_coefficient=0.0,
            envy_freeness=1.0,
            proportional_fairness=1.0,
            max_min_fairness=1.0,
            jain_fairness_index=1.0,
            atkinson_index=0.0,
            welfare=0.0
        )
    
    def generate_report(self, 
                       allocations: Dict[int, float],
                       contributions: Optional[Dict[int, float]] = None) -> str:
        """Generate a comprehensive fairness report.
        
        Args:
            allocations: Current resource allocations
            contributions: Agent contributions/efforts
            
        Returns:
            Formatted fairness report string
        """
        metrics = self.analyze_allocation(allocations, contributions)
        
        report = f"""
📊 FAIRNESS ANALYSIS REPORT
═══════════════════════════

📈 INEQUALITY METRICS:
  • Gini Coefficient: {metrics.gini_coefficient:.3f} (0=perfect equality, 1=max inequality)
  • Atkinson Index:   {metrics.atkinson_index:.3f} (0=perfect equality, 1=max inequality)

🤝 FAIRNESS METRICS:
  • Envy-Freeness:     {metrics.envy_freeness:.3f} (fraction of non-envious agents)
  • Proportional Fair: {metrics.proportional_fairness:.3f} (allocation matches contribution)
  • Max-Min Fairness:  {metrics.max_min_fairness:.3f} (min/max ratio)
  • Jain's Index:      {metrics.jain_fairness_index:.3f} (1=perfect fairness)

💰 WELFARE:
  • Total Welfare:     {metrics.welfare:.3f}
  • Agents Count:      {len(allocations)}
  • Average Allocation: {metrics.welfare/len(allocations):.3f}

📋 INTERPRETATION:
"""
        
        # Add interpretations
        if metrics.gini_coefficient < 0.3:
            report += "  ✅ Low inequality - relatively fair distribution\n"
        elif metrics.gini_coefficient < 0.6:
            report += "  ⚠️  Moderate inequality - some fairness concerns\n"
        else:
            report += "  ❌ High inequality - significant fairness issues\n"
        
        if metrics.envy_freeness > 0.8:
            report += "  ✅ Most agents are satisfied with their allocation\n"
        elif metrics.envy_freeness > 0.5:
            report += "  ⚠️  Some agents may envy others' allocations\n"
        else:
            report += "  ❌ Many agents envy others - potential conflicts\n"
        
        if metrics.jain_fairness_index > 0.8:
            report += "  ✅ High fairness according to Jain's index\n"
        else:
            report += "  ⚠️  Room for improvement in fairness\n"
        
        return report
    
    def plot_fairness_over_time(self, save_path: Optional[str] = None) -> None:
        """Plot fairness metrics over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.history:
            print("No fairness history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Fairness Metrics Over Time', fontsize=16)
        
        timesteps = range(len(self.history))
        
        # Plot Gini coefficient
        gini_values = [m.gini_coefficient for m in self.history]
        axes[0, 0].plot(timesteps, gini_values, 'b-', linewidth=2)
        axes[0, 0].set_title('Gini Coefficient')
        axes[0, 0].set_ylabel('Inequality')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Envy-freeness
        envy_values = [m.envy_freeness for m in self.history]
        axes[0, 1].plot(timesteps, envy_values, 'g-', linewidth=2)
        axes[0, 1].set_title('Envy-Freeness')
        axes[0, 1].set_ylabel('Fraction Non-Envious')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Jain's fairness index
        jain_values = [m.jain_fairness_index for m in self.history]
        axes[1, 0].plot(timesteps, jain_values, 'r-', linewidth=2)
        axes[1, 0].set_title("Jain's Fairness Index")
        axes[1, 0].set_ylabel('Fairness')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot Total welfare
        welfare_values = [m.welfare for m in self.history]
        axes[1, 1].plot(timesteps, welfare_values, 'm-', linewidth=2)
        axes[1, 1].set_title('Total Welfare')
        axes[1, 1].set_ylabel('Welfare')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness plot saved to {save_path}")
        else:
            plt.show()
    
    def compare_allocations(self, 
                          allocation_schemes: Dict[str, Dict[int, float]]) -> Dict[str, FairnessMetrics]:
        """Compare fairness across different allocation schemes.
        
        Args:
            allocation_schemes: Dict mapping scheme name to allocations
            
        Returns:
            Dict mapping scheme name to fairness metrics
        """
        results = {}
        
        for scheme_name, allocations in allocation_schemes.items():
            metrics = self.analyze_allocation(allocations)
            results[scheme_name] = metrics
        
        return results
    
    def suggest_improvements(self, 
                           allocations: Dict[int, float],
                           contributions: Optional[Dict[int, float]] = None) -> List[str]:
        """Suggest improvements to increase fairness.
        
        Args:
            allocations: Current allocations
            contributions: Agent contributions
            
        Returns:
            List of improvement suggestions
        """
        metrics = self.analyze_allocation(allocations, contributions)
        suggestions = []
        
        if metrics.gini_coefficient > 0.5:
            # High inequality
            agent_values = list(allocations.values())
            max_agent = max(allocations, key=allocations.get)
            min_agent = min(allocations, key=allocations.get)
            
            suggestions.append(
                f"High inequality detected (Gini={metrics.gini_coefficient:.3f}). "
                f"Consider redistributing from agent {max_agent} "
                f"(has {allocations[max_agent]:.2f}) to agent {min_agent} "
                f"(has {allocations[min_agent]:.2f})."
            )
        
        if metrics.envy_freeness < 0.7:
            suggestions.append(
                f"Low envy-freeness ({metrics.envy_freeness:.3f}). "
                "Consider implementing lottery-based allocation or "
                "rotating assignment schemes to reduce envy."
            )
        
        if contributions and metrics.proportional_fairness < 0.7:
            suggestions.append(
                f"Low proportional fairness ({metrics.proportional_fairness:.3f}). "
                "Allocations don't match contributions well. "
                "Consider merit-based allocation adjustments."
            )
        
        if metrics.jain_fairness_index < 0.8:
            suggestions.append(
                f"Jain's fairness index is low ({metrics.jain_fairness_index:.3f}). "
                "Consider implementing max-min fair allocation or "
                "proportional share mechanisms."
            )
        
        if not suggestions:
            suggestions.append("Allocation appears reasonably fair across all metrics!")
        
        return suggestions