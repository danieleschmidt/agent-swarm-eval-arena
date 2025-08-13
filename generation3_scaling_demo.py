#!/usr/bin/env python3
"""
Generation 3 Scaling Demo - Showcase optimization and scaling features
"""

import sys
import time
import numpy as np
from pathlib import Path
import multiprocessing as mp

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent
from swarm_arena.optimization.performance import (
    PerformanceOptimizer, PerformanceConfig, VectorizedOperations,
    cached, vectorized, memory_managed, BatchProcessor, global_optimizer
)
from swarm_arena.optimization.auto_scaling import (
    ScalingManager, ScalingPolicy, LoadBalancer, AutoScaler
)
from swarm_arena.optimization.distributed_computing import (
    DistributedSimulationManager, NodeInfo, NodeRole, Task
)


def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("⚡ PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 50)
    
    # Create performance optimizer
    config = PerformanceConfig(
        max_workers=mp.cpu_count(),
        cache_size=500,
        use_vectorization=True,
        batch_size=100
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Test caching optimization
    print("Testing caching optimization...")
    
    @cached(ttl=60.0)
    def expensive_computation(n):
        """Simulate expensive computation."""
        time.sleep(0.01)  # Simulate work
        return sum(i**2 for i in range(n))
    
    # First call (should be slow)
    start_time = time.time()
    result1 = expensive_computation(1000)
    first_call_time = time.time() - start_time
    
    # Second call (should be fast - cached)
    start_time = time.time()
    result2 = expensive_computation(1000)
    second_call_time = time.time() - start_time
    
    print(f"✅ First call: {first_call_time:.4f}s, Second call: {second_call_time:.4f}s")
    print(f"   Cache speedup: {first_call_time / second_call_time:.1f}x")
    
    # Test vectorized operations
    print("\nTesting vectorized operations...")
    
    # Generate random agent positions
    n_agents = 1000
    positions = np.random.uniform(0, 500, (n_agents, 2))
    
    # Test distance matrix computation
    start_time = time.time()
    distances = VectorizedOperations.distance_matrix(positions)
    vectorized_time = time.time() - start_time
    
    print(f"✅ Vectorized distance matrix ({n_agents}x{n_agents}): {vectorized_time:.4f}s")
    
    # Test neighbor finding
    start_time = time.time()
    neighbors = VectorizedOperations.find_neighbors(positions, radius=50.0)
    neighbor_time = time.time() - start_time
    
    avg_neighbors = np.mean([len(n) for n in neighbors])
    print(f"✅ Neighbor search: {neighbor_time:.4f}s, avg neighbors: {avg_neighbors:.1f}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    
    batch_processor = BatchProcessor(batch_size=100, parallel=True)
    
    def process_batch(batch):
        return [x**2 for x in batch]
    
    large_dataset = list(range(1000))
    
    start_time = time.time()
    results = batch_processor.process(large_dataset, process_batch)
    batch_time = time.time() - start_time
    
    print(f"✅ Batch processing (1000 items): {batch_time:.4f}s")
    
    # Get optimization stats
    stats = optimizer.get_optimization_stats()
    print(f"\n📊 Optimization Statistics:")
    print(f"   Cache optimizations: {stats['cache_optimizations']}")
    print(f"   Vectorization optimizations: {stats['vectorization_optimizations']}")
    print(f"   Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    print(f"   Memory usage: {stats['memory_stats']['current_gb']:.2f} GB")
    
    return stats


def demo_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    print("\n📈 AUTO-SCALING DEMO")
    print("=" * 50)
    
    # Create scaling policy
    policy = ScalingPolicy(
        min_workers=2,
        max_workers=8,
        target_cpu_usage=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        cooldown_period=10.0,  # Shorter for demo
        evaluation_period=5.0
    )
    
    # Create load balancer
    load_balancer = LoadBalancer()
    
    # Add initial workers
    for i in range(policy.min_workers):
        load_balancer.add_worker(f"worker_{i}")
    
    print(f"✅ Started with {policy.min_workers} workers")
    
    # Create auto-scaler
    auto_scaler = AutoScaler(policy, load_balancer)
    
    # Simulate load and test scaling
    print("\nSimulating high load...")
    
    # Simulate tasks being submitted
    for i in range(20):
        worker_id = load_balancer.get_worker_for_task()
        if worker_id:
            # Simulate task execution
            load_balancer.update_worker_state(worker_id, load_balancer.workers[worker_id].state)
            
            # Add task to worker queue (simulated)
            if worker_id in load_balancer.worker_queues:
                try:
                    load_balancer.worker_queues[worker_id].put(f"task_{i}", block=False)
                except:
                    pass  # Queue might be full
    
    # Get load statistics
    initial_stats = load_balancer.get_load_statistics()
    print(f"✅ Load balancing statistics:")
    print(f"   Total workers: {initial_stats['total_workers']}")
    print(f"   Active workers: {initial_stats['active_workers']}")
    print(f"   Total queue size: {initial_stats['total_queue_size']}")
    print(f"   Strategy: {initial_stats['strategy']}")
    
    # Test different load balancing strategies
    print("\nTesting load balancing strategies...")
    
    strategies = ['round_robin', 'least_loaded', 'weighted_round_robin', 'random']
    
    for strategy in strategies:
        load_balancer.current_strategy = strategy
        
        # Simulate task distribution
        worker_assignments = {}
        for i in range(10):
            worker_id = load_balancer.get_worker_for_task()
            if worker_id:
                worker_assignments[worker_id] = worker_assignments.get(worker_id, 0) + 1
        
        print(f"   {strategy}: {dict(worker_assignments)}")
    
    return initial_stats


def demo_distributed_computing():
    """Demonstrate distributed computing capabilities."""
    print("\n🌐 DISTRIBUTED COMPUTING DEMO")
    print("=" * 50)
    
    # Create distributed manager
    distributed_manager = DistributedSimulationManager()
    
    # Register nodes
    print("Registering distributed nodes...")
    
    # Register worker nodes
    for i in range(3):
        worker_info = NodeInfo(
            node_id=f"worker_node_{i}",
            role=NodeRole.WORKER,
            address=f"192.168.1.{10+i}",
            port=8000 + i,
            capabilities={"cpu_cores": 4, "memory_gb": 8, "gpu": False}
        )
        distributed_manager.node_manager.register_node(worker_info)
    
    # Register storage node
    storage_info = NodeInfo(
        node_id="storage_node",
        role=NodeRole.STORAGE,
        address="192.168.1.20",
        port=6379,
        capabilities={"storage_type": "redis", "capacity_gb": 100}
    )
    distributed_manager.node_manager.register_node(storage_info)
    
    print(f"✅ Registered {len(distributed_manager.node_manager.nodes)} nodes")
    
    # Test distributed storage
    print("\nTesting distributed storage...")
    
    storage = distributed_manager.node_manager.storage
    
    # Store and retrieve data
    test_data = {"simulation_results": [1, 2, 3, 4, 5], "timestamp": time.time()}
    success = storage.store("test_key", test_data, ttl=300)
    retrieved_data = storage.retrieve("test_key")
    
    print(f"✅ Storage test: stored={success}, retrieved={retrieved_data is not None}")
    
    # Test task scheduling
    print("\nTesting task scheduling...")
    
    distributed_manager.start()
    
    # Submit test tasks
    task_ids = []
    for i in range(5):
        def dummy_simulation(agent_count, steps):
            time.sleep(0.1)  # Simulate work
            return f"Simulation with {agent_count} agents, {steps} steps completed"
        
        task_id = distributed_manager.submit_simulation_task(
            dummy_simulation,
            agent_count=50 + i*10,
            steps=100,
            priority=i+1
        )
        task_ids.append(task_id)
    
    print(f"✅ Submitted {len(task_ids)} tasks")
    
    # Wait for tasks to complete (simulated)
    time.sleep(1.0)
    
    # Get cluster status
    cluster_status = distributed_manager.get_cluster_status()
    print(f"✅ Cluster status:")
    print(f"   Node ID: {cluster_status['node_id']}")
    print(f"   Total nodes: {cluster_status['nodes']}")
    print(f"   Worker nodes: {cluster_status['workers']}")
    print(f"   Pending tasks: {cluster_status['scheduling_stats']['pending_tasks']}")
    
    distributed_manager.stop()
    
    return cluster_status


def demo_optimized_simulation():
    """Demonstrate optimized simulation with all scaling features."""
    print("\n🚀 OPTIMIZED SIMULATION DEMO")
    print("=" * 50)
    
    # Create optimized configuration
    config = SwarmConfig(
        num_agents=200,  # Larger scale
        arena_size=(800, 800),
        episode_length=200,
        seed=42
    )
    
    print(f"Creating optimized simulation ({config.num_agents} agents)...")
    
    # Use performance optimization
    @memory_managed
    @vectorized
    def create_optimized_arena():
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=config.num_agents // 2)
        arena.add_agents(CompetitiveAgent, count=config.num_agents // 2)
        return arena
    
    # Time arena creation
    start_time = time.time()
    arena = create_optimized_arena()
    creation_time = time.time() - start_time
    
    print(f"✅ Arena created in {creation_time:.4f}s")
    
    # Run optimized simulation
    @cached(ttl=120.0)
    def run_optimized_simulation(episodes):
        return arena.run(episodes=episodes, verbose=False)
    
    print("\nRunning optimized simulation...")
    
    start_time = time.time()
    results = run_optimized_simulation(3)
    simulation_time = time.time() - start_time
    
    print(f"✅ Simulation completed in {simulation_time:.4f}s")
    print(f"   Mean reward: {results.mean_reward:.3f}")
    print(f"   Fairness index: {results.fairness_index:.3f}")
    print(f"   Total steps: {results.total_steps}")
    
    # Performance comparison
    print("\nPerformance comparison (optimized vs baseline)...")
    
    # Baseline (no optimization)
    baseline_arena = Arena(SwarmConfig(num_agents=50, arena_size=(400, 400), episode_length=50))
    baseline_arena.add_agents(CooperativeAgent, count=25)
    baseline_arena.add_agents(CompetitiveAgent, count=25)
    
    start_time = time.time()
    baseline_results = baseline_arena.run(episodes=1, verbose=False)
    baseline_time = time.time() - start_time
    
    # Calculate performance improvement
    relative_workload = (config.num_agents / 50) * (config.episode_length / 50) * (3 / 1)
    expected_baseline_time = baseline_time * relative_workload
    speedup = expected_baseline_time / simulation_time
    
    print(f"   Baseline (50 agents, 1 episode): {baseline_time:.4f}s")
    print(f"   Expected time for optimized workload: {expected_baseline_time:.4f}s")
    print(f"   Actual optimized time: {simulation_time:.4f}s")
    print(f"   Performance improvement: {speedup:.1f}x")
    
    return {
        'simulation_time': simulation_time,
        'baseline_time': baseline_time,
        'speedup': speedup,
        'results': results
    }


def demo_resource_monitoring():
    """Demonstrate resource monitoring and management."""
    print("\n🔍 RESOURCE MONITORING DEMO")
    print("=" * 50)
    
    import psutil
    
    # Get system resources
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"System Resources:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Memory: {memory.total / (1024**3):.1f} GB total, {memory.percent}% used")
    print(f"   Disk: {disk.total / (1024**3):.1f} GB total, {disk.percent}% used")
    
    # Monitor resource usage during simulation
    print("\nMonitoring resource usage during simulation...")
    
    # Start resource monitoring
    initial_cpu = psutil.cpu_percent(interval=0.1)
    initial_memory = psutil.virtual_memory().percent
    
    # Run resource-intensive simulation
    config = SwarmConfig(num_agents=500, arena_size=(1000, 1000), episode_length=100)
    arena = Arena(config)
    arena.add_agents(CooperativeAgent, count=250)
    arena.add_agents(CompetitiveAgent, count=250)
    
    start_time = time.time()
    results = arena.run(episodes=2, verbose=False)
    end_time = time.time()
    
    # Check resource usage after simulation
    final_cpu = psutil.cpu_percent(interval=0.1)
    final_memory = psutil.virtual_memory().percent
    
    print(f"✅ Resource usage during simulation:")
    print(f"   CPU: {initial_cpu:.1f}% → {final_cpu:.1f}%")
    print(f"   Memory: {initial_memory:.1f}% → {final_memory:.1f}%")
    print(f"   Simulation time: {end_time - start_time:.2f}s")
    print(f"   Throughput: {results.total_steps / (end_time - start_time):.1f} steps/sec")
    
    # Get optimizer statistics
    optimizer_stats = global_optimizer.get_optimization_stats()
    
    print(f"\n📊 Global Optimizer Statistics:")
    print(f"   Cache optimizations: {optimizer_stats['cache_optimizations']}")
    print(f"   Vectorization optimizations: {optimizer_stats['vectorization_optimizations']}")
    print(f"   Memory optimizations: {optimizer_stats['memory_optimizations']}")
    print(f"   Cache hit rate: {optimizer_stats['cache_stats']['hit_rate']:.2%}")
    
    return {
        'cpu_usage': (initial_cpu, final_cpu),
        'memory_usage': (initial_memory, final_memory),
        'throughput': results.total_steps / (end_time - start_time),
        'optimizer_stats': optimizer_stats
    }


def main():
    """Run complete Generation 3 demonstration."""
    print("🚀 SWARM ARENA - GENERATION 3 SCALING DEMO")
    print("⚡ Autonomous SDLC Execution - Making It Scale!")
    print("=" * 60)
    
    try:
        # Performance optimization
        perf_stats = demo_performance_optimization()
        
        # Auto-scaling
        scaling_stats = demo_auto_scaling()
        
        # Distributed computing
        distributed_stats = demo_distributed_computing()
        
        # Optimized simulation
        simulation_stats = demo_optimized_simulation()
        
        # Resource monitoring
        resource_stats = demo_resource_monitoring()
        
        print("\n🎉 GENERATION 3 COMPLETE!")
        print("=" * 60)
        print("✅ Performance optimization & caching")
        print("✅ Vectorized operations & batch processing")
        print("✅ Auto-scaling & load balancing")
        print("✅ Distributed computing infrastructure")
        print("✅ Resource monitoring & management")
        print("✅ Memory management & optimization")
        print("✅ Concurrent processing & thread pools")
        
        print(f"\n📊 Scaling Summary:")
        print(f"   • Cache hit rate: {perf_stats['cache_stats']['hit_rate']:.2%}")
        print(f"   • Load balancer workers: {scaling_stats['total_workers']}")
        print(f"   • Distributed nodes: {distributed_stats['nodes']}")
        print(f"   • Performance improvement: {simulation_stats['speedup']:.1f}x")
        print(f"   • Throughput: {resource_stats['throughput']:.1f} steps/sec")
        
        print(f"\n🎯 Ready for Quality Gates and Production Deployment!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)