#!/usr/bin/env python3
"""
krpsim - Resource Process Simulator

A program that simulates resource transformation processes and optimizes
for time and/or stock quantities.
"""

import sys
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq


@dataclass
class Process:
    """Represents a process that transforms resources."""
    name: str
    inputs: Dict[str, int]  # resource_name -> quantity needed
    outputs: Dict[str, int]  # resource_name -> quantity produced
    delay: int  # time to complete the process
    
    def can_execute(self, stocks: Dict[str, int]) -> bool:
        """Check if this process can be executed with current stocks."""
        for resource, qty in self.inputs.items():
            if stocks.get(resource, 0) < qty:
                return False
        return True
    
    def consume_inputs(self, stocks: Dict[str, int]) -> None:
        """Consume input resources from stocks."""
        for resource, qty in self.inputs.items():
            stocks[resource] -= qty
    
    def produce_outputs(self, stocks: Dict[str, int]) -> None:
        """Add output resources to stocks."""
        for resource, qty in self.outputs.items():
            stocks[resource] = stocks.get(resource, 0) + qty


@dataclass
class ScheduledProcess:
    """A process scheduled to complete at a specific cycle."""
    end_cycle: int
    start_cycle: int
    process: Process
    
    def __lt__(self, other):
        return self.end_cycle < other.end_cycle


@dataclass
class Configuration:
    """Holds the parsed configuration."""
    stocks: Dict[str, int] = field(default_factory=dict)
    processes: Dict[str, Process] = field(default_factory=dict)
    optimize: List[str] = field(default_factory=list)  # list of optimization targets


def parse_resources(resource_str: str) -> Dict[str, int]:
    """Parse a resource string like 'euro:8;materiel:2' into a dict."""
    resources = {}
    if not resource_str or resource_str == "":
        return resources
    
    for item in resource_str.split(';'):
        item = item.strip()
        if ':' in item:
            parts = item.split(':')
            name = parts[0].strip()
            qty = int(parts[1].strip())
            resources[name] = qty
    return resources


def parse_configuration(filename: str) -> Configuration:
    """Parse the configuration file and return a Configuration object."""
    config = Configuration()
    
    with open(filename, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        # Remove comments and strip whitespace
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Check if it's an optimize line
        if line.startswith('optimize:'):
            # Parse optimize:(time;stock1;stock2;...)
            match = re.match(r'optimize:\(([^)]+)\)', line)
            if match:
                targets = match.group(1).split(';')
                config.optimize = [t.strip() for t in targets]
            continue
        
        # Check if it's a process definition
        # Format: name:(need1:qty1;need2:qty2):(result1:qty1;result2:qty2):delay
        process_match = re.match(r'(\w+):\(([^)]*)\):\(([^)]*)\):(\d+)', line)
        if process_match:
            name = process_match.group(1)
            inputs = parse_resources(process_match.group(2))
            outputs = parse_resources(process_match.group(3))
            delay = int(process_match.group(4))
            config.processes[name] = Process(name, inputs, outputs, delay)
            continue
        
        # Check if it's a stock definition
        # Format: name:quantity
        stock_match = re.match(r'(\w+):(\d+)', line)
        if stock_match:
            name = stock_match.group(1)
            qty = int(stock_match.group(2))
            config.stocks[name] = qty
            continue
    
    return config


def find_dependencies(process: Process, all_processes: Dict[str, Process]) -> List[str]:
    """Return processes reachable from this process's outputs (BFS over consumers)."""
    deps: List[str] = []
    visited = set()
    queue: List[Process] = [process]
    while queue:
        current = queue.pop(0)
        for produced in current.outputs.keys():
            for candidate in all_processes.values():
                if candidate.name in visited:
                    continue
                if produced in candidate.inputs:
                    visited.add(candidate.name)
                    deps.append(candidate.name)
                    queue.append(candidate)
    return deps


def calculate_chain_efficiency(
    process: Process,
    all_processes: Dict[str, Process],
    optimize_targets: List[str],
    visited: frozenset = frozenset()
) -> Dict[str, float]:
    """
    For each non-time optimize target, return the best achievable
    target_units/cycle ratio for a chain starting at this process.

    Example: apple_to_juice (delay=15) -> juice_sale (delay=10, euro:2)
      chain_eff[euro] = 1 * (2/10) * 10/(15+10) = 0.08
    vs apple_sale (delay=35, euro:1)
      chain_eff[euro] = 1/35 ≈ 0.029
    """
    if process.name in visited:
        return {}

    visited = visited | frozenset([process.name])
    non_time_targets = [t for t in optimize_targets if t != 'time']
    result: Dict[str, float] = defaultdict(float)

    for res, qty in process.outputs.items():
        # Direct production of a target resource
        for target in non_time_targets:
            if res == target:
                eff = qty / max(1, process.delay)
                if eff > result[target]:
                    result[target] = eff

        # Chain through downstream processes that consume this resource
        for downstream in all_processes.values():
            if res not in downstream.inputs or downstream.name in visited:
                continue
            needed = downstream.inputs[res]
            # Normalize by total inputs consumed: a process that needs 2 apples
            # to produce 1 juice is half as efficient per apple as one needing 1.
            total_inputs = sum(process.inputs.values()) if process.inputs else 1
            ratio = (qty / needed) / total_inputs  # target units per input unit
            ds_eff = calculate_chain_efficiency(
                downstream, all_processes, optimize_targets, visited
            )
            for target, eff in ds_eff.items():
                # Amortise our delay into the chain:
                #   chain_eff = ratio * downstream_eff * (downstream_delay / total_delay)
                chain_eff = ratio * eff * downstream.delay / (process.delay + downstream.delay)
                if chain_eff > result[target]:
                    result[target] = chain_eff

    return dict(result)


def calculate_process_score(process: Process, stocks: Dict[str, int],
                           optimize_targets: List[str],
                           all_processes: Dict[str, Process],
                           pending_outputs: Dict[str, int]) -> float:
    """
    Score a process using full chain efficiency towards optimize targets.

    For each target the scorer computes the best target_units/cycle ratio
    achievable by this process including all downstream chains, then scales
    it by 10 000 so the values remain comparable to the time heuristic.
    """
    score = 0.0

    # Time optimisation: reward shorter delays
    if 'time' in optimize_targets:
        score += 1000.0 / (process.delay + 1)

    # Resource optimisation: use chain efficiency so indirect paths that
    # deliver more target units per combined cycle beat direct slow paths
    chain_eff = calculate_chain_efficiency(process, all_processes, optimize_targets)
    for target, eff in chain_eff.items():
        score += 10000.0 * eff

    # Heavy penalty for consuming a target resource directly.
    # Without this, processes that eat the target (e.g. apple_sale when
    # optimizing apple) score 0 and still run due to the runnable bonus.
    for target in optimize_targets:
        if target != 'time' and target in process.inputs:
            score -= 50000.0 * process.inputs[target]

    # Tiny bonus if immediately runnable (tie-break)
    if process.can_execute(stocks):
        score += 0.1

    return score


def simulate(config: Configuration, max_delay: float) -> List[Tuple[int, str]]:
    """
    Run the simulation and return a list of (cycle, process_name) tuples.
    Uses a greedy scheduling algorithm with priority queue for running processes.
    """
    stocks = dict(config.stocks)
    processes = config.processes
    optimize_targets = config.optimize
    
    trace: List[Tuple[int, str]] = []
    running_processes: List[ScheduledProcess] = []  # heap of scheduled processes
    current_cycle = 0
    
    start_time = time.time()
    
    # Track if system is making progress
    last_progress_cycle = 0
    max_idle_cycles = 1000  # Maximum cycles without progress before stopping
    
    # Track optimization target quantities for self-sustaining detection
    target_counts = defaultdict(int)
    initial_target_check = True
    min_target_runs = 3  # Minimum times to produce optimization targets
    
    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_delay:
            break
        
        # Complete any processes that finish at current cycle
        while running_processes and running_processes[0].end_cycle <= current_cycle:
            scheduled = heapq.heappop(running_processes)
            scheduled.process.produce_outputs(stocks)
            
            # Track optimization target production
            for target in optimize_targets:
                if target != 'time' and target in scheduled.process.outputs:
                    target_counts[target] += scheduled.process.outputs[target]
        
        # Try to start new processes
        started_any = True
        while started_any:
            started_any = False
            
            # Score and sort processes
            pending_outputs = defaultdict(int)
            for scheduled in running_processes:
                for res, qty in scheduled.process.outputs.items():
                    pending_outputs[res] += qty

            executable = []
            for process in processes.values():
                if process.can_execute(stocks):
                    score = calculate_process_score(
                        process, stocks, optimize_targets, processes, pending_outputs
                    )
                    executable.append((score, process))
            
            # Sort by score (highest first)
            executable.sort(key=lambda x: -x[0])
            # if the best-scoring process does not have a positive score,
            # there's no benefit to running anything further; stop trying.
            if not executable or executable[0][0] <= 0:
                # if there are no currently running processes, nothing
                # will ever change from this point on, so we can exit the
                # entire simulation and avoid artificially incrementing
                # `current_cycle` during idle iterations.
                if not running_processes:
                    return trace, stocks, current_cycle
                break
            
            # Try to start the best process
            for score, process in executable:
                if process.can_execute(stocks):
                    process.consume_inputs(stocks)
                    scheduled = ScheduledProcess(
                        end_cycle=current_cycle + process.delay,
                        start_cycle=current_cycle,
                        process=process
                    )
                    heapq.heappush(running_processes, scheduled)
                    trace.append((current_cycle, process.name))
                    started_any = True
                    last_progress_cycle = current_cycle
                    break
        
        # Check if we should stop
        if not running_processes:
            # No running processes and none can start
            can_start_any = any(p.can_execute(stocks) for p in processes.values())
            if not can_start_any:
                break
        
        # Check for self-sustaining system stop condition
        non_time_targets = [t for t in optimize_targets if t != 'time']
        if non_time_targets:
            all_targets_met = all(target_counts[t] >= min_target_runs for t in non_time_targets)
            if all_targets_met and current_cycle > 0:
                # System has produced optimization targets multiple times
                # Check if we can continue or should stop
                if current_cycle - last_progress_cycle > max_idle_cycles:
                    break
        
        # Advance time
        if running_processes:
            # Jump to next process completion
            next_completion = running_processes[0].end_cycle
            current_cycle = next_completion
        else:
            current_cycle += 1
        
        # Safety check for infinite loops
        if current_cycle > 1000000:
            break
    
    # Complete any remaining processes
    while running_processes:
        scheduled = heapq.heappop(running_processes)
        current_cycle = scheduled.end_cycle
        scheduled.process.produce_outputs(stocks)
    
    return trace, stocks, current_cycle


def display_results(trace: List[Tuple[int, str]], final_stocks: Dict[str, int], 
                   final_cycle: int, config: Configuration) -> None:
    """Display the simulation results."""
    print("=" * 60)
    print("KRPSIM - Simulation Results")
    print("=" * 60)
    resource_opt_targets = [t for t in config.optimize if t != 'time']
    header_opt_count = len(resource_opt_targets)
    print(f"{len(config.processes)} processes, {len(config.stocks)} stocks, {header_opt_count} to optimize")
    print()
    
    print("Initial stocks:")
    for name, qty in sorted(config.stocks.items()):
        print(f"  {name}: {qty}")
    print()
    
    print("Processes executed:")
    if trace:
        for cycle, process_name in trace:
            print(f"  {cycle}:{process_name}")
    else:
        print("  (none)")
    print()
    
    print("Final stocks:")
    for name, qty in sorted(final_stocks.items()):
        print(f"  {name}: {qty}")
    print()
    
    print(f"Final cycle: {final_cycle}")
    print()
    
    # Show optimization targets status
    print("Optimization targets:")
    for target in config.optimize:
        if target == 'time':
            print(f"  time: {final_cycle} cycles")
        else:
            qty = final_stocks.get(target, 0)
            initial = config.stocks.get(target, 0)
            print(f"  {target}: {initial} -> {qty} (+{qty - initial})")
    if not config.optimize:
        print("  (none)")
    print("=" * 60)


def print_trace_only(trace: List[Tuple[int, str]]) -> None:
    """Print trace in verification format."""
    for cycle, process_name in trace:
        print(f"{cycle}:{process_name}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file> <delay>", file=sys.stderr)
        print("  <file>  - Configuration file with stocks and processes", file=sys.stderr)
        print("  <delay> - Maximum execution time in seconds", file=sys.stderr)
        sys.exit(1)
    
    config_file = sys.argv[1]
    try:
        max_delay = float(sys.argv[2])
    except ValueError:
        print(f"Error: delay must be a number, got '{sys.argv[2]}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        config = parse_configuration(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not config.processes:
        print("Error: No processes defined in configuration", file=sys.stderr)
        sys.exit(1)
    
    # Run simulation
    trace, final_stocks, final_cycle = simulate(config, max_delay)
    
    # Display results
    display_results(trace, final_stocks, final_cycle, config)
    
    print("\n--- Verification Output ---")
    print_trace_only(trace)


if __name__ == "__main__":
    main()
