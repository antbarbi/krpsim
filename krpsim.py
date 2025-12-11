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


def calculate_process_score(process: Process, stocks: Dict[str, int], 
                           optimize_targets: List[str]) -> float:
    """
    Calculate a priority score for a process based on optimization targets.
    Higher score = higher priority.
    """
    score = 0.0
    
    # Check if process outputs any optimization target
    for target in optimize_targets:
        if target == 'time':
            # For time optimization, prefer shorter processes
            score += 1000.0 / (process.delay + 1)
        elif target in process.outputs:
            # This process produces an optimization target
            score += 100.0 * process.outputs[target]
    
    # Bonus for processes that can be executed immediately
    if process.can_execute(stocks):
        score += 50.0
    
    # Consider process efficiency (output/delay ratio)
    total_output = sum(process.outputs.values())
    score += total_output / (process.delay + 1)
    
    return score


def find_dependencies(process: Process, all_processes: Dict[str, Process]) -> List[str]:
    """Find processes that produce resources needed by this process."""
    dependencies = []
    for resource in process.inputs:
        for p in all_processes.values():
            if resource in p.outputs:
                dependencies.append(p.name)
    return dependencies


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
            executable = []
            for process in processes.values():
                if process.can_execute(stocks):
                    score = calculate_process_score(process, stocks, optimize_targets)
                    executable.append((score, process))
            
            # Sort by score (highest first)
            executable.sort(key=lambda x: -x[0])
            
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
