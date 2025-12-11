#!/usr/bin/env python3
"""
krpsim_verif - Resource Process Simulator Trace Verifier

A program that verifies execution traces from krpsim against a configuration file.
"""

import sys
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


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


@dataclass
class Configuration:
    """Holds the parsed configuration."""
    stocks: Dict[str, int] = field(default_factory=dict)
    processes: Dict[str, Process] = field(default_factory=dict)
    optimize: List[str] = field(default_factory=list)


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
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('optimize:'):
            match = re.match(r'optimize:\(([^)]+)\)', line)
            if match:
                targets = match.group(1).split(';')
                config.optimize = [t.strip() for t in targets]
            continue
        
        process_match = re.match(r'(\w+):\(([^)]*)\):\(([^)]*)\):(\d+)', line)
        if process_match:
            name = process_match.group(1)
            inputs = parse_resources(process_match.group(2))
            outputs = parse_resources(process_match.group(3))
            delay = int(process_match.group(4))
            config.processes[name] = Process(name, inputs, outputs, delay)
            continue
        
        stock_match = re.match(r'(\w+):(\d+)', line)
        if stock_match:
            name = stock_match.group(1)
            qty = int(stock_match.group(2))
            config.stocks[name] = qty
            continue
    
    return config


def parse_trace(filename: str) -> List[Tuple[int, str]]:
    """Parse a trace file and return list of (cycle, process_name) tuples."""
    trace = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse cycle:process_name format
            match = re.match(r'(\d+):(\w+)', line)
            if match:
                cycle = int(match.group(1))
                process_name = match.group(2)
                trace.append((cycle, process_name))
            else:
                # Try to skip lines that don't match the format
                # (might be header/footer info from krpsim output)
                pass
    
    return trace


def verify_trace(config: Configuration, trace: List[Tuple[int, str]]) -> Tuple[bool, str, Dict[str, int], int]:
    """
    Verify a trace against a configuration.
    
    Returns:
        (is_valid, error_message, final_stocks, last_cycle)
    """
    stocks = dict(config.stocks)
    processes = config.processes
    
    # Track running processes
    running: List[ScheduledProcess] = []
    last_cycle = 0
    
    for trace_idx, (cycle, process_name) in enumerate(trace):
        # Validate process exists
        if process_name not in processes:
            return (False, 
                    f"Error at cycle {cycle}: Unknown process '{process_name}'",
                    stocks, cycle)
        
        process = processes[process_name]
        
        # Complete any processes that finish before or at this cycle
        completed = []
        still_running = []
        for scheduled in running:
            if scheduled.end_cycle <= cycle:
                completed.append(scheduled)
            else:
                still_running.append(scheduled)
        
        # Sort completed by end time and apply outputs
        completed.sort(key=lambda x: x.end_cycle)
        for scheduled in completed:
            scheduled.process.produce_outputs(stocks)
        
        running = still_running
        
        # Check if process can be executed with current stocks
        if not process.can_execute(stocks):
            missing = []
            for resource, qty in process.inputs.items():
                available = stocks.get(resource, 0)
                if available < qty:
                    missing.append(f"{resource} (need {qty}, have {available})")
            
            return (False,
                    f"Error at cycle {cycle}: Cannot execute process '{process_name}' - "
                    f"insufficient resources: {', '.join(missing)}",
                    stocks, cycle)
        
        # Consume inputs and schedule process
        process.consume_inputs(stocks)
        scheduled = ScheduledProcess(
            end_cycle=cycle + process.delay,
            start_cycle=cycle,
            process=process
        )
        running.append(scheduled)
        last_cycle = max(last_cycle, cycle)
    
    # Complete all remaining processes
    running.sort(key=lambda x: x.end_cycle)
    for scheduled in running:
        scheduled.process.produce_outputs(stocks)
        last_cycle = max(last_cycle, scheduled.end_cycle)
    
    return (True, "Trace is valid", stocks, last_cycle)


def display_verification_result(is_valid: bool, message: str, 
                                 final_stocks: Dict[str, int], last_cycle: int,
                                 config: Configuration) -> None:
    """Display the verification results."""
    print("=" * 60)
    print("KRPSIM_VERIF - Trace Verification Results")
    print("=" * 60)
    print()
    
    if is_valid:
        print("✓ VERIFICATION PASSED")
        print(f"  {message}")
    else:
        print("✗ VERIFICATION FAILED")
        print(f"  {message}")
    print()
    
    print("Initial stocks:")
    for name, qty in sorted(config.stocks.items()):
        print(f"  {name}: {qty}")
    print()
    
    print("Final stocks:")
    # Show all resources that exist in either initial or final state
    all_resources = set(config.stocks.keys()) | set(final_stocks.keys())
    for name in sorted(all_resources):
        initial = config.stocks.get(name, 0)
        final = final_stocks.get(name, 0)
        diff = final - initial
        diff_str = f"+{diff}" if diff >= 0 else str(diff)
        print(f"  {name}: {final} ({diff_str} from initial)")
    print()
    
    print(f"Last cycle: {last_cycle}")
    print()
    
    # Show optimization targets status
    if config.optimize:
        print("Optimization targets:")
        for target in config.optimize:
            if target == 'time':
                print(f"  time: {last_cycle} cycles")
            else:
                qty = final_stocks.get(target, 0)
                initial = config.stocks.get(target, 0)
                print(f"  {target}: {initial} -> {qty} (+{qty - initial})")
    
    print("=" * 60)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file> <result_to_test>", file=sys.stderr)
        print("  <file>           - Configuration file with stocks and processes", file=sys.stderr)
        print("  <result_to_test> - Trace file to verify", file=sys.stderr)
        sys.exit(1)
    
    config_file = sys.argv[1]
    trace_file = sys.argv[2]
    
    # Parse configuration
    try:
        config = parse_configuration(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse trace
    try:
        trace = parse_trace(trace_file)
    except FileNotFoundError:
        print(f"Error: Trace file '{trace_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing trace: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not trace:
        print("Warning: Trace file is empty or contains no valid entries")
    
    # Verify trace
    is_valid, message, final_stocks, last_cycle = verify_trace(config, trace)
    
    # Display results
    display_verification_result(is_valid, message, final_stocks, last_cycle, config)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
