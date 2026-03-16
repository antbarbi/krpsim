#!/usr/bin/env python3
"""
krpsim_parser - Configuration file parser for krpsim.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Process:
    """Represents a process that transforms resources."""
    name: str
    inputs: Dict[str, int]   # resource_name -> quantity needed
    outputs: Dict[str, int]  # resource_name -> quantity produced
    delay: int               # time to complete the process

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
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # optimize:(time;stock1;stock2;...)
        if line.startswith('optimize:'):
            match = re.match(r'optimize:\(([^)]+)\)', line)
            if match:
                targets = match.group(1).split(';')
                config.optimize = [t.strip() for t in targets]
            continue

        # Process: name:(need1:qty1;...):(result1:qty1;...):delay
        process_match = re.match(r'(\w+):\(([^)]*)\):\(([^)]*)\):(\d+)', line)
        if process_match:
            name = process_match.group(1)
            inputs = parse_resources(process_match.group(2))
            outputs = parse_resources(process_match.group(3))
            delay = int(process_match.group(4))
            config.processes[name] = Process(name, inputs, outputs, delay)
            continue

        # Stock: name:quantity
        stock_match = re.match(r'(\w+):(\d+)', line)
        if stock_match:
            name = stock_match.group(1)
            qty = int(stock_match.group(2))
            config.stocks[name] = qty
            continue

    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_file>", file=sys.stderr)
        sys.exit(1)

    config = parse_configuration(sys.argv[1])

    print(f"Stocks ({len(config.stocks)}):")
    for name, qty in sorted(config.stocks.items()):
        print(f"  {name}: {qty}")

    print(f"\nProcesses ({len(config.processes)}):")
    for p in config.processes.values():
        print(f"  {p.name}: inputs={p.inputs}, outputs={p.outputs}, delay={p.delay}")

    print(f"\nOptimize: {config.optimize}")
