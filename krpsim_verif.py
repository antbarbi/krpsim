import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Process:
    name: str
    inputs: Dict[str, int]
    outputs: Dict[str, int]
    delay: int

    def can_execute(self, stocks: Dict[str, int]) -> bool:
        for resource, qty in self.inputs.items():
            if stocks.get(resource, 0) < qty:
                return False
        return True

    def consume_inputs(self, stocks: Dict[str, int]) -> None:
        for resource, qty in self.inputs.items():
            stocks[resource] -= qty

    def produce_outputs(self, stocks: Dict[str, int]) -> None:
        for resource, qty in self.outputs.items():
            stocks[resource] = stocks.get(resource, 0) + qty


@dataclass
class ScheduledProcess:
    end_cycle: int
    start_cycle: int
    process: Process


@dataclass
class Configuration:
    stocks: Dict[str, int] = field(default_factory=dict)
    processes: Dict[str, Process] = field(default_factory=dict)
    optimize: List[str] = field(default_factory=list)


def parse_resources(resource_str: str) -> Dict[str, int]:
    resources: Dict[str, int] = {}
    if not resource_str:
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
    config = Configuration()

    with open(filename, 'r') as f:
        content = f.read()

    for line in content.split('\n'):
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

    return config


def parse_trace(filename: str) -> List[Tuple[int, str]]:
    trace: List[Tuple[int, str]] = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            match = re.match(r'(\d+):(\w+)', line)
            if match:
                cycle = int(match.group(1))
                process_name = match.group(2)
                trace.append((cycle, process_name))

    return trace


def verify_trace(config: Configuration, trace: List[Tuple[int, str]]) -> Tuple[bool, str, Dict[str, int], int]:
    stocks = dict(config.stocks)
    processes = config.processes
    running: List[ScheduledProcess] = []
    last_cycle = 0

    for cycle, process_name in trace:
        if process_name not in processes:
            return (
                False,
                f"Error at cycle {cycle}: Unknown process '{process_name}'",
                stocks,
                cycle,
            )

        process = processes[process_name]

        completed: List[ScheduledProcess] = []
        still_running: List[ScheduledProcess] = []
        for scheduled in running:
            if scheduled.end_cycle <= cycle:
                completed.append(scheduled)
            else:
                still_running.append(scheduled)

        completed.sort(key=lambda x: x.end_cycle)
        for scheduled in completed:
            scheduled.process.produce_outputs(stocks)

        running = still_running

        if not process.can_execute(stocks):
            missing: List[str] = []
            for resource, qty in process.inputs.items():
                available = stocks.get(resource, 0)
                if available < qty:
                    missing.append(f"{resource} (need {qty}, have {available})")

            return (
                False,
                f"Error at cycle {cycle}: Cannot execute process '{process_name}' - insufficient resources: {', '.join(missing)}",
                stocks,
                cycle,
            )

        process.consume_inputs(stocks)
        running.append(
            ScheduledProcess(
                end_cycle=cycle + process.delay,
                start_cycle=cycle,
                process=process,
            )
        )
        last_cycle = max(last_cycle, cycle)

    running.sort(key=lambda x: x.end_cycle)
    for scheduled in running:
        scheduled.process.produce_outputs(stocks)
        last_cycle = max(last_cycle, scheduled.end_cycle)

    return (True, "Trace is valid", stocks, last_cycle)


def display_verification_result(
    is_valid: bool,
    message: str,
    final_stocks: Dict[str, int],
    last_cycle: int,
    config: Configuration,
) -> None:
    print("=" * 60)
    print("KRPSIM_VERIF - Trace Verification Results")
    print("=" * 60)
    print()

    if is_valid:
        print("VERIFICATION PASSED")
        print(f"  {message}")
    else:
        print("VERIFICATION FAILED")
        print(f"  {message}")
    print()

    print("Initial stocks:")
    for name, qty in sorted(config.stocks.items()):
        print(f"  {name}: {qty}")
    print()

    print("Final stocks:")
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


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file> <result_to_test>", file=sys.stderr)
        print("  <file>           - Configuration file with stocks and processes", file=sys.stderr)
        print("  <result_to_test> - Trace file to verify", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    trace_file = sys.argv[2]

    try:
        config = parse_configuration(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        sys.exit(1)

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

    is_valid, message, final_stocks, last_cycle = verify_trace(config, trace)
    display_verification_result(is_valid, message, final_stocks, last_cycle, config)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
