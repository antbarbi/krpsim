#!/usr/bin/env python3
"""
krpsim_greedy - Greedy planner for krpsim configurations.

This planner greedily starts the currently executable process with the best
target-oriented score, while respecting delays and overlapping executions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from krpsim_parser import Configuration, Process, parse_configuration


@dataclass(frozen=True, order=True)
class RunningProc:
    """A process instance currently running."""

    end_cycle: int
    process_name: str


@dataclass(frozen=True)
class Action:
    """A process start action in trace format."""

    cycle: int
    process_name: str


@dataclass
class GreedyResult:
    """Result object returned by greedy planning."""

    reached: bool
    actions: List[Action]
    final_stocks: Dict[str, int]
    final_cycle: int
    starts: int
    timed_out: bool = False
    stopped_by_limits: bool = False


class GreedyPlanner:
    """Greedy planner over krpsim process/resource dynamics."""

    def __init__(self, config: Configuration, target_resource: str):
        self.config = config
        self.target_resource = target_resource
        self.process_names = sorted(config.processes.keys())

    def _score(self, process: Process) -> tuple[float, float, int, int, int, str]:
        target_out = process.outputs.get(self.target_resource, 0)
        target_in = process.inputs.get(self.target_resource, 0)
        net_target = target_out - target_in
        total_out = sum(process.outputs.values())
        total_in = sum(process.inputs.values())
        delay = process.delay if process.delay > 0 else 1

        return (
            net_target / delay,
            target_out / delay,
            net_target,
            total_out - total_in,
            -process.delay,
            process.name,
        )

    def _choose_process(self, stocks: Dict[str, int]) -> Optional[Process]:
        best_process: Optional[Process] = None
        best_score: Optional[tuple[float, float, int, int, int, str]] = None

        for name in self.process_names:
            process = self.config.processes[name]
            if not process.can_execute(stocks):
                continue

            score = self._score(process)
            if best_score is None or score > best_score:
                best_score = score
                best_process = process

        return best_process

    @staticmethod
    def _apply_completions(cycle: int, stocks: Dict[str, int], running: List[RunningProc], processes: Dict[str, Process]) -> None:
        if not running:
            return

        still_running: List[RunningProc] = []
        for proc in sorted(running):
            if proc.end_cycle <= cycle:
                processes[proc.process_name].produce_outputs(stocks)
            else:
                still_running.append(proc)

        running.clear()
        running.extend(still_running)

    def run(
        self,
        max_starts: int = 100000,
        max_cycle: int = 100000000,
        delay_seconds: float = 0.0,
    ) -> GreedyResult:
        stocks = dict(self.config.stocks)
        running: List[RunningProc] = []
        actions: List[Action] = []

        cycle = 0
        starts = 0
        timed_out = False
        stopped_by_limits = False

        started_at = time.perf_counter()
        start_qty = stocks.get(self.target_resource, 0)

        while True:
            if delay_seconds > 0 and (time.perf_counter() - started_at) >= delay_seconds:
                timed_out = True
                break

            if cycle > max_cycle or starts >= max_starts:
                stopped_by_limits = True
                break

            self._apply_completions(cycle, stocks, running, self.config.processes)

            process = self._choose_process(stocks)
            if process is not None:
                process.consume_inputs(stocks)
                running.append(
                    RunningProc(
                        end_cycle=cycle + process.delay,
                        process_name=process.name,
                    )
                )
                actions.append(Action(cycle=cycle, process_name=process.name))
                starts += 1
                continue

            if not running:
                break

            cycle = min(proc.end_cycle for proc in running)

        final_cycle = cycle
        if running and not timed_out and not stopped_by_limits:
            final_cycle = max(proc.end_cycle for proc in running)

        return GreedyResult(
            reached=stocks.get(self.target_resource, 0) > start_qty,
            actions=actions,
            final_stocks=stocks,
            final_cycle=final_cycle,
            starts=starts,
            timed_out=timed_out,
            stopped_by_limits=stopped_by_limits,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy planner for krpsim resource/process configurations",
    )
    parser.add_argument("config_file", help="Path to krpsim configuration file")
    parser.add_argument("delay", type=float, help="Maximum planning time in seconds")
    return parser.parse_args()


def infer_target_from_config(config: Configuration) -> str:
    target_resource = ""

    for entry in config.optimize:
        if entry != "time":
            target_resource = entry
            break

    if not target_resource:
        raise ValueError(
            "No target resource found. Add optimize:(resource) in config or pass --target-resource."
        )

    return target_resource


def action_trace_lines(actions: Sequence[Action]) -> List[str]:
    return [f"{a.cycle}:{a.process_name}" for a in actions]


def print_summary(target_resource: str, initial_qty: int, result: GreedyResult) -> None:
    print("=" * 60)
    print("KRPSIM_GREEDY - Planning Results")
    print("=" * 60)
    print(f"Objective: maximize {target_resource}")
    print(f"Initial {target_resource}: {initial_qty}")
    print(f"Final cycle: {result.final_cycle}")
    print(f"Final {target_resource}: {result.final_stocks.get(target_resource, 0)}")
    print(f"Process starts in plan: {result.starts}")

    if result.timed_out:
        print("Result: stopped by timeout")
    elif result.stopped_by_limits:
        print("Result: stopped by safety limits")
    elif result.reached:
        print("Result: improved target quantity")
    else:
        print("Result: no improvement")

    print("=" * 60)


def main() -> int:
    args = parse_args()

    try:
        config = parse_configuration(args.config_file)
    except FileNotFoundError:
        print(f"Error: configuration file not found: {args.config_file}")
        return 1
    except Exception as exc:
        print(f"Error: failed to parse configuration: {exc}")
        return 1

    if args.delay < 0:
        print("Error: delay must be >= 0")
        return 1

    try:
        target_resource = infer_target_from_config(config)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    planner = GreedyPlanner(config=config, target_resource=target_resource)
    initial_qty = config.stocks.get(target_resource, 0)

    max_starts = 100000
    max_cycle = 100000000

    result = planner.run(
        max_starts=max_starts,
        max_cycle=max_cycle,
        delay_seconds=args.delay,
    )
    print_summary(target_resource, initial_qty, result)

    trace_lines = action_trace_lines(result.actions)

    if trace_lines:
        print("Trace (cycle:process):")
        for line in trace_lines:
            print(line)
    else:
        print("Trace (cycle:process):")

    return 0 if result.reached else 2


if __name__ == "__main__":
    raise SystemExit(main())