#!/usr/bin/env python3
"""
krpsim_greedy - CLI wrapper for krpsim beam-greedy planner.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

from srcs.krpsim_parser import Configuration, parse_configuration
from srcs.krpsim_greedy_core import Action, GreedyPlanner, GreedyResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy planner for krpsim resource/process configurations",
    )
    parser.add_argument("config_file", help="Path to krpsim configuration file")
    parser.add_argument("delay", type=float, help="Maximum planning time in seconds")
    return parser.parse_args()


def infer_targets_from_config(config: Configuration) -> tuple[List[str], bool]:
    target_resources: List[str] = []
    optimize_time = False

    for entry in config.optimize:
        if entry == "time":
            optimize_time = True
            continue
        if entry not in target_resources:
            target_resources.append(entry)

    if not target_resources and not optimize_time:
        raise ValueError(
            "No valid optimization target found. Use optimize:(time|stock1;time|stock2;...)."
        )

    return target_resources, optimize_time


def action_trace_lines(actions: Sequence[Action]) -> List[str]:
    return [f"{a.cycle}:{a.process_name}" for a in actions]


def print_summary(target_resources: Sequence[str], optimize_time: bool, initial_stocks: Dict[str, int], result: GreedyResult) -> None:
    print("=" * 60)
    print("KRPSIM_GREEDY - Planning Results")
    print("=" * 60)
    if target_resources:
        print(f"Objective: maximize {';'.join(target_resources)}")
    if optimize_time:
        print("Objective: minimize completion time")

    for target in target_resources:
        initial_qty = initial_stocks.get(target, 0)
        final_qty = result.final_targets.get(target, result.final_stocks.get(target, 0))
        print(f"Initial {target}: {initial_qty}")
        print(f"Final {target}: {final_qty}")

    print(f"Final cycle: {result.final_cycle}")
    print(f"Process starts in plan: {result.starts}")
    print(f"Beam width: {result.beam_width}")

    if optimize_time and not target_resources:
        if result.timed_out:
            print("Result: stopped by timeout")
        elif result.stopped_by_limits:
            print("Result: stopped by safety limits")
        else:
            print("Result: completed (time objective)")
    elif result.timed_out:
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
        target_resources, optimize_time = infer_targets_from_config(config)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    planner = GreedyPlanner(
        config=config,
        target_resources=target_resources,
        optimize_time=optimize_time,
    )

    max_starts = 100000
    max_cycle = 100000000

    result = planner.run(
        max_starts=max_starts,
        max_cycle=max_cycle,
        delay_seconds=args.delay,
    )
    print_summary(target_resources, optimize_time, config.stocks, result)

    trace_lines = action_trace_lines(result.actions)

    if trace_lines:
        print("Trace (cycle:process):")
        for line in trace_lines:
            print(line)
    else:
        print("Trace (cycle:process):")

    if optimize_time and not target_resources:
        # Time-only objective is considered successful when we complete naturally
        # before timeout/limits.
        return 0 if not result.timed_out and not result.stopped_by_limits else 2

    return 0 if result.reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
