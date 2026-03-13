#!/usr/bin/env python3
"""
krpsim_astar - Minimal A* planner for krpsim configurations.

This planner searches for a sequence of process starts that reaches a target
resource quantity. It supports process delays and overlapping executions.
"""

from __future__ import annotations

import argparse
import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from krpsim_parser import Configuration, Process, parse_configuration


@dataclass(frozen=True, order=True)
class RunningProc:
    """A process instance currently running."""

    end_cycle: int
    process_name: str


@dataclass(frozen=True)
class State:
    """Search state: current cycle, stocks, and running process instances."""

    cycle: int
    stocks: Tuple[int, ...]
    running: Tuple[RunningProc, ...]


@dataclass(frozen=True)
class Action:
    """Transition used to reconstruct a plan."""

    kind: str  # "start" or "wait"
    cycle: int
    process_name: Optional[str] = None


@dataclass
class SearchResult:
    """Result object returned by A* search."""

    reached: bool
    goal_state: Optional[State]
    actions: List[Action]
    expanded_nodes: int
    best_qty: int = 0
    timed_out: bool = False


class Planner:
    """A* planner over krpsim process/resource dynamics."""

    def __init__(self, config: Configuration, target_resource: str):
        self.config = config
        self.target_resource = target_resource

        self.resource_names = self._collect_resources(config, target_resource)
        self.resource_index = {name: idx for idx, name in enumerate(self.resource_names)}

        self.process_names = sorted(config.processes.keys())
        self.target_idx = self.resource_index[target_resource]

    @staticmethod
    def _collect_resources(config: Configuration, target_resource: str) -> List[str]:
        all_names = set(config.stocks.keys())
        all_names.add(target_resource)

        for process in config.processes.values():
            all_names.update(process.inputs.keys())
            all_names.update(process.outputs.keys())

        return sorted(all_names)

    def initial_state(self) -> State:
        stocks = [0] * len(self.resource_names)
        for name, qty in self.config.stocks.items():
            stocks[self.resource_index[name]] = qty

        state = State(cycle=0, stocks=tuple(stocks), running=tuple())
        return self._normalize(state)

    def _state_to_stocks_dict(self, state: State) -> Dict[str, int]:
        return {name: state.stocks[idx] for idx, name in enumerate(self.resource_names)}

    def _normalize(self, state: State) -> State:
        """Complete all running processes that end at or before state's cycle."""
        if not state.running:
            return state

        stocks = list(state.stocks)
        still_running: List[RunningProc] = []

        for running in state.running:
            if running.end_cycle <= state.cycle:
                process = self.config.processes[running.process_name]
                for resource, qty in process.outputs.items():
                    stocks[self.resource_index[resource]] += qty
            else:
                still_running.append(running)

        return State(
            cycle=state.cycle,
            stocks=tuple(stocks),
            running=tuple(sorted(still_running)),
        )

    def _can_start(self, state: State, process: Process) -> bool:
        for resource, qty in process.inputs.items():
            if state.stocks[self.resource_index[resource]] < qty:
                return False
        return True

    def _start_process(self, state: State, process: Process) -> State:
        stocks = list(state.stocks)

        for resource, qty in process.inputs.items():
            stocks[self.resource_index[resource]] -= qty

        running = list(state.running)
        running.append(RunningProc(end_cycle=state.cycle + process.delay, process_name=process.name))

        next_state = State(
            cycle=state.cycle,
            stocks=tuple(stocks),
            running=tuple(sorted(running)),
        )
        return self._normalize(next_state)

    def _wait_to_next_completion(self, state: State) -> Optional[State]:
        if not state.running:
            return None

        next_cycle = min(proc.end_cycle for proc in state.running)
        return self._normalize(State(cycle=next_cycle, stocks=state.stocks, running=state.running))

    def neighbors(self, state: State) -> List[Tuple[State, Action]]:
        out: List[Tuple[State, Action]] = []

        for name in self.process_names:
            process = self.config.processes[name]
            if self._can_start(state, process):
                next_state = self._start_process(state, process)
                out.append((next_state, Action(kind="start", cycle=state.cycle, process_name=name)))

        waited = self._wait_to_next_completion(state)
        if waited is not None:
            out.append((waited, Action(kind="wait", cycle=state.cycle)))

        return out

    def run_astar(self, delay_seconds: float = 0.0) -> SearchResult:
        start = self.initial_state()

        open_heap: List[Tuple[int, int, int, State]] = []
        counter = 0
        seen: set[State] = set()
        came_from: Dict[State, Tuple[State, Action]] = {}

        start_qty = start.stocks[self.target_idx]
        best_state = start
        best_qty = start_qty

        # Priority: larger target stock first, then earlier cycle.
        heapq.heappush(open_heap, (-start_qty, start.cycle, counter, start))

        expanded = 0

        started_at = time.perf_counter()

        while open_heap:
            if delay_seconds > 0 and (time.perf_counter() - started_at) >= delay_seconds:
                return SearchResult(
                    reached=best_qty > start_qty,
                    goal_state=best_state,
                    actions=self._reconstruct_actions(came_from, best_state),
                    expanded_nodes=expanded,
                    best_qty=best_qty,
                    timed_out=True,
                )

            _, _, _, current = heapq.heappop(open_heap)

            if current in seen:
                continue
            seen.add(current)

            curr_qty = current.stocks[self.target_idx]
            if curr_qty > best_qty or (curr_qty == best_qty and current.cycle < best_state.cycle):
                best_qty = curr_qty
                best_state = current

            expanded += 1

            for neighbor, action in self.neighbors(current):
                if neighbor in seen:
                    continue
                if neighbor not in came_from:
                    came_from[neighbor] = (current, action)
                counter += 1
                heapq.heappush(open_heap, (-neighbor.stocks[self.target_idx], neighbor.cycle, counter, neighbor))

        return SearchResult(
            reached=best_qty > start_qty,
            goal_state=best_state,
            actions=self._reconstruct_actions(came_from, best_state),
            expanded_nodes=expanded,
            best_qty=best_qty,
            timed_out=False,
        )

    @staticmethod
    def _reconstruct_actions(came_from: Dict[State, Tuple[State, Action]], end_state: State) -> List[Action]:
        actions: List[Action] = []
        current = end_state

        while current in came_from:
            previous, action = came_from[current]
            actions.append(action)
            current = previous

        actions.reverse()
        return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A* planner for krpsim resource/process configurations",
    )
    parser.add_argument("config_file", help="Path to krpsim configuration file")
    parser.add_argument(
        "--target-resource",
        default="",
        help="Optional resource target override (otherwise inferred from optimize:(...))",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Search timeout in seconds (0 means no timeout)",
    )
    parser.add_argument(
        "--trace-output",
        default="",
        help="Optional output file to save trace lines cycle:process",
    )
    return parser.parse_args()


def infer_target_from_config(config: Configuration, target_resource_arg: str) -> str:
    """Infer missing target settings from the configuration file."""
    target_resource = target_resource_arg.strip()

    if not target_resource:
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
    return [f"{a.cycle}:{a.process_name}" for a in actions if a.kind == "start" and a.process_name]


def print_summary(planner: Planner, result: SearchResult) -> None:
    print("=" * 60)
    print("KRPSIM_ASTAR - Planning Results")
    print("=" * 60)
    print(f"Objective: maximize {planner.target_resource}")
    print(f"Expanded states: {result.expanded_nodes}")
    initial_qty = planner.config.stocks.get(planner.target_resource, 0)
    print(f"Initial {planner.target_resource}: {initial_qty}")

    if result.goal_state is None:
        if result.timed_out:
            print("Result: no plan found before timeout")
        else:
            print("Result: no plan found")
        print("=" * 60)
        return

    goal = result.goal_state
    stocks = planner._state_to_stocks_dict(goal)

    if result.timed_out:
        print("Result: best plan found before timeout")
    else:
        print("Result: best plan found")

    print(f"Best cycle: {goal.cycle}")
    print(f"Best {planner.target_resource}: {stocks.get(planner.target_resource, 0)}")

    start_actions = [a for a in result.actions if a.kind == "start"]
    print(f"Process starts in plan: {len(start_actions)}")
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

    try:
        target_resource = infer_target_from_config(
            config,
            args.target_resource,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    planner = Planner(
        config=config,
        target_resource=target_resource,
    )

    if args.delay < 0:
        print("Error: delay must be >= 0")
        return 1

    result = planner.run_astar(delay_seconds=args.delay)
    print_summary(planner, result)

    if not result.reached:
        return 2

    trace_lines = action_trace_lines(result.actions)

    print("Trace (cycle:process):")
    for line in trace_lines:
        print(line)

    if args.trace_output:
        with open(args.trace_output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(trace_lines))
            if trace_lines:
                fh.write("\n")
        print(f"Trace saved to: {args.trace_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
