#!/usr/bin/env python3
"""
Core classes and planner logic for krpsim greedy/beam search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from srcs.krpsim_parser import Configuration, Process


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
class BeamState:
    """A single search state kept in the beam frontier."""

    cycle: int
    stocks: Dict[str, int]
    running: List[RunningProc]
    actions: List[Action]
    starts: int


@dataclass
class GreedyResult:
    """Result object returned by greedy planning."""

    reached: bool
    actions: List[Action]
    final_stocks: Dict[str, int]
    final_cycle: int
    starts: int
    beam_width: int
    final_targets: Dict[str, int]
    timed_out: bool = False
    stopped_by_limits: bool = False


class GreedyPlanner:
    """Beam-search planner with greedy process scoring."""

    def __init__(self, config: Configuration, target_resources: List[str], optimize_time: bool):
        self.config = config
        self.target_resources = target_resources
        self.optimize_time = optimize_time
        self.process_names = sorted(config.processes.keys())
        self.resource_names = self._collect_resources()
        self.target_value_per_unit = self._build_target_value_map()

    def _collect_resources(self) -> List[str]:
        resources = set(self.config.stocks.keys())
        resources.update(self.target_resources)
        for process in self.config.processes.values():
            resources.update(process.inputs.keys())
            resources.update(process.outputs.keys())
        return sorted(resources)

    def _score(self, process: Process) -> tuple[float, float, int, int, int, str]:
        target_out = sum(process.outputs.get(target, 0) for target in self.target_resources)
        target_in = sum(process.inputs.get(target, 0) for target in self.target_resources)
        net_target = target_out - target_in
        total_out = sum(process.outputs.values())
        total_in = sum(process.inputs.values())
        delay = process.delay if process.delay > 0 else 1

        weighted_out = sum(
            qty * self.target_value_per_unit.get(resource, 0.0)
            for resource, qty in process.outputs.items()
        )
        weighted_in = sum(
            qty * self.target_value_per_unit.get(resource, 0.0)
            for resource, qty in process.inputs.items()
        )
        weighted_net = weighted_out - weighted_in

        if self.optimize_time:
            return (
                weighted_net / delay,
                weighted_out / delay,
                net_target,
                total_out - total_in,
                -process.delay,
                process.name,
            )

        return (
            float(net_target),
            float(target_out),
            net_target,
            total_out - total_in,
            0,
            process.name,
        )

    def _build_target_value_map(self) -> Dict[str, float]:
        """Estimate direct target value carried by each resource unit."""
        values: Dict[str, float] = {target: 1.0 for target in self.target_resources}

        for process in self.config.processes.values():
            target_out = sum(process.outputs.get(target, 0) for target in self.target_resources)
            if target_out <= 0:
                continue

            for resource, qty in process.inputs.items():
                if qty <= 0:
                    continue
                ratio = target_out / qty
                if ratio > values.get(resource, 0.0):
                    values[resource] = ratio

        return values

    def _iter_executable_processes(self, stocks: Dict[str, int]) -> List[Process]:
        executable: List[Tuple[tuple[float, float, int, int, int, str], Process]] = []
        for name in self.process_names:
            process = self.config.processes[name]
            if not process.can_execute(stocks):
                continue
            executable.append((self._score(process), process))

        executable.sort(key=lambda x: x[0], reverse=True)
        return [proc for _, proc in executable]

    @staticmethod
    def _max_repetitions(process: Process, stocks: Dict[str, int]) -> int:
        """Return max times a process can be started immediately with current stocks."""
        if not process.inputs:
            return 1

        max_repeat: Optional[int] = None
        for resource, qty in process.inputs.items():
            if qty <= 0:
                continue
            possible = stocks.get(resource, 0) // qty
            if max_repeat is None or possible < max_repeat:
                max_repeat = possible

        if max_repeat is None:
            return 1
        return max(1, max_repeat)

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

    def _incoming_target(self, running: List[RunningProc]) -> float:
        incoming = 0.0
        for proc in running:
            process = self.config.processes[proc.process_name]
            incoming += sum(
                qty * self.target_value_per_unit.get(resource, 0.0)
                for resource, qty in process.outputs.items()
            )
        return incoming

    def _target_potential(self, stocks: Dict[str, int]) -> float:
        potential = 0.0
        for resource, qty in stocks.items():
            if qty <= 0:
                continue
            potential += qty * self.target_value_per_unit.get(resource, 0.0)
        return potential

    def _state_score(self, state: BeamState) -> tuple[float, int, int]:
        if not self.target_resources:
            # Time-only optimization: prefer shorter completion paths.
            return (
                -float(state.cycle),
                -len(state.actions),
                0,
            )

        current_target = sum(state.stocks.get(target, 0) for target in self.target_resources)
        incoming_target = self._incoming_target(state.running)
        target_potential = self._target_potential(state.stocks)
        # Primary: maximize current target, near-future target and convertibility potential.
        # Secondary depends on optimize mode.
        combined_target = current_target + 0.50 * incoming_target + 0.15 * target_potential
        if self.optimize_time:
            return (
                combined_target,
                -state.cycle,
                -len(state.actions),
            )

        return (
            combined_target,
            -len(state.actions),
            0,
        )

    def _state_signature(self, state: BeamState) -> tuple[int, tuple[int, ...], tuple[RunningProc, ...]]:
        stocks_tuple = tuple(state.stocks.get(r, 0) for r in self.resource_names)
        running_tuple = tuple(sorted(state.running))
        return (state.cycle, stocks_tuple, running_tuple)

    @staticmethod
    def _adaptive_beam_width(delay_seconds: float) -> int:
        """Choose beam width automatically from the given delay budget."""
        if delay_seconds <= 0:
            return 16
        if delay_seconds < 0.25:
            return 4
        if delay_seconds < 1.0:
            return 8
        if delay_seconds < 3.0:
            return 12
        if delay_seconds < 8.0:
            return 16
        return 24

    def _better_state(self, candidate: BeamState, current: BeamState) -> bool:
        return self._state_score(candidate) > self._state_score(current)

    def _pick_best_state(self, states: Sequence[BeamState]) -> Optional[BeamState]:
        if not states:
            return None
        return max(states, key=self._state_score)

    def _evaluate_actions(self, actions: Sequence[Action]) -> tuple[Dict[str, int], int]:
        stocks = dict(self.config.stocks)
        running: List[RunningProc] = []
        last_cycle = 0

        for action in actions:
            cycle = action.cycle
            self._apply_completions(cycle, stocks, running, self.config.processes)

            process = self.config.processes[action.process_name]
            if process.can_execute(stocks):
                process.consume_inputs(stocks)
                running.append(RunningProc(end_cycle=cycle + process.delay, process_name=process.name))
                last_cycle = max(last_cycle, cycle)

        if running:
            running.sort()
            for proc in running:
                self.config.processes[proc.process_name].produce_outputs(stocks)
                last_cycle = max(last_cycle, proc.end_cycle)

        return stocks, last_cycle

    def _targets_improved(self, final_stocks: Dict[str, int]) -> bool:
        if not self.target_resources:
            return True
        for target in self.target_resources:
            if final_stocks.get(target, 0) > self.config.stocks.get(target, 0):
                return True
        return False

    def run(
        self,
        max_starts: int = 100000,
        max_cycle: int = 100000000,
        delay_seconds: float = 0.0,
    ) -> GreedyResult:
        beam_width = self._adaptive_beam_width(delay_seconds)

        frontier = [
            BeamState(
                cycle=0,
                stocks=dict(self.config.stocks),
                running=[],
                actions=[],
                starts=0,
            )
        ]
        terminals: List[BeamState] = []

        timed_out = False
        stopped_by_limits = False

        started_at = time.perf_counter()
        deadline = (started_at + delay_seconds) if delay_seconds > 0 else None

        def is_timed_out() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        while frontier:
            if is_timed_out():
                timed_out = True
                break

            candidates: List[BeamState] = []
            expansion_timed_out = False

            for state in frontier:
                if is_timed_out():
                    timed_out = True
                    expansion_timed_out = True
                    break

                if state.cycle > max_cycle or state.starts >= max_starts:
                    stopped_by_limits = True
                    continue

                stocks = dict(state.stocks)
                running = list(state.running)
                actions = list(state.actions)
                cycle = state.cycle

                self._apply_completions(cycle, stocks, running, self.config.processes)
                executable = self._iter_executable_processes(stocks)

                if executable:
                    for process in executable:
                        if is_timed_out():
                            timed_out = True
                            expansion_timed_out = True
                            break

                        next_stocks = dict(stocks)
                        next_running = list(running)
                        next_actions = list(actions)

                        process.consume_inputs(next_stocks)
                        next_running.append(
                            RunningProc(
                                end_cycle=cycle + process.delay,
                                process_name=process.name,
                            )
                        )
                        next_actions.append(Action(cycle=cycle, process_name=process.name))

                        candidates.append(
                            BeamState(
                                cycle=cycle,
                                stocks=next_stocks,
                                running=next_running,
                                actions=next_actions,
                                starts=state.starts + 1,
                            )
                        )

                    if expansion_timed_out:
                        break

                # If no process is executable, advance to next completion cycle.
                if not executable and running:
                    next_cycle = min(proc.end_cycle for proc in running)
                    candidates.append(
                        BeamState(
                            cycle=next_cycle,
                            stocks=stocks,
                            running=running,
                            actions=actions,
                            starts=state.starts,
                        )
                    )
                elif not executable:
                    terminals.append(
                        BeamState(
                            cycle=cycle,
                            stocks=stocks,
                            running=[],
                            actions=actions,
                            starts=state.starts,
                        )
                    )

            if expansion_timed_out:
                break

            if not candidates:
                break

            dedup: Dict[tuple[int, tuple[int, ...], tuple[RunningProc, ...]], BeamState] = {}
            for state in candidates:
                sig = self._state_signature(state)
                prev = dedup.get(sig)
                if prev is None or self._better_state(state, prev):
                    dedup[sig] = state

            ranked = sorted(dedup.values(), key=self._state_score, reverse=True)
            frontier = ranked[:beam_width]

        best_state = self._pick_best_state(terminals + frontier)
        if best_state is None:
            best_state = BeamState(
                cycle=0,
                stocks=dict(self.config.stocks),
                running=[],
                actions=[],
                starts=0,
            )

        final_stocks, final_cycle = self._evaluate_actions(best_state.actions)
        final_targets = {target: final_stocks.get(target, 0) for target in self.target_resources}

        return GreedyResult(
            reached=self._targets_improved(final_stocks),
            actions=list(best_state.actions),
            final_stocks=final_stocks,
            final_cycle=final_cycle,
            starts=len(best_state.actions),
            beam_width=beam_width,
            final_targets=final_targets,
            timed_out=timed_out,
            stopped_by_limits=stopped_by_limits,
        )
