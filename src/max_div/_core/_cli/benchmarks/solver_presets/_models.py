from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass

from max_div._core.solver import Score, SolverPreset, TargetTimeDuration


# =================================================================================================
#  Models
# =================================================================================================
@dataclass(frozen=True)
class SolverPresetBenchmarkParams:
    """A SolverPresetBenchmarkParams identifies one benchmark run.

    `n_workers=1` is a plain single solve; a higher count runs the parallel solver with that many
    workers in the default (dynamic) grouping, so records carry the actual worker count of the
    machine that ran them.
    """

    preset: SolverPreset
    problem_name: str
    problem_size: int
    duration: TargetTimeDuration
    seed: int
    n_workers: int = 1

    @property
    def is_parallel(self) -> bool:
        """Return whether this run uses the parallel solver."""
        return self.n_workers > 1

    def column_label(self) -> str:
        """Return the label identifying this run's results column in reports.

        The bracket part states the worker count: "(1 worker)" for a single solve,
        "(12 workers)" for a parallel solve.
        """
        label = self.preset.value.upper()
        if not self.is_parallel:
            return f"{label} (1 worker)"
        return f"{label} ({self.n_workers} workers)"

    def to_dict(self) -> dict:
        return {
            "preset": self.preset.value,
            "problem_name": self.problem_name,
            "problem_size": self.problem_size,
            "duration_sec": self.duration.value(),
            "seed": self.seed,
            "n_workers": self.n_workers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SolverPresetBenchmarkParams:
        return cls(
            preset=SolverPreset(data["preset"]),
            problem_name=data["problem_name"],
            problem_size=data["problem_size"],
            duration=TargetTimeDuration(t_target_sec=data["duration_sec"]),
            seed=data["seed"],
            n_workers=data.get("n_workers", 1),
        )


@dataclass(frozen=True)
class SolverPresetBenchmarkExecutionInfo:
    pid: int
    t_start: float
    t_end: float

    @property
    def dt_start(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.t_start)

    @property
    def dt_end(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.t_end)

    @property
    def elapsed_sec(self) -> float:
        return (self.dt_end - self.dt_start).total_seconds()


@dataclass(frozen=True)
class SolverPresetBenchmarkResult:
    params: SolverPresetBenchmarkParams
    execution_info: SolverPresetBenchmarkExecutionInfo
    t_elapsed_sec: float
    n_iterations: int
    score: Score

    def to_dict(self) -> dict:
        result_dict = asdict(self)
        result_dict["params"] = self.params.to_dict()
        result_dict["execution_info"] = asdict(self.execution_info)
        return result_dict

    @classmethod
    def from_dict(cls, data: dict) -> SolverPresetBenchmarkResult:
        data["params"] = SolverPresetBenchmarkParams.from_dict(data["params"])
        data["execution_info"] = SolverPresetBenchmarkExecutionInfo(**data["execution_info"])
        data["score"] = Score(**data["score"])
        return cls(**data)


# =================================================================================================
#  (De)serialization
# =================================================================================================
def results_to_json(results: list[SolverPresetBenchmarkResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=4)


def results_from_json(json_str: str) -> list[SolverPresetBenchmarkResult]:
    return [SolverPresetBenchmarkResult.from_dict(d) for d in json.loads(json_str)]
