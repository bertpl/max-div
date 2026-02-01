from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass

from max_div.solver import Score, SolverPreset, TargetTimeDuration


# =================================================================================================
#  Models
# =================================================================================================
@dataclass(frozen=True)
class SolverPresetBenchmarkParams:
    preset: SolverPreset
    problem_name: str
    problem_size: int
    duration: TargetTimeDuration
    seed: int

    def to_dict(self) -> dict:
        return dict(
            preset=self.preset.value,
            problem_name=self.problem_name,
            problem_size=self.problem_size,
            duration_sec=self.duration.value(),
            seed=self.seed,
        )

    @classmethod
    def from_dict(cls, data: dict) -> SolverPresetBenchmarkParams:
        return cls(
            preset=SolverPreset(data["preset"]),
            problem_name=data["problem_name"],
            problem_size=data["problem_size"],
            duration=TargetTimeDuration(t_target_sec=data["duration_sec"]),
            seed=data["seed"],
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
