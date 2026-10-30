from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class CaseInput:
    """Only model-visible fields; never reference answers or diagnosis-bearing titles.

    Logical isolation, not a security sandbox. Each benchmark must whitelist its
    payload. Different input schemas can support future QA/image benchmarks.
    """
    case_id: str
    schema: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Example:
    input: CaseInput
    reference: dict[str, Any] = field(repr=False)
    stratum: str = 'default'


StageRunner = Callable[[str, Callable[[], dict]], dict]


class Method(Protocol):
    def predict(self, case: CaseInput, run: StageRunner) -> None: ...


class Benchmark(Protocol):
    schema: str
    def load(self, dataset, image_root) -> list[Example]: ...
    def evaluate(self, example: Example, records: dict, run: StageRunner, judge: str) -> None: ...
