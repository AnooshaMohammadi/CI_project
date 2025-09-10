# benchmark.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BenchmarkFunction:
    name: str
    range: Tuple[float, float]
    dimension: int
    global_minima: int
    type: str  # "unimodal" or "multimodal"

benchmark_functions = [
    BenchmarkFunction("ackleyn2", (-5.12, 5.12), 2, 0.0, "unimodal"),
    BenchmarkFunction("ackley", (-32.768, 32.768), 2, 0.0, "multimodal"),
    BenchmarkFunction("zimmerman", (-5.12, 5.12), 2, 0.0, "multimodal"),
]
