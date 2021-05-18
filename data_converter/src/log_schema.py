from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

SCHEMA_VERSION = "1.0.0"


@dataclass
class Step:
    obs: np.ndarray = None
    reward: float = None
    action: List[float] = field(default_factory=list)
    done: bool = False

@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    version: str = SCHEMA_VERSION