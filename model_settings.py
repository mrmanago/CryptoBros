from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ModelSettings:
    q: int # Probability of adding or exiting
    p: float # Probability of being natted
    outgoing_dist: np.array # Distribution of outgoing links