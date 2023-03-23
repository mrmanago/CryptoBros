from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ModelSettings:
    q: int # Probability of adding or exiting
    p: float # Probability of being natted
    outgoing_nat: int # Default # of outgoing links for natted nodes
    outgoing: int # Default # of outgoing links for non-natted nodes