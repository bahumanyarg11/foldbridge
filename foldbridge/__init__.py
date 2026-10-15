"""foldbridge: one job spec -> Boltz-2 / Chai-1 / ColabFold -> uniform scores -> MD / agents."""
from .score import score_structure
from .spec import Chain, JobSpec, Ligand, load_spec
from .structure import Structure, read_structure

__version__ = "0.1.0"
__all__ = ["JobSpec", "Chain", "Ligand", "load_spec", "Structure", "read_structure", "score_structure"]
