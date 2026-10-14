"""flow_bed: flow matching with likelihood guidance for Bayesian experimental design."""

from flow_bed.models.flow_matching import LikelihoodGuidanceModel, LinearPath, PriorFlowModel
from flow_bed.models.nodewrappers import CNFVectorField, GuidedVectorField, PosteriorSampler

__all__ = [
    "LinearPath",
    "PriorFlowModel",
    "LikelihoodGuidanceModel",
    "GuidedVectorField",
    "CNFVectorField",
    "PosteriorSampler",
]

__version__ = "0.1.0"
