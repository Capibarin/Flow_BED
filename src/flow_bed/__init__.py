"""flow_bed: flow matching with likelihood guidance for Bayesian experimental design."""

from flow_bed.models.flow_matching import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.models.posterior_sampler import PosteriorSampler, VectorField

__all__ = [
    "LikelihoodGuidanceModel",
    "PosteriorSampler",
    "PriorFlowModel",
    "VectorField",
]

__version__ = "0.1.0"
