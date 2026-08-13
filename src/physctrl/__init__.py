"""Differentiable physical-optimisation benchmark."""

from .acoustic import ArrayConfig, evaluate, focus_phases, optimise_phases
from .control import optimise_controls, tracking_objective
from .dynamics import Params, simulate

__all__ = [
    "ArrayConfig",
    "Params",
    "evaluate",
    "focus_phases",
    "optimise_controls",
    "optimise_phases",
    "simulate",
    "tracking_objective",
]
