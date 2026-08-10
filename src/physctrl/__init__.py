"""Differentiable physical-control benchmark."""

from .control import optimise_controls, tracking_objective
from .dynamics import Params, simulate

__all__ = ["Params", "optimise_controls", "simulate", "tracking_objective"]
