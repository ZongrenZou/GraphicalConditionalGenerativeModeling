"""Backward-compatible re-export of the core CFM API (replaces per-example models.py)."""

from gcm.core.cfm import ModelCfg, TrainCfg, train_cfm_flax
from gcm.core.velocity_mlp import VelocityMLP

__all__ = ["VelocityMLP", "ModelCfg", "TrainCfg", "train_cfm_flax"]
