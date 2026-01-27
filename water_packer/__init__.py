"""
water_packer - Pack water molecules into atomic systems.

This package provides functionality to pack water molecules into atomic
systems represented as ASE Atoms, hyobj PeriodicSystem, or raw arrays.
"""

from .packer import WaterPacker, pack_water
from .distance_manager import DistanceManager
from .water_geometry import get_water_template, O_H_DISTANCE, H_O_H_ANGLE
from .constants import WATER_MOLAR_MASS, WATER_DENSITY_FACTOR
from .relaxation import WaterRelaxer, pack_with_relaxation

__version__ = "0.2.5"

__all__ = [
    "WaterPacker",
    "pack_water",
    "pack_with_relaxation",
    "WaterRelaxer",
    "DistanceManager",
    "get_water_template",
    "O_H_DISTANCE",
    "H_O_H_ANGLE",
    "WATER_MOLAR_MASS",
    "WATER_DENSITY_FACTOR",
]
