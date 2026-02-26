"""
BANKAI Tools — Utility scripts for trajectory conversion and data preparation.

Available tools:
    convert_trajectory  - Convert GROMACS .xtc/.trr to BANKAI .npy format
"""

from .convert_trajectory import convert

__all__ = ["convert"]
