"""Water molecule geometry template."""

import numpy as np
from scipy.spatial.transform import Rotation

# TIP3P/SPC-like water geometry
O_H_DISTANCE = 0.9572  # Å
H_O_H_ANGLE = 104.52   # degrees


def get_water_template():
    """
    Return water molecule template (symbols, coordinates centered on O).
    
    Returns
    -------
    symbols : list of str
        ['O', 'H', 'H']
    coords : np.ndarray
        (3, 3) array of coordinates in Angstrom, O at origin.
    """
    angle_rad = np.deg2rad(H_O_H_ANGLE / 2)
    
    # O at origin, H atoms in the y-z plane
    coords = np.array([
        [0.0, 0.0, 0.0],  # O
        [0.0, O_H_DISTANCE * np.sin(angle_rad), O_H_DISTANCE * np.cos(angle_rad)],   # H1
        [0.0, -O_H_DISTANCE * np.sin(angle_rad), O_H_DISTANCE * np.cos(angle_rad)],  # H2
    ])
    
    return ['O', 'H', 'H'], coords


def rotate_water(coords, rotation=None, rng=None):
    """
    Apply rotation to water coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        (3, 3) water coordinates.
    rotation : scipy.spatial.transform.Rotation, optional
        Specific rotation to apply.
    rng : np.random.Generator, optional
        Random number generator for random rotation.
    
    Returns
    -------
    np.ndarray
        Rotated coordinates.
    """
    if rotation is None:
        if rng is None:
            rng = np.random.default_rng()
        rotation = Rotation.random(random_state=rng)
    
    return rotation.apply(coords)
