"""Physical constants for water packing calculations."""

# Fundamental constants
AVOGADRO = 6.02214076e23  # molecules/mol
WATER_MOLAR_MASS = 18.015  # g/mol

# Unit conversions
ANGSTROM_TO_CM = 1e-8  # 1 Å = 1e-8 cm

# Water density conversions
# 1 g/cm³ water → molecules/Å³
# = (1 g/cm³) * (1 mol / 18.015 g) * (6.022e23 molecules/mol) * (1e-8 cm/Å)³
# ≈ 0.0334 molecules/Å³
WATER_DENSITY_FACTOR = AVOGADRO / WATER_MOLAR_MASS * (ANGSTROM_TO_CM**3)
# ≈ 0.03342 molecules/Å³ per (g/cm³)
