# BANKAI Tools

Utility scripts for trajectory conversion and data preparation.

## Tools

### convert_trajectory.py

Convert GROMACS `.xtc` / `.trr` trajectories to BANKAI-compatible `.npy` format.

### Preparing GROMACS Data
```bash
pip install mdtraj
python tools/convert_trajectory.py trajectory.xtc topology.pdb --output-dir ./data
```

**Output files:**

| File | Description |
|------|-------------|
| `trajectory.npy` | Atomic coordinates (n_frames × n_atoms × 3), Å units |
| `protein.npy` | Protein atom indices |
| `residue_atom_mapping.json` | Residue → atom index mapping (for Two-stage analysis) |
| `metadata.json` | System metadata (temperature, timestep, sequence, etc.) |
