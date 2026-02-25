# BANKAI Sample Data: Chignolin (CLN025)

## Overview
Chignolin is a 10-residue synthetic miniprotein (YYDPETGTWY) commonly used
as a benchmark system in molecular dynamics studies due to its small size
and well-characterized folding behavior.

## Dataset Specifications
- **Protein**: Chignolin (CLN025), PDB: 1UAO
- **Atoms**: 166
- **Residues**: 10
- **Frames**: 10,001
- **Timestep**: 0.01 ps (10 fs)
- **Total simulation time**: 100 ps (0.1 ns)
- **Force field**: AMBER ff14SB
- **Water model**: TIP3P (removed for analysis)
- **Temperature**: 300 K
- **Software**: GROMACS 2023.x

## Files
- `trajectory_stable.npy` - Coordinate trajectory (10001, 166, 3) float32
- `metadata_stable.json`  - Simulation metadata (timestep, n_atoms, etc.)
- `protein.npy`           - Protein atom indices array
- `residue_atom_mapping.json` - Residue-to-atom index mapping
- `5awl.pdb`              - Reference PDB structure
- `README.md`             - This file

## Usage
```bash
# Run example analysis (easiest)
bankai example

# Or manually
bankai analyze data/chignolin/trajectory_stable.npy \
               data/chignolin/metadata_stable.json \
               data/chignolin/protein.npy
```

## Citation
If you use this sample data, please cite:
- Honda, S. et al. (2004) Structure 12(8):1507-1518
- BANKAI: https://github.com/miosync-masa/bankai
