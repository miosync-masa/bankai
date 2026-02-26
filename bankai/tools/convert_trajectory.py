#!/usr/bin/env python3
"""
BANKAI Trajectory Converter
Convert GROMACS .xtc/.trr trajectories to BANKAI-compatible .npy format.

Usage:
    python convert_trajectory.py trajectory.xtc topology.pdb
    python convert_trajectory.py trajectory.xtc topology.pdb --output-dir ./data
"""

import argparse
import json
import numpy as np

try:
    import mdtraj as md
except ImportError:
    print("❌ mdtraj required: pip install mdtraj")
    exit(1)


def convert(xtc_path, top_path, output_dir="."):
    print(f"Loading trajectory: {xtc_path}")
    traj = md.load(xtc_path, top=top_path)

    print(f"  Atoms:  {traj.n_atoms}")
    print(f"  Frames: {traj.n_frames}")
    print(f"  Time:   {traj.time[-1] / 1000:.2f} ns")

    # nm → Å
    coords = traj.xyz * 10
    np.save(f"{output_dir}/trajectory.npy", coords)

    # Protein atom indices
    protein_atoms = traj.topology.select("protein")
    np.save(f"{output_dir}/protein.npy", protein_atoms)

    # Residue-atom mapping
    residue_map = {}
    for res in traj.topology.residues:
        if res.is_protein:
            residue_map[f"{res.name}{res.resSeq}"] = [a.index for a in res.atoms]

    with open(f"{output_dir}/residue_atom_mapping.json", "w") as f:
        json.dump(residue_map, f, indent=2)

    # Metadata
    protein_residues = [r for r in traj.topology.residues if r.is_protein]
    dt_ps = traj.timestep  # mdtraj returns ps
    sequence = "".join(
        md.core.residue_names.amino_acid_codes.get(r.name, "X")
        for r in protein_residues
    )

    metadata = {
        "system_name": top_path.replace(".pdb", ""),
        "temperature": 300.0,
        "time_step_ps": float(dt_ps),
        "n_frames": traj.n_frames,
        "n_atoms": len(protein_atoms),
        "protein": {
            "n_residues": len(protein_residues),
            "n_atoms": len(protein_atoms),
            "sequence": sequence,
        },
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Conversion complete!")
    print(f"  trajectory.npy          ({coords.shape})")
    print(f"  protein.npy             ({len(protein_atoms)} atoms)")
    print(f"  residue_atom_mapping.json ({len(residue_map)} residues)")
    print(f"  metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GROMACS trajectory for BANKAI")
    parser.add_argument("trajectory", help=".xtc or .trr file")
    parser.add_argument("topology", help=".pdb or .gro file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()
    convert(args.trajectory, args.topology, args.output_dir)
