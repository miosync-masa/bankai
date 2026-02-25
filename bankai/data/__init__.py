"""
BANKAI Sample Data
==================

Provides access to bundled example datasets for testing and demonstration.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

# サンプルデータのルートディレクトリ
DATA_DIR = Path(__file__).parent
CHIGNOLIN_DIR = DATA_DIR / 'chignolin'


def get_data_dir() -> Path:
    """サンプルデータのルートディレクトリを返す"""
    return DATA_DIR


def get_chignolin_dir() -> Path:
    """Chignolinサンプルデータのディレクトリを返す"""
    return CHIGNOLIN_DIR


def chignolin_available() -> bool:
    """Chignolinサンプルデータが利用可能か確認"""
    required = ['trajectory_stable.npy', 'metadata_stable.json', 'protein.npy']
    return all((CHIGNOLIN_DIR / f).exists() for f in required)


def load_chignolin() -> dict[str, Any]:
    if not chignolin_available():
        raise FileNotFoundError(
            f"Chignolin sample data not found. "
            f"Expected files in: {CHIGNOLIN_DIR}\n"
            f"Run 'bankai example --generate' to create synthetic test data, "
            f"or place your own data files in the directory above."
        )

    paths = {
        'trajectory': str(CHIGNOLIN_DIR / 'trajectory_stable.npy'),
        'metadata': str(CHIGNOLIN_DIR / 'metadata_stable.json'),
        'protein_indices': str(CHIGNOLIN_DIR / 'protein.npy'),
    }

    result = {
        'trajectory': np.load(paths['trajectory']),
        'metadata': json.loads((CHIGNOLIN_DIR / 'metadata_stable.json').read_text()),
        'protein_indices': np.load(paths['protein_indices']),
        'paths': paths,
    }

    atom_mapping_path = CHIGNOLIN_DIR / 'residue_atom_mapping.json'
    if atom_mapping_path.exists():
        result['atom_mapping'] = json.loads(atom_mapping_path.read_text())
        result['paths']['atom_mapping'] = str(atom_mapping_path)

    pdb_path = CHIGNOLIN_DIR / '5awl.pdb'
    if pdb_path.exists():
        result['paths']['topology'] = str(pdb_path)

    return result


def get_chignolin_paths() -> dict[str, str]:
    if not chignolin_available():
        raise FileNotFoundError(f"Chignolin sample data not found in: {CHIGNOLIN_DIR}")

    paths = {
        'trajectory': str(CHIGNOLIN_DIR / 'trajectory_stable.npy'),
        'metadata': str(CHIGNOLIN_DIR / 'metadata_stable.json'),
        'protein_indices': str(CHIGNOLIN_DIR / 'protein.npy'),
    }

    atom_mapping_path = CHIGNOLIN_DIR / 'residue_atom_mapping.json'
    if atom_mapping_path.exists():
        paths['atom_mapping'] = str(atom_mapping_path)

    pdb_path = CHIGNOLIN_DIR / '5awl.pdb'
    if pdb_path.exists():
        paths['topology'] = str(pdb_path)

    return paths


def generate_synthetic_chignolin(output_dir: Optional[str] = None) -> dict[str, str]:
    out = Path(output_dir) if output_dir else CHIGNOLIN_DIR
    out.mkdir(parents=True, exist_ok=True)

    n_frames = 10_001
    n_atoms = 166
    n_residues = 10

    residue_names = ['TYR', 'TYR', 'ASP', 'PRO', 'GLU',
                     'THR', 'GLY', 'THR', 'TRP', 'TYR']
    atoms_per_residue = [21, 21, 12, 14, 15, 14, 7, 14, 24, 24]

    rng = np.random.default_rng(42)

    coords_0 = np.zeros((n_atoms, 3), dtype=np.float32)
    atom_idx = 0
    for res_i, n_at in enumerate(atoms_per_residue):
        center = np.array([res_i * 0.38, 0.0, 0.0])
        coords_0[atom_idx:atom_idx + n_at] = center + rng.normal(0, 0.15, (n_at, 3))
        atom_idx += n_at

    trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    trajectory[0] = coords_0
    for t in range(1, n_frames):
        noise = rng.normal(0, 0.002, (n_atoms, 3)).astype(np.float32)
        restoring = -0.01 * (trajectory[t - 1] - coords_0)
        collective = 0.005 * np.sin(2 * np.pi * t / 500) * np.ones((n_atoms, 1))
        trajectory[t] = trajectory[t - 1] + noise + restoring
        trajectory[t, :, 0] += collective[:, 0]

    np.save(out / 'trajectory.npy', trajectory)

    metadata = {
        'n_frames': n_frames, 'n_atoms': n_atoms, 'n_residues': n_residues,
        'timestep_ps': 0.01, 'total_time_ps': (n_frames - 1) * 0.01,
        'protein_name': 'Chignolin_CLN025_synthetic', 'pdb_id': '1UAO',
        'sequence': 'YYDPETGTWY', 'force_field': 'synthetic (not real MD)',
        'temperature_K': 300,
        'note': 'Synthetic data for pipeline testing. Not from actual MD simulation.',
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    np.save(out / 'protein_indices.npy', np.arange(n_atoms, dtype=np.int64))

    atom_mapping = {}
    atom_idx = 0
    for res_i, n_at in enumerate(atoms_per_residue):
        atom_mapping[str(res_i)] = {
            'residue_name': residue_names[res_i],
            'residue_index': res_i,
            'atom_indices': list(range(atom_idx, atom_idx + n_at)),
        }
        atom_idx += n_at

    with open(out / 'atom_mapping.json', 'w') as f:
        json.dump(atom_mapping, f, indent=2)

    return {
        'trajectory': str(out / 'trajectory.npy'),
        'metadata': str(out / 'metadata.json'),
        'protein_indices': str(out / 'protein_indices.npy'),
        'atom_mapping': str(out / 'atom_mapping.json'),
    }
