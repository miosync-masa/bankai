"""
BANKAI Test Configuration
=========================

Shared fixtures and markers for the test suite.
"""

import pytest
import sys
import os
import numpy as np
from pathlib import Path


# ===============================
# Custom Markers
# ===============================

def pytest_configure(config):
    """カスタムマーカー登録"""
    config.addinivalue_line("markers", "gpu: requires GPU (skip in CI)")
    config.addinivalue_line("markers", "slow: takes >30s")
    config.addinivalue_line("markers", "integration: requires sample data files")


# ===============================
# GPU Detection
# ===============================

def _gpu_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_collection_modifyitems(config, items):
    """GPU非搭載環境でgpuマーカー付きテストを自動スキップ"""
    if not GPU_AVAILABLE:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ===============================
# Fixtures
# ===============================

@pytest.fixture(scope="session")
def sample_trajectory():
    """テスト用の小さなトラジェクトリ (100 frames, 20 atoms)"""
    rng = np.random.default_rng(42)
    n_frames = 100
    n_atoms = 20

    coords_0 = rng.normal(0, 1, (n_atoms, 3)).astype(np.float32)
    trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    trajectory[0] = coords_0

    for t in range(1, n_frames):
        noise = rng.normal(0, 0.01, (n_atoms, 3)).astype(np.float32)
        trajectory[t] = trajectory[t - 1] + noise

    return trajectory


@pytest.fixture(scope="session")
def sample_metadata():
    """テスト用メタデータ"""
    return {
        "n_frames": 100,
        "n_atoms": 20,
        "n_residues": 4,
        "timestep_ps": 0.01,
        "total_time_ps": 0.99,
        "protein_name": "test_protein",
    }


@pytest.fixture(scope="session")
def sample_protein_indices():
    """テスト用プロテインインデックス"""
    return np.arange(20, dtype=np.int64)


@pytest.fixture(scope="session")
def sample_atom_mapping():
    """テスト用残基→原子マッピング"""
    return {
        "0": [0, 1, 2, 3, 4],
        "1": [5, 6, 7, 8, 9],
        "2": [10, 11, 12, 13, 14],
        "3": [15, 16, 17, 18, 19],
    }


@pytest.fixture
def tmp_data_dir(tmp_path, sample_trajectory, sample_metadata,
                 sample_protein_indices, sample_atom_mapping):
    """一時ディレクトリにテストデータを書き出す"""
    import json

    np.save(tmp_path / "trajectory_stable.npy", sample_trajectory)

    with open(tmp_path / "metadata_stable.json", "w") as f:
        json.dump(sample_metadata, f)

    np.save(tmp_path / "protein.npy", sample_protein_indices)

    with open(tmp_path / "residue_atom_mapping.json", "w") as f:
        json.dump(sample_atom_mapping, f)

    return tmp_path
