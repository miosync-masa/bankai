"""
Pipeline Integration Tests
============================

Test that the analysis pipeline runs end-to-end with synthetic data.
These tests verify the pipeline structure, not scientific correctness.
"""

import pytest
import numpy as np
import json
from pathlib import Path


class TestSyntheticPipelineSetup:
    """合成データからのパイプライン前処理テスト"""

    def test_synthetic_data_loadable(self, tmp_path):
        """合成データが正しくロードできる"""
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))

        traj = np.load(paths['trajectory'])
        indices = np.load(paths['protein_indices'])

        with open(paths['metadata']) as f:
            meta = json.load(f)

        with open(paths['atom_mapping']) as f:
            mapping = json.load(f)

        assert traj.shape[0] == meta['n_frames']
        assert traj.shape[1] == meta['n_atoms']
        assert traj.shape[2] == 3
        assert len(indices) == meta['n_atoms']
        assert len(mapping) == meta['n_residues']

    def test_atom_mapping_covers_all_atoms(self, tmp_path):
        """マッピングが全原子をカバー"""
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        traj = np.load(paths['trajectory'])
        n_atoms = traj.shape[1]

        with open(paths['atom_mapping']) as f:
            mapping = json.load(f)

        all_indices = set()
        for res_data in mapping.values():
            all_indices.update(res_data['atom_indices'])

        assert all_indices == set(range(n_atoms))

    def test_trajectory_continuity(self, tmp_path):
        """トラジェクトリが連続的（巨大ジャンプなし）"""
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        traj = np.load(paths['trajectory'])

        # フレーム間の最大変位
        displacements = np.diff(traj, axis=0)
        max_disp = np.max(np.abs(displacements))

        # 合成データでは0.1nm以下のはず
        assert max_disp < 0.5, f"Max displacement {max_disp} too large"


class TestFixtureData:
    """conftest.pyフィクスチャのテスト"""

    def test_fixture_trajectory_shape(self, sample_trajectory):
        assert sample_trajectory.shape == (100, 20, 3)
        assert sample_trajectory.dtype == np.float32

    def test_fixture_metadata_keys(self, sample_metadata):
        required = ['n_frames', 'n_atoms', 'n_residues', 'timestep_ps']
        for key in required:
            assert key in sample_metadata

    def test_fixture_consistency(self, sample_trajectory, sample_metadata,
                                 sample_protein_indices):
        assert sample_trajectory.shape[0] == sample_metadata['n_frames']
        assert sample_trajectory.shape[1] == sample_metadata['n_atoms']
        assert len(sample_protein_indices) == sample_metadata['n_atoms']

    def test_tmp_data_dir_files(self, tmp_data_dir):
        assert (tmp_data_dir / "trajectory_stable.npy").exists()
        assert (tmp_data_dir / "metadata_stable.json").exists()
        assert (tmp_data_dir / "protein.npy").exists()
        assert (tmp_data_dir / "residue_atom_mapping.json").exists()
