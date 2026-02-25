"""
Sample Data Tests
==================

Test synthetic data generation and data loading utilities.
"""

import pytest
import numpy as np
import json
from pathlib import Path


class TestSyntheticGeneration:
    """合成データ生成のテスト"""

    def test_generate_creates_files(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))

        assert Path(paths['trajectory']).exists()
        assert Path(paths['metadata']).exists()
        assert Path(paths['protein_indices']).exists()
        assert Path(paths['atom_mapping']).exists()

    def test_trajectory_shape(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        traj = np.load(paths['trajectory'])

        assert traj.shape == (10_001, 166, 3)
        assert traj.dtype == np.float32

    def test_trajectory_no_nan(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        traj = np.load(paths['trajectory'])

        assert not np.any(np.isnan(traj))
        assert not np.any(np.isinf(traj))

    def test_metadata_content(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        with open(paths['metadata']) as f:
            meta = json.load(f)

        assert meta['n_frames'] == 10_001
        assert meta['n_atoms'] == 166
        assert meta['n_residues'] == 10
        assert meta['timestep_ps'] == 0.01
        assert meta['sequence'] == 'YYDPETGTWY'
        assert 'synthetic' in meta.get('note', '').lower()

    def test_protein_indices(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        indices = np.load(paths['protein_indices'])

        assert indices.shape == (166,)
        assert indices[0] == 0
        assert indices[-1] == 165

    def test_atom_mapping_residues(self, tmp_path):
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        with open(paths['atom_mapping']) as f:
            mapping = json.load(f)

        assert len(mapping) == 10  # 10 residues

        # 全原子カバー確認
        all_atoms = []
        for res_id, info in mapping.items():
            all_atoms.extend(info['atom_indices'])

        assert sorted(all_atoms) == list(range(166))

    def test_atom_mapping_chignolin_sequence(self, tmp_path):
        """Chignolinの残基名がYYDPETGTWY"""
        from bankai.data import generate_synthetic_chignolin

        paths = generate_synthetic_chignolin(str(tmp_path))
        with open(paths['atom_mapping']) as f:
            mapping = json.load(f)

        expected = ['TYR', 'TYR', 'ASP', 'PRO', 'GLU',
                    'THR', 'GLY', 'THR', 'TRP', 'TYR']

        for i, name in enumerate(expected):
            assert mapping[str(i)]['residue_name'] == name

    def test_deterministic(self, tmp_path):
        """同じシードなら同じデータ"""
        from bankai.data import generate_synthetic_chignolin

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        paths1 = generate_synthetic_chignolin(str(dir1))
        paths2 = generate_synthetic_chignolin(str(dir2))

        traj1 = np.load(paths1['trajectory'])
        traj2 = np.load(paths2['trajectory'])

        np.testing.assert_array_equal(traj1, traj2)


class TestDataLoader:
    """データローダーのテスト"""

    def test_chignolin_available_false_when_empty(self, tmp_path, monkeypatch):
        """データがない場合 False"""
        from bankai import data as data_mod
        monkeypatch.setattr(data_mod, 'CHIGNOLIN_DIR', tmp_path)
        assert not data_mod.chignolin_available()

    def test_chignolin_available_true_after_generate(self, tmp_path, monkeypatch):
        """生成後は True"""
        from bankai import data as data_mod
        monkeypatch.setattr(data_mod, 'CHIGNOLIN_DIR', tmp_path)

        data_mod.generate_synthetic_chignolin(str(tmp_path))
        assert data_mod.chignolin_available()

    def test_load_chignolin_raises_when_missing(self, tmp_path, monkeypatch):
        from bankai import data as data_mod
        monkeypatch.setattr(data_mod, 'CHIGNOLIN_DIR', tmp_path)

        with pytest.raises(FileNotFoundError):
            data_mod.load_chignolin()

    def test_load_chignolin_returns_dict(self, tmp_path, monkeypatch):
        from bankai import data as data_mod
        monkeypatch.setattr(data_mod, 'CHIGNOLIN_DIR', tmp_path)

        data_mod.generate_synthetic_chignolin(str(tmp_path))
        result = data_mod.load_chignolin()

        assert 'trajectory' in result
        assert 'metadata' in result
        assert 'protein_indices' in result
        assert 'paths' in result
        assert result['trajectory'].shape == (10_001, 166, 3)

    def test_get_chignolin_paths(self, tmp_path, monkeypatch):
        from bankai import data as data_mod
        monkeypatch.setattr(data_mod, 'CHIGNOLIN_DIR', tmp_path)

        data_mod.generate_synthetic_chignolin(str(tmp_path))
        paths = data_mod.get_chignolin_paths()

        assert 'trajectory' in paths
        assert 'metadata' in paths
        assert 'protein_indices' in paths
        assert 'atom_mapping' in paths
        assert all(Path(p).exists() for p in paths.values())
