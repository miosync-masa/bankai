"""
Import Tests
=============

Verify that all public modules and classes can be imported.
These tests must pass even without GPU (CPU-only CI environment).
"""

import pytest


class TestPackageImport:
    """パッケージレベルのインポート"""

    def test_import_bankai(self):
        import bankai
        assert hasattr(bankai, '__version__')
        assert hasattr(bankai, '__author__')

    def test_version_format(self):
        import bankai
        parts = bankai.__version__.split('.')
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_gpu_globals_exist(self):
        from bankai import (
            GPU_AVAILABLE, GPU_NAME, GPU_MEMORY,
            GPU_COMPUTE_CAPABILITY, CUDA_VERSION_STR, HAS_CUPY,
        )
        assert isinstance(GPU_AVAILABLE, bool)
        assert isinstance(GPU_NAME, str)
        assert isinstance(GPU_MEMORY, float)
        assert isinstance(HAS_CUPY, bool)

    def test_get_gpu_info(self):
        from bankai import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'name' in info
        assert 'memory_gb' in info
        assert 'has_cupy' in info

    def test_set_log_level(self):
        from bankai import set_log_level
        set_log_level('WARNING')
        set_log_level('INFO')  # 元に戻す

    def test_set_log_level_invalid(self):
        from bankai import set_log_level
        with pytest.raises(ValueError):
            set_log_level('INVALID_LEVEL')


class TestLazyImports:
    """遅延インポートの動作確認（GPUなしでもImportErrorにならないこと）"""

    def test_import_error_handling(self):
        """存在しない属性でAttributeError"""
        import bankai
        with pytest.raises(AttributeError):
            _ = bankai.NonExistentClass

    def test_all_exports_listed(self):
        """__all__ に登録された名前が存在確認"""
        import bankai
        assert isinstance(bankai.__all__, list)
        assert len(bankai.__all__) > 0


class TestCLIImport:
    """CLIモジュールのインポート"""

    def test_import_cli(self):
        from bankai.cli import main, build_parser, print_banner
        assert callable(main)
        assert callable(build_parser)
        assert callable(print_banner)

    def test_build_parser(self):
        from bankai.cli import build_parser
        parser = build_parser()
        assert parser.prog == 'bankai'


class TestDataImport:
    """データモジュールのインポート"""

    def test_import_data(self):
        from bankai.data import (
            get_data_dir, get_chignolin_dir,
            chignolin_available, generate_synthetic_chignolin,
        )
        assert callable(chignolin_available)
        assert callable(generate_synthetic_chignolin)

    def test_data_dir_exists(self):
        from bankai.data import get_data_dir
        data_dir = get_data_dir()
        assert data_dir.exists()
