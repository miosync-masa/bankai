"""
GPU Environment Tests
======================

Test GPU detection, fallback behavior, and environment configuration.
"""

import pytest
import os


class TestGPUEnvironment:
    """GPU環境検出"""

    def test_gpu_info_dict_keys(self):
        from bankai import get_gpu_info
        info = get_gpu_info()
        expected_keys = {'available', 'name', 'memory_gb',
                         'cuda_version', 'compute_capability', 'has_cupy'}
        assert set(info.keys()) == expected_keys

    def test_gpu_info_types(self):
        from bankai import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info['available'], bool)
        assert isinstance(info['name'], str)
        assert isinstance(info['memory_gb'], (int, float))
        assert isinstance(info['cuda_version'], str)
        assert isinstance(info['compute_capability'], str)
        assert isinstance(info['has_cupy'], bool)

    def test_memory_non_negative(self):
        from bankai import GPU_MEMORY
        assert GPU_MEMORY >= 0.0

    def test_globals_consistent_with_info(self):
        """グローバル変数とget_gpu_info()の整合性"""
        from bankai import (
            GPU_AVAILABLE, GPU_NAME, GPU_MEMORY,
            GPU_COMPUTE_CAPABILITY, CUDA_VERSION_STR, HAS_CUPY,
            get_gpu_info,
        )
        info = get_gpu_info()
        assert GPU_AVAILABLE == info['available']
        assert GPU_NAME == info['name']
        assert GPU_MEMORY == info['memory_gb']
        assert GPU_COMPUTE_CAPABILITY == info['compute_capability']
        assert CUDA_VERSION_STR == info['cuda_version']
        assert HAS_CUPY == info['has_cupy']


class TestGPUDeviceSwitch:
    """GPUデバイス切り替え"""

    @pytest.mark.gpu
    def test_set_gpu_device(self):
        from bankai import set_gpu_device
        set_gpu_device(0)  # デバイス0が存在する環境でのみ

    def test_set_gpu_device_no_gpu(self, monkeypatch):
        """GPUなし環境で警告が出るがクラッシュしない"""
        import bankai
        monkeypatch.setattr(bankai, 'GPU_AVAILABLE', False)
        bankai.set_gpu_device(0)  # 警告のみ、エラーなし


class TestEnvironmentVariables:
    """環境変数による設定"""

    def test_debug_env(self, monkeypatch):
        """BANKAI_DEBUG=1 でDEBUGレベル"""
        import logging
        from bankai import logger
        # 現在のレベルを保存
        original = logger.level
        try:
            monkeypatch.setenv('BANKAI_DEBUG', '1')
            from bankai import set_log_level
            set_log_level('DEBUG')
            assert logger.level == logging.DEBUG
        finally:
            logger.setLevel(original)

    def test_no_banner_env(self, monkeypatch):
        """BANKAI_NO_BANNER で抑制"""
        monkeypatch.setenv('BANKAI_NO_BANNER', '1')
        from bankai.cli import _should_show_banner
        assert not _should_show_banner()
