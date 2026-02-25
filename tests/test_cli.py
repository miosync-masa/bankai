"""
CLI Tests
==========

Test command-line interface argument parsing and banner display.
"""

import pytest
import sys
from unittest.mock import patch
from io import StringIO


class TestParser:
    """argparseパーサーのテスト"""

    def test_parser_creation(self):
        from bankai.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_version_flag(self):
        from bankai.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(['--version'])
        assert exc.value.code == 0

    def test_no_args_returns_none_command(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_analyze_requires_args(self):
        from bankai.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['analyze'])

    def test_analyze_parses_positional(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            'analyze', 'traj.npy', 'meta.json', 'prot.npy'
        ])
        assert args.command == 'analyze'
        assert args.trajectory == 'traj.npy'
        assert args.metadata == 'meta.json'
        assert args.protein == 'prot.npy'

    def test_analyze_default_options(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            'analyze', 'traj.npy', 'meta.json', 'prot.npy'
        ])
        assert args.output == './bankai_results'
        assert args.enable_third_impact is False
        assert args.no_two_stage is False
        assert args.no_viz is False
        assert args.verbose is False
        assert args.third_impact_top_n == 10

    def test_analyze_all_options(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            'analyze', 'traj.npy', 'meta.json', 'prot.npy',
            '--enable-third-impact',
            '--atom-mapping', 'map.json',
            '--third-impact-top-n', '15',
            '--topology', 'top.pdb',
            '--output', './out',
            '--no-two-stage',
            '--no-viz',
            '--verbose',
        ])
        assert args.enable_third_impact is True
        assert args.atom_mapping == 'map.json'
        assert args.third_impact_top_n == 15
        assert args.topology == 'top.pdb'
        assert args.output == './out'
        assert args.no_two_stage is True
        assert args.no_viz is True
        assert args.verbose is True

    def test_benchmark_subcommand(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['benchmark'])
        assert args.command == 'benchmark'

    def test_info_subcommand(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['info'])
        assert args.command == 'info'

    def test_check_gpu_subcommand(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['check-gpu'])
        assert args.command == 'check-gpu'

    def test_example_subcommand(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['example'])
        assert args.command == 'example'
        assert args.generate is False

    def test_example_generate_flag(self):
        from bankai.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['example', '--generate'])
        assert args.generate is True


class TestBanner:
    """バナー表示のテスト"""

    def test_banner_suppressed_by_env(self, monkeypatch):
        monkeypatch.setenv('BANKAI_NO_BANNER', '1')
        from bankai.cli import _should_show_banner
        assert not _should_show_banner()

    def test_banner_suppressed_non_tty(self, monkeypatch):
        monkeypatch.delenv('BANKAI_NO_BANNER', raising=False)
        # StringIOはisatty() = False
        monkeypatch.setattr(sys, 'stdout', StringIO())
        from bankai.cli import _should_show_banner
        assert not _should_show_banner()

    def test_gpu_status_line_cpu(self, monkeypatch):
        import bankai.cli as cli_mod
        monkeypatch.setattr(cli_mod, 'GPU_AVAILABLE', False)
        line = cli_mod._gpu_status_line()
        assert 'CPU' in line

    def test_gpu_status_line_gpu(self, monkeypatch):
        import bankai.cli as cli_mod
        monkeypatch.setattr(cli_mod, 'GPU_AVAILABLE', True)
        monkeypatch.setattr(cli_mod, 'GPU_NAME', 'Test GPU')
        monkeypatch.setattr(cli_mod, 'GPU_MEMORY', 8.0)
        line = cli_mod._gpu_status_line()
        assert 'Test GPU' in line
        assert '8.0' in line

    def test_each_banner_style_runs(self, monkeypatch):
        """各バナースタイルがエラーなく実行される"""
        import bankai.cli as cli_mod
        monkeypatch.setattr(cli_mod, 'GPU_AVAILABLE', False)

        from bankai.cli import _banner_simple, _banner_ascii, _banner_matrix, _banner_tamaki

        # 出力キャプチャしつつエラーなく動くか確認
        for fn in [_banner_simple, _banner_ascii, _banner_matrix, _banner_tamaki]:
            fn()  # エラーが出なければOK


class TestMainEntrypoint:
    """__main__.py のテスト"""

    def test_main_module_exists(self):
        """python -m bankai 用のモジュール存在確認"""
        import importlib
        spec = importlib.util.find_spec('bankai.__main__')
        assert spec is not None
