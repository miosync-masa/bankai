"""
Lambda³ GPU版解析モジュール
MD軌道の完全GPU化解析パイプライン
Version 4.0 + Third Impact対応版
"""

# 🆕🆕 最強レポート生成機能！（v4.0対応）
from .maximum_report_generator import (
    generate_maximum_report_from_results_v4,
)
from .md_lambda3_detector_gpu import MDConfig, MDLambda3DetectorGPU, MDLambda3Result

# 🆕 フル解析パイプライン
from .run_full_analysis import (
    run_quantum_validation_pipeline,
)

# 🔺 Third Impact Analytics
from .third_impact_analytics import (
    # ネットワーク解析（v3.0新機能！）
    AtomicNetworkGPU,
    AtomicNetworkLink,
    AtomicNetworkResult,
    # 原子レベル
    AtomicQuantumTrace,
    # 起源情報
    EventOrigin,
    ResidueBridge,
    # メインクラス
    ThirdImpactAnalyzer,
    ThirdImpactResult,
    # 実行関数
    run_third_impact_analysis,
)
from .two_stage_analyzer_gpu import (
    ResidueAnalysisConfig,
    ResidueEvent,
    ResidueLevelAnalysis,
    TwoStageAnalyzerGPU,
    TwoStageLambda3Result,
    perform_two_stage_analysis_gpu,
)

__all__ = [
    # メイン検出器
    "MDLambda3DetectorGPU",
    "MDConfig",
    "MDLambda3Result",
    # 2段階解析
    "TwoStageAnalyzerGPU",
    "ResidueAnalysisConfig",
    "TwoStageLambda3Result",
    "ResidueEvent",
    "ResidueLevelAnalysis",
    "perform_two_stage_analysis_gpu",
    # 🔺 Third Impact Analytics v3.0
    "ThirdImpactAnalyzer",
    "ThirdImpactResult",
    "AtomicQuantumTrace",
    "AtomicNetworkGPU",  # 追加！
    "AtomicNetworkResult",  # 追加！
    "AtomicNetworkLink",  # 追加！
    "ResidueBridge",  # 追加！
    "EventOrigin",  # 追加！
    "run_third_impact_analysis",
    # 評価
    "PerformanceEvaluatorGPU",
    "PerformanceMetrics",
    "EventDetectionResult",
    "evaluate_two_stage_performance",
    # フル解析パイプライン
    "run_quantum_validation_pipeline",
    # 🆕 最強レポート生成（v4.0）
    "generate_maximum_report_from_results_v4",
]

__version__ = "1.5.0"  # v4.0 + Third Impact対応メジャーアップデート！

# ========================================
# 便利な一括実行関数
# ========================================


def run_full_analysis(
    trajectory_path: str,
    metadata_path: str,
    protein_indices_path: str,
    enable_quantum: bool = True,
    enable_third_impact: bool = False,
    **kwargs,
):
    """
    Lambda³ GPU フル解析の便利関数

    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルパス
    metadata_path : str
        メタデータファイルパス
    protein_indices_path : str
        タンパク質インデックスファイルパス
    enable_quantum : bool
        量子検証を実行するか
    enable_third_impact : bool
        Third Impact解析を実行するか
    **kwargs
        その他のオプション

    Returns
    -------
    dict
        解析結果（v4.0形式）

    Examples
    --------
    >>> from bankai.analysis import run_full_analysis
    >>> results = run_full_analysis('traj.npy', 'meta.json', 'protein.npy')
    """
    from .run_full_analysis import run_quantum_validation_pipeline

    # デフォルト設定
    kwargs.setdefault("enable_two_stage", True)
    kwargs.setdefault("enable_visualization", True)
    kwargs.setdefault("output_dir", "./bankai_results")

    # Third Impact設定
    kwargs["enable_third_impact"] = enable_third_impact

    # 量子検証の制御
    if not enable_quantum:
        # 量子検証をスキップする方法を追加する必要があるかも
        pass

    return run_quantum_validation_pipeline(
        trajectory_path, metadata_path, protein_indices_path, **kwargs
    )


# ========================================
# 最強レポート生成の便利関数（v4.0対応）
# ========================================


def generate_max_report(results_or_path, **kwargs):
    """
    最強レポートを生成する超便利関数！（v4.0対応版）

    Parameters
    ----------
    results_or_path : dict or str
        解析結果の辞書、またはトラジェクトリパス

    Examples
    --------
    # パターン1: 既存の結果から
    >>> results = run_full_analysis('traj.npy', 'meta.json', 'protein.npy')
    >>> report = generate_max_report(results)

    # パターン2: ファイルから一気に
    >>> report = generate_max_report('traj.npy', metadata_path='meta.json', protein_indices_path='protein.npy')
    """
    if isinstance(results_or_path, dict):
        # 既存の結果から（v4.0形式）
        return generate_maximum_report_from_results_v4(
            lambda_result=results_or_path.get("lambda_result"),
            two_stage_result=results_or_path.get("two_stage_result"),
            quantum_assessments=results_or_path.get("quantum_assessments"),
            third_impact_results=results_or_path.get("third_impact_results"),  # 🔺 追加
            **kwargs,
        )
    else:
        # ファイルから解析して最強レポート
        results = run_full_analysis(
            results_or_path,
            kwargs.pop("metadata_path", None),
            kwargs.pop("protein_indices_path", None),
            **kwargs,
        )
        return generate_maximum_report_from_results_v4(
            lambda_result=results["lambda_result"],
            two_stage_result=results.get("two_stage_result"),
            quantum_assessments=results.get("quantum_assessments"),
            third_impact_results=results.get("third_impact_results"),  # 🔺 追加
            **kwargs,
        )


# ========================================
# ショートカット（さらに便利に）
# ========================================


def analyze(
    trajectory_path: str, metadata_path: str, protein_indices_path: str, **kwargs
):
    """
    最も簡単な実行方法

    Examples
    --------
    >>> from bankai.analysis import analyze
    >>> results = analyze('traj.npy', 'meta.json', 'protein.npy')
    """
    return run_full_analysis(
        trajectory_path, metadata_path, protein_indices_path, **kwargs
    )


def analyze_with_impact(
    trajectory_path: str, metadata_path: str, protein_indices_path: str, **kwargs
):
    """
    Third Impact込みの完全解析！

    Examples
    --------
    >>> from bankai.analysis import analyze_with_impact
    >>> results = analyze_with_impact('traj.npy', 'meta.json', 'protein.npy')
    >>> # 自動的にThird Impactも実行される！
    """
    kwargs["enable_third_impact"] = True
    return run_full_analysis(
        trajectory_path, metadata_path, protein_indices_path, **kwargs
    )


def max_report(results):
    """
    超簡単な最強レポート生成！（v4.0対応版）

    Examples
    --------
    >>> from bankai.analysis import analyze, max_report
    >>> results = analyze('traj.npy', 'meta.json', 'protein.npy')
    >>> report = max_report(results)  # これだけ！
    """
    return generate_maximum_report_from_results_v4(
        lambda_result=results.get("lambda_result"),
        two_stage_result=results.get("two_stage_result"),
        quantum_assessments=results.get("quantum_assessments"),
        third_impact_results=results.get("third_impact_results"),  # 🔺 追加
    )


# ========================================
# 実験的機能：ワンライナー解析
# ========================================


def quick_quantum_check(
    trajectory_path: str, metadata_path: str, protein_indices_path: str
):
    """
    超高速量子チェック（Third Impact v3.0込み）
    """
    results = analyze_with_impact(
        trajectory_path,
        metadata_path,
        protein_indices_path,
        enable_visualization=False,
        verbose=False,
    )

    # Third Impact v3.0結果から起源原子を抽出
    quantum_atoms = []
    network_hubs = []  # v3.0: ハブ原子も重要！
    bridges = []  # v3.0: ブリッジ情報も！

    if "third_impact_results" in results and results["third_impact_results"]:
        for impact_result in results["third_impact_results"].values():
            # v3.0: origin.genesis_atomsを使用
            quantum_atoms.extend(impact_result.origin.genesis_atoms)

            # v3.0: ネットワーク情報も抽出
            if impact_result.atomic_network:
                network_hubs.extend(impact_result.atomic_network.hub_atoms[:3])
                bridges.extend(impact_result.atomic_network.residue_bridges[:2])

    return {
        "quantum_atoms": quantum_atoms,
        "network_hubs": network_hubs,  # v3.0新機能！
        "bridges": bridges,  # v3.0新機能！
    }


# ========================================
# バージョン情報
# ========================================


def get_version_info():
    """
    Lambda³ GPU バージョン情報取得

    Returns
    -------
    dict
        バージョン情報
    """
    return {
        "version": __version__,
        "features": {
            "bankai_core": True,
            "two_stage": True,
            "quantum_validation": True,
            "third_impact": True,  # 🔺 新機能！
            "gpu_acceleration": True,
            "maximum_report": True,
        },
        "description": "Lambda³ GPU Analysis Pipeline with Third Impact Analytics",
    }
