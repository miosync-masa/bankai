"""
Quantum Validation Module for Lambda³ - Version 4.0
====================================================

Lambda³統合型量子起源判定モジュール

主な特徴：
- Lambda³が検出した構造変化の量子性を判定
- 3パターン（瞬間/遷移/カスケード）の明確な分類
- 原子レベル証拠の活用
- 現実的かつ調整可能な判定基準

Version: 4.0.0 - Complete Refactoring
Authors: 環ちゃん & ご主人さま 💕
"""

from .quantum_validation_gpu import (
    # メインクラス
    QuantumValidatorV4,
    # データクラス
    LambdaAnomaly,
    AtomicEvidence,
    QuantumAssessment,
    # Enumクラス
    StructuralEventPattern,
    QuantumSignature,
    # 便利関数
    validate_lambda_events,
)

# バージョン情報（4.0.0！）
__version__ = "4.0.0"

# 公開するAPI
__all__ = [
    # Main class
    "QuantumValidatorV4",
    # Data classes
    "LambdaAnomaly",
    "AtomicEvidence",
    "QuantumAssessment",
    # Enums
    "StructuralEventPattern",
    "QuantumSignature",
    # Convenience functions
    "validate_lambda_events",
    "quick_validate",
    "batch_validate",
    # Utilities
    "check_dependencies",
    "test_quantum_module",
    "create_assessment_report",
]

# ============================================
# Dependency Checking
# ============================================


def check_dependencies():
    """依存関係のチェックとステータス表示"""
    import_status = {}

    # NumPy（必須）
    try:
        import numpy as np

        import_status["numpy"] = f"✅ {np.__version__}"
    except ImportError:
        import_status["numpy"] = "❌ Not installed (REQUIRED)"

    # SciPy（統計・信号処理）
    try:
        import scipy

        import_status["scipy"] = f"✅ {scipy.__version__}"
    except ImportError:
        import_status["scipy"] = "❌ Not installed (REQUIRED)"

    # CuPy（GPU - オプション）
    try:
        import cupy as cp

        if cp.cuda.is_available():
            import_status["cupy"] = (
                f"✅ Available (CUDA {cp.cuda.runtime.runtimeGetVersion()})"
            )
        else:
            import_status["cupy"] = "⚠️ Installed but no GPU detected"
    except ImportError:
        import_status["cupy"] = "ℹ️ Not installed (optional for GPU acceleration)"

    # MDAnalysis（トポロジー処理 - オプション）
    try:
        import MDAnalysis

        import_status["MDAnalysis"] = f"✅ {MDAnalysis.__version__}"
    except ImportError:
        import_status["MDAnalysis"] = "⚠️ Not installed (optional for topology)"

    # Lambda³ GPU本体
    try:
        from ..analysis import MDLambda3AnalyzerGPU

        import_status["bankai"] = "✅ Available"
    except ImportError:
        import_status["bankai"] = "⚠️ Lambda³ GPU not found (standalone mode)"

    # Matplotlib（可視化）
    try:
        import matplotlib

        import_status["matplotlib"] = f"✅ {matplotlib.__version__}"
    except ImportError:
        import_status["matplotlib"] = "⚠️ Not installed (optional for plots)"

    return import_status


# ============================================
# Quick Validation Functions
# ============================================


def quick_validate(event, lambda_result, trajectory=None, **kwargs):
    """
    単一イベントのクイック検証

    Parameters
    ----------
    event : dict
        構造変化イベント
    lambda_result : Any
        Lambda³解析結果
    trajectory : np.ndarray, optional
        原子座標トラジェクトリ
    **kwargs
        追加設定

    Returns
    -------
    QuantumAssessment
        量子性評価結果

    Examples
    --------
    >>> assessment = quick_validate(event, lambda_result, trajectory)
    >>> print(f"Quantum: {assessment.is_quantum}")
    >>> print(f"Signature: {assessment.signature.value}")
    """
    validator = QuantumValidatorV4(trajectory=trajectory, **kwargs)
    return validator.validate_event(event, lambda_result)


def batch_validate(lambda_result, trajectory=None, max_events=100, **kwargs):
    """
    Lambda³結果の一括検証

    Parameters
    ----------
    lambda_result : Any
        Lambda³解析結果
    trajectory : np.ndarray, optional
        原子座標トラジェクトリ
    max_events : int
        処理する最大イベント数
    **kwargs
        追加設定

    Returns
    -------
    dict
        検証結果サマリー
    """
    # イベント抽出
    events = []

    # critical_eventsから抽出
    if hasattr(lambda_result, "critical_events"):
        for e in lambda_result.critical_events[:max_events]:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                events.append(
                    {
                        "frame_start": int(e[0]),
                        "frame_end": int(e[1]),
                        "type": "critical",
                    }
                )

    # eventsディクショナリから抽出
    if hasattr(lambda_result, "events") and isinstance(lambda_result.events, dict):
        for event_type, event_list in lambda_result.events.items():
            for e in event_list[:10]:  # 各タイプ最大10個
                if len(events) >= max_events:
                    break

                if isinstance(e, dict):
                    events.append(
                        {
                            "frame_start": e.get("frame", e.get("start", 0)),
                            "frame_end": e.get("end", e.get("frame", 0)),
                            "type": event_type,
                        }
                    )

    if not events:
        print("⚠️ No events found in lambda_result")
        return {"error": "No events found"}

    # バリデーター作成と実行
    validator = QuantumValidatorV4(trajectory=trajectory, **kwargs)
    assessments = validator.validate_events(events, lambda_result)

    # サマリー生成
    summary = validator.generate_summary(assessments)

    # 詳細追加
    summary["assessments"] = assessments
    summary["validator"] = validator

    # サマリー表示
    validator.print_summary(assessments)

    return summary


# ============================================
# Report Generation
# ============================================


def create_assessment_report(assessments, output_file="quantum_assessment_v4.txt"):
    """
    量子性評価レポートの生成

    Parameters
    ----------
    assessments : List[QuantumAssessment]
        評価結果リスト
    output_file : str
        出力ファイル名

    Returns
    -------
    str
        レポート内容
    """
    report = []
    report.append("=" * 70)
    report.append("Quantum Assessment Report - Version 4.0")
    report.append("Lambda³ Integrated Quantum Origin Validation")
    report.append("=" * 70)
    report.append("")

    # 統計サマリー
    total = len(assessments)
    quantum = sum(1 for a in assessments if a.is_quantum)

    report.append(f"Total Events Analyzed: {total}")
    report.append(f"Quantum Events: {quantum} ({quantum / total * 100:.1f}%)")
    report.append("")

    # パターン別統計
    report.append("Pattern Distribution:")
    for pattern in StructuralEventPattern:
        count = sum(1 for a in assessments if a.pattern == pattern)
        if count > 0:
            quantum_count = sum(
                1 for a in assessments if a.pattern == pattern and a.is_quantum
            )
            report.append(
                f"  {pattern.value}: {count} events, "
                f"{quantum_count} quantum ({quantum_count / count * 100:.1f}%)"
            )
    report.append("")

    # シグネチャー分布
    report.append("Quantum Signatures Detected:")
    for sig in QuantumSignature:
        if sig == QuantumSignature.NONE:
            continue
        count = sum(1 for a in assessments if a.signature == sig)
        if count > 0:
            report.append(f"  {sig.value}: {count}")
    report.append("")

    # 個別イベント詳細（最初の10個）
    report.append("-" * 70)
    report.append("Individual Event Details (First 10 Quantum Events):")
    report.append("")

    quantum_events = [a for a in assessments if a.is_quantum][:10]
    for i, assessment in enumerate(quantum_events, 1):
        report.append(f"Event {i}:")
        report.append(f"  Pattern: {assessment.pattern.value}")
        report.append(f"  Signature: {assessment.signature.value}")
        report.append(f"  Confidence: {assessment.confidence:.1%}")
        report.append(f"  Explanation: {assessment.explanation}")

        if assessment.criteria_met:
            report.append("  Criteria met:")
            for criterion in assessment.criteria_met[:3]:  # 最初の3個
                report.append(f"    - {criterion}")

        if assessment.bell_inequality is not None:
            report.append(f"  Bell inequality: S = {assessment.bell_inequality:.3f}")

        report.append("")

    # レポート文字列作成
    report_text = "\n".join(report)

    # ファイル保存
    with open(output_file, "w") as f:
        f.write(report_text)

    print(f"📄 Report saved to {output_file}")
    return report_text


# ============================================
# Module Testing
# ============================================


def test_quantum_module():
    """モジュールの簡易テスト（v4.0版）"""
    import numpy as np

    print("\n🧪 Testing Quantum Validation Module v4.0...")

    # ダミーデータ作成
    np.random.seed(42)  # 再現性のため
    trajectory = np.random.randn(100, 100, 3) * 10  # 100 frames, 100 atoms

    # ダミーLambda結果
    class DummyLambdaResult:
        def __init__(self):
            self.structures = {
                "lambda_f": np.random.randn(100) * 0.1
                + np.sin(np.linspace(0, 10, 100)),
                "rho_t": np.abs(np.random.randn(100)),
                "sigma_s": np.random.rand(100),
            }
            self.critical_events = [
                (10, 10),  # 瞬間的
                (20, 25),  # 遷移
                (30, 35),  # 遷移
                (50, 50),  # 瞬間的
            ]

    try:
        # 1. インポートテスト
        print("   Testing imports...")
        from .quantum_validation_v4 import (
            QuantumValidatorV4,
            QuantumAssessment,
            StructuralEventPattern,
            QuantumSignature,
        )

        print("   ✅ Imports successful")

        # 2. バリデーター初期化
        print("   Testing validator initialization...")
        validator = QuantumValidatorV4(
            trajectory=trajectory, dt_ps=100.0, temperature_K=300.0
        )
        print("   ✅ Validator initialized")

        # 3. イベント検証
        print("   Testing event validation...")
        test_event = {"frame_start": 10, "frame_end": 10, "type": "test"}

        assessment = validator.validate_event(test_event, DummyLambdaResult())

        print("   ✅ Validation completed")
        print(f"      Pattern: {assessment.pattern.value}")
        print(f"      Quantum: {assessment.is_quantum}")
        print(f"      Confidence: {assessment.confidence:.1%}")

        # 4. バッチ処理
        print("   Testing batch processing...")
        events = [
            {"frame_start": 10, "frame_end": 10, "type": "test1"},
            {"frame_start": 20, "frame_end": 25, "type": "test2"},
            {"frame_start": 50, "frame_end": 50, "type": "test3"},
        ]

        assessments = validator.validate_events(events, DummyLambdaResult())

        print("   ✅ Batch processing completed")
        print(f"      Processed: {len(assessments)} events")
        print(f"      Quantum: {sum(1 for a in assessments if a.is_quantum)}")

        # 5. パターン分類テスト
        print("   Testing pattern classification...")
        patterns_found = set(a.pattern for a in assessments)
        print(f"   ✅ Patterns detected: {[p.value for p in patterns_found]}")

        # 6. サマリー生成
        print("   Testing summary generation...")
        summary = validator.generate_summary(assessments)
        print(f"   ✅ Summary generated with {len(summary)} fields")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ============================================
# Initialization Info
# ============================================


def _print_init_info():
    """初期化情報の表示"""
    print("🌌 Quantum Validation Module v4.0 Loaded")
    print("   Lambda³ Integrated Edition")
    print(f"   Version: {__version__}")
    print("   Key Features:")
    print("   - Lambda structure anomaly evaluation")
    print("   - 3-pattern classification (instant/transition/cascade)")
    print("   - Atomic-level evidence gathering")
    print("   - Adjustable quantum criteria")


# 環境変数でデバッグモード制御
import os

if os.environ.get("QUANTUM_DEBUG", "").lower() == "true":
    _print_init_info()
    print("\n📋 Dependencies:")
    status = check_dependencies()
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")

# ============================================
# Main Execution (for testing)
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Quantum Validation Module for Lambda³ - Version 4.0")
    print("Complete Refactoring with Lambda³ Integration")
    print("=" * 70)

    # 依存関係チェック
    print("\n📋 Checking dependencies...")
    status = check_dependencies()
    all_ok = True
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")
        if "❌" in stat and lib in ["numpy", "scipy"]:
            all_ok = False

    if not all_ok:
        print("\n⚠️ Critical dependencies missing!")
        print("Install with: pip install numpy scipy")
    else:
        # テスト実行
        if test_quantum_module():
            print("\n✨ Module v4.0 is ready for production!")
            print("\nUsage Examples:")
            print("  from bankai.quantum import QuantumValidatorV4")
            print("  validator = QuantumValidatorV4(trajectory=traj)")
            print("  assessment = validator.validate_event(event, lambda_result)")
            print("")
            print("  # Or use convenience functions:")
            print("  from bankai.quantum import quick_validate")
            print("  assessment = quick_validate(event, lambda_result, trajectory)")
            print("")
            print("New in v4.0:")
            print("  ✅ Lambda³ structure anomaly as primary input")
            print("  ✅ Clear 3-pattern classification")
            print("  ✅ Trajectory-based atomic evidence")
            print("  ✅ Realistic and adjustable criteria")
            print("  ✅ No more forced classical for 10+ frames!")
        else:
            print("\n⚠️ Module test failed. Please check installation.")
