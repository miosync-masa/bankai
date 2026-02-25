"""
Quantum Validation Module v5.0 - Simplified Three-Axis Edition
===============================================================

Lambda³構造変化イベントの量子性を3軸で判定：
1. 空間的異常（距離的な大変化）
2. 同期的異常（相関・協調的変化）
3. 時間的異常（速度的な異常）

Version: 5.0.0 - Complete Refactoring
Authors: 環ちゃん & ご主人さま
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("quantum_validation_v5")

# ============================================
# Enums (外部互換性のため維持)
# ============================================


class StructuralEventPattern(Enum):
    """構造変化パターン分類"""

    INSTANTANEOUS = "instantaneous"  # 単一フレーム
    TRANSITION = "transition"  # 連続フレーム
    CASCADE = "cascade"  # ネットワークカスケード


class QuantumSignature(Enum):
    """量子的シグネチャー"""

    ENTANGLEMENT = "quantum_entanglement"
    TUNNELING = "quantum_tunneling"
    COHERENCE = "quantum_coherence"
    PHASE_TRANSITION = "quantum_phase_transition"
    INFORMATION_TRANSFER = "quantum_info_transfer"
    NONE = "classical"


# ============================================
# Data Classes (簡略化版)
# ============================================


@dataclass
class LambdaAnomaly:
    """Lambda構造の異常性評価（簡略化）"""

    lambda_jump: float = 0.0  # Λの変化量
    lambda_zscore: float = 0.0  # 全体分布からのZ-score
    rho_t_spike: float = 0.0  # テンション密度
    sigma_s_value: float = 0.0  # 構造同期率
    # 削除: coordination, statistical_rarity, thermal_comparison


@dataclass
class AtomicEvidence:
    """原子レベルの証拠（簡略化）"""

    max_velocity: float = 0.0  # 最大速度 (Å/ps)
    max_acceleration: float = 0.0  # 最大加速度 (Å/ps²)
    correlation_coefficient: float = 0.0  # 原子運動の相関
    n_bond_anomalies: int = 0  # 結合異常の数
    n_dihedral_flips: int = 0  # 二面角フリップの数


@dataclass
class QuantumAssessment:
    """量子性評価結果"""

    pattern: StructuralEventPattern
    signature: QuantumSignature
    confidence: float = 0.0
    is_quantum: bool = False

    lambda_anomaly: Optional[LambdaAnomaly] = None
    atomic_evidence: Optional[AtomicEvidence] = None
    criteria_met: list[str] = field(default_factory=list)
    explanation: str = ""

    # CASCADE用（互換性維持）
    bell_inequality: Optional[float] = None
    async_bonds_used: list[dict] = field(default_factory=list)


@dataclass
class AnomalyAxes:
    """3軸異常判定結果"""

    spatial: float = 0.0  # 空間的異常度 (0-1)
    sync: float = 0.0  # 同期的異常度 (0-1)
    temporal: float = 0.0  # 時間的異常度 (0-1)


# ============================================
# Main Validator Class
# ============================================


class QuantumValidatorV4:  # クラス名は互換性のため維持
    """
    Lambda³統合型量子判定器（簡略化版）
    3軸判定: 空間・同期・時間
    """

    def __init__(
        self,
        trajectory: Optional[np.ndarray] = None,
        topology: Optional[Any] = None,
        dt_ps: float = 100.0,
        temperature_K: float = 300.0,
        config: Optional[dict] = None,
    ):

        self.trajectory = trajectory
        self.topology = topology
        self.dt_ps = dt_ps
        self.temperature = temperature_K

        # 閾値設定（シンプル化）
        self.thresholds = {
            # 空間的異常
            "lambda_jump_high": 0.1,
            "lambda_zscore_high": 3.0,
            "velocity_factor": 3.0,
            # 同期的異常
            "sigma_s_high": 0.7,
            "correlation_high": 0.6,
            "async_bonds_min": 1,
            # 時間的異常
            "instant_frames": 1,
            "fast_transition_factor": 0.1,
            "coherence_duration": 300.0,  # ps
            # 判定閾値
            "quantum_confidence": 0.3,
        }

        if config:
            self.thresholds.update(config)

        logger.info("Quantum Validator v5.0 initialized")

    # ========================================
    # Main Entry Point (インターフェース維持)
    # ========================================

    def validate_event(
        self, event: dict, lambda_result: Any, network_result: Optional[Any] = None
    ) -> QuantumAssessment:
        """イベントの量子性を判定"""

        # パターン分類
        pattern = self._classify_pattern(event, network_result)

        # Lambda異常評価
        lambda_anomaly = self._evaluate_lambda_anomaly(event, lambda_result)

        # 原子レベル証拠
        atomic_evidence = None
        if self.trajectory is not None:
            atomic_evidence = self._gather_atomic_evidence(event)

        # 3軸異常度計算
        axes = self._calculate_anomaly_axes(
            event, lambda_anomaly, atomic_evidence, network_result
        )

        # パターンと異常軸からシグネチャー判定
        signature = self._determine_signature(pattern, axes)

        # 信頼度計算
        confidence = self._calculate_confidence(axes, pattern)

        # Assessment作成
        assessment = QuantumAssessment(
            pattern=pattern,
            signature=signature,
            confidence=confidence,
            is_quantum=(confidence >= self.thresholds["quantum_confidence"]),
            lambda_anomaly=lambda_anomaly,
            atomic_evidence=atomic_evidence,
        )

        # 判定根拠
        assessment.criteria_met = self._generate_criteria(axes, pattern)
        assessment.explanation = self._generate_explanation(assessment)

        # CASCADE特有（互換性）
        if pattern == StructuralEventPattern.CASCADE and network_result:
            self._add_cascade_info(assessment, network_result)

        return assessment

    # ========================================
    # Pattern Classification
    # ========================================

    def _classify_pattern(
        self, event: dict, network_result: Optional[Any]
    ) -> StructuralEventPattern:
        """イベントパターンの分類"""
        duration = (
            event.get("frame_end", event.get("frame", 0))
            - event.get("frame_start", event.get("frame", 0))
            + 1
        )

        # CASCADE: async_bondsがある
        if network_result and hasattr(network_result, "async_strong_bonds"):
            if len(network_result.async_strong_bonds) > 0:
                return StructuralEventPattern.CASCADE

        # INSTANTANEOUS: 単一フレーム
        if duration <= self.thresholds["instant_frames"]:
            return StructuralEventPattern.INSTANTANEOUS

        # TRANSITION: 複数フレーム
        return StructuralEventPattern.TRANSITION

    # ========================================
    # Lambda Anomaly Evaluation
    # ========================================

    def _evaluate_lambda_anomaly(
        self, event: dict, lambda_result: Any
    ) -> LambdaAnomaly:
        """Lambda構造の異常性を評価"""
        anomaly = LambdaAnomaly()

        if not hasattr(lambda_result, "lambda_structures"):
            return anomaly

        structures = lambda_result.lambda_structures
        frame = event.get("frame_start", event.get("frame", 0))

        # Lambda値の変化
        if "lambda_F_mag" in structures and frame < len(structures["lambda_F_mag"]):
            lambda_vals = np.array(structures["lambda_F_mag"])

            # ジャンプ量
            if frame > 0:
                anomaly.lambda_jump = abs(lambda_vals[frame] - lambda_vals[frame - 1])

            # Z-score
            if len(lambda_vals) > 10:
                mean = np.mean(lambda_vals)
                std = np.std(lambda_vals)
                if std > 1e-10:
                    anomaly.lambda_zscore = abs(lambda_vals[frame] - mean) / std

        # その他の構造パラメータ
        if "rho_T" in structures and frame < len(structures["rho_T"]):
            anomaly.rho_t_spike = structures["rho_T"][frame]

        if "sigma_s" in structures and frame < len(structures["sigma_s"]):
            anomaly.sigma_s_value = structures["sigma_s"][frame]

        return anomaly

    # ========================================
    # Atomic Evidence Gathering
    # ========================================

    def _gather_atomic_evidence(self, event: dict) -> AtomicEvidence:
        """原子レベルの証拠を収集"""
        evidence = AtomicEvidence()

        frame_start = event.get("frame_start", event.get("frame", 0))
        frame_end = event.get("frame_end", frame_start)

        if frame_start >= len(self.trajectory):
            return evidence

        frame_end = min(frame_end, len(self.trajectory) - 1)

        # 速度・加速度
        if frame_start > 0:
            disp = self.trajectory[frame_start] - self.trajectory[frame_start - 1]
            velocities = disp / self.dt_ps
            evidence.max_velocity = np.max(np.linalg.norm(velocities, axis=1))

            if frame_start > 1:
                prev_disp = (
                    self.trajectory[frame_start - 1] - self.trajectory[frame_start - 2]
                )
                prev_vel = prev_disp / self.dt_ps
                acc = (velocities - prev_vel) / self.dt_ps
                evidence.max_acceleration = np.max(np.linalg.norm(acc, axis=1))

        # 相関計算
        if frame_end > frame_start + 1:
            correlations = self._calculate_correlations(frame_start, frame_end)
            if correlations:
                evidence.correlation_coefficient = max(correlations)

        # 結合異常（簡易チェック）
        if self.topology is not None:
            evidence.n_bond_anomalies = self._count_bond_anomalies(
                self.trajectory[frame_start]
            )

        return evidence

    def _calculate_correlations(self, frame_start: int, frame_end: int) -> list[float]:
        """フレーム間の相関を計算"""
        correlations = []
        n_frames = min(frame_end - frame_start + 1, 10)

        displacements = []
        for i in range(n_frames):
            f = frame_start + i
            if f < len(self.trajectory) and f > 0:
                disp = self.trajectory[f] - self.trajectory[f - 1]
                displacements.append(disp.flatten())

        if len(displacements) >= 2:
            try:
                corr_matrix = np.corrcoef(displacements)
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                if not np.isnan(corr_matrix).any():
                    correlations = list(np.abs(corr_matrix[mask]))
            except:
                pass

        return correlations

    def _count_bond_anomalies(self, coords: np.ndarray) -> int:
        """結合異常の数をカウント"""
        count = 0
        if hasattr(self.topology, "bonds"):
            for bond in self.topology.bonds[:100]:
                i, j = bond[0], bond[1]
                if i < len(coords) and j < len(coords):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < 0.8 or dist > 2.0:
                        count += 1
        return count

    # ========================================
    # 3-Axis Anomaly Calculation
    # ========================================

    def _calculate_anomaly_axes(
        self,
        event: dict,
        lambda_anomaly: LambdaAnomaly,
        atomic_evidence: Optional[AtomicEvidence],
        network_result: Optional[Any],
    ) -> AnomalyAxes:
        """3軸の異常度を計算"""
        axes = AnomalyAxes()

        # 1. 空間的異常（距離的な大変化）
        spatial_scores = []

        if lambda_anomaly.lambda_jump > self.thresholds["lambda_jump_high"]:
            spatial_scores.append(min(lambda_anomaly.lambda_jump / 0.5, 1.0))

        if lambda_anomaly.lambda_zscore > self.thresholds["lambda_zscore_high"]:
            spatial_scores.append(min(lambda_anomaly.lambda_zscore / 5.0, 1.0))

        if atomic_evidence:
            typical_vel = 0.02  # Å/ps at 300K
            if (
                atomic_evidence.max_velocity
                > typical_vel * self.thresholds["velocity_factor"]
            ):
                spatial_scores.append(
                    min(atomic_evidence.max_velocity / (typical_vel * 10), 1.0)
                )

            if atomic_evidence.n_bond_anomalies > 0:
                spatial_scores.append(min(atomic_evidence.n_bond_anomalies / 10, 1.0))

        axes.spatial = np.mean(spatial_scores) if spatial_scores else 0.0

        # 2. 同期的異常（相関・協調）
        sync_scores = []

        if lambda_anomaly.sigma_s_value > self.thresholds["sigma_s_high"]:
            sync_scores.append((lambda_anomaly.sigma_s_value - 0.5) / 0.5)

        if (
            atomic_evidence
            and atomic_evidence.correlation_coefficient
            > self.thresholds["correlation_high"]
        ):
            sync_scores.append(atomic_evidence.correlation_coefficient)

        if network_result and hasattr(network_result, "async_strong_bonds"):
            n_bonds = len(network_result.async_strong_bonds)
            if n_bonds >= self.thresholds["async_bonds_min"]:
                sync_scores.append(min(n_bonds / 10, 1.0))

        axes.sync = np.mean(sync_scores) if sync_scores else 0.0

        # 3. 時間的異常（速度的な異常）
        temporal_scores = []

        duration = (
            event.get("frame_end", event.get("frame", 0))
            - event.get("frame_start", event.get("frame", 0))
            + 1
        )

        # 瞬間的変化
        if duration <= self.thresholds["instant_frames"]:
            temporal_scores.append(1.0)

        # 速い遷移
        elif duration > 1:
            transition_time = duration * self.dt_ps

            # Lambda jumpから期待される時間と比較
            if lambda_anomaly.lambda_jump > 0.01:
                expected_time = 1000.0 * lambda_anomaly.lambda_jump  # 経験的
                if (
                    transition_time
                    < expected_time * self.thresholds["fast_transition_factor"]
                ):
                    temporal_scores.append(1.0 - transition_time / expected_time)

            # 長時間コヒーレンス
            if (
                lambda_anomaly.sigma_s_value > 0.7
                and transition_time > self.thresholds["coherence_duration"]
            ):
                temporal_scores.append(min(transition_time / 1000.0, 1.0))

        axes.temporal = np.mean(temporal_scores) if temporal_scores else 0.0

        return axes

    # ========================================
    # Signature Determination
    # ========================================

    def _determine_signature(
        self, pattern: StructuralEventPattern, axes: AnomalyAxes
    ) -> QuantumSignature:
        """パターンと3軸異常からシグネチャーを判定"""

        # 閾値
        threshold = 0.3

        spatial_high = axes.spatial > threshold
        sync_high = axes.sync > threshold
        temporal_high = axes.temporal > threshold

        # INSTANTANEOUS（瞬間的）
        if pattern == StructuralEventPattern.INSTANTANEOUS:
            if spatial_high and sync_high:
                return QuantumSignature.PHASE_TRANSITION
            elif spatial_high:
                return QuantumSignature.TUNNELING
            elif sync_high:
                return QuantumSignature.ENTANGLEMENT

        # TRANSITION（遷移）
        elif pattern == StructuralEventPattern.TRANSITION:
            if temporal_high and spatial_high:
                return QuantumSignature.TUNNELING
            elif sync_high and temporal_high:
                return QuantumSignature.COHERENCE
            elif spatial_high and sync_high:
                return QuantumSignature.PHASE_TRANSITION

        # CASCADE（カスケード）
        elif pattern == StructuralEventPattern.CASCADE:
            if sync_high:
                return QuantumSignature.INFORMATION_TRANSFER
            elif spatial_high and temporal_high:
                return QuantumSignature.PHASE_TRANSITION

        return QuantumSignature.NONE

    # ========================================
    # Confidence Calculation
    # ========================================

    def _calculate_confidence(
        self, axes: AnomalyAxes, pattern: StructuralEventPattern
    ) -> float:
        """信頼度を計算"""

        # 基本信頼度（3軸の平均）
        base_confidence = (axes.spatial + axes.sync + axes.temporal) / 3.0

        # パターンによる補正
        if pattern == StructuralEventPattern.INSTANTANEOUS:
            # 瞬間的は時間異常がデフォルトなので少し割引
            confidence = base_confidence * 0.9
        elif pattern == StructuralEventPattern.CASCADE:
            # カスケードは同期が重要
            confidence = base_confidence * 0.7 + axes.sync * 0.3
        else:
            confidence = base_confidence

        return min(confidence, 1.0)

    # ========================================
    # Criteria and Explanation Generation
    # ========================================

    def _generate_criteria(
        self, axes: AnomalyAxes, pattern: StructuralEventPattern
    ) -> list[str]:
        """判定根拠を生成"""
        criteria = []

        if axes.spatial > 0.3:
            criteria.append(f"Spatial anomaly: {axes.spatial:.2f}")
        if axes.sync > 0.3:
            criteria.append(f"Synchronization anomaly: {axes.sync:.2f}")
        if axes.temporal > 0.3:
            criteria.append(f"Temporal anomaly: {axes.temporal:.2f}")

        if pattern == StructuralEventPattern.INSTANTANEOUS:
            criteria.append("Instantaneous change")
        elif pattern == StructuralEventPattern.CASCADE:
            criteria.append("Network cascade detected")

        return criteria

    def _generate_explanation(self, assessment: QuantumAssessment) -> str:
        """説明文を生成"""
        if not assessment.is_quantum:
            return f"Classical {assessment.pattern.value} process (confidence: {assessment.confidence:.1%})"

        explanations = {
            QuantumSignature.ENTANGLEMENT: "Quantum entanglement via instantaneous correlation",
            QuantumSignature.TUNNELING: "Quantum tunneling through energy barrier",
            QuantumSignature.COHERENCE: "Sustained quantum coherence",
            QuantumSignature.PHASE_TRANSITION: "Quantum phase transition",
            QuantumSignature.INFORMATION_TRANSFER: "Non-local quantum information transfer",
        }

        base = explanations.get(assessment.signature, "Quantum behavior detected")
        return f"{base} (confidence: {assessment.confidence:.1%})"

    def _add_cascade_info(self, assessment: QuantumAssessment, network_result: Any):
        """CASCADE固有情報を追加（互換性）"""
        if hasattr(network_result, "async_strong_bonds"):
            bonds = network_result.async_strong_bonds[:5]
            assessment.async_bonds_used = [
                {
                    "residues": (b.from_res, b.to_res),
                    "strength": b.strength,
                    "lag": getattr(b, "lag", 0),
                }
                for b in bonds
            ]

            # 簡易Bell不等式
            if bonds:
                max_strength = max(b.strength for b in bonds)
                assessment.bell_inequality = 2.0 + max_strength * 0.8

    # ========================================
    # Batch Processing (インターフェース維持)
    # ========================================

    def validate_events(
        self,
        events: list[dict],
        lambda_result: Any,
        network_results: Optional[list[Any]] = None,
    ) -> list[QuantumAssessment]:
        """複数イベントの一括処理"""
        assessments = []

        for i, event in enumerate(events):
            network = (
                network_results[i]
                if network_results and i < len(network_results)
                else None
            )

            try:
                assessment = self.validate_event(event, lambda_result, network)
                assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to process event {i}: {e}")
                assessments.append(
                    QuantumAssessment(
                        pattern=StructuralEventPattern.INSTANTANEOUS,
                        signature=QuantumSignature.NONE,
                        is_quantum=False,
                        explanation=f"Processing failed: {e}",
                    )
                )

        return assessments

    # ========================================
    # Summary and Reporting (インターフェース維持)
    # ========================================

    def generate_summary(self, assessments: list[QuantumAssessment]) -> dict:
        """サマリー生成"""
        total = len(assessments)
        quantum_count = sum(1 for a in assessments if a.is_quantum)

        pattern_stats = {}
        for pattern in StructuralEventPattern:
            count = sum(1 for a in assessments if a.pattern == pattern)
            quantum = sum(
                1 for a in assessments if a.pattern == pattern and a.is_quantum
            )
            pattern_stats[pattern.value] = {
                "total": count,
                "quantum": quantum,
                "ratio": quantum / count if count > 0 else 0,
            }

        signature_stats = {}
        for sig in QuantumSignature:
            count = sum(1 for a in assessments if a.signature == sig)
            if count > 0:
                signature_stats[sig.value] = count

        confidences = [a.confidence for a in assessments if a.is_quantum]

        return {
            "total_events": total,
            "quantum_events": quantum_count,
            "quantum_ratio": quantum_count / total if total > 0 else 0,
            "pattern_statistics": pattern_stats,
            "signature_distribution": signature_stats,
            "confidence_stats": {
                "mean": np.mean(confidences) if confidences else 0,
                "std": np.std(confidences) if confidences else 0,
                "min": np.min(confidences) if confidences else 0,
                "max": np.max(confidences) if confidences else 0,
            },
        }

    def print_summary(self, assessments: list[QuantumAssessment]):
        """サマリー表示"""
        summary = self.generate_summary(assessments)

        print("\n" + "=" * 70)
        print("🌌 QUANTUM VALIDATION SUMMARY v5.0")
        print("=" * 70)
        print("\n📊 Overall Statistics:")
        print(f"   Total events: {summary['total_events']}")
        print(
            f"   Quantum events: {summary['quantum_events']} ({summary['quantum_ratio']:.1%})"
        )

        print("\n🎯 Pattern Analysis:")
        for pattern, stats in summary["pattern_statistics"].items():
            if stats["total"] > 0:
                print(
                    f"   {pattern}: {stats['quantum']}/{stats['total']} quantum ({stats['ratio']:.1%})"
                )

        print("\n⚛️ Quantum Signatures:")
        for sig, count in summary["signature_distribution"].items():
            if sig != "classical":
                print(f"   {sig}: {count}")


# ============================================
# Convenience Functions (互換性維持)
# ============================================


def validate_lambda_events(
    lambda_result: Any,
    trajectory: Optional[np.ndarray] = None,
    network_results: Optional[list[Any]] = None,
    **kwargs,
) -> list[QuantumAssessment]:
    """Lambda³イベントの量子性を検証"""

    events = []
    if hasattr(lambda_result, "critical_events"):
        for e in lambda_result.critical_events:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                events.append(
                    {
                        "frame_start": int(e[0]),
                        "frame_end": int(e[1]),
                        "type": "critical",
                    }
                )

    validator = QuantumValidatorV4(trajectory=trajectory, **kwargs)
    assessments = validator.validate_events(events, lambda_result, network_results)
    validator.print_summary(assessments)

    return assessments


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example: テスト用のダミーデータ
    print("Quantum Validator v4.0.1 (FIXED) - Test Run")

    # ダミーイベント
    test_event = {"frame_start": 100, "frame_end": 100, "type": "critical"}

    # ダミーLambda結果（修正版のキー名）
    class DummyLambdaResult:
        def __init__(self):
            self.lambda_structures = {
                "lambda_F_mag": np.random.randn(1000) * 0.1 + 1.0,
                "rho_T": np.random.rand(1000) * 2.0,
                "sigma_s": np.random.rand(1000),
            }

    # バリデーター作成
    validator = QuantumValidatorV4(dt_ps=100.0, temperature_K=300.0)

    # 検証実行
    assessment = validator.validate_event(test_event, DummyLambdaResult())

    # 結果表示
    print(f"\nPattern: {assessment.pattern.value}")
    print(f"Quantum: {assessment.is_quantum}")
    print(f"Signature: {assessment.signature.value}")
    print(f"Confidence: {assessment.confidence:.1%}")
    print(f"Explanation: {assessment.explanation}")
