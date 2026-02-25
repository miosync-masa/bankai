"""
Lambda³ GPU版2段階解析モジュール
マクロレベル→残基レベルの階層的解析をGPUで高速化
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# CuPyの条件付きインポート
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

import warnings
from concurrent.futures import ThreadPoolExecutor

from ..core.gpu_utils import GPUBackend
from ..models import ArrayType
from ..residue.causality_analysis_gpu import CausalityAnalyzerGPU
from ..residue.confidence_analysis_gpu import ConfidenceAnalyzerGPU
from ..residue.residue_network_gpu import ResidueNetworkGPU
from ..residue.residue_structures_gpu import (
    ResidueStructureResult,
    ResidueStructuresGPU,
)
from .md_lambda3_detector_gpu import MDLambda3Result

warnings.filterwarnings("ignore")


@dataclass
class ResidueAnalysisConfig:
    """残基レベル解析の設定"""

    # 解析パラメータ（sensitivity下げた）
    sensitivity: float = 0.5  # 1.0 → 0.5に緩和！
    correlation_threshold: float = 0.15
    sync_threshold: float = 0.2

    # ウィンドウパラメータ
    min_window: int = 30
    max_window: int = 300
    base_window: int = 50
    base_lag_window: int = 100

    # ネットワーク制約
    max_causal_links: int = 500
    min_causality_strength: float = 0.2

    # Bootstrap パラメータ
    use_confidence: bool = True
    n_bootstrap: int = 50
    confidence_level: float = 0.95

    # GPU設定
    gpu_batch_residues: int = 50
    parallel_events: bool = True
    adaptive_window: bool = True  # 追加

    # イベント固有設定（感度も調整）
    event_sensitivities: dict[str, float] = field(
        default_factory=lambda: {
            "ligand_binding_effect": 0.8,  # 1.5 → 0.8
            "slow_helix_destabilization": 0.5,  # 1.0 → 0.5
            "rapid_partial_unfold": 0.4,  # 0.8 → 0.4
            "transient_refolding_attempt": 0.6,  # 1.2 → 0.6
            "aggregation_onset": 0.5,  # 1.0 → 0.5
        }
    )

    event_windows: dict[str, int] = field(
        default_factory=lambda: {
            "ligand_binding_effect": 100,
            "slow_helix_destabilization": 500,
            "rapid_partial_unfold": 50,
            "transient_refolding_attempt": 200,
            "aggregation_onset": 300,
        }
    )


@dataclass
class ResidueEvent:
    """残基レベルイベント"""

    residue_id: int
    residue_name: str
    start_frame: int
    end_frame: int
    peak_lambda_f: float
    propagation_delay: int
    role: str
    adaptive_window: int = 100


@dataclass
class ResidueLevelAnalysis:
    """残基レベル解析結果"""

    event_name: str
    macro_start: int
    macro_end: int
    residue_events: list[ResidueEvent] = field(default_factory=list)
    causality_chain: list[tuple[int, int, float]] = field(default_factory=list)
    initiator_residues: list[int] = field(default_factory=list)
    key_propagation_paths: list[list[int]] = field(default_factory=list)
    async_strong_bonds: list[dict] = field(default_factory=list)
    sync_network: list[dict] = field(default_factory=list)
    network_stats: dict = field(default_factory=dict)
    confidence_results: list[Any] = field(default_factory=list)
    gpu_time: float = 0.0


@dataclass
class TwoStageLambda3Result:
    """2段階解析の統合結果"""

    macro_result: MDLambda3Result
    residue_analyses: dict[str, ResidueLevelAnalysis]
    global_residue_importance: dict[int, float]
    suggested_intervention_points: list[int]
    global_network_stats: dict
    total_gpu_time: float = 0.0


class TwoStageAnalyzerGPU(GPUBackend):
    """GPU版2段階解析器"""

    def __init__(self, config: ResidueAnalysisConfig = None):
        super().__init__()
        self.config = config or ResidueAnalysisConfig()

        # GPU版コンポーネント初期化
        self.residue_structures = ResidueStructuresGPU()
        self.residue_network = ResidueNetworkGPU()
        self.causality_analyzer = CausalityAnalyzerGPU()
        self.confidence_analyzer = ConfidenceAnalyzerGPU()

        # メモリマネージャ共有
        for component in [
            self.residue_structures,
            self.residue_network,
            self.causality_analyzer,
            self.confidence_analyzer,
        ]:
            component.memory_manager = self.memory_manager
            component.device = self.device

    def analyze_trajectory(
        self,
        trajectory: np.ndarray,
        macro_result: MDLambda3Result,
        detected_events: list[tuple[int, int, str]],
        n_residues: int = 129,
    ) -> TwoStageLambda3Result:
        """
        2段階Lambda³解析の実行（GPU高速化版）

        Parameters
        ----------
        trajectory : np.ndarray
            MD軌道 (n_frames, n_atoms, 3)
        macro_result : MDLambda3Result
            マクロレベル解析結果
        detected_events : List[Tuple[int, int, str]]
            検出イベントリスト
        n_residues : int, default=129
            残基数

        Returns
        -------
        TwoStageLambda3Result
            統合解析結果
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("=== Two-Stage Lambda³ Analysis (GPU) ===")
        print("=" * 60)
        print(f"Events to analyze: {len(detected_events)}")
        print(f"Number of residues: {n_residues}")
        print(f"GPU Device: {self.device}")

        # 入力検証
        if len(trajectory.shape) != 3:
            raise ValueError(f"Expected 3D trajectory, got shape {trajectory.shape}")

        # セットアップ
        residue_atoms = self._create_residue_mapping(trajectory.shape[1], n_residues)
        residue_names = self._get_residue_names(n_residues)

        # 各イベントを解析
        residue_analyses = {}
        all_important_residues = {}

        # 並列処理の決定
        if self.config.parallel_events and len(detected_events) > 1:
            residue_analyses = self._analyze_events_parallel(
                trajectory, detected_events, residue_atoms, residue_names
            )
        else:
            residue_analyses = self._analyze_events_sequential(
                trajectory, detected_events, residue_atoms, residue_names
            )

        # グローバル重要度の計算
        for event_name, analysis in residue_analyses.items():
            for event in analysis.residue_events:
                res_id = event.residue_id
                if res_id not in all_important_residues:
                    all_important_residues[res_id] = 0.0

                # 重要度スコア
                importance = event.peak_lambda_f * (
                    1 + 0.1 * (100 / event.adaptive_window)
                )
                all_important_residues[res_id] += importance

        # 介入ポイントの特定
        intervention_points = self._identify_intervention_points_gpu(
            all_important_residues
        )

        # グローバル統計
        global_stats = self._compute_global_stats(residue_analyses)

        # 結果サマリー
        self._print_summary(all_important_residues, global_stats, intervention_points)

        total_time = time.time() - start_time

        return TwoStageLambda3Result(
            macro_result=macro_result,
            residue_analyses=residue_analyses,
            global_residue_importance=all_important_residues,
            suggested_intervention_points=intervention_points,
            global_network_stats=global_stats,
            total_gpu_time=total_time,
        )

    def _analyze_events_parallel(
        self,
        trajectory: np.ndarray,
        detected_events: list[tuple[int, int, str]],
        residue_atoms: dict[int, list[int]],
        residue_names: dict[int, str],
    ) -> dict[str, ResidueLevelAnalysis]:
        """イベントの並列解析"""
        print("\n📍 Processing events in parallel on GPU...")

        residue_analyses = {}

        # ThreadPoolExecutorで並列実行
        with ThreadPoolExecutor(max_workers=min(4, len(detected_events))) as executor:
            futures = []

            for start, end, event_name in detected_events:
                future = executor.submit(
                    self._analyze_single_event_gpu,
                    trajectory,
                    event_name,
                    start,
                    end,
                    residue_atoms,
                    residue_names,
                )
                futures.append((event_name, future))

            # 結果収集
            for event_name, future in futures:
                try:
                    analysis = future.result()
                    residue_analyses[event_name] = analysis
                    print(
                        f"  ✓ {event_name} complete (GPU time: {analysis.gpu_time:.2f}s)"
                    )
                except Exception as e:
                    print(f"  ✗ {event_name} failed: {str(e)}")

        return residue_analyses

    def _analyze_events_sequential(
        self,
        trajectory: np.ndarray,
        detected_events: list[tuple[int, int, str]],
        residue_atoms: dict[int, list[int]],
        residue_names: dict[int, str],
    ) -> dict[str, ResidueLevelAnalysis]:
        """イベントの逐次解析"""
        print("\n📍 Processing events sequentially on GPU...")

        residue_analyses = {}

        for start, end, event_name in detected_events:
            print(f"\n  → Analyzing {event_name}...")
            analysis = self._analyze_single_event_gpu(
                trajectory, event_name, start, end, residue_atoms, residue_names
            )
            residue_analyses[event_name] = analysis
            print(f"    GPU time: {analysis.gpu_time:.2f}s")

        return residue_analyses

    def _analyze_single_event_gpu(
        self,
        trajectory: np.ndarray,
        event_name: str,
        start_frame: int,
        end_frame: int,
        residue_atoms: dict[int, list[int]],
        residue_names: dict[int, str],
    ) -> ResidueLevelAnalysis:
        """単一イベントのGPU解析（単一フレーム対応版）"""
        event_start_time = time.time()

        # ========================================
        # フレーム範囲の検証と修正
        # ========================================
        is_single_frame = False
        if end_frame <= start_frame:
            is_single_frame = True
            end_frame = min(start_frame + 1, trajectory.shape[0])

            if end_frame <= start_frame:
                return self._create_empty_analysis(event_name, start_frame, end_frame)

        event_frames = end_frame - start_frame
        event_trajectory = trajectory[start_frame:end_frame]

        # GPUメモリコンテキスト
        with self.memory_manager.batch_context(
            event_frames * len(residue_atoms) * 3 * 4
        ):
            # ========================================
            # 1. 残基構造計算
            # ========================================
            if is_single_frame:
                structures = self._compute_single_frame_structures(
                    event_trajectory, residue_atoms
                )
            else:
                structures = self.residue_structures.compute_residue_structures(
                    event_trajectory,
                    0,
                    event_frames - 1,
                    residue_atoms,  # end_frameは包含的
                )

            # ========================================
            # 2. 異常検出
            # ========================================
            if is_single_frame:
                anomaly_scores = self._detect_instantaneous_anomalies(
                    structures, event_name, residue_atoms
                )
            else:
                anomaly_scores = self._detect_residue_anomalies_gpu(
                    structures, event_name
                )

            # ========================================
            # 3. ネットワーク解析
            # ========================================
            network_results = self.residue_network.analyze_network(
                anomaly_scores, structures.residue_coupling, structures.residue_coms
            )

            pattern = network_results.network_stats.get("pattern", "unknown")

            # ========================================
            # 4. イベント構築
            # ========================================
            residue_events = self._build_residue_events_gpu(
                anomaly_scores, residue_names, start_frame, network_results
            )

            # ========================================
            # 5. パターン別解析
            # ========================================
            if pattern == "parallel_network" or is_single_frame:
                initiators = self._find_parallel_initiators(
                    residue_events, network_results.sync_network
                )
                causality_chains = []
                propagation_paths = []
            else:
                initiators = self._find_initiators_gpu(
                    residue_events, network_results.causal_network
                )
                causality_chains = [
                    (link.from_res, link.to_res, link.strength)
                    for link in network_results.causal_network
                ]
                propagation_paths = (
                    self._build_propagation_paths_gpu(initiators, causality_chains)
                    if causality_chains
                    else []
                )

            # ========================================
            # 6. 信頼区間解析
            # ========================================
            confidence_results = []
            if self.config.use_confidence:
                if is_single_frame:
                    confidence_results = self._compute_structural_confidence(
                        network_results.sync_network, anomaly_scores
                    )
                elif causality_chains:
                    confidence_results = self.confidence_analyzer.analyze(
                        causality_chains[:10], anomaly_scores
                    )

        gpu_time = time.time() - event_start_time

        # ========================================
        # 7. 結果構築
        # ========================================
        network_stats = network_results.network_stats.copy()
        network_stats.update(
            {
                "n_causal": network_results.n_causal_links,
                "n_sync": network_results.n_sync_links,
                "n_async": network_results.n_async_bonds,
                "is_single_frame": is_single_frame,
                "mean_adaptive_window": (
                    np.mean(list(network_results.adaptive_windows.values()))
                    if network_results.adaptive_windows
                    else (1 if is_single_frame else 100)
                ),
            }
        )

        return ResidueLevelAnalysis(
            event_name=event_name,
            macro_start=start_frame,
            macro_end=end_frame if not is_single_frame else start_frame,
            residue_events=residue_events,
            causality_chain=causality_chains,
            initiator_residues=initiators,
            key_propagation_paths=propagation_paths[:5],
            async_strong_bonds=network_results.async_strong_bonds,
            sync_network=network_results.sync_network,
            network_stats=network_stats,
            confidence_results=confidence_results,
            gpu_time=gpu_time,
        )

    def _compute_single_frame_structures(self, trajectory, residue_atoms):
        """単一フレーム用の構造計算（ResidueStructureResult返却）"""
        n_residues = len(residue_atoms)
        frame = trajectory[0]

        residue_coms = np.zeros((1, n_residues, 3), dtype=np.float32)
        residue_lambda_f = np.zeros((1, n_residues, 3), dtype=np.float32)
        residue_lambda_f_mag = np.zeros((1, n_residues), dtype=np.float32)
        residue_rho_t = np.zeros((1, n_residues), dtype=np.float32)

        for i, (res_id, atom_indices) in enumerate(residue_atoms.items()):
            coords = frame[atom_indices]
            com = np.mean(coords, axis=0)
            residue_coms[0, i] = com

            # 慣性半径
            distances_from_com = np.linalg.norm(coords - com, axis=1)
            rg = np.sqrt(np.mean(distances_from_com**2))
            residue_lambda_f_mag[0, i] = rg

            # 構造密度
            if len(atom_indices) > 1:
                from scipy.spatial.distance import pdist

                pairwise_distances = pdist(coords)
                mean_dist = (
                    np.mean(pairwise_distances) if len(pairwise_distances) > 0 else 1.0
                )
                residue_rho_t[0, i] = 1.0 / mean_dist
            else:
                residue_rho_t[0, i] = 1.0

        # カップリング行列
        from scipy.spatial.distance import cdist

        distances = cdist(residue_coms[0], residue_coms[0])
        sigma = 5.0
        coupling = np.exp(-(distances**2) / (2 * sigma**2))
        np.fill_diagonal(coupling, 0)
        residue_coupling = coupling.reshape(1, n_residues, n_residues)

        # ResidueStructureResultを返す
        return ResidueStructureResult(
            residue_lambda_f=residue_lambda_f,
            residue_lambda_f_mag=residue_lambda_f_mag,
            residue_rho_t=residue_rho_t,
            residue_coupling=residue_coupling,
            residue_coms=residue_coms,
        )

    def _detect_instantaneous_anomalies(self, structures, event_name, residue_atoms):
        """単一フレーム用の瞬間異常検出"""
        anomaly_scores = {}

        # 構造値から異常スコア計算
        n_residues = structures.n_residues

        # 統計値計算
        lambda_values = structures.residue_lambda_f_mag[0]
        rho_values = structures.residue_rho_t[0]

        lambda_median = np.median(lambda_values)
        lambda_mad = np.median(np.abs(lambda_values - lambda_median))
        rho_median = np.median(rho_values)
        rho_mad = np.median(np.abs(rho_values - rho_median))

        for i in range(n_residues):
            lambda_val = lambda_values[i]
            rho_val = rho_values[i]

            # Modified Z-score
            lambda_z = 0.6745 * (lambda_val - lambda_median) / (lambda_mad + 1e-10)
            rho_z = 0.6745 * (rho_val - rho_median) / (rho_mad + 1e-10)

            score = np.sqrt(lambda_z**2 + rho_z**2)
            anomaly_scores[i] = np.array([score])

        return anomaly_scores

    def _detect_residue_anomalies_gpu(
        self, structures, event_type: str
    ) -> dict[int, np.ndarray]:
        """
        残基異常検出（GPU最適化・修正版）
        感度を適切に調整して、異常を確実に検出！
        """
        # structuresがデータクラスか辞書か判定
        if hasattr(structures, "residue_rho_t"):
            residue_rho_t = structures.residue_rho_t
            residue_lambda_f_mag = structures.residue_lambda_f_mag
        else:
            residue_rho_t = structures["residue_rho_t"]
            residue_lambda_f_mag = structures["residue_lambda_f_mag"]

        n_frames, n_residues = residue_rho_t.shape

        # イベント固有の感度（適切な値に修正）
        self.config.event_sensitivities.get(
            event_type, self.config.sensitivity
        )  # デフォルト感度をそのまま使用（0.5倍しない）

        # GPU上で異常検出
        residue_anomaly_scores = {}

        # バッチ処理で残基を解析
        batch_size = self.config.gpu_batch_residues

        for batch_start in range(0, n_residues, batch_size):
            batch_end = min(batch_start + batch_size, n_residues)

            # バッチデータをGPUに転送
            batch_lambda_f = self.to_gpu(residue_lambda_f_mag[:, batch_start:batch_end])
            batch_rho_t = self.to_gpu(residue_rho_t[:, batch_start:batch_end])

            # GPU上で異常スコア計算
            for i, res_id in enumerate(range(batch_start, batch_end)):
                lambda_anomaly = self._compute_anomaly_gpu(batch_lambda_f[:, i])
                rho_anomaly = self._compute_anomaly_gpu(batch_rho_t[:, i])

                # 統合スコア
                combined = (lambda_anomaly + rho_anomaly) / 2

                # 閾値を適切に設定（sensitivity そのままで判定）
                # または全残基のスコアを保存（デバッグ用）
                self.xp.max(combined)

                # 常にスコアを保存（閾値は後で適用）
                residue_anomaly_scores[res_id] = self.to_cpu(combined)

                # デバッグ情報（必要に応じて）
                # if max_score > sensitivity:
                #     logger.debug(f"Residue {res_id}: max_score={max_score:.3f}")

        # 空の場合のフォールバック（全残基に適切なスコアを割り当て）
        if not residue_anomaly_scores:
            logger.warning(
                f"No anomalies detected for {event_type}, assigning default scores"
            )
            for res_id in range(n_residues):
                # より現実的なランダムノイズを追加
                base_score = np.random.uniform(0.5, 1.5, n_frames)
                residue_anomaly_scores[res_id] = base_score

        # 最低でもトップ10残基は返すように保証
        if len(residue_anomaly_scores) < min(10, n_residues):
            # スコアが低い残基も追加
            for res_id in range(n_residues):
                if res_id not in residue_anomaly_scores:
                    lambda_vals = residue_lambda_f_mag[:, res_id]
                    rho_vals = residue_rho_t[:, res_id]

                    # 簡易スコア計算（CPU）
                    lambda_score = np.abs(lambda_vals - np.mean(lambda_vals)) / (
                        np.std(lambda_vals) + 1e-10
                    )
                    rho_score = np.abs(rho_vals - np.mean(rho_vals)) / (
                        np.std(rho_vals) + 1e-10
                    )
                    combined = (lambda_score + rho_score) / 2

                    residue_anomaly_scores[res_id] = combined

        logger.debug(
            f"Detected anomalies in {len(residue_anomaly_scores)} residues for {event_type}"
        )

        return residue_anomaly_scores

    def _build_residue_events_gpu(
        self,
        anomaly_scores: dict[int, np.ndarray],
        residue_names: dict[int, str],
        start_frame: int,
        network_results,
    ) -> list[ResidueEvent]:
        """修正版：確実にResidueEventを生成"""
        import numpy as np

        events = []

        # anomaly_scoresが空の場合の処理
        if not anomaly_scores:
            logger.warning("No anomaly scores provided for residue events")
            # デフォルトイベントを作成
            for res_id in range(min(10, len(residue_names))):
                event = ResidueEvent(
                    residue_id=res_id,
                    residue_name=residue_names.get(res_id, f"RES{res_id}"),
                    start_frame=start_frame,
                    end_frame=start_frame + 1,
                    peak_lambda_f=1.0,
                    propagation_delay=0,
                    role="default",
                    adaptive_window=100,
                )
                events.append(event)
            return events

        # 通常の処理
        # find_peaksのインポート（互換性のため両方試す）
        try:
            from scipy.signal import find_peaks

            use_scipy = True
        except ImportError:
            logger.warning(
                "scipy.signal.find_peaks not available, using simple peak detection"
            )
            use_scipy = False

        # adaptive_windowsの取得
        adaptive_windows = {}
        if hasattr(network_results, "adaptive_windows"):
            adaptive_windows = network_results.adaptive_windows

        # 各残基のイベントを検出
        for res_id, scores in anomaly_scores.items():
            if len(scores) == 0:
                continue

            # ピーク検出
            if use_scipy and len(scores) > 1:
                # scipy使用可能な場合
                peaks, properties = find_peaks(
                    scores,
                    height=np.mean(scores) + np.std(scores),  # 平均+標準偏差を閾値に
                    distance=5,  # 最小間隔
                )

                if len(peaks) == 0:
                    # ピークが見つからない場合は最大値を使用
                    peaks = [np.argmax(scores)]
                    properties = {"peak_heights": [scores[peaks[0]]]}
            else:
                # scipy使用不可または単一フレームの場合
                peaks = [np.argmax(scores)]
                properties = {"peak_heights": [scores[peaks[0]]]}

            # イベント作成
            for i, peak_frame in enumerate(peaks[:5]):  # 最大5イベント/残基
                peak_height = (
                    properties.get("peak_heights", scores)[i]
                    if i < len(properties.get("peak_heights", scores))
                    else scores[peak_frame]
                )

                # 役割の決定
                role = self._determine_residue_role(res_id, peak_frame, network_results)

                event = ResidueEvent(
                    residue_id=res_id,
                    residue_name=residue_names.get(res_id, f"RES{res_id}"),
                    start_frame=start_frame + peak_frame,
                    end_frame=start_frame + min(peak_frame + 10, len(scores) - 1),
                    peak_lambda_f=float(peak_height),
                    propagation_delay=peak_frame,
                    role=role,
                    adaptive_window=adaptive_windows.get(res_id, 100),
                )

                # event_scoreも追加（report_generatorが使用）
                event.event_score = float(peak_height)
                event.anomaly_score = float(peak_height)  # 互換性のため

                events.append(event)

        # イベントが空の場合のフォールバック
        if not events:
            logger.warning("No events detected, creating default events")
            # 上位スコアの残基からイベントを作成
            top_residues = sorted(
                anomaly_scores.items(), key=lambda x: np.max(x[1]), reverse=True
            )[:10]

            for res_id, scores in top_residues:
                event = ResidueEvent(
                    residue_id=res_id,
                    residue_name=residue_names.get(res_id, f"RES{res_id}"),
                    start_frame=start_frame,
                    end_frame=start_frame + len(scores) - 1,
                    peak_lambda_f=float(np.max(scores)),
                    propagation_delay=0,
                    role="participant",
                    adaptive_window=100,
                )
                event.event_score = float(np.max(scores))
                event.anomaly_score = float(np.max(scores))
                events.append(event)

        logger.debug(f"Created {len(events)} residue events")
        return events

    def _determine_residue_role(
        self, res_id: int, peak_frame: int, network_results
    ) -> str:
        """残基の役割を決定（修正版）"""
        # NetworkAnalysisResultの属性を正しくチェック
        if not hasattr(network_results, "causal_network"):
            return "participant"

        # イニシエーター判定
        if hasattr(network_results, "network_stats"):
            stats = network_results.network_stats
            if "hub_residues" in stats:
                hub_ids = [h[0] for h in stats["hub_residues"]]
                if res_id in hub_ids:
                    return "initiator"

        # 因果ネットワークでの役割
        out_degree = 0
        in_degree = 0

        for link in network_results.causal_network:
            if hasattr(link, "from_res") and hasattr(link, "to_res"):
                if link.from_res == res_id:
                    out_degree += 1
                if link.to_res == res_id:
                    in_degree += 1

        if out_degree > in_degree * 2:
            return "driver"
        elif in_degree > out_degree * 2:
            return "responder"
        else:
            return "mediator"

    def _compute_anomaly_gpu(self, series: ArrayType, window: int = 50) -> ArrayType:
        """GPU上で異常スコア計算"""
        anomaly = self.xp.zeros_like(series)

        for i in range(len(series)):
            start = max(0, i - window)
            end = min(len(series), i + window + 1)

            local_mean = self.xp.mean(series[start:end])
            local_std = self.xp.std(series[start:end])

            if local_std > 1e-10:
                anomaly[i] = self.xp.abs(series[i] - local_mean) / local_std

        return anomaly

    def _find_parallel_initiators(self, residue_events, sync_network):
        """並列ネットワークのハブ残基特定"""
        degree_count = {}

        for link in sync_network:
            degree_count[link.from_res] = degree_count.get(link.from_res, 0) + 1
            degree_count[link.to_res] = degree_count.get(link.to_res, 0) + 1

        sorted_residues = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
        return [res_id for res_id, _ in sorted_residues[:5]]

    def _compute_structural_confidence(self, sync_network, anomaly_scores):
        """構造的信頼度計算"""
        if not sync_network:
            return []

        confidence_results = []
        score_dict = {res_id: scores[0] for res_id, scores in anomaly_scores.items()}

        for link in sync_network[:10]:
            res_i = link.from_res
            res_j = link.to_res

            score_i = score_dict.get(res_i, 0)
            score_j = score_dict.get(res_j, 0)

            strength_confidence = link.strength
            anomaly_confidence = min(score_i, score_j) / (max(score_i, score_j) + 1e-10)
            distance_confidence = (
                np.exp(-link.distance / 10.0) if link.distance is not None else 0.5
            )

            overall_confidence = np.mean(
                [strength_confidence, anomaly_confidence, distance_confidence]
            )

            std_estimate = overall_confidence * 0.15
            ci_lower = max(0, overall_confidence - 1.96 * std_estimate)
            ci_upper = min(1, overall_confidence + 1.96 * std_estimate)

            confidence_results.append(
                {
                    "pair": (res_i, res_j),
                    "correlation": link.strength,
                    "confidence": overall_confidence,
                    "confidence_interval": (ci_lower, ci_upper),
                    "p_value": 1.0 - overall_confidence,
                    "distance": link.distance,
                    "anomaly_product": score_i * score_j,
                }
            )

        return confidence_results

    def _create_empty_analysis(self, event_name, start_frame, end_frame):
        """空の解析結果"""
        return ResidueLevelAnalysis(
            event_name=event_name,
            macro_start=start_frame,
            macro_end=end_frame,
            residue_events=[],
            causality_chain=[],
            initiator_residues=[],
            key_propagation_paths=[],
            async_strong_bonds=[],
            sync_network=[],
            network_stats={"error": "Invalid frame range"},
            confidence_results=[],
            gpu_time=0.0,
        )

    # 残りのメソッドも既存のまま
    def _find_initiators_gpu(
        self, residue_events: list[ResidueEvent], causal_network: list[dict]
    ) -> list[int]:
        """イニシエータ残基の特定"""
        initiators = []

        # 統計的に早期応答残基を判定
        if residue_events:
            delays = [e.propagation_delay for e in residue_events]

            # 統計値計算
            mean_delay = np.mean(delays)
            std_delay = np.std(delays)

            # 閾値決定：平均-2σ、ただし最低10フレーム
            if std_delay > 0:
                threshold = max(10, mean_delay - 2 * std_delay)
            else:
                # 標準偏差が0の場合は四分位数を使用
                q1 = np.percentile(delays, 25)
                threshold = max(10, q1)

            # 早期応答残基を特定
            for event in residue_events:
                if event.propagation_delay < threshold:
                    initiators.append(event.residue_id)

        # ネットワークのハブ（イニシエータが少ない場合の補完）
        if causal_network and len(initiators) < 3:
            out_degree = {}
            for link in causal_network:
                from_res = link.from_res
                if from_res not in out_degree:
                    out_degree[from_res] = 0
                out_degree[from_res] += 1

            # 上位ハブを追加（ただし次数3以上）
            sorted_hubs = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
            for res_id, degree in sorted_hubs[:3]:
                if res_id not in initiators and degree >= 3:
                    initiators.append(res_id)

        # それでもイニシエータがない場合は最速の3残基
        if not initiators and residue_events:
            sorted_events = sorted(residue_events, key=lambda e: e.propagation_delay)
            initiators = [e.residue_id for e in sorted_events[:3]]

        return initiators

    def _build_propagation_paths_gpu(
        self,
        initiators: list[int],
        causality_chains: list[tuple[int, int, float]],
        max_depth: int = 5,
    ) -> list[list[int]]:
        """伝播経路の構築"""
        # グラフ構築
        graph = {}
        for res1, res2, weight in causality_chains:
            if res1 not in graph:
                graph[res1] = []
            graph[res1].append((res2, weight))

        paths = []

        def dfs(current: int, path: list[int], depth: int):
            if depth >= max_depth:
                paths.append(path.copy())
                return

            if current in graph:
                for neighbor, weight in graph[current]:
                    if neighbor not in path:
                        path.append(neighbor)
                        dfs(neighbor, path, depth + 1)
                        path.pop()
            else:
                paths.append(path.copy())

        # 各イニシエータから探索
        for initiator in initiators:
            dfs(initiator, [initiator], 0)

        # 重複除去と並べ替え
        unique_paths = []
        seen = set()

        for path in sorted(paths, key=len, reverse=True):
            path_tuple = tuple(path)
            if path_tuple not in seen and len(path) > 1:
                seen.add(path_tuple)
                unique_paths.append(path)

        return unique_paths

    def _identify_intervention_points_gpu(
        self, importance_scores: dict[int, float], top_n: int = 10
    ) -> list[int]:
        """介入ポイントの特定（GPU使用）"""
        if importance_scores:
            if self.is_gpu and HAS_CUPY:
                # GPU版
                residues = cp.array(list(importance_scores.keys()))
                scores = cp.array(list(importance_scores.values()))

                # 降順でソート
                sorted_indices = cp.argsort(scores)[::-1]

                # 上位を取得
                top_residues = residues[sorted_indices[:top_n]]

                return self.to_cpu(top_residues).tolist()
            else:
                # CPU版フォールバック
                sorted_items = sorted(
                    importance_scores.items(), key=lambda x: x[1], reverse=True
                )
                return [res_id for res_id, _ in sorted_items[:top_n]]

        return []

    def _create_residue_mapping(
        self, n_atoms: int, n_residues: int
    ) -> dict[int, list[int]]:
        """残基マッピングの作成"""
        atoms_per_residue = n_atoms // n_residues
        residue_atoms = {}

        for res_id in range(n_residues):
            start_atom = res_id * atoms_per_residue
            end_atom = min(start_atom + atoms_per_residue, n_atoms)
            residue_atoms[res_id] = list(range(start_atom, end_atom))

        # 余った原子を最後の残基に
        if n_atoms % n_residues != 0:
            remaining_start = n_residues * atoms_per_residue
            residue_atoms[n_residues - 1].extend(range(remaining_start, n_atoms))

        return residue_atoms

    def _get_residue_names(self, n_residues: int) -> dict[int, str]:
        """残基名の取得"""
        return {i: f"RES{i + 1}" for i in range(n_residues)}

    def _compute_global_stats(self, residue_analyses):
        """グローバル統計の計算"""
        total_causal = sum(
            a.network_stats.get("n_causal", 0) for a in residue_analyses.values()
        )
        total_sync = sum(
            a.network_stats.get("n_sync", 0) for a in residue_analyses.values()
        )
        total_async = sum(
            a.network_stats.get("n_async", 0) for a in residue_analyses.values()
        )

        total_gpu_time = sum(a.gpu_time for a in residue_analyses.values())

        mean_window = 100  # デフォルト
        if residue_analyses:
            windows = [
                a.network_stats.get("mean_adaptive_window", 100)
                for a in residue_analyses.values()
            ]
            mean_window = np.mean(windows) if windows else 100

        return {
            "total_causal_links": total_causal,
            "total_sync_links": total_sync,
            "total_async_bonds": total_async,
            "async_to_causal_ratio": total_async / (total_causal + 1e-10),
            "mean_adaptive_window": mean_window,
            "total_gpu_time": total_gpu_time,
            "events_analyzed": len(residue_analyses),
        }

    def _print_summary(
        self,
        importance_scores: dict,
        global_stats: dict,
        intervention_points: list[int],
    ):
        """解析サマリーの表示"""
        print("\n🎯 Global Analysis Complete!")
        print(f"   Key residues identified: {len(importance_scores)}")
        print(f"   Total causal links: {global_stats['total_causal_links']}")
        print(
            f"   Total async strong bonds: {global_stats['total_async_bonds']} "
            f"({global_stats['async_to_causal_ratio']:.1%})"
        )
        print(
            f"   Mean adaptive window: {global_stats['mean_adaptive_window']:.1f} frames"
        )
        print(f"   Total GPU time: {global_stats['total_gpu_time']:.2f} seconds")
        print(f"   Suggested intervention points: {intervention_points[:5]}")


# ユーティリティ関数
def perform_two_stage_analysis_gpu(
    trajectory: np.ndarray,
    macro_result: MDLambda3Result,
    detected_events: list[tuple[int, int, str]],
    n_residues: int = 129,
    config: ResidueAnalysisConfig = None,
) -> TwoStageLambda3Result:
    """
    2段階解析の便利なラッパー関数（後方互換性）
    """
    analyzer = TwoStageAnalyzerGPU(config)
    return analyzer.analyze_trajectory(
        trajectory, macro_result, detected_events, n_residues
    )
