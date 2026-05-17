# geometric_validation_gpu.py — README

## What this module does

`geometric_validation_gpu.py` classifies structural events detected by the
BANKAI-MD / Lambda³ pipeline into a small set of **geometric anomaly
signatures**. The classification is a description of *coordinate-level event
patterns* — it does not assert any specific physical mechanism. Higher-
resolution analyses may reveal underlying classical mechanisms; this module
makes no claim about the physical origin of detected events.

The framework operates exclusively on atomic coordinate trajectories
(x, y, z over time) produced by classical molecular dynamics. No wave
functions, electron densities, or quantum-mechanical calculations are
performed.

---

## Signature taxonomy

| Signature (enum value)                       | Geometric meaning |
|----------------------------------------------|-------------------|
| `instantaneous_correlation_signature`        | Spatially distant atoms displaced within a single frame window (r > 0.8, separation > 5 Å) |
| `barrier_crossing_signature`                 | Displacement crossing a local energy-surface boundary (Q_λ sign reversal) |
| `sustained_coordination_signature`           | Cooperative structural coordination sustained over a long window (> 300 ps) |
| `cooperative_phase_signature`                | Cooperative phase-like transition (combined spatial + synchronization anomaly) |
| `causal_cascade_signature`                   | Causal cascade propagation through async bonds |
| `thermal_baseline`                           | No anomaly detected; consistent with thermal noise (Z < 2.0) |

These labels are a **geometric classification of frame-to-frame coordinate
change patterns**, not a mechanism inference.

---

## Three-axis classification

Each event is scored on three orthogonal axes:

- **Spatial** — displacement magnitude exceeding the thermal baseline
  (ΔΛC jump, Z-score, atomic velocity)
- **Synchronization** — cooperative/correlated multi-atom response
  (σS, pairwise correlation coefficient, async bonds)
- **Temporal** — timescale inconsistent with thermal diffusion
  (instantaneous events, fast transitions, sustained coherence)

The combination of pattern (instantaneous / transition / cascade) and
axis profile determines the assigned signature.

---

## API

```python
from bankai.analysis.run_full_analysis import run_geometric_validation_pipeline

results = run_geometric_validation_pipeline(
    trajectory_path="trajectory.npy",
    metadata_path="metadata.json",
    protein_indices_path="protein_idx.npy",
    topology_path="structure.pdb",            # optional, for atom-name resolution
    atom_mapping_path="residue_map.json",     # optional, residue→atom mapping
)

# Each result includes:
#   - geometric_assessments: list[GeometricAssessment]
#   - lambda_result, two_stage_result
#   - third_impact_results (if enabled)
```

The lower-level validator can be used directly:

```python
from bankai.geometric import GeometricValidatorV4

validator = GeometricValidatorV4(trajectory=traj, dt_ps=100.0, temperature_K=300.0)
assessments = validator.validate_events(events, lambda_result, network_results)
validator.print_summary(assessments)
```

---

## On the relationship between geometry and force-field provenance

The chemical specificity of patterns detected by this module — directional
cooperativity of aromatic rings, synchronized carboxylate displacements,
correlated indole-edge fluctuations — exists in the coordinate data because
classical force-field parameters (AMBER, CHARMM, OPLS, GROMOS) themselves
derive from quantum-mechanical reference calculations (HF, MP2, CCSD(T), DFT)
and experimental measurements. The force field encodes those constraints as
parameters; classical MD integration propagates them into atomic coordinates;
this module decodes the geometric consequences.

That provenance does not make the analysis itself a quantum-mechanical
analysis. The detection method is purely geometric.

---

## Citation

```
Iizumi, M. (2025). BANKAI-MD: Discrete Geometric Feature Extraction for
Sub-Picosecond Cooperative Event Detection in Molecular Dynamics Trajectories.
Cureus. [submitted]
```

Please also acknowledge the force-field developers whose parameterization
work makes this analysis possible.
