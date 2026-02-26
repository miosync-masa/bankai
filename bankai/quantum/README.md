# quantum_validation_gpu.py — README

## Why "Quantum"?

**This module does not perform electronic structure calculations.**

No wave functions. No electron densities. No Schrödinger equation.
We know. We did it on purpose.

---

### What This Module Actually Does

`quantum_validation_gpu.py` detects **cooperative geometric anomalies** in atomic
coordinate trajectories produced by classical molecular dynamics simulations.
It classifies detected events into signature types based on their spatiotemporal
geometry:

| Internal Label | What It Actually Detects |
|---|---|
| `quantum_jump` | Abrupt displacement exceeding thermal baseline (Z > 3.0) |
| `tunneling` | Barrier-crossing displacement: atom traverses a local energy minimum boundary |
| `entanglement` | Spatially instantaneous correlation: distant atoms displaced in the same sub-frame window |
| `quantum_anomaly` | Cooperative multi-atom event not classifiable into the above |
| `thermal` | Stochastic fluctuation consistent with thermal noise (null class) |

These labels are **geometric metaphors**, not claims about electronic phenomena.

---

### So Why Not Just Call It "geometric_anomaly_gpu.py"?

Because that would be scientifically misleading in the opposite direction.

Here's the thing people forget:

#### Classical force fields are not "classical physics."

The parameters that drive every MD simulation—bond lengths, angles, dihedrals,
partial charges, Lennard-Jones coefficients—were not derived from classical
mechanics. They were derived from:

- **Ab initio quantum-mechanical calculations** (HF, MP2, CCSD(T))
- **Density functional theory** (B3LYP, ωB97X-D, etc.)
- **Experimental measurements** that themselves reflect quantum-mechanical reality

The AMBER `parm99` tyrosine parameters encode the **conjugated π-electron system**
as a planar dihedral constraint with specific barrier heights. The CHARMM
carboxylate parameters encode **resonance-stabilized charge delocalization** as
symmetric partial charges on OE1/OE2. The OPLS aromatic stacking parameters
encode **dispersion interactions** derived from quantum-mechanical perturbation
theory.

When a classical MD trajectory is generated using these parameters, the resulting
atomic coordinates carry a **re-encoded imprint** of the original quantum-mechanical
potential energy surface. The electronic degrees of freedom have not disappeared—
they have been **dimensionally reduced** into effective geometric constraints.

```
Quantum PES (continuous, ∞-dimensional)
    ↓  Force field parameterization (lossy compression)
Classical parameters (discrete, finite-dimensional)
    ↓  MD integration (Newton's equations)
Coordinate trajectory (discrete, 3N-dimensional)
    ↓  This module
Geometric anomaly detection → chemically specific patterns
```

What this module detects—the directional cooperativity of TYR aromatic rings,
the synchronized displacement of GLU carboxylate oxygens, the correlated
fluctuation of TRP indole ring edges—these patterns exist in the coordinate
data **because the force field preserved the quantum-mechanical interaction
geometry** through parameterization.

Calling these detections purely "geometric" would erase the provenance of the
information. The geometry is quantum-mechanically informed. The detection method
is classical. Both statements are simultaneously true.

---

### The Actual Scientific Position

To be explicit:

1. **We do not compute electronic structure.** No wave functions, no electron
   densities, no orbital interactions are calculated by this code.

2. **We do not claim to perform quantum mechanics.** The analysis operates
   exclusively on atomic coordinate trajectories (x, y, z positions over time).

3. **The chemical specificity of detected patterns originates from force field
   parameterization,** which itself derives from quantum-mechanical reference
   calculations. We detect the geometric consequences of quantum-mechanical
   constraints, not the constraints themselves.

4. **The word "quantum" in this module's name refers to the provenance of the
   information being decoded,** not to the method of decoding.

Or, more concisely:

> The force field is the encoder. This module is the decoder.
> The encoded information is quantum-mechanical.
> The decoding method is geometric.

---

### A Note on "First Principles"

Some may argue that only explicit electronic structure methods qualify as
"quantum" analysis. We respectfully note that:

- **DFT** uses approximate exchange-correlation functionals, many of which
  (B3LYP, M06-2X) contain empirically fitted parameters. It is an approximation.
- **MP2** truncates the perturbation expansion at second order. It is an
  approximation.
- **CCSD(T)**, the "gold standard," treats triple excitations perturbatively
  rather than iteratively. It is an approximation.
- **Classical force fields** compress the quantum PES into analytical functional
  forms with fitted parameters. They are an approximation.

Every level of theory is an approximation. The question is not whether
approximations are present, but whether the approximation preserves the
information relevant to the phenomenon being studied. For the geometric
interaction patterns detected by this module—aromatic cooperativity, carboxylate
synchronization, stacking dynamics—the answer is empirically yes: these patterns
are consistently recovered from force-field-generated trajectories and align with
known chemistry.

---

### Acknowledgment

The ability of this module to detect chemically meaningful patterns from
classical coordinate data is entirely dependent on the decades of careful
parameterization by the AMBER, CHARMM, OPLS, and GROMOS force field development
communities. Without their work encoding quantum-mechanical interaction geometries
into force field parameters, this module would detect only isotropic thermal noise.

---

### Module API Reference

```python
from bankai.analysis.quantum_validation_gpu import (
    run_quantum_validation_pipeline,
    AtomicQuantumTrace,
    ThirdImpactAnalyzer,
    AtomicNetworkGPU,
)

# Run full pipeline
results = run_quantum_validation_pipeline(
    trajectory_path="trajectory.xtc",
    metadata_path="metadata.json",
    topology_path="structure.pdb",           # PDB for atom name resolution
    protein_indices_path="protein_idx.npy",  # Optional: protein atom indices
    atom_mapping_path="residue_map.json",    # Optional: residue-atom mapping
)

# Each result contains:
#   - quantum_atoms: dict[int, AtomicQuantumTrace]
#   - atomic_network: AtomicNetworkResult (sync/causal/async links)
#   - drug_target_atoms: list[int] (hub + bridge + high-confidence atoms)
#   - origin: EventOrigin (genesis_atoms, first_wave_atoms)
```

### Signature Classification Criteria

| Signature | Geometric Criterion |
|---|---|
| `quantum_jump` | Single-frame displacement Z-score > 3.0, isolated to ≤3 atoms |
| `tunneling` | Displacement crosses local energy minimum boundary (detected via sign reversal of Q_λ) |
| `entanglement` | Correlation coefficient > 0.8 between atoms separated by > 5Å, within single frame |
| `quantum_anomaly` | Multi-atom cooperative event (≥4 atoms, Z > 2.5) not matching above patterns |
| `thermal` | Z-score < 2.0 or isotropic displacement pattern consistent with Boltzmann distribution |

All thresholds are configurable. Default values are optimized for protein MD
trajectories at 300K with 0.01 ps frame intervals.

---

### Citation

If you use this module, please cite:

```
Iizumi, M. (2025). BANKAI-MD: Discrete Geometric Feature Extraction for
Sub-Picosecond Cooperative Event Detection in Molecular Dynamics Trajectories.
Cureus. [submitted]
```

And please also acknowledge the force field developers whose parameterization
work makes this analysis possible.

---

*"The force field developers encoded quantum mechanics into geometry.*
*We just learned to read it."*

— Miosync, Inc.
