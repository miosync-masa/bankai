"""
Quantum Validation Module for Lambda³ - Version 4.0
====================================================

Lambda³ integrated quantum origin validation module.

Features:
- Quantum origin assessment for structural changes detected by Lambda³
- 3-pattern classification (instantaneous/transition/cascade)
- Atomic-level evidence integration
- Adjustable quantum criteria
"""

from .quantum_validation_gpu import (
    AtomicEvidence,
    LambdaAnomaly,
    QuantumAssessment,
    QuantumSignature,
    QuantumValidatorV4,
    StructuralEventPattern,
    validate_lambda_events,
)

__all__ = [
    "QuantumValidatorV4",
    "LambdaAnomaly",
    "AtomicEvidence",
    "QuantumAssessment",
    "StructuralEventPattern",
    "QuantumSignature",
    "validate_lambda_events",
]
