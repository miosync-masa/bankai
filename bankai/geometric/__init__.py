"""
Geometric Validation Module for Lambda³ - Version 4.0
=====================================================

Lambda³ integrated geometric anomaly validation module.

Features:
- Geometric anomaly assessment for structural changes detected by Lambda³
- 3-pattern classification (instantaneous/transition/cascade)
- Atomic-level evidence integration
- Adjustable cooperative-event criteria
"""

from .geometric_validation_gpu import (
    AtomicEvidence,
    GeometricAssessment,
    GeometricSignature,
    GeometricValidatorV4,
    LambdaAnomaly,
    StructuralEventPattern,
    validate_lambda_events,
)

__all__ = [
    "GeometricValidatorV4",
    "LambdaAnomaly",
    "AtomicEvidence",
    "GeometricAssessment",
    "StructuralEventPattern",
    "GeometricSignature",
    "validate_lambda_events",
]
