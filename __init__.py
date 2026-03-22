"""
GeoVerify: Geometric Equivalence Checking for Mathematical Objects
"""
from .core import GeoVerifier, EquivalenceResult, VerificationResult
from .structural import LaTeXStructuralAnalyzer

__version__ = "0.1.0"
__all__ = ['GeoVerifier', 'EquivalenceResult', 'VerificationResult', 'LaTeXStructuralAnalyzer']
