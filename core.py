"""
GeoVerify: Geometric Equivalence Checking for Mathematical Objects
===================================================================

Core thesis: Mathematical equivalence is a structural property, not a
surface-form property. Two expressions are equivalent iff they occupy
the same point on the mathematical object manifold, regardless of the
coordinate system (notation) used to describe them.

Architecture:
  Layer 0: Surface normalization (LaTeX cleanup, notation standardization)
  Layer 1: Symbolic parsing (LaTeX -> SymPy expression tree)
  Layer 2: Algebraic canonicalization (canonical form via SymPy simplification)
  Layer 3: Geometric signature extraction (structural invariants)
  Layer 4: Multi-signal equivalence decision

The key insight: we don't need a 120B parameter model to determine if
two mathematical objects are the same. We need proper compression into
a representation where equivalence is measurable by structure, not by
pattern matching on strings.
"""

import re
import hashlib
import signal
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def with_timeout(func, timeout_sec=5, default=None):
    """Run a function with a timeout. Returns default if it times out."""
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        try:
            result = func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
    except (TimeoutError, Exception):
        return default

import sympy
from sympy import (
    sympify, simplify, expand, factor, cancel, trigsimp,
    symbols, Symbol, Function, Eq, oo, pi, I, E,
    sin, cos, tan, exp, log, sqrt, Abs, Rational,
    Matrix, Piecewise, Interval, FiniteSet, Union,
    latex as sympy_latex
)
from sympy.core.relational import Relational
from sympy.parsing.latex import parse_latex


class EquivalenceResult(Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    UNCERTAIN = "uncertain"


@dataclass
class VerificationResult:
    decision: EquivalenceResult
    confidence: float  # 0.0 to 1.0
    method: str  # which layer made the decision
    details: Dict[str, Any]


# =============================================================================
# Layer 0: Surface Normalization
# =============================================================================

class LaTeXNormalizer:
    """
    Normalize LaTeX surface form without changing mathematical content.
    Handles the exact failure cases documented in the Principia paper:
      - Term/factor reordering
      - Variable symbol differences in dummy indices
      - LaTeX command variations
      - Text span removal
    """

    @staticmethod
    def normalize(latex_str: str) -> str:
        """Apply all surface normalizations."""
        s = latex_str.strip()

        # Remove surrounding text descriptions
        s = LaTeXNormalizer._strip_text_wrappers(s)

        # Normalize LaTeX commands
        s = LaTeXNormalizer._normalize_commands(s)

        # Normalize whitespace
        s = LaTeXNormalizer._normalize_whitespace(s)

        # Normalize common equivalent notations
        s = LaTeXNormalizer._normalize_notation(s)

        return s

    @staticmethod
    def _strip_text_wrappers(s: str) -> str:
        """Remove natural language wrappers around math expressions."""
        # Remove common preambles
        patterns = [
            r'^The\s+(generating\s+function|answer|result|solution|expression)\s+(is|equals?)\s*',
            r'^(?:Therefore|Thus|Hence|So),?\s*',
            r'^\s*(?:for|where|when)\s+.*?[,;:]\s*',
        ]
        for pat in patterns:
            s = re.sub(pat, '', s, flags=re.IGNORECASE)

        # Remove trailing conditions that are separate from the expression
        # but keep conditions that are part of piecewise/domain specs
        s = s.strip().rstrip('.')

        return s

    @staticmethod
    def _normalize_commands(s: str) -> str:
        """Normalize LaTeX command variations."""
        # \mathrm{i} -> i (imaginary unit)
        s = re.sub(r'\\mathrm\{i\}', 'i', s)
        s = re.sub(r'\\mathrm\{e\}', 'e', s)
        s = re.sub(r'\\mathrm\{d\}', 'd', s)

        # \text{...} removal for common math terms
        s = re.sub(r'\\text\{(\s*(?:if|for|when|otherwise|and|or|where)\s*)\}', r' \1 ', s)
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\textrm\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\textit\{([^}]*)\}', r'\1', s)

        # \left and \right are purely visual
        s = re.sub(r'\\left\s*', '', s)
        s = re.sub(r'\\right\s*', '', s)

        # \displaystyle, \scriptstyle etc
        s = re.sub(r'\\(?:display|script|scriptscript|text)style\s*', '', s)

        # Normalize fraction forms: both \frac{a}{b} and \dfrac{a}{b}
        s = re.sub(r'\\dfrac', r'\\frac', s)
        s = re.sub(r'\\tfrac', r'\\frac', s)

        # \cdot vs \times vs * (multiplication) - remove them as they're implicit
        s = re.sub(r'\s*\\cdot\s*', ' ', s)
        s = re.sub(r'\s*\\times\s*', ' ', s)

        # \geq vs \ge, \leq vs \le
        s = re.sub(r'\\geq\b', r'\\ge', s)
        s = re.sub(r'\\leq\b', r'\\le', s)

        # \ln vs \log (context-dependent but normalize for comparison)
        # Actually keep these distinct - they're mathematically different

        # \mathbf, \boldsymbol -> plain symbol for scalar comparison
        s = re.sub(r'\\(?:mathbf|boldsymbol|mathbb|mathcal|mathfrak)\{([^}]*)\}', r'\1', s)

        # \, \; \: \! (spacing) -> nothing
        s = re.sub(r'\\[,;:!]', ' ', s)

        # \quad, \qquad -> space
        s = re.sub(r'\\q?quad\s*', ' ', s)

        return s

    @staticmethod
    def _normalize_whitespace(s: str) -> str:
        """Collapse whitespace."""
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    @staticmethod
    def _normalize_notation(s: str) -> str:
        """Normalize equivalent mathematical notations."""
        # Product notation: \prod_{n>=1} vs \prod_{m=1}^{\infty}
        # These are structurally equivalent - handled at symbolic level

        # exp(-x) vs e^{-x}
        # Handled at symbolic level after parsing

        # |x| vs \|x\| vs \lvert x \rvert
        s = re.sub(r'\\lvert\s*', '|', s)
        s = re.sub(r'\\rvert\s*', '|', s)
        s = re.sub(r'\\\|', '|', s)

        return s


# =============================================================================
# Layer 1: Symbolic Parsing
# =============================================================================

class SymbolicParser:
    """
    Parse normalized LaTeX into SymPy expression trees.
    Uses multiple parsing strategies with fallbacks.
    """

    @staticmethod
    def parse(latex_str: str) -> Optional[sympy.Basic]:
        """Attempt to parse LaTeX string into a SymPy expression."""
        # Strategy 1: Direct sympy.parsing.latex
        result = SymbolicParser._try_sympy_latex_parser(latex_str)
        if result is not None:
            return result

        # Strategy 2: latex2sympy2
        result = SymbolicParser._try_latex2sympy(latex_str)
        if result is not None:
            return result

        # Strategy 3: Manual preprocessing + sympy parse
        result = SymbolicParser._try_manual_parse(latex_str)
        if result is not None:
            return result

        return None

    @staticmethod
    def _try_sympy_latex_parser(latex_str: str) -> Optional[sympy.Basic]:
        try:
            return parse_latex(latex_str)
        except Exception:
            return None

    @staticmethod
    def _try_latex2sympy(latex_str: str) -> Optional[sympy.Basic]:
        try:
            from latex2sympy2 import latex2sympy
            return latex2sympy(latex_str)
        except Exception:
            return None

    @staticmethod
    def _try_manual_parse(latex_str: str) -> Optional[sympy.Basic]:
        """Last resort: manual LaTeX -> Python string -> sympify."""
        try:
            s = latex_str
            # Basic substitutions
            s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'((\1)/(\2))', s)
            s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
            s = re.sub(r'\\pi', 'pi', s)
            s = re.sub(r'\\infty', 'oo', s)
            s = re.sub(r'\\sin', 'sin', s)
            s = re.sub(r'\\cos', 'cos', s)
            s = re.sub(r'\\tan', 'tan', s)
            s = re.sub(r'\\exp', 'exp', s)
            s = re.sub(r'\\ln', 'log', s)
            s = re.sub(r'\\log', 'log', s)
            s = re.sub(r'\^{([^}]*)}', r'**(\1)', s)
            s = re.sub(r'\^(.)', r'**\1', s)
            s = re.sub(r'_\{[^}]*\}', '', s)  # remove subscripts for parsing
            s = re.sub(r'\\[a-zA-Z]+', '', s)  # remove remaining commands
            s = re.sub(r'[{}]', '', s)
            s = s.strip()
            if s:
                return sympify(s)
        except Exception:
            pass
        return None


# =============================================================================
# Layer 2: Algebraic Canonicalization
# =============================================================================

class AlgebraicCanonicalizer:
    """
    Reduce expressions to canonical forms using algebraic identities.
    This is where commutativity, associativity, and algebraic equivalences
    are resolved.
    """

    @staticmethod
    def canonicalize(expr: sympy.Basic) -> sympy.Basic:
        """Apply canonicalization cascade."""
        if expr is None:
            return None

        try:
            # Step 1: Expand and simplify
            canon = with_timeout(lambda: sympy.expand(expr), timeout_sec=2, default=expr)

            # Step 2: Cancel common factors
            canon = with_timeout(lambda: sympy.cancel(canon), timeout_sec=2, default=canon)

            # Step 3: Trigonometric simplification
            canon = with_timeout(lambda: sympy.trigsimp(canon), timeout_sec=2, default=canon)

            # Step 4: General simplification
            canon = with_timeout(lambda: sympy.simplify(canon), timeout_sec=2, default=canon)

            return canon
        except Exception:
            return expr


# =============================================================================
# Layer 3: Geometric Signature Extraction
# =============================================================================

class GeometricSignature:
    """
    Extract structural invariants from expression trees.

    The geometric insight: two equivalent mathematical objects have the
    same structure even when expressed differently. We extract invariants
    that are preserved under the transformations that don't change
    mathematical meaning (variable renaming, term reordering, notation changes).

    Invariants extracted:
    1. Expression tree depth and branching structure
    2. Operation spectrum (histogram of operations)
    3. Numeric constants present (sorted)
    4. Structural hash (order-independent)
    5. Degree/order properties
    6. Coefficient spectrum (for polynomials)
    """

    @dataclass
    class Signature:
        tree_depth: int
        num_operations: int
        operation_spectrum: Dict[str, int]  # op_name -> count
        constants: List[float]  # sorted numeric constants
        num_free_symbols: int
        structural_hash: str
        degree_info: Optional[Dict[str, int]]  # variable -> degree
        complexity: int  # sympy count_ops

        def distance(self, other: 'GeometricSignature.Signature') -> float:
            """
            Compute distance between two signatures.
            Low distance = likely equivalent.
            This is the geometric core: equivalence as proximity
            in signature space.
            """
            if self.structural_hash == other.structural_hash:
                return 0.0

            d = 0.0

            # Tree structure distance
            d += abs(self.tree_depth - other.tree_depth) * 0.5
            d += abs(self.num_operations - other.num_operations) * 0.3
            d += abs(self.num_free_symbols - other.num_free_symbols) * 2.0

            # Operation spectrum distance (L1)
            all_ops = set(self.operation_spectrum.keys()) | set(other.operation_spectrum.keys())
            for op in all_ops:
                d += abs(self.operation_spectrum.get(op, 0) -
                        other.operation_spectrum.get(op, 0)) * 0.2

            # Constants distance
            c1 = self.constants
            c2 = other.constants
            if len(c1) != len(c2):
                d += abs(len(c1) - len(c2)) * 1.0
            else:
                for a, b in zip(c1, c2):
                    if a != b:
                        d += 0.5

            # Complexity distance
            d += abs(self.complexity - other.complexity) * 0.1

            return d

    @staticmethod
    def extract(expr: sympy.Basic) -> Optional['GeometricSignature.Signature']:
        """Extract geometric signature from a SymPy expression."""
        if expr is None:
            return None

        try:
            # Tree depth
            depth = GeometricSignature._tree_depth(expr)

            # Operation spectrum
            op_spec = GeometricSignature._operation_spectrum(expr)
            num_ops = sum(op_spec.values())

            # Constants
            constants = sorted([float(n) for n in expr.atoms(sympy.Number)
                              if n not in (sympy.S.Zero, sympy.S.One, sympy.S.NegativeOne)])

            # Free symbols
            num_free = len(expr.free_symbols)

            # Structural hash (order-independent)
            struct_hash = GeometricSignature._structural_hash(expr)

            # Degree info
            degree_info = GeometricSignature._degree_info(expr)

            # Complexity
            try:
                complexity = sympy.count_ops(expr)
            except Exception:
                complexity = 0

            return GeometricSignature.Signature(
                tree_depth=depth,
                num_operations=num_ops,
                operation_spectrum=op_spec,
                constants=constants,
                num_free_symbols=num_free,
                structural_hash=struct_hash,
                degree_info=degree_info,
                complexity=complexity
            )
        except Exception:
            return None

    @staticmethod
    def _tree_depth(expr: sympy.Basic) -> int:
        if not expr.args:
            return 0
        return 1 + max(GeometricSignature._tree_depth(a) for a in expr.args)

    @staticmethod
    def _operation_spectrum(expr: sympy.Basic) -> Dict[str, int]:
        """Count occurrences of each operation type."""
        spec = {}
        def walk(e):
            if e.args:
                name = type(e).__name__
                spec[name] = spec.get(name, 0) + 1
                for a in e.args:
                    walk(a)
        walk(expr)
        return spec

    @staticmethod
    def _structural_hash(expr: sympy.Basic) -> str:
        """
        Hash that is invariant to variable naming.
        Replace all free symbols with canonical names based on
        their structural role, then hash.
        """
        try:
            # Sort free symbols by their first appearance in the tree
            free = sorted(expr.free_symbols, key=lambda s: str(s))
            # Create substitution to canonical names
            canonical_syms = [Symbol(f'x_{i}') for i in range(len(free))]
            sub_dict = dict(zip(free, canonical_syms))
            canonical_expr = expr.subs(sub_dict)
            # Hash the canonical string representation
            canon_str = str(canonical_expr)
            return hashlib.md5(canon_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(expr).encode()).hexdigest()

    @staticmethod
    def _degree_info(expr: sympy.Basic) -> Optional[Dict[str, int]]:
        """Extract polynomial degree info if applicable."""
        try:
            free = expr.free_symbols
            if not free:
                return {}
            result = {}
            for s in free:
                try:
                    p = sympy.Poly(expr, s)
                    result[str(s)] = p.degree()
                except Exception:
                    pass
            return result if result else None
        except Exception:
            return None


# =============================================================================
# Layer 4: Multi-Signal Equivalence Decision
# =============================================================================

class GeoVerifier:
    """
    Main equivalence checker. Uses a cascade of methods from cheapest
    (string comparison) to most expensive (geometric signature comparison),
    with early exit on high-confidence decisions.
    """

    def __init__(self, verbose: bool = False):
        self.normalizer = LaTeXNormalizer()
        self.parser = SymbolicParser()
        self.canonicalizer = AlgebraicCanonicalizer()
        self.verbose = verbose

    def verify(self, reference: str, prediction: str,
               problem_statement: str = "", _depth: int = 0) -> VerificationResult:
        """
        Determine if reference and prediction are mathematically equivalent.

        Args:
            reference: Ground truth answer (LaTeX string)
            prediction: Model prediction (LaTeX string)
            problem_statement: Optional context (not used in v0 but reserved)

        Returns:
            VerificationResult with decision, confidence, and method info
        """
        details = {}

        # =========================
        # Check 0: Exact string match (after normalization)
        # =========================
        norm_ref = self.normalizer.normalize(reference)
        norm_pred = self.normalizer.normalize(prediction)
        details['normalized_ref'] = norm_ref
        details['normalized_pred'] = norm_pred

        if norm_ref == norm_pred:
            return VerificationResult(
                decision=EquivalenceResult.EQUIVALENT,
                confidence=1.0,
                method="exact_string_match",
                details=details
            )

        # =========================
        # Check 0.5: Structural/alpha equivalence (pre-symbolic)
        # Handles products, sums, integrals with dummy variable renaming
        # =========================
        try:
            from structural import LaTeXStructuralAnalyzer
            sa = LaTeXStructuralAnalyzer()
            struct_equiv, struct_conf, struct_method = sa.structural_equivalent(norm_ref, norm_pred)
            details['structural_method'] = struct_method
            details['structural_confidence'] = struct_conf

            if struct_equiv and struct_conf >= 0.85:
                return VerificationResult(
                    decision=EquivalenceResult.EQUIVALENT,
                    confidence=struct_conf,
                    method=f"structural:{struct_method}",
                    details=details
                )
            elif not struct_equiv and struct_conf >= 0.70:
                # Structural analysis says NOT equivalent with decent confidence
                # Store this but don't return yet - let other methods confirm
                details['structural_non_equiv'] = True
                details['structural_non_equiv_conf'] = struct_conf
        except Exception as e:
            details['structural_error'] = str(e)

        # =========================
        # Check 1: Symbolic equivalence via SymPy
        # =========================
        ref_expr = self.parser.parse(norm_ref)
        pred_expr = self.parser.parse(norm_pred)
        details['ref_parsed'] = ref_expr is not None
        details['pred_parsed'] = pred_expr is not None

        if ref_expr is not None and pred_expr is not None:
            # Direct symbolic comparison
            try:
                diff = with_timeout(
                    lambda: sympy.simplify(ref_expr - pred_expr),
                    timeout_sec=3, default=None)
                if diff is not None and diff == 0:
                    return VerificationResult(
                        decision=EquivalenceResult.EQUIVALENT,
                        confidence=0.98,
                        method="symbolic_simplify_zero",
                        details=details
                    )
            except Exception:
                pass

            # Canonicalize and compare
            canon_ref = with_timeout(
                lambda: self.canonicalizer.canonicalize(ref_expr),
                timeout_sec=3, default=ref_expr)
            canon_pred = with_timeout(
                lambda: self.canonicalizer.canonicalize(pred_expr),
                timeout_sec=3, default=pred_expr)

            if canon_ref is not None and canon_pred is not None:
                try:
                    diff = with_timeout(
                        lambda: sympy.simplify(canon_ref - canon_pred),
                        timeout_sec=3, default=None)
                    if diff is not None and diff == 0:
                        return VerificationResult(
                            decision=EquivalenceResult.EQUIVALENT,
                            confidence=0.97,
                            method="canonical_simplify_zero",
                            details=details
                        )
                except Exception:
                    pass

                # Ratio test: if a/b simplifies to a constant
                try:
                    if canon_pred != 0:
                        ratio = with_timeout(
                            lambda: sympy.simplify(canon_ref / canon_pred),
                            timeout_sec=3, default=None)
                        if ratio is not None and ratio == 1:
                            return VerificationResult(
                                decision=EquivalenceResult.EQUIVALENT,
                                confidence=0.96,
                                method="ratio_test",
                                details=details
                            )
                except Exception:
                    pass

                # Numerical evaluation test
                result = self._numerical_equivalence_test(
                    canon_ref, canon_pred, details)
                if result is not None:
                    return result

            # =========================
            # Check 2: Geometric signature comparison
            # =========================
            sig_ref = GeometricSignature.extract(
                canon_ref if canon_ref is not None else ref_expr)
            sig_pred = GeometricSignature.extract(
                canon_pred if canon_pred is not None else pred_expr)

            if sig_ref is not None and sig_pred is not None:
                distance = sig_ref.distance(sig_pred)
                details['signature_distance'] = distance
                details['sig_ref_hash'] = sig_ref.structural_hash
                details['sig_pred_hash'] = sig_pred.structural_hash

                if sig_ref.structural_hash == sig_pred.structural_hash:
                    return VerificationResult(
                        decision=EquivalenceResult.EQUIVALENT,
                        confidence=0.95,
                        method="structural_hash_match",
                        details=details
                    )

                if distance == 0.0:
                    return VerificationResult(
                        decision=EquivalenceResult.EQUIVALENT,
                        confidence=0.90,
                        method="zero_signature_distance",
                        details=details
                    )

                if distance > 5.0:
                    return VerificationResult(
                        decision=EquivalenceResult.NOT_EQUIVALENT,
                        confidence=min(0.95, 0.5 + distance * 0.05),
                        method="high_signature_distance",
                        details=details
                    )

        # =========================
        # Check 3: Fuzzy string comparison (fallback)
        # =========================
        result = self._fuzzy_string_comparison(norm_ref, norm_pred, details)
        if result is not None:
            return result

        # =========================
        # Check 4: Equation-aware comparison
        # =========================
        if _depth < 2:
            result = self._equation_aware_comparison(norm_ref, norm_pred, details, _depth)
            if result is not None:
                return result

        # =========================
        # Check 5: Prefix/suffix difference detection
        # =========================
        result = self._prefix_difference_detection(norm_ref, norm_pred, details)
        if result is not None:
            return result

        # =========================
        # Default: use structural non-equiv signal if available
        # =========================
        if details.get('structural_non_equiv', False):
            return VerificationResult(
                decision=EquivalenceResult.NOT_EQUIVALENT,
                confidence=details.get('structural_non_equiv_conf', 0.5),
                method="structural_non_equiv_fallback",
                details=details
            )

        return VerificationResult(
            decision=EquivalenceResult.UNCERTAIN,
            confidence=0.3,
            method="no_conclusive_method",
            details=details
        )

    def _equation_aware_comparison(
        self, s1: str, s2: str, details: Dict, _depth: int = 0
    ) -> Optional[VerificationResult]:
        """
        If expressions contain '=', split into LHS and RHS and compare
        each side. This catches cases like 'E = expr1' vs 'E = expr2'
        where the full expression doesn't simplify but the RHS does.
        """
        if '=' not in s1 or '=' not in s2:
            return None
        # Don't trigger on \leq, \geq, \neq etc.
        if re.search(r'\\[lg]eq|\\neq|\\approx|<=|>=|!=', s1):
            return None

        try:
            # Split at first standalone = (not inside \frac, etc.)
            lhs1, rhs1 = self._split_equation(s1)
            lhs2, rhs2 = self._split_equation(s2)

            if lhs1 is None or lhs2 is None:
                return None

            # Compare LHS strings (after normalization)
            lhs_match = (lhs1.strip() == lhs2.strip())

            if lhs_match:
                # Same LHS, compare RHS
                rhs_result = self.verify(rhs1.strip(), rhs2.strip(), _depth=_depth+1)
                if rhs_result.decision == EquivalenceResult.EQUIVALENT:
                    return VerificationResult(
                        decision=EquivalenceResult.EQUIVALENT,
                        confidence=rhs_result.confidence * 0.95,
                        method=f"equation_rhs:{rhs_result.method}",
                        details=details
                    )
                elif rhs_result.decision == EquivalenceResult.NOT_EQUIVALENT:
                    return VerificationResult(
                        decision=EquivalenceResult.NOT_EQUIVALENT,
                        confidence=rhs_result.confidence * 0.95,
                        method=f"equation_rhs_diff:{rhs_result.method}",
                        details=details
                    )
            else:
                # Different LHS = definitely different equation
                # But check if it's just a prefix issue (2X = ... vs X = ...)
                pass
        except Exception:
            pass
        return None

    def _split_equation(self, s: str) -> Tuple[Optional[str], Optional[str]]:
        """Split at top-level = sign, respecting braces."""
        depth = 0
        for i, c in enumerate(s):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            elif c == '=' and depth == 0:
                # Check it's not part of \leq, \geq etc.
                before = s[max(0,i-4):i]
                if '\\' not in before or before.endswith(' '):
                    return s[:i], s[i+1:]
        return None, None

    def _prefix_difference_detection(
        self, s1: str, s2: str, details: Dict
    ) -> Optional[VerificationResult]:
        """
        Detect cases where one expression is the other with a prefix
        modification: leading constant, negation, or added term.
        
        Cases: '2 expr' vs 'expr', '-expr' vs 'expr'
        """
        # Strip all whitespace for comparison
        c1 = re.sub(r'\s+', '', s1)
        c2 = re.sub(r'\s+', '', s2)

        # Check if one is a prefix of the other (with extra stuff)
        # Case: '2expr' vs 'expr' or '-expr' vs 'expr'
        if len(c1) > len(c2) and c1.endswith(c2):
            prefix = c1[:len(c1)-len(c2)]
            if prefix in ('2', '3', '-', '-1', '2*', '3*', '-1*'):
                return VerificationResult(
                    decision=EquivalenceResult.NOT_EQUIVALENT,
                    confidence=0.88,
                    method=f"prefix_diff:{prefix}",
                    details=details
                )
        if len(c2) > len(c1) and c2.endswith(c1):
            prefix = c2[:len(c2)-len(c1)]
            if prefix in ('2', '3', '-', '-1', '2*', '3*', '-1*'):
                return VerificationResult(
                    decision=EquivalenceResult.NOT_EQUIVALENT,
                    confidence=0.88,
                    method=f"prefix_diff:{prefix}",
                    details=details
                )

        # Case: 'expr' vs '-expr' (negation prefix)
        if c2 == '-' + c1 or c1 == '-' + c2:
            return VerificationResult(
                decision=EquivalenceResult.NOT_EQUIVALENT,
                confidence=0.92,
                method="negation_prefix",
                details=details
            )

        # Case: same string except one char changed
        # Good for catching subscript changes like \mu_0 -> \mu_2
        if len(c1) == len(c2):
            diffs = [(i, c1[i], c2[i]) for i in range(len(c1)) if c1[i] != c2[i]]
            if 1 <= len(diffs) <= 2:
                # 1-2 character differences in same-length string
                # Check if the differences are in numeric characters
                all_numeric_diff = all(a.isdigit() or b.isdigit() for _, a, b in diffs)
                if all_numeric_diff:
                    return VerificationResult(
                        decision=EquivalenceResult.NOT_EQUIVALENT,
                        confidence=0.85,
                        method=f"char_diff:{diffs}",
                        details=details
                    )
                # Check if diffs are in function names (e.g., sqrt vs cbrt)
                if len(diffs) >= 1:
                    return VerificationResult(
                        decision=EquivalenceResult.NOT_EQUIVALENT,
                        confidence=0.78,
                        method=f"char_diff_func:{len(diffs)}_diffs",
                        details=details
                    )

        # Case: strings are very similar but not identical
        # One is slightly longer (1-3 chars) — likely a small modification
        len_diff = abs(len(c1) - len(c2))

        # Special case: subscript changes like \varepsilon_0 -> \varepsilon_2
        # or \int_0 -> \int_2 — same length but a subscript digit changed
        if len_diff == 0 and len(c1) > 5:
            diffs_with_context = []
            for i in range(len(c1)):
                if c1[i] != c2[i]:
                    # Get surrounding context
                    ctx_before = c1[max(0,i-3):i]
                    diffs_with_context.append((i, c1[i], c2[i], ctx_before))
            # If 1-2 diffs and at least one is near a subscript '_'
            if 1 <= len(diffs_with_context) <= 2:
                has_subscript_context = any('_' in ctx for _, _, _, ctx in diffs_with_context)
                has_numeric_change = any(a.isdigit() and b.isdigit() for _, a, b, _ in diffs_with_context)
                if has_subscript_context and has_numeric_change:
                    return VerificationResult(
                        decision=EquivalenceResult.NOT_EQUIVALENT,
                        confidence=0.88,
                        method=f"subscript_digit_change:{diffs_with_context}",
                        details=details
                    )

        if 0 < len_diff <= 3 and min(len(c1), len(c2)) > 10:
            # Check character overlap
            shorter = c1 if len(c1) < len(c2) else c2
            longer = c1 if len(c1) >= len(c2) else c2

            # If the shorter is a subsequence of the longer with 1-3 insertions
            # this is likely a small modification (non-equivalent)
            # Simple heuristic: count matching prefix + suffix
            prefix_match = 0
            for i in range(min(len(shorter), len(longer))):
                if shorter[i] == longer[i]:
                    prefix_match += 1
                else:
                    break
            suffix_match = 0
            for i in range(1, min(len(shorter), len(longer)) + 1):
                if shorter[-i] == longer[-i]:
                    suffix_match += 1
                else:
                    break

            coverage = (prefix_match + suffix_match) / len(shorter)
            if coverage > 0.90:
                return VerificationResult(
                    decision=EquivalenceResult.NOT_EQUIVALENT,
                    confidence=0.80,
                    method=f"near_match_diff:{len_diff}chars",
                    details=details
                )

    def _numerical_equivalence_test(
        self, expr1: sympy.Basic, expr2: sympy.Basic,
        details: Dict
    ) -> Optional[VerificationResult]:
        """
        Test equivalence by numerical evaluation at random points.
        This catches cases where symbolic simplification fails but
        the expressions are numerically identical.
        """
        try:
            free1 = expr1.free_symbols
            free2 = expr2.free_symbols

            if len(free1) != len(free2):
                return None

            # Map symbols by sorted name
            syms1 = sorted(free1, key=str)
            syms2 = sorted(free2, key=str)

            # Test at multiple random points
            np.random.seed(42)
            n_tests = 10
            n_pass = 0

            for _ in range(n_tests):
                vals = {s: np.random.uniform(0.1, 5.0) for s in syms1}
                vals2 = {s2: vals[s1] for s1, s2 in zip(syms1, syms2)}

                try:
                    v1 = complex(expr1.subs(vals))
                    v2 = complex(expr2.subs(vals2))

                    if abs(v1) < 1e-15 and abs(v2) < 1e-15:
                        n_pass += 1
                    elif abs(v1) > 1e-15:
                        rel_diff = abs(v1 - v2) / abs(v1)
                        if rel_diff < 1e-8:
                            n_pass += 1
                    elif abs(v2) > 1e-15:
                        rel_diff = abs(v1 - v2) / abs(v2)
                        if rel_diff < 1e-8:
                            n_pass += 1
                except Exception:
                    continue

            details['numerical_tests_passed'] = n_pass
            details['numerical_tests_total'] = n_tests

            if n_pass >= 9:
                return VerificationResult(
                    decision=EquivalenceResult.EQUIVALENT,
                    confidence=0.92,
                    method="numerical_equivalence",
                    details=details
                )
            elif n_pass <= 2:
                return VerificationResult(
                    decision=EquivalenceResult.NOT_EQUIVALENT,
                    confidence=0.85,
                    method="numerical_nonequivalence",
                    details=details
                )

        except Exception:
            pass

        return None

    def _fuzzy_string_comparison(
        self, s1: str, s2: str, details: Dict
    ) -> Optional[VerificationResult]:
        """
        Last-resort comparison on normalized strings.
        Handles cases where parsing fails but strings are
        clearly similar or clearly different.
        """
        # Remove all whitespace for comparison
        c1 = re.sub(r'\s+', '', s1)
        c2 = re.sub(r'\s+', '', s2)

        if c1 == c2:
            return VerificationResult(
                decision=EquivalenceResult.EQUIVALENT,
                confidence=0.85,
                method="fuzzy_string_exact",
                details=details
            )

        # Sort characters and compare (catches pure reordering)
        if sorted(c1) == sorted(c2) and len(c1) == len(c2) and len(c1) > 3:
            return VerificationResult(
                decision=EquivalenceResult.EQUIVALENT,
                confidence=0.60,
                method="fuzzy_char_sort",
                details=details
            )

        # Levenshtein-like: if very short edit distance relative to length
        if len(c1) > 5 and len(c2) > 5:
            max_len = max(len(c1), len(c2))
            # Simple check: character overlap
            set1 = set(c1)
            set2 = set(c2)
            overlap = len(set1 & set2) / max(len(set1 | set2), 1)

            if overlap < 0.3:
                return VerificationResult(
                    decision=EquivalenceResult.NOT_EQUIVALENT,
                    confidence=0.70,
                    method="fuzzy_low_overlap",
                    details=details
                )

        return None
