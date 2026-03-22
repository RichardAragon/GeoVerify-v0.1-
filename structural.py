"""
LaTeX Structural Analyzer (Layer 2.5)
======================================

When SymPy's parser can't handle complex LaTeX (products, sums, integrals,
binomials), we need to work directly on the LaTeX structure itself.

Core insight: Two mathematical expressions with different bound variable
names are alpha-equivalent. This is the same problem as lambda calculus
alpha-equivalence, applied to mathematical notation.

The approach:
1. Parse LaTeX into a lightweight structural tree (not full symbolic)
2. Identify bound variables (summation indices, integration variables, 
   product indices) and their binding sites
3. Canonicalize bound variables by structural position (de Bruijn indices)
4. Compare canonicalized structures

This is the geometric contribution: we're not doing string matching or
LLM reasoning. We're computing structural invariants of the expression
tree that are preserved under exactly the transformations that don't
change mathematical meaning.
"""

import re
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class LaTeXToken:
    """A token in the LaTeX structural tree."""
    type: str  # 'cmd', 'group', 'text', 'sub', 'sup', 'infix'
    value: str
    children: List['LaTeXToken'] = field(default_factory=list)

    def __repr__(self):
        if self.children:
            return f"{self.type}({self.value}, [{', '.join(str(c) for c in self.children)}])"
        return f"{self.type}({self.value})"


class LaTeXStructuralAnalyzer:
    """
    Analyze LaTeX at the structural level without full symbolic parsing.
    
    Key operations:
    1. Identify binding constructs (\sum, \prod, \int) and their bound vars
    2. Canonicalize bound variables (alpha-normalization)
    3. Normalize equivalent notations at the structural level
    4. Compare structural fingerprints
    """

    # Binding constructs: commands that introduce bound variables
    BINDING_CONSTRUCTS = {
        r'\sum', r'\prod', r'\int', r'\oint',
        r'\bigcup', r'\bigcap', r'\bigoplus',
        r'\coprod', r'\bigotimes',
    }

    # Notation equivalences at structural level
    LIMIT_EQUIVALENCES = [
        # n >= 1 is the same as n=1 to infinity
        (r'(\w+)\s*\\ge\s*(\d+)', r'\1=\2', 'lower_bound_equiv'),
        (r'(\w+)\s*\\geq\s*(\d+)', r'\1=\2', 'lower_bound_equiv'),
        (r'(\w+)\s*>=\s*(\d+)', r'\1=\2', 'lower_bound_equiv'),
    ]

    def structural_equivalent(self, latex1: str, latex2: str) -> Tuple[bool, float, str]:
        """
        Determine structural equivalence between two LaTeX strings.
        
        Returns: (is_equivalent, confidence, method)
        """
        # Step 1: Structural normalization
        norm1 = self._structural_normalize(latex1)
        norm2 = self._structural_normalize(latex2)

        if norm1 == norm2:
            return True, 0.95, "structural_normalize_exact"

        # Step 2: Alpha-normalize bound variables
        alpha1 = self._alpha_normalize(norm1)
        alpha2 = self._alpha_normalize(norm2)

        if alpha1 == alpha2:
            return True, 0.93, "alpha_normalized_exact"

        # Step 3: Structural fingerprint comparison
        fp1 = self._structural_fingerprint(alpha1)
        fp2 = self._structural_fingerprint(alpha2)

        if fp1 == fp2:
            return True, 0.88, "structural_fingerprint_match"

        # Step 4: Decompose and compare parts
        parts_equiv, parts_conf = self._compare_decomposed(norm1, norm2)
        if parts_equiv:
            return True, parts_conf, "decomposed_comparison"

        # Step 5: Token-level structural distance
        dist = self._token_edit_distance(alpha1, alpha2)
        max_len = max(len(alpha1), len(alpha2), 1)
        normalized_dist = dist / max_len

        if normalized_dist < 0.05:
            return True, 0.80, f"low_edit_distance({normalized_dist:.3f})"
        elif normalized_dist > 0.5:
            return False, 0.75, f"high_edit_distance({normalized_dist:.3f})"

        return False, 0.40, "structural_inconclusive"

    def _structural_normalize(self, s: str) -> str:
        """
        Normalize structural notation variants without changing semantics.
        This handles the product/sum notation differences.
        """
        s = s.strip()

        # Normalize limit bounds
        # \prod_{n \ge 1} -> \prod_{n=1}^{\infty}
        # \sum_{k \ge 0} -> \sum_{k=0}^{\infty}
        s = re.sub(
            r'(\\(?:prod|sum|bigcup|bigcap))\s*_\s*\{?\s*(\w+)\s*(?:\\ge(?:q)?|>=)\s*(\d+)\s*\}?',
            r'\1_{\2=\3}^{\\infty}',
            s
        )

        # Normalize \infty notation
        s = re.sub(r'\\infinity', r'\\infty', s)

        # Normalize spacing in subscripts/superscripts
        s = re.sub(r'_\s*\{', '_{', s)
        s = re.sub(r'\^\s*\{', '^{', s)

        # Remove \limits (purely visual)
        s = re.sub(r'\\limits\s*', '', s)

        # Normalize multiplication signs
        s = re.sub(r'\\cdot\s*', '*', s)
        s = re.sub(r'\\times\s*', '*', s)

        # Normalize d in integrals: \,dt -> dt, \mathrm{d}t -> dt
        s = re.sub(r'\\,\s*d(\w)', r' d\1', s)
        s = re.sub(r'\\mathrm\{d\}', 'd', s)

        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        return s

    def _alpha_normalize(self, s: str) -> str:
        """
        Alpha-normalize bound variables using de Bruijn-style canonical naming.
        
        Strategy:
        1. Find all binding sites (subscripts of \sum, \prod, \int, etc.)
        2. Extract the bound variable name from each
        3. Replace with canonical names (α₀, α₁, α₂, ...) in order of appearance
        4. Replace all occurrences of each bound variable
        
        This is the core geometric operation: projecting from the space of
        all possible variable namings to a canonical representative.
        """
        # Find binding constructs and their bound variables
        bound_vars = []

        # Pattern: \sum_{VAR=...} or \prod_{VAR=...} or \prod_{VAR \ge ...}
        binding_pattern = re.compile(
            r'(\\(?:sum|prod|int|oint|bigcup|bigcap|coprod|bigoplus|bigotimes))'
            r'\s*_\s*\{?\s*([a-zA-Z])\s*[=\\]'
        )

        for match in binding_pattern.finditer(s):
            var = match.group(2)
            if var not in bound_vars:
                bound_vars.append(var)

        # Also catch integral variables: \int ... dVAR at the end
        int_var_pattern = re.compile(r'\\int.*?\s+d([a-zA-Z])\b')
        for match in int_var_pattern.finditer(s):
            var = match.group(1)
            if var not in bound_vars:
                bound_vars.append(var)

        # Now replace each bound variable with a canonical name
        # Use characters unlikely to collide: ξ₀, ξ₁, etc.
        # But for string comparison, use _BV0_, _BV1_, etc.
        result = s
        for i, var in enumerate(bound_vars):
            canonical = f'_BV{i}_'
            # Replace the variable everywhere it appears as a standalone symbol
            # Be careful not to replace partial matches
            # Also replace in differential form: dVAR -> d_BVi_
            result = re.sub(
                r'(?<![a-zA-Z])d' + re.escape(var) + r'(?![a-zA-Z])',
                f'd{canonical}',
                result
            )
            result = re.sub(
                r'(?<![a-zA-Z_])' + re.escape(var) + r'(?![a-zA-Z_0-9])',
                canonical,
                result
            )

        return result

    def _structural_fingerprint(self, s: str) -> str:
        """
        Extract a structural fingerprint that captures the shape of the
        expression but not the specific variable names or exact formatting.
        
        This is a hash of the expression's skeleton:
        - What commands are used and in what order
        - The nesting structure
        - The numeric constants
        - The operation types
        """
        # Extract command sequence
        commands = re.findall(r'\\[a-zA-Z]+', s)
        cmd_seq = ' '.join(commands)

        # Extract structural delimiters
        delims = re.findall(r'[{}()\[\]_^]', s)
        delim_seq = ''.join(delims)

        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', s)
        num_seq = ','.join(sorted(numbers))

        # Extract operators
        ops = re.findall(r'[+\-*/=<>]', s)
        op_seq = ''.join(sorted(ops))

        return f"C:{cmd_seq}|D:{delim_seq}|N:{num_seq}|O:{op_seq}"

    def _compare_decomposed(self, s1: str, s2: str) -> Tuple[bool, float]:
        """
        Decompose expressions at top-level operators and compare parts.
        
        For example:
          \frac{A}{B} \prod_{...} C  vs  \frac{A}{B} \prod_{...} C'
        
        If the non-binding parts match and binding parts are alpha-equivalent,
        the whole expression is equivalent.
        """
        # Split at major multiplication points (spaces between terms)
        parts1 = self._split_multiplicative(s1)
        parts2 = self._split_multiplicative(s2)

        if len(parts1) != len(parts2):
            return False, 0.0

        if len(parts1) <= 1:
            return False, 0.0

        # For each part, alpha-normalize independently and compare
        matched = 0
        for p1 in parts1:
            an1 = self._alpha_normalize(p1.strip())
            for p2 in parts2:
                an2 = self._alpha_normalize(p2.strip())
                if an1 == an2:
                    matched += 1
                    break

        if matched == len(parts1):
            return True, 0.90

        # Partial match
        if matched > 0 and matched >= len(parts1) - 1:
            return True, 0.75

        return False, 0.0

    def _split_multiplicative(self, s: str) -> List[str]:
        """
        Split expression at top-level multiplicative boundaries.
        E.g., '\frac{2t^2}{1-t^2} \prod_{...} \frac{1}{...}'
        -> ['\frac{2t^2}{1-t^2}', '\prod_{...} \frac{1}{...}']
        """
        parts = []
        depth = 0
        current = []
        i = 0

        while i < len(s):
            c = s[i]
            if c == '{':
                depth += 1
                current.append(c)
            elif c == '}':
                depth -= 1
                current.append(c)
            elif c == ' ' and depth == 0:
                # Check if this is a split point (space between terms)
                chunk = ''.join(current).strip()
                if chunk:
                    # Check if next part starts with a binding construct
                    rest = s[i:].strip()
                    is_binding = any(rest.startswith(bc) for bc in
                                   [r'\prod', r'\sum', r'\int', r'\oint'])
                    if is_binding and chunk:
                        parts.append(chunk)
                        current = []
                    else:
                        current.append(c)
                else:
                    current.append(c)
            else:
                current.append(c)
            i += 1

        final = ''.join(current).strip()
        if final:
            parts.append(final)

        return parts if len(parts) > 1 else [s]

    def _token_edit_distance(self, s1: str, s2: str) -> int:
        """
        Compute edit distance on tokens (not characters).
        Tokens are: commands, symbols, numbers, operators.
        """
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)

        n, m = len(tokens1), len(tokens2)
        if n == 0:
            return m
        if m == 0:
            return n

        # Use a bounded DP for efficiency
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if tokens1[i-1] == tokens2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )

        return dp[n][m]

    def _tokenize(self, s: str) -> List[str]:
        """Tokenize LaTeX string into structural tokens."""
        tokens = []
        i = 0
        while i < len(s):
            if s[i] == '\\':
                # Command
                j = i + 1
                while j < len(s) and s[j].isalpha():
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i].isdigit() or (s[i] == '.' and i+1 < len(s) and s[i+1].isdigit()):
                # Number
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i].isalpha() or s[i] == '_':
                # Identifier (including our _BV0_ canonical names)
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i] in ' \t\n':
                i += 1
            else:
                tokens.append(s[i])
                i += 1
        return tokens
