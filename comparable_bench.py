"""
Comparable Benchmark for GeoVerify
====================================

Meta's VerifyBench methodology:
1. Take real mathematical object answers from PrincipiaBench
2. Generate model predictions (which may be equivalent in different form)
3. Find cases where math-verify and o3 disagree
4. Have humans label the ground truth

Our comparable methodology:
1. Use real mathematical object answers from PrincipiaBench (sampled from
   the dataset viewer)
2. Apply systematic perturbations that mirror what real models produce:
   - Surface-form equivalences (notation, ordering, variable naming)
   - Non-equivalences (wrong coefficients, wrong signs, wrong functions)
3. Include both easy and adversarial cases
4. Ground truth is known by construction

Key design principle: we weight the test set toward adversarial cases
(the kind that math-verify gets wrong) to match VerifyBench's adversarial
sampling. VerifyBench is NOT a random sample - it's specifically cases
where verifiers disagree. So our benchmark should be similarly hard.

We construct ~170 instances to match VerifyBench's 168.
"""

import random
import re
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class Label(Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"


@dataclass
class BenchInstance:
    id: int
    reference: str
    prediction: str
    label: Label
    perturbation_type: str
    difficulty: str  # 'easy', 'medium', 'adversarial'
    source_domain: str  # 'physics', 'math', 'engineering'
    description: str


# ============================================================
# Real mathematical object answers from PrincipiaBench
# (sampled from the HuggingFace dataset viewer)
# ============================================================

REAL_ANSWERS = [
    # Physics expressions
    (r"\frac{\nu_e - \nu_s}{\nu_s} \approx \frac{\Phi(\mathbf{r_s}) - \Phi(\mathbf{r_e})}{c^2}", "physics"),
    (r"\frac{2T^2 + 4Tmc^2}{mc^2}", "physics"),
    (r"v = \frac{\sqrt{3}}{2} c", "physics"),
    (r"m = m_0 \sqrt{\frac{c - v}{c + v}}", "physics"),
    (r"\sigma = \pi R^2 \left(1 + \frac{2GM}{V_0^2 R}\right)", "physics"),
    (r"\mathbf{J} = \mathbf{J_0} e^{-\frac{\lambda}{m} t}", "physics"),
    (r"\omega' = \sqrt{3\omega^2 + \frac{K}{m}}", "physics"),
    (r"v_0 = R \omega_0", "physics"),
    (r"V(r) = \frac{-m h^2}{2r^2}", "physics"),
    (r"y = \frac{\omega}{V}(R - x)^2", "physics"),
    (r"U = -m\omega (x\dot{y} - \dot{x}y) - \frac{1}{2} m\omega^2 (x^2 + y^2)", "physics"),
    (r"\frac{Q^2}{32\pi \epsilon_0 R^2}", "physics"),
    (r"C = \frac{\pi}{\ln \frac{2b}{a}}", "physics"),
    (r"\frac{\varepsilon_0 \pi r^2 (d' - d) V^2}{2d^2}", "physics"),
    (r"I = \frac{4 \pi K V^2}{\ln(b/a)}", "physics"),
    (r"\delta = \sqrt{\frac{2}{\omega \mu \sigma}}", "physics"),
    (r"R = \left( \frac{1 - \sqrt{\varepsilon}}{1 + \sqrt{\varepsilon}} \right)^2", "physics"),
    (r"B = \frac{\mu_r \mu_0 NI}{2\pi R + (\mu_r - 1)s}", "physics"),
    (r"q = \frac{B \pi a^2}{R}", "physics"),
    (r"\dot{S} = k \left(\frac{\nabla T}{T}\right)^2", "physics"),
    (r"\rho(E) = \frac{2}{\sqrt{\pi}(kT)^{3/2}} E^{1/2} e^{-E/kT}", "physics"),
    (r"p_1(t) = \frac{p_0}{2} \left(1 + e^{-\frac{Avt}{2V}}\right)", "physics"),
    (r"c_1 = \frac{N k \ln 2}{1 - \ln 2}", "physics"),
    (r"C_p = \frac{5}{2} R", "physics"),
    (r"\Delta E = \frac{2e^2 R^2}{3a_0^3}", "physics"),
    (r"-\frac{q^2 E^2}{2m\omega^2}", "physics"),
    (r"\frac{3}{2} \hbar \omega - \frac{3 \lambda^2}{2m \omega^2}", "physics"),
    (r"E = \frac{5}{4}m_t c^2", "physics"),
    (r"W = C\left(\sqrt{\tau_h} - \sqrt{\tau_l}\right)^2", "physics"),
    (r"T = T_0", "physics"),
    # Math expressions
    (r"\frac{2t^2}{1 - t^2} \prod_{n \geq 1} \frac{1}{1 - t^n}", "math"),
    (r"\sum_{k=0}^{n} \binom{n}{k} x^k", "math"),
    (r"\frac{x^2 - 1}{x + 1}", "math"),
    (r"(x+1)^2 + (y-1)^2 = r^2", "math"),
    (r"\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}", "math"),
]


# ============================================================
# Perturbation generators
# Each produces (perturbed_string, label, difficulty)
# ============================================================

def perturb_reorder_terms(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Reorder additive terms, respecting LaTeX grouping (braces)."""
    results = []

    # Split at top-level + signs only (not inside braces)
    parts = []
    depth = 0
    current = []
    for ch in expr:
        if ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
        elif ch == '+' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))

    if len(parts) >= 2 and all(p.strip() for p in parts):
        shuffled = parts[::-1]
        new_expr = '+'.join(shuffled)
        # Verify the result looks like valid LaTeX (balanced braces)
        if new_expr.count('{') == new_expr.count('}'):
            results.append((new_expr, Label.EQUIVALENT,
                          'adversarial', 'term_reorder_additive'))

    return results


def perturb_reorder_multiplicative(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Reorder multiplicative factors in denominators/products."""
    results = []

    # Swap factors in denominator
    # e.g., 2m\omega^2 -> 2\omega^2 m
    match = re.search(r'(\d+)([a-z])\\([a-z]+)', expr)
    if match:
        new = expr[:match.start()] + match.group(1) + '\\' + match.group(3) + ' ' + match.group(2) + expr[match.end():]
        results.append((new, Label.EQUIVALENT, 'adversarial', 'factor_reorder'))

    return results


def perturb_variable_rename(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Rename bound/dummy variables."""
    results = []

    # Sum index rename
    for old_var, new_var in [('n', 'm'), ('k', 'j'), ('i', 'l'), ('t', 's')]:
        if f'_{{{old_var}=' in expr or f'_{old_var}' in expr:
            renamed = re.sub(
                r'(?<![a-zA-Z])' + old_var + r'(?![a-zA-Z])',
                new_var, expr
            )
            if renamed != expr:
                results.append((renamed, Label.EQUIVALENT,
                              'adversarial', f'var_rename_{old_var}_to_{new_var}'))
                break

    return results


def perturb_notation_variant(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Change notation without changing meaning."""
    results = []

    # \frac{a}{b} -> (a)/(b) — only for simple fractions without nested \frac
    if r'\frac{' in expr:
        # Find outermost \frac and replace only if contents are simple
        i = expr.find(r'\frac{')
        if i >= 0:
            # Extract numerator (balanced braces)
            j = i + 6  # after \frac{
            depth = 1
            while j < len(expr) and depth > 0:
                if expr[j] == '{': depth += 1
                if expr[j] == '}': depth -= 1
                j += 1
            num = expr[i+6:j-1]

            # Extract denominator
            if j < len(expr) and expr[j] == '{':
                k = j + 1
                depth = 1
                while k < len(expr) and depth > 0:
                    if expr[k] == '{': depth += 1
                    if expr[k] == '}': depth -= 1
                    k += 1
                den = expr[j+1:k-1]

                # Only replace if no nested \frac in num or den
                if r'\frac' not in num and r'\frac' not in den:
                    new = expr[:i] + f'\\dfrac{{{num}}}{{{den}}}' + expr[k:]
                    results.append((new, Label.EQUIVALENT, 'medium', 'frac_to_dfrac'))

    # e^{x} -> \exp(x)
    m = re.search(r'e\^\{([^{}]+)\}', expr)
    if m:
        new = expr[:m.start()] + f'\\exp({m.group(1)})' + expr[m.end():]
        results.append((new, Label.EQUIVALENT, 'medium', 'exp_notation'))

    # \sqrt{x} -> x^{1/2}
    m = re.search(r'\\sqrt\{([^{}]+)\}', expr)
    if m:
        content = m.group(1)
        # Only if content is simple (no nested braces)
        if '{' not in content:
            new = expr[:m.start()] + f'{{{content}}}^{{1/2}}' + expr[m.end():]
            results.append((new, Label.EQUIVALENT, 'medium', 'sqrt_to_power'))

    # \ge -> \geq  (trivial but tests robustness)
    if r'\ge ' in expr or r'\geq' in expr:
        new = expr.replace(r'\ge ', r'\geq ')
        new = new.replace(r'\geq', r'\ge')
        if new != expr:
            results.append((new, Label.EQUIVALENT, 'easy', 'ge_geq'))

    # Product notation: \prod_{n \geq 1} -> \prod_{n=1}^{\infty}
    m = re.search(r'\\prod_\{?(\w+)\s*\\geq?\s*(\d+)\}?', expr)
    if m:
        new = expr[:m.start()] + f'\\prod_{{{m.group(1)}={m.group(2)}}}^{{\\infty}}' + expr[m.end():]
        results.append((new, Label.EQUIVALENT, 'adversarial', 'prod_notation'))

    return results


def perturb_latex_formatting(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Change LaTeX formatting without changing meaning."""
    results = []

    # Add \left \right
    if '(' in expr and r'\left' not in expr:
        new = expr.replace('(', r'\left(').replace(')', r'\right)')
        results.append((new, Label.EQUIVALENT, 'easy', 'add_left_right'))

    # \mathrm{} around certain symbols
    if 'i' in expr and r'\mathrm' not in expr:
        new = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z])', r'\\mathrm{i}', expr, count=1)
        if new != expr:
            results.append((new, Label.EQUIVALENT, 'adversarial', 'mathrm_i'))

    # \cdot for multiplication
    # Insert \cdot between adjacent terms
    new = re.sub(r'(\})(\s*)([a-zA-Z])', r'\1 \\cdot \3', expr, count=1)
    if new != expr:
        results.append((new, Label.EQUIVALENT, 'easy', 'add_cdot'))

    # Add text wrapper
    results.append((f'The answer is {expr}', Label.EQUIVALENT,
                   'medium', 'text_wrapper'))

    return results


def perturb_wrong_coefficient(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Change a coefficient to make it wrong."""
    results = []

    # Find a number and change it
    numbers = list(re.finditer(r'(?<![a-zA-Z])(\d+)(?![a-zA-Z\d])', expr))
    if numbers:
        m = numbers[0]
        old_num = int(m.group(1))
        new_num = old_num + 1 if old_num > 0 else 2
        new = expr[:m.start()] + str(new_num) + expr[m.end():]
        results.append((new, Label.NOT_EQUIVALENT, 'medium',
                       f'wrong_coeff_{old_num}_to_{new_num}'))

    return results


def perturb_wrong_sign(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Flip a sign to make it wrong."""
    results = []

    if '+' in expr:
        new = expr.replace('+', '-', 1)
        results.append((new, Label.NOT_EQUIVALENT, 'medium', 'flip_plus_to_minus'))

    if '- ' in expr:
        new = expr.replace('- ', '+ ', 1)
        results.append((new, Label.NOT_EQUIVALENT, 'medium', 'flip_minus_to_plus'))

    return results


def perturb_wrong_function(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Change a function to a different one."""
    results = []

    replacements = [
        (r'\sin', r'\cos'), (r'\cos', r'\sin'),
        (r'\ln', r'\log_{10}'),
        (r'\sqrt', r'\cbrt'),
        (r'e^{', r'2^{'),
    ]
    for old, new_func in replacements:
        if old in expr:
            new = expr.replace(old, new_func, 1)
            results.append((new, Label.NOT_EQUIVALENT, 'medium',
                           f'wrong_func_{old}_to_{new_func}'))
            break

    return results


def perturb_wrong_exponent(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Change an exponent."""
    results = []

    m = re.search(r'\^{?(\d+)}?', expr)
    if m:
        old = int(m.group(1))
        new_exp = old + 1
        if m.group(0).startswith('^{'):
            new = expr[:m.start()] + f'^{{{new_exp}}}' + expr[m.end():]
        else:
            new = expr[:m.start()] + f'^{new_exp}' + expr[m.end():]
        results.append((new, Label.NOT_EQUIVALENT, 'medium',
                       f'wrong_exp_{old}_to_{new_exp}'))

    return results


def perturb_missing_term(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Remove a term to make it wrong."""
    results = []

    # Remove a term after +
    m = re.search(r'\s*\+\s*[^+\-]+', expr)
    if m and m.end() < len(expr):
        new = expr[:m.start()] + expr[m.end():]
        if len(new.strip()) > 3:
            results.append((new, Label.NOT_EQUIVALENT, 'adversarial', 'missing_term'))

    return results


def perturb_extra_constant(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Add a multiplicative constant to make it wrong."""
    results = []
    results.append((f'2 {expr}', Label.NOT_EQUIVALENT, 'adversarial', 'extra_factor_2'))
    results.append((f'-{expr}', Label.NOT_EQUIVALENT, 'medium', 'negated'))
    return results


def perturb_algebraic_equivalent(expr: str) -> List[Tuple[str, Label, str, str]]:
    """Create algebraically equivalent forms."""
    results = []

    # For simple fractions, multiply top and bottom by same thing
    m = re.match(r'\\frac\{(.+)\}\{(.+)\}$', expr.strip())
    if m:
        num, den = m.group(1), m.group(2)
        # Not a true algebraic equivalence at string level, but tests the verifier
        results.append((f'\\frac{{{num} \\cdot 1}}{{{den}}}', Label.EQUIVALENT,
                       'easy', 'multiply_by_one'))

    return results


def build_comparable_benchmark() -> List[BenchInstance]:
    """
    Build a benchmark that is comparable to Meta's VerifyBench.

    VerifyBench composition (from paper):
    - 168 instances total
    - 75 labeled "Equivalent" (44.6%)
    - 93 labeled "Not Equivalent" (55.4%)
    - Adversarially selected (cases where math-verify and o3 disagree)

    Our target:
    - ~170 instances
    - ~45% equivalent, ~55% not equivalent (matching their ratio)
    - Heavy weighting toward adversarial cases
    """
    instances = []
    id_counter = 0

    equiv_generators = [
        perturb_reorder_terms,
        perturb_reorder_multiplicative,
        perturb_variable_rename,
        perturb_notation_variant,
        perturb_latex_formatting,
        perturb_algebraic_equivalent,
    ]

    non_equiv_generators = [
        perturb_wrong_coefficient,
        perturb_wrong_sign,
        perturb_wrong_function,
        perturb_wrong_exponent,
        perturb_missing_term,
        perturb_extra_constant,
    ]

    # Generate equivalences
    for answer, domain in REAL_ANSWERS:
        for gen in equiv_generators:
            perturbations = gen(answer)
            for pred, label, difficulty, ptype in perturbations:
                id_counter += 1
                instances.append(BenchInstance(
                    id=id_counter,
                    reference=answer,
                    prediction=pred,
                    label=label,
                    perturbation_type=ptype,
                    difficulty=difficulty,
                    source_domain=domain,
                    description=f"{ptype} on {domain} expr"
                ))

    # Generate non-equivalences
    for answer, domain in REAL_ANSWERS:
        for gen in non_equiv_generators:
            perturbations = gen(answer)
            for pred, label, difficulty, ptype in perturbations:
                id_counter += 1
                instances.append(BenchInstance(
                    id=id_counter,
                    reference=answer,
                    prediction=pred,
                    label=label,
                    perturbation_type=ptype,
                    difficulty=difficulty,
                    source_domain=domain,
                    description=f"{ptype} on {domain} expr"
                ))

    # Target ratio: ~45% equiv, ~55% non-equiv
    equiv = [i for i in instances if i.label == Label.EQUIVALENT]
    non_equiv = [i for i in instances if i.label == Label.NOT_EQUIVALENT]

    print(f"Generated: {len(equiv)} equivalent, {len(non_equiv)} non-equivalent")
    print(f"Total: {len(instances)}")

    # If we have too many, subsample to ~170 maintaining ratio
    random.seed(42)
    target_total = 170
    target_equiv = int(target_total * 0.45)
    target_non_equiv = target_total - target_equiv

    if len(equiv) > target_equiv:
        # Prioritize adversarial cases
        adv_equiv = [i for i in equiv if i.difficulty == 'adversarial']
        other_equiv = [i for i in equiv if i.difficulty != 'adversarial']
        random.shuffle(other_equiv)
        equiv = adv_equiv + other_equiv
        equiv = equiv[:target_equiv]

    if len(non_equiv) > target_non_equiv:
        adv_non = [i for i in non_equiv if i.difficulty == 'adversarial']
        other_non = [i for i in non_equiv if i.difficulty != 'adversarial']
        random.shuffle(other_non)
        non_equiv = adv_non + other_non
        non_equiv = non_equiv[:target_non_equiv]

    final = equiv + non_equiv
    random.shuffle(final)

    # Reassign IDs
    for i, inst in enumerate(final):
        inst.id = i + 1

    print(f"Final benchmark: {len(final)} instances")
    print(f"  Equivalent: {sum(1 for i in final if i.label == Label.EQUIVALENT)}")
    print(f"  Not equivalent: {sum(1 for i in final if i.label == Label.NOT_EQUIVALENT)}")

    # Difficulty breakdown
    for d in ['easy', 'medium', 'adversarial']:
        count = sum(1 for i in final if i.difficulty == d)
        print(f"  {d}: {count}")

    return final


if __name__ == "__main__":
    bench = build_comparable_benchmark()

    # Show some examples
    print("\nSample instances:")
    for inst in bench[:5]:
        print(f"\n  [{inst.perturbation_type}] {inst.label.value}")
        print(f"  Ref:  {inst.reference[:80]}")
        print(f"  Pred: {inst.prediction[:80]}")
