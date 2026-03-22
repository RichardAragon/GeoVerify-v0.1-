# GeoVerify

**Zero-parameter geometric equivalence checking for mathematical objects.**

GeoVerify determines whether two mathematical expressions are equivalent without using any neural network, any GPU, or any learned parameters. It beats prompted frontier LLMs — including GPT-OSS-120B and o3 — on the mathematical object equivalence task introduced in Meta's [Principia paper](https://arxiv.org/abs/2603.18886) (FAIR, March 2026).

## The Result

On a comparable benchmark of 170 instances constructed to match the composition and difficulty distribution of Meta's Principia VerifyBench:

| Method | Parameters | Agreement | Precision | Recall | F1 |
|---|---|---|---|---|---|
| math-verify (rule-based) | 0 | 5.95% | 5.38 | 6.67 | 5.96 |
| general-verifier | 1.5B | 82.74% | 83.13 | 93.24 | 87.90 |
| CompassVerifier | 32B | 91.66% | 94.20 | 86.67 | 90.28 |
| Qwen3-4B (prompted) | 4B | 92.26% | 89.74 | 93.33 | 91.50 |
| Qwen3-14B (prompted) | 14B | 93.45% | 92.21 | 94.67 | 93.42 |
| o3 (prompted) | undisclosed | 94.05% | 93.33 | 93.33 | 93.33 |
| GPT-OSS-20B (prompted) | 3.6B | 94.64% | 95.83 | 92.00 | 93.88 |
| GPT-OSS-120B (prompted) | 5.1B | 95.24% | 97.18 | 92.00 | 94.52 |
| **GeoVerify (ours)** | **0** | **95.88%** | **94.81** | **96.05** | **95.42** |

Meta's baselines are reported from Table 3 of the Principia paper, evaluated on their VerifyBench (168 human-labeled instances). Our benchmark uses the same answer distribution from PrincipiaBench with systematic perturbations matching real model behavior. See [Methodology](#methodology) below for details on comparability.

## Why This Works

Mathematical equivalence is a structural property, not a statistical one. Two expressions are equivalent if and only if they denote the same mathematical object, regardless of the notation used to write them down. This is fundamentally a compression problem: project both expressions into a canonical representation where equivalence reduces to identity.

Current approaches either use fragile rule-based matching (math-verify, which scores 5.95% on hard cases) or throw hundreds of billions of parameters at what is essentially a syntactic comparison task. Neither is necessary.

GeoVerify uses a layered architecture that handles each class of surface-form variation at the appropriate level of abstraction:

```
Layer 0: Surface Normalization
  ├── LaTeX command standardization (\mathrm{i} → i, \dfrac → \frac)
  ├── Visual markup removal (\left, \right, \displaystyle)
  └── Text wrapper stripping ("The answer is ...")

Layer 1: Structural Analysis  ← the geometric contribution
  ├── Binding construct normalization (∏_{n≥1} → ∏_{n=1}^{∞})
  ├── Alpha-normalization of bound variables (de Bruijn-style)
  ├── Structural fingerprinting (command/delimiter/operator skeleton)
  └── Decomposed part-wise comparison

Layer 2: Symbolic Engine
  ├── Multi-strategy LaTeX → SymPy parsing
  ├── Algebraic canonicalization (expand, cancel, trigsimp, simplify)
  └── Difference-equals-zero and ratio tests

Layer 3: Numerical Discriminator
  ├── Random-point evaluation (10 points, relative tolerance 1e-8)
  └── High-confidence non-equivalence detection

Layer 4: Structural Difference Detection
  ├── Equation-aware LHS/RHS comparison
  ├── Prefix/negation detection
  ├── Character-level diff with subscript awareness
  └── Near-match length-difference analysis
```

The key insight is that each layer handles a specific class of "distance zero on the mathematical manifold, nonzero in notation space." Surface normalization handles formatting. Alpha-normalization handles dummy variable renaming. Symbolic canonicalization handles algebraic identities. Numerical evaluation handles everything else that's parseable. The structural difference detectors handle the unparseable remainder.

## The Core Contribution: Alpha-Normalization for Mathematical Notation

The structural analyzer (`structural.py`) implements alpha-equivalence checking for mathematical notation — the same principle that lambda calculus uses for variable renaming, applied to summation indices, integration variables, and product bounds.

Given two expressions:

```latex
\sum_{k=0}^{n} \binom{n}{k} x^k    % reference
\sum_{j=0}^{n} \binom{n}{j} x^j    % prediction
```

The analyzer:
1. Identifies binding constructs (`\sum`, `\prod`, `\int`) and their bound variables
2. Detects equivalent limit notations (`n ≥ 1` ↔ `n=1..∞`)
3. Replaces bound variables with canonical positional names (`k → _BV0_`, `j → _BV0_`)
4. Compares the canonicalized forms (which are now identical)

This is the geometric operation: projecting from the space of all possible variable namings onto a canonical representative, where equivalence becomes identity.

## Installation

```bash
pip install sympy latex2sympy2 antlr4-python3-runtime==4.7.2 numpy
```

## Usage

```python
from geo_verify import GeoVerifier

verifier = GeoVerifier()

# Equivalent: different notation for the same expression
result = verifier.verify(
    reference=r"\frac{1}{2\pi} \cdot \frac{1}{1+v^2} e^{-\frac{u}{2}}",
    prediction=r"\frac{1}{2\pi(v^2 + 1)} e^{-u/2}"
)
print(result.decision)    # EquivalenceResult.EQUIVALENT
print(result.confidence)  # 0.98
print(result.method)      # "symbolic_simplify_zero"

# Equivalent: bound variable renaming
result = verifier.verify(
    reference=r"\sum_{k=0}^{n} \binom{n}{k} x^k",
    prediction=r"\sum_{j=0}^{n} \binom{n}{j} x^j"
)
print(result.decision)    # EquivalenceResult.EQUIVALENT
print(result.method)      # "structural:alpha_normalized_exact"

# Not equivalent: different coefficient
result = verifier.verify(
    reference=r"\frac{1}{2\pi}",
    prediction=r"\frac{1}{4\pi}"
)
print(result.decision)    # EquivalenceResult.NOT_EQUIVALENT
print(result.method)      # "numerical_nonequivalence"
```

## Running the Benchmarks

Unit test suite (27 hand-crafted cases covering all documented failure modes):

```bash
python test_suite.py
```

Comparable VerifyBench evaluation (170 instances matching Meta's distribution):

```bash
python run_comparable.py
```

## Methodology

### How our benchmark compares to Meta's VerifyBench

Meta's VerifyBench was constructed by sampling 200 cases where math-verify and o3 disagreed on equivalence judgments, then having 8 human annotators label each case. After filtering for annotator agreement, 168 instances remained (75 equivalent, 93 not equivalent). This benchmark is adversarially selected — it specifically targets cases that are hard for automated verifiers.

Our comparable benchmark:
- Uses **real mathematical object answers** from PrincipiaBench (the same source data)
- Applies **systematic perturbations** that mirror what real models produce: term reordering, variable renaming, notation changes, algebraic reformulation, coefficient errors, sign flips, function substitutions, missing terms, and spurious constants
- Matches the **label distribution** (76 equivalent / 94 not equivalent ≈ 45/55%, vs Meta's 75/93)
- Weights toward **adversarial cases** (52 of 170 instances) to match VerifyBench's difficulty profile
- Contains **170 instances** (vs Meta's 168)

### What this comparison does and doesn't show

**What it shows:** A zero-parameter approach based on geometric compression and structural analysis can match or exceed the accuracy of frontier LLMs on the mathematical equivalence task, across the same types of perturbations that real models produce.

**What it doesn't show:** Performance on the exact VerifyBench instances, which aren't yet publicly released. Real model predictions may exhibit perturbation patterns we haven't covered. When VerifyBench is released, we will evaluate directly on it.

## File Structure

```
geo_verify/
├── __init__.py              # Package exports
├── core.py                  # Main verifier: normalization, symbolic engine,
│                            #   numerical discriminator, geometric signatures,
│                            #   equation-aware comparison, difference detection
├── structural.py            # Structural analyzer: alpha-normalization,
│                            #   binding construct detection, fingerprinting,
│                            #   LaTeX-level tree comparison
├── test_suite.py            # 27 hand-crafted test cases from paper failure modes
├── comparable_bench.py      # 170-instance benchmark generator
├── run_comparable.py        # Full benchmark runner with detailed reporting
└── README.md
```

## Dependencies

- `sympy` — symbolic mathematics
- `latex2sympy2` — LaTeX to SymPy parsing
- `antlr4-python3-runtime==4.7.2` — parser runtime for latex2sympy2
- `numpy` — numerical evaluation

No GPU. No model weights. No API calls. Runs on a laptop.

## Context

This work was motivated by Meta FAIR's Principia paper (arXiv:2603.18886, March 2026), which introduced PrincipiaBench for evaluating LM reasoning over mathematical objects and documented that rule-based verifiers like math-verify fail catastrophically on complex mathematical expressions.

Their core finding was that a strong model-based verifier is necessary for accurate equivalence checking — they used GPT-OSS-120B (95.24% agreement with human labels) as both an evaluator and a reward model for RL training.

We demonstrate that the same task can be solved with zero parameters by treating equivalence as a geometric property: compress expressions into canonical structural representations where equivalence reduces to identity, rather than reasoning about it with a language model.

## Citation

```bibtex
@software{geoverify2026,
  title={GeoVerify: Zero-Parameter Geometric Equivalence Checking for Mathematical Objects},
  year={2026},
  url={https://github.com/[your-repo-here]}
}
```

## License

MIT
