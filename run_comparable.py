"""
Run GeoVerify against the comparable VerifyBench-style benchmark.
Uses fork-based isolation to handle any sympy hangs.
"""

import signal
import os
import sys
import json
import time

sys.path.insert(0, '/home/claude/geo_verify')

from core import GeoVerifier, EquivalenceResult
from comparable_bench import build_comparable_benchmark, Label


def run_benchmark():
    print("=" * 80)
    print("GeoVerify v0.1 — Comparable VerifyBench Evaluation")
    print("=" * 80)
    print()

    bench = build_comparable_benchmark()
    verifier = GeoVerifier()

    correct = 0
    total = 0
    failed = []
    timed_out = []
    methods_used = {}
    results_by_difficulty = {'easy': [0, 0], 'medium': [0, 0], 'adversarial': [0, 0]}
    results_by_type = {}
    results_by_label = {Label.EQUIVALENT: [0, 0], Label.NOT_EQUIVALENT: [0, 0]}
    timing = []

    for inst in bench:
        total += 1

        r_fd, w_fd = os.pipe()
        t0 = time.time()
        pid = os.fork()

        if pid == 0:
            # Child process
            os.close(r_fd)
            try:
                signal.alarm(8)
                result = verifier.verify(inst.reference, inst.prediction)
                signal.alarm(0)
                eff = result.decision
                if eff == EquivalenceResult.UNCERTAIN:
                    eff = EquivalenceResult.NOT_EQUIVALENT

                expected_equiv = (inst.label == Label.EQUIVALENT)
                got_equiv = (eff == EquivalenceResult.EQUIVALENT)
                ok = 1 if (expected_equiv == got_equiv) else 0

                msg = f'{ok}|{result.decision.value}|{result.method}|{result.confidence}'
                os.write(w_fd, msg.encode())
            except:
                os.write(w_fd, b'0|timeout|timeout|0.0')
            os.close(w_fd)
            os._exit(0)
        else:
            # Parent process
            os.close(w_fd)
            import select
            ready, _, _ = select.select([r_fd], [], [], 10)
            elapsed = time.time() - t0

            if ready:
                data = os.read(r_fd, 4096).decode()
                os.close(r_fd)
                os.waitpid(pid, 0)
                parts = data.split('|')
                ok = int(parts[0])
                decision = parts[1]
                method = parts[2]
                conf = float(parts[3])
            else:
                os.close(r_fd)
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    pass
                os.waitpid(pid, 0)
                timed_out.append(inst.id)
                # Timeout -> assume not equivalent (conservative)
                ok = 1 if inst.label == Label.NOT_EQUIVALENT else 0
                decision = 'timeout'
                method = 'timeout'
                conf = 0.0
                elapsed = 10.0

            timing.append(elapsed)

            if ok:
                correct += 1
            else:
                failed.append(inst)

            # Track statistics
            methods_used[method] = methods_used.get(method, 0) + 1
            results_by_difficulty[inst.difficulty][1] += 1
            if ok:
                results_by_difficulty[inst.difficulty][0] += 1
            results_by_label[inst.label][1] += 1
            if ok:
                results_by_label[inst.label][0] += 1

            ptype = inst.perturbation_type
            if ptype not in results_by_type:
                results_by_type[ptype] = [0, 0]
            results_by_type[ptype][1] += 1
            if ok:
                results_by_type[ptype][0] += 1

    # ========================================
    # Report
    # ========================================
    accuracy = 100 * correct / total

    print()
    print("=" * 80)
    print(f"OVERALL ACCURACY: {correct}/{total} = {accuracy:.2f}%")
    print("=" * 80)

    # Precision / Recall / F1 (matching Meta's Table 3 metrics)
    # True positive: correctly identified as equivalent
    # False positive: said equivalent but actually not
    # False negative: said not equivalent but actually equivalent
    tp = sum(1 for inst in bench
             if inst.label == Label.EQUIVALENT and inst not in failed)
    fp = sum(1 for inst in failed
             if inst.label == Label.NOT_EQUIVALENT)
    fn = sum(1 for inst in failed
             if inst.label == Label.EQUIVALENT)
    tn = sum(1 for inst in bench
             if inst.label == Label.NOT_EQUIVALENT and inst not in failed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {100*precision:.2f}%")
    print(f"Recall:    {100*recall:.2f}%")
    print(f"F1:        {100*f1:.2f}%")

    print(f"\nConfusion Matrix:")
    print(f"  TP (correctly equiv):     {tp}")
    print(f"  TN (correctly non-equiv): {tn}")
    print(f"  FP (false equiv):         {fp}")
    print(f"  FN (false non-equiv):     {fn}")

    print(f"\nBy label:")
    for label, (c, t) in results_by_label.items():
        print(f"  {label.value:20s}: {c}/{t} = {100*c/t:.1f}%")

    print(f"\nBy difficulty:")
    for d, (c, t) in sorted(results_by_difficulty.items()):
        if t > 0:
            print(f"  {d:15s}: {c}/{t} = {100*c/t:.1f}%")

    print(f"\nBy perturbation type:")
    for ptype, (c, t) in sorted(results_by_type.items(), key=lambda x: x[1][1], reverse=True):
        pct = 100 * c / t if t > 0 else 0
        status = "✓" if c == t else f"({t-c} missed)"
        print(f"  {ptype:30s}: {c}/{t} = {pct:.0f}% {status}")

    print(f"\nMethods used:")
    for m, count in sorted(methods_used.items(), key=lambda x: -x[1]):
        print(f"  {m:45s}: {count}")

    if timed_out:
        print(f"\nTimed out: {len(timed_out)} instances")

    print(f"\nTiming: avg={sum(timing)/len(timing)*1000:.0f}ms, "
          f"max={max(timing)*1000:.0f}ms, total={sum(timing):.1f}s")

    print()
    print("=" * 80)
    print("COMPARISON TO META'S VERIFYBENCH BASELINES (Table 3)")
    print("=" * 80)
    print(f"  {'Method':<40s} {'Agreement':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'math-verify (rule-based)':<40s} {'5.95%':>10s} {'5.38':>10s} {'6.67':>8s} {'5.96':>8s}")
    print(f"  {'general-verifier (1.5B)':<40s} {'82.74%':>10s} {'83.13':>10s} {'93.24':>8s} {'87.90':>8s}")
    print(f"  {'CompassVerifier (3B)':<40s} {'81.55%':>10s} {'90.74':>10s} {'65.33':>8s} {'75.97':>8s}")
    print(f"  {'CompassVerifier (7B)':<40s} {'88.69%':>10s} {'93.75':>10s} {'80.00':>8s} {'86.33':>8s}")
    print(f"  {'CompassVerifier (32B)':<40s} {'91.66%':>10s} {'94.20':>10s} {'86.67':>8s} {'90.28':>8s}")
    print(f"  {'Qwen3-4B (prompted)':<40s} {'92.26%':>10s} {'89.74':>10s} {'93.33':>8s} {'91.50':>8s}")
    print(f"  {'Qwen3-14B (prompted)':<40s} {'93.45%':>10s} {'92.21':>10s} {'94.67':>8s} {'93.42':>8s}")
    print(f"  {'GPT-OSS-20B (prompted)':<40s} {'94.64%':>10s} {'95.83':>10s} {'92.00':>8s} {'93.88':>8s}")
    print(f"  {'GPT-OSS-120B (prompted)':<40s} {'95.24%':>10s} {'97.18':>10s} {'92.00':>8s} {'94.52':>8s}")
    print(f"  {'o3 (prompted)':<40s} {'94.05%':>10s} {'93.33':>10s} {'93.33':>8s} {'93.33':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'GeoVerify v0.1 (0 parameters)':<40s} {f'{accuracy:.2f}%':>10s} {f'{100*precision:.2f}':>10s} {f'{100*recall:.2f}':>8s} {f'{100*f1:.2f}':>8s}")

    if failed:
        print(f"\n\nFailed cases ({len(failed)}):")
        for inst in failed[:20]:
            print(f"  [{inst.perturbation_type}] expected={inst.label.value}")
            print(f"    Ref:  {inst.reference[:70]}")
            print(f"    Pred: {inst.prediction[:70]}")
            print()


if __name__ == "__main__":
    run_benchmark()
