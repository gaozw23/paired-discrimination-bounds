"""Independent P3.1 oracles; no production rank, prefix, root or extrema imports.

Binary expected values use only direct positive/negative score comparisons.
Global Decimal expectations interpolate independently enumerated binary extrema.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product


def compare_twice(x, y):
    return 2 if x > y else 1 if x == y else 0


def direct_binary(a, b, y):
    n = len(a)
    m = sum(y)
    if m in (0, n):
        return None
    numerator_twice = sum(
        compare_twice(a[i], a[j]) - compare_twice(b[i], b[j])
        for i in range(n) if y[i]
        for j in range(n) if not y[j]
    )
    return F(numerator_twice, 2 * m * (n - m))


def binary_completions(a, b, r, observed):
    unknown = [i for i in range(len(a)) if not r[i]]
    values = {}
    for bits in product((0, 1), repeat=len(unknown)):
        y = list(observed)
        for i, bit in zip(unknown, bits):
            y[i] = bit
        m = sum(y)
        value = direct_binary(a, b, y)
        values.setdefault(m, []).append((value, y))
    return values






def as_decimal(x):
    x = F(x)
    return Decimal(x.numerator) / Decimal(x.denominator)


def global_oracle(a, b, r, observed, intervals):
    """80-digit independent segment search from enumerated binary pairs.

    Per frozen API contract, a float exactly equal to float(m/n) denotes that
    rational knot. Other floats denote their exact binary values. nextafter is
    never epsilon-snapped. This is input interpretation, not extrema sharing.
    """
    with localcontext() as context:
        context.prec = 80
        n = len(a)
        p = sum(int(observed[i]) for i in range(n) if r[i])
        u = sum(not value for value in r)
        feasible_lo, feasible_hi = F(p, n), F(p+u, n)
        def canonical_input(x):
            if isinstance(x, float):
                m = round(n*x)
                if x == m/n:
                    return F(m, n)
            return F(x)
        components = []
        for raw_lo, raw_hi in intervals:
            lo, hi = max(canonical_input(raw_lo), feasible_lo), min(canonical_input(raw_hi), feasible_hi)
            if lo <= hi and hi > 0 and lo < 1:
                components.append((lo, hi))
        if not components:
            return {"status": "empty", "lower": None, "upper": None, "candidates": []}
        completions = binary_completions(a, b, r, observed)
        if not u:
            value = direct_binary(a, b, [int(x) for x in observed])
            return {"status": "ok", "lower": float(value), "upper": float(value),
                    "lower_attained": True, "upper_attained": True, "candidates": []}
        knots = {}
        for m, group in completions.items():
            vals = [F(0) if value is None else value * F(m*(n-m), n*n) for value, _ in group]
            knots[m] = (min(vals), max(vals))
        candidates = []
        for branch_index, branch in enumerate(("lower", "upper")):
            for k in range(u):
                x0, x1 = F(p+k, n), F(p+k+1, n)
                v0, v1 = knots[p+k][branch_index], knots[p+k+1][branch_index]
                slope = (v1-v0)/(x1-x0)
                intercept = v0-slope*x0
                da, db = as_decimal(intercept), as_decimal(slope)
                for component, (domain_lo, domain_hi) in enumerate(components):
                    lo, hi = max(x0, domain_lo), min(x1, domain_hi)
                    if lo > hi or hi <= 0 or lo >= 1:
                        continue
                    roots = []
                    if slope == 0:
                        if intercept:
                            roots = [Decimal("0.5")]
                        elif lo < hi:
                            roots = [(as_decimal(lo)+as_decimal(hi))/2]
                    else:
                        disc = 4*da*da+4*da*db
                        if disc >= 0:
                            roots = [(-2*da+disc.sqrt())/(2*db), (-2*da-disc.sqrt())/(2*db)]
                    points = [(as_decimal(lo), "boundary"), (as_decimal(hi), "boundary")]
                    points += [(root, "constant" if not slope and not intercept else "stationary")
                               for root in roots if as_decimal(lo) < root < as_decimal(hi)]
                    for x, kind in points:
                        if x == 0:
                            if intercept:
                                raise AssertionError("nonzero numerator at zero")
                            value, attained, kind = db, False, "limit_0"
                        elif x == 1:
                            if intercept+slope:
                                raise AssertionError("nonzero numerator at one")
                            value, attained, kind = -db, False, "limit_1"
                        else:
                            value, attained = (da+db*x)/(x*(1-x)), True
                        candidates.append({"branch": branch, "segment": k, "component": component,
                                           "pi": str(x), "value": str(value), "attained": attained,
                                           "kind": kind, "a": str(intercept), "b": str(slope)})
        answer = {"status": "ok", "candidates": candidates}
        for branch in ("lower", "upper"):
            selected = [row for row in candidates if row["branch"] == branch]
            extreme = (min if branch == "lower" else max)(Decimal(row["value"]) for row in selected)
            # 80-digit rounding may differ in the last few digits across exact
            # representations of the same rational value; this is not float atol.
            winners = [row for row in selected if abs(Decimal(row["value"])-extreme) <= Decimal("1e-70")]
            answer[branch] = float(extreme)
            answer[branch+"_decimal"] = str(extreme)
            answer[branch+"_attained"] = any(row["attained"] for row in winners)
            answer[branch+"_winners"] = winners
        return answer
