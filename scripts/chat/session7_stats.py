"""Pool session-7's seed-replicate results per `docs/chat/PREDICTION_v7.md`
§3: Welch's t-test on n=4 per-seed headline/component values per recipe,
read from the committed `_quality/*.json` files -- never from a remembered
or hand-transcribed summary, the rule `docs/chat/CONVENTIONS.md` exists to
enforce. Every number in `docs/chat/QUALITY_v7.md` §2-3 traces to this
script's output.

Stdlib-only (no `scipy` on this machine): implements the regularized
incomplete beta function (Numerical Recipes' betacf/betai) to get an exact
Welch-Satterthwaite p-value instead of reading off a critical-value table.

Not part of the research protocol. Reads only under `experiments/chat/`.

    python scripts/chat/session7_stats.py
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q = REPO / "experiments" / "chat" / "_quality"

V3D = ["chat-v3d-aligned", "chat-v3d-aligned-s1", "chat-v3d-aligned-s2", "chat-v3d-aligned-s3"]
INSTA = ["chat-v6-inst-a", "chat-v6-inst-a-s1", "chat-v6-inst-a-s2", "chat-v6-inst-a-s3"]
FIELDS = ["headline", "topic", "list", "social", "fallback_rate"]


def load_row(label):
    d = json.loads((Q / f"{label}.json").read_text(encoding="utf-8"))
    r = d["report"]
    row = next(s for s in r["sweep"] if s["n"] == 1)
    out = {"headline": row["headline"], "fallback_rate": row["fallback_rate"]}
    for k in ("topic", "list", "social"):
        out[k] = row["by_kind"].get(k, 0.0)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# --- regularized incomplete beta function (Numerical Recipes betai/betacf) ---
def _betacf(a, b, x, maxit=200, eps=3e-12, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch(xs, ys):
    n1, n2 = len(xs), len(ys)
    m1, m2 = mean(xs), mean(ys)
    s1, s2 = sd(xs), sd(ys)
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    se = math.sqrt(v1 + v2)
    t = (m1 - m2) / se if se > 0 else float("inf")
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1)) if (v1 + v2) > 0 else float("nan")
    x = df / (df + t * t) if math.isfinite(df) and math.isfinite(t) else 0.0
    p = betai(df / 2.0, 0.5, x) if math.isfinite(df) else float("nan")
    return {"n1": n1, "n2": n2, "mean1": m1, "mean2": m2, "sd1": s1, "sd2": s2,
            "diff": m1 - m2, "se_diff": se, "t": t, "df": df, "p": p}


def main():
    v3d_rows = {lbl: load_row(lbl) for lbl in V3D}
    insta_rows = {lbl: load_row(lbl) for lbl in INSTA}

    print("=== per-seed values ===")
    for lbl, row in {**v3d_rows, **insta_rows}.items():
        print(f"{lbl:28s} " + " ".join(f"{k}={row[k]:.4f}" for k in FIELDS))

    print("\n=== Welch t-test, v3d-aligned recipe (n=4) vs chat-v6-inst-a recipe (n=4) ===")
    for field in FIELDS:
        xs = [v3d_rows[l][field] for l in V3D]
        ys = [insta_rows[l][field] for l in INSTA]
        r = welch(xs, ys)
        verdict = "RESOLVED" if r["p"] < 0.05 else "unresolved"
        print(f"{field:14s} v3d mean={r['mean1']:.4f} sd={r['sd1']:.4f} | "
              f"insta mean={r['mean2']:.4f} sd={r['sd2']:.4f} | "
              f"diff(insta-v3d)={-r['diff']:.4f} se={r['se_diff']:.4f} "
              f"t={r['t']:.3f} df={r['df']:.2f} p={r['p']:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
