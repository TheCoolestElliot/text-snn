"""Score the v15 recall round against the bars fixed before it was trained.

NOT PART OF THE RESEARCH PROTOCOL. `snnchat` is outside the Phase-1..5 study.
ZERO GPU: this file samples nothing and loads no checkpoint. It reads the
artifacts `session15_driver.py` leaves behind and fires the bars. It adopts
nothing and edits `experiments/chat/SHIPPED` under no outcome: a ship-candidate
is a recommendation to the owner, with transcripts.

ORDERING, ENFORCED RATHER THAN PROMISED
---------------------------------------
Every threshold below is a constant at the top of this file, fixed by
`docs/chat/PREDICTION_v15.md`. `session15_driver.py` refuses to start unless
this file, that document, `snnchat/recall.py` and `memory_probe_v2.py` are
committed and unmodified, and it records the HEAD it checked -- so the commit
that holds these constants precedes every v15 checkpoint by construction, not
by a sentence in a commit message.

THE TIERS (both training seeds must clear; a near-miss is a miss)
-----------------------------------------------------------------
* **resolved** -- on `memory_probe_v2` S1 (trained values, HELD-OUT phrasings,
  distance 0) told beats untold by exact McNemar, p < 0.05, in the told
  direction, at the same sampler seeds. 6/80 against 0/80 is the least that
  clears.
* **ship-candidate** -- resolved, and ALL of: (i) the committed
  `memory_probe.py` at distance 0 reads told > 15.5/80 and untold < 4.5/80;
  (ii) S1 told > 7.5/80; (iii) the two-fact stratum S4 beats its SWAPPED-ROLES
  control by the same McNemar rule; and every guard below passes. Two tiers
  because the committed probe's phrasings are the generator's TRAIN templates:
  passing it alone is recitation, which `canned_rate` already flags.

Every count bar is a half-integer on the /80 lattice (`CONVENTIONS.md` section
4 rule 2), so none can be landed on. 15/80 fails (i); 16/80 passes it.

Per seed the verdict is `ship-candidate`, `resolved`, `floor` or `not run`.
Overall: both ship-candidate -> `ship-candidate`; both at least resolved ->
`resolved`; both floor -> `floor`; ANYTHING ELSE -- a straddle, a missing or
incomplete seed -- is `unresolved`, which is a verdict and is reported as one.
n = 2 training seeds was fixed in advance. It is never enlarged, and a seed
that rolled back is reported as rolled back and is not replaced.

THE GUARDS, listed individually in the output
---------------------------------------------
* `wrong_value` -- S1 told: replies naming a DIFFERENT value of the slot are
  strictly fewer than hits ("Your name is Tom" must lose to "Your name is
  Alice"). Flags, not categories: a reply naming both counts in both.
* `swap`        -- S4 told: replies naming the OTHER told fact's value are
  strictly fewer than hits.
* `confabulation` -- S1 untold: replies naming ANY value of the slot are no
  more than replies in the "haven't told me" family. Read it beside hazard (a)
  in `snnchat/recall.py`: 0.1241 of trained answers had their evidence cut off
  by the window, which trains exactly this.
* `bpc_<source>` -- held-out bpc on soda, stories_topic and tinystories, whose
  mix weights did not change, is not WORSE than the same-index incumbent seed
  by more than `BPC_SIGMAS` x the per-source SD_seed of the four incumbents
  (2.83 = 2*sqrt(2): two sigma on a difference of two single runs). A source
  that comes out BETTER by more than the band is printed and fails nothing; a
  guard exists to catch damage. Weighted bpc is NOT read: the mix differs, and
  `ckpt_best` was SELECTED on a weighted val that now contains the recall
  source at a very low bpc (hazard (c)). alpaca gave up 0.06 of the mix and is
  expected to rise; it is reported, as are persona, dolly and oasst1.
* `heldout`     -- `echo_holdout.py --set heldout --seeds 6 --n 256`, the
  shipped decoder: the selected rate is not resolved BELOW 60/120 (Wilson).
* `social`, `identity` -- `scripts/chat/quality.py`'s battery at its pinned
  reading row (n = 1, lambda = 0, `baseline_picks`): social > 19.5/20,
  identity > 14.5/16.

Reported, gating nothing: distances 1-2, S3 (unseen values), S5 (held-out
frames), the n=256 pass, the acknowledgement-echo column, the binding probe,
the committed probe's item 1 ('elliot', a trained value) on its own line, and
the template-capture rate -- `canned_rate` extended to `snnchat.recall`'s own
bot lines, because the templated share of the mix rises from 8 % to 14 %.

THE DECISION RULE CARRIED FORWARD
---------------------------------
If distance 0 passes (overall resolved or better) and S2's told rate at
distance 2 is below half its distance-0 rate on both seeds, carried-state
training is the next round. If distance 0 FAILS on trained values -- overall
`floor`, and neither S2 at distance 0 nor the committed probe resolves on
either seed -- stop the memory programme: longer windows, carried state and the
retrieval envelope are moot for recall.

    python scripts/chat/score_v15.py --out-root <the driver's --out-root>
    python scripts/chat/score_v15.py --check-bands <main tree>/experiments/chat
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat import recall  # noqa: E402
from snnchat.quality import (  # noqa: E402
    _SUBSTANTIVE,
    _is_canned,
    rate_ci,
    resolves_against,
)

# ---------------------------------------------------------------------------
# THE BARS. Fixed by docs/chat/PREDICTION_v15.md; nothing below computes one.
# ---------------------------------------------------------------------------

#: The two training seeds, fixed in advance. Never enlarged, never replaced.
RUNS: tuple[str, ...] = ("chat-v15-recall-s0", "chat-v15-recall-s1")
#: The incumbent seed each run's per-source bpc is compared with.
REFERENCE: dict[str, str] = {"chat-v15-recall-s0": "chat-v3d-aligned",
                             "chat-v15-recall-s1": "chat-v3d-aligned-s1"}
#: Every probe stratum is 10 items x 8 sampler seeds.
LATTICE = 80
#: Two-sided exact McNemar, told vs control at the same sampler seed.
ALPHA = 0.05
#: (i) the committed probe at distance 0.
COMMITTED_TOLD_MIN = 15.5
COMMITTED_UNTOLD_MAX = 4.5
#: (ii) S1 told at distance 0.
S1_TOLD_MIN = 7.5
#: HELDOUT selected at the shipped n=256 decoder: not resolved below this.
HELDOUT_FLOOR = (60, 120)
#: Battery, n=1 lambda=0 reading row.
SOCIAL_MIN = (19.5, 20)
IDENTITY_MIN = (14.5, 16)
#: 2 * sqrt(2): two sigma on the difference of two single runs.
BPC_SIGMAS = 2.83
#: Sources whose mix weight is unchanged AND whose band the panel named.
GUARDED_SOURCES: tuple[str, ...] = ("soda", "stories_topic", "tinystories")
#: The recipe's step count; a summary at any other count is not a finished run.
STEPS = 14_000

#: Held-out per-source bpc of the four committed replicates of the shipped
#: recipe, read from `experiments/chat/<run>/summary.json` -> `val`. `SD_SEED`
#: is derived from THESE in `_check_transcription`, so a band cannot drift from
#: its evidence. Regenerate / verify against the files themselves with
#:
#:     python scripts/chat/score_v15.py --check-bands <main tree>/experiments/chat
INCUMBENT_VAL: dict[str, dict[str, float]] = {
    "chat-v3d-aligned": {
        "alpaca": 1.6660379026748378, "dolly": 1.9443344445881836,
        "oasst1": 1.8700833937931003, "persona": 0.22944459126017322,
        "soda": 1.1948112867208713, "stories_topic": 1.0452434404919475,
        "tinystories": 1.0545297660240587,
    },
    "chat-v3d-aligned-s1": {
        "alpaca": 1.6686341527722628, "dolly": 1.9636568355466317,
        "oasst1": 1.8732680538666773, "persona": 0.23155319888296425,
        "soda": 1.1938854811802568, "stories_topic": 1.0435722029040249,
        "tinystories": 1.0531219007567492,
    },
    "chat-v3d-aligned-s2": {
        "alpaca": 1.6611927501071804, "dolly": 1.9507840742635625,
        "oasst1": 1.8756756275305957, "persona": 0.22896815632162051,
        "soda": 1.1942887346344897, "stories_topic": 1.043846113939299,
        "tinystories": 1.053446261469465,
    },
    "chat-v3d-aligned-s3": {
        "alpaca": 1.6619048156093028, "dolly": 1.9541825533602566,
        "oasst1": 1.880909036665331, "persona": 0.22814054714189352,
        "soda": 1.195735458426303, "stories_topic": 1.0423770209373637,
        "tinystories": 1.0527272922970994,
    },
}

#: Sample SD over the four replicates above (n - 1), to 7 places, and the band
#: `BPC_SIGMAS` x SD. Transcribed; `_check_transcription` recomputes both.
SD_SEED: dict[str, float] = {
    "alpaca": 0.0035174, "dolly": 0.0080569, "oasst1": 0.0045660,
    "persona": 0.0014545, "soda": 0.0007991, "stories_topic": 0.0011770,
    "tinystories": 0.0007737,
}
BPC_BAND: dict[str, float] = {
    "soda": 0.0022614, "stories_topic": 0.0033309, "tinystories": 0.0021895,
}


def _check_transcription() -> dict:
    """The bands must equal what their stated evidence implies. Fails loudly."""
    bad = {}
    for source, want in SD_SEED.items():
        sd = statistics.stdev(v[source] for v in INCUMBENT_VAL.values())
        if round(sd, 7) != want:
            bad[f"sd:{source}"] = (round(sd, 7), want)
        if source in BPC_BAND and round(BPC_SIGMAS * sd, 7) != BPC_BAND[source]:
            bad[f"band:{source}"] = (round(BPC_SIGMAS * sd, 7), BPC_BAND[source])
    if set(BPC_BAND) != set(GUARDED_SOURCES):
        bad["guarded"] = (sorted(BPC_BAND), sorted(GUARDED_SOURCES))
    if bad:
        raise SystemExit(f"TRANSCRIPTION ERROR: a bpc band does not match the four "
                         f"replicates it is derived from: {bad}")
    return {"n_replicates": len(INCUMBENT_VAL), "recomputed_ok": True}


def check_bands(chat_root: Path) -> int:
    """Re-read the four incumbent summaries and compare with `INCUMBENT_VAL`."""
    bad = []
    for run, want in INCUMBENT_VAL.items():
        val = json.loads((chat_root / run / "summary.json").read_text(encoding="utf-8"))["val"]
        for source, w in want.items():
            if val.get(source) != w:
                bad.append((run, source, val.get(source), w))
    for row in bad:
        print("MISMATCH", *row)
    _check_transcription()
    for source in SD_SEED:
        band = f"  band {BPC_BAND[source]:.7f}" if source in BPC_BAND else ""
        print(f"  {source:<14} SD_seed {SD_SEED[source]:.7f} (n=4){band}")
    print("embedded values match the summaries" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs. The same arithmetic as
    `memory_probe.mcnemar`, restated so that this file imports no torch-loading
    script; `tests/test_snnchat_v15.py` holds the two equal."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n))


def paired(told_only: int, control_only: int) -> dict:
    """`above` / `below` / `unresolved`, with direction as part of the gate.

    `p < 0.05` alone is "these differ", not "told won" -- the defect
    `score_v12.py` records catching in itself.
    """
    p = mcnemar(control_only, told_only)
    if p < ALPHA and told_only > control_only:
        verdict = "above"
    elif p < ALPHA and control_only > told_only:
        verdict = "below"
    else:
        verdict = "unresolved"
    return {"told_only": told_only, "control_only": control_only, "mcnemar_p": p,
            "verdict": verdict}


def fmt(k: int, n: int) -> str:
    """`0.1250 (6/48), 95% CI [0.059, 0.247]` -- `CONVENTIONS.md` section 1."""
    ci = rate_ci(k, n)
    return f"{ci['rate']:.4f} ({k}/{n}), 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def _template_regex(template: str) -> re.Pattern:
    pat = re.escape(template.lower())
    pat = pat.replace(r"\{v\}", r"[a-z' ]+?").replace(r"\{x\}", r"[a-z]+")
    return re.compile(pat.replace(r"\{a\}", r"an?"))


def _recall_bot_lines() -> tuple[re.Pattern, ...]:
    templates: list[str] = []
    for table in (recall.ANSWERS, recall.UNTOLD_ANSWERS, recall._ACKS_ECHO):
        for forms in table.values():
            templates += forms
    templates += recall._ACKS_PLAIN + recall._ACKS_PLAIN_NAME
    templates += [bot for _user, bot in recall.TRAIN_FILLERS]
    return tuple(_template_regex(t) for t in dict.fromkeys(templates))


_RECALL_LINES = _recall_bot_lines()


def is_recall_template(text: str) -> bool:
    """True if the reply OPENS with one of `snnchat.recall`'s own bot lines.

    `snnchat.quality._is_canned` extended to the new source. Every line the
    generator can put in the model's mouth -- answers, refusals,
    acknowledgements, filler replies -- with its placeholders as wildcards.
    """
    low = re.sub(r"\s+", " ", text.replace("’", "'").strip().lower())
    return any(p.match(low) for p in _RECALL_LINES)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _n_divergences(log: Path) -> int | None:
    if not log.exists():
        return None
    n = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if json.loads(line).get("event") == "divergence":
                n += 1
        except Exception:
            continue
    return n


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------


def _stratum(v2: dict | None, stratum: str, distance: int = 0) -> dict | None:
    """A complete /80 cell of a plain-decoder v2 artifact, or None."""
    if not v2 or not v2.get("complete") or v2.get("decoder", {}).get("n") != 8:
        return None
    cell = v2.get("summary", {}).get(stratum, {}).get(str(distance))
    return cell if cell and cell.get("n") == LATTICE else None


def _guard(passed: bool | None, detail: str) -> dict:
    status = "not run" if passed is None else ("pass" if passed else "FAIL")
    return {"status": status, "detail": detail}


def bpc_guards(run: str, summary: dict | None) -> dict:
    out = {}
    val = (summary or {}).get("val") or {}
    ref = INCUMBENT_VAL[REFERENCE[run]]
    for source in GUARDED_SOURCES:
        if source not in val:
            out[f"bpc_{source}"] = _guard(None, "no held-out bpc for this source")
            continue
        delta = val[source] - ref[source]
        band = BPC_BAND[source]
        side = "worse" if delta > band else ("better" if delta < -band else "within")
        out[f"bpc_{source}"] = {
            **_guard(side != "worse",
                     f"{val[source]:.7f} vs {REFERENCE[run]} {ref[source]:.7f}: "
                     f"delta {delta:+.7f}, band +-{band:.7f} -> {side}"),
            "delta": delta, "band": band, "side": side}
    return out


def battery_guards(battery: dict | None) -> tuple[dict, dict]:
    """(guards, reported) from a `scripts/chat/quality.py --out` artifact."""
    picks = ((battery or {}).get("report") or {}).get("baseline_picks")
    guards, reported = {}, {}
    for kind, (thr, den) in (("social", SOCIAL_MIN), ("identity", IDENTITY_MIN)):
        sub = [p for p in picks or [] if p["kind"] == kind]
        if len(sub) != den:
            guards[kind] = _guard(None, f"{len(sub)} {kind} picks, the bar assumes {den}")
            continue
        k = int(sum(1 for p in sub if p["hit"] >= 1.0))
        guards[kind] = _guard(k > thr, f"{fmt(k, den)}; bar > {thr}/{den}")
    if picks:
        subst = [p for p in picks if p["kind"] in _SUBSTANTIVE]
        persona = sum(_is_canned(p["text"]) for p in subst)
        rec = sum(is_recall_template(p["text"]) for p in subst)
        either = sum(_is_canned(p["text"]) or is_recall_template(p["text"]) for p in subst)
        reported["template_capture"] = {
            "over": "substantive picks (topic, list, fact) at n=1, lambda=0",
            "persona": rate_ci(persona, len(subst)),
            "recall": rate_ci(rec, len(subst)),
            "either": rate_ci(either, len(subst)),
            "recall_over_all_picks": rate_ci(
                sum(is_recall_template(p["text"]) for p in picks), len(picks)),
        }
    return guards, reported


def heldout_guard(heldout: dict | None) -> dict:
    rows = (heldout or {}).get("rows")
    floor_k, den = HELDOUT_FLOOR
    if not rows or len(rows) != den or heldout.get("n") != 256:
        return _guard(None, f"needs echo_holdout --set heldout at n=256 over {den} draws")
    k = int(sum(r["echo_hit"] for r in rows))
    side = resolves_against(k, den, floor_k / den)
    return _guard(side != "below", f"{fmt(k, den)} against {floor_k}/{den} -> {side}")


def score_seed(run: str, art: dict) -> dict:
    """The verdict for one training seed from its loaded artifacts.

    `art` keys: `committed`, `v2`, `v2_shipped`, `binding`, `battery`,
    `heldout`, `summary` (each a parsed JSON blob or None) and `n_divergences`.
    Pure: no file is opened here, which is what lets a test hand it synthetic
    artifacts for every tier and every guard.
    """
    out: dict = {"run": run, "reference": REFERENCE[run]}
    summary = art.get("summary")
    finished = bool(summary) and summary.get("steps") == STEPS
    out["finished"] = finished
    out["n_divergences"] = art.get("n_divergences")
    out["rolled_back"] = bool(art.get("n_divergences"))

    s1, s4 = _stratum(art.get("v2"), "S1"), _stratum(art.get("v2"), "S4")
    com = ((art.get("committed") or {}).get("summary") or {}).get("0")
    if com and (com["told"][1] != LATTICE or com["untold"][1] != LATTICE):
        com = None

    # ---- tier: resolved ---------------------------------------------------
    primary = paired(s1["discordant"]["told_only"], s1["discordant"]["control_only"]) \
        if s1 else None
    out["primary_S1"] = primary and {
        **primary, "told": fmt(s1["told"]["hit"]["k"], LATTICE),
        "untold": fmt(s1["control"]["hit"]["k"], LATTICE)}
    resolved = bool(primary) and primary["verdict"] == "above"

    # ---- tier: ship-candidate ---------------------------------------------
    ship = {}
    if com:
        kt, ku = com["told"][0], com["untold"][0]
        ship["i_committed_probe"] = _guard(
            kt > COMMITTED_TOLD_MIN and ku < COMMITTED_UNTOLD_MAX,
            f"told {fmt(kt, LATTICE)}, bar > {COMMITTED_TOLD_MIN}/{LATTICE}; "
            f"untold {fmt(ku, LATTICE)}, bar < {COMMITTED_UNTOLD_MAX}/{LATTICE}")
    else:
        ship["i_committed_probe"] = _guard(None, "no complete committed-probe artifact")
    if s1:
        k = s1["told"]["hit"]["k"]
        ship["ii_unseen_phrasing"] = _guard(
            k > S1_TOLD_MIN, f"S1 told {fmt(k, LATTICE)}; bar > {S1_TOLD_MIN}/{LATTICE}")
    else:
        ship["ii_unseen_phrasing"] = _guard(None, "no complete S1 cell")
    if s4:
        two = paired(s4["discordant"]["told_only"], s4["discordant"]["control_only"])
        ship["iii_two_fact"] = {**_guard(
            two["verdict"] == "above",
            f"S4 told {fmt(s4['told']['hit']['k'], LATTICE)} vs swapped-roles "
            f"{fmt(s4['control']['hit']['k'], LATTICE)}, +{two['told_only']}/"
            f"-{two['control_only']} p={two['mcnemar_p']:.4f} -> {two['verdict']}"), **two}
    else:
        ship["iii_two_fact"] = _guard(None, "no complete S4 cell")
    out["ship_bars"] = ship

    # ---- guards ------------------------------------------------------------
    guards: dict = {}
    if s1:
        h, w = s1["told"]["hit"]["k"], s1["told"]["wrong_value"]["k"]
        guards["wrong_value"] = _guard(
            w < h, f"S1 told: wrong value {fmt(w, LATTICE)} must be below hit {fmt(h, LATTICE)}")
        a, r = s1["control"]["any_value"]["k"], s1["control"]["refusal"]["k"]
        guards["confabulation"] = _guard(
            a <= r, f"S1 untold: any value {fmt(a, LATTICE)} must not exceed "
                    f"\"haven't told me\" {fmt(r, LATTICE)}")
    else:
        guards["wrong_value"] = _guard(None, "no complete S1 cell")
        guards["confabulation"] = _guard(None, "no complete S1 cell")
    if s4:
        h, sw = s4["told"]["hit"]["k"], s4["told"]["swap"]["k"]
        guards["swap"] = _guard(
            sw < h, f"S4 told: swap {fmt(sw, LATTICE)} must be below hit {fmt(h, LATTICE)}")
    else:
        guards["swap"] = _guard(None, "no complete S4 cell")
    guards.update(bpc_guards(run, summary if finished else None))
    guards["heldout"] = heldout_guard(art.get("heldout"))
    b_guards, reported = battery_guards(art.get("battery"))
    guards.update(b_guards)
    out["guards"] = guards

    # ---- verdict -----------------------------------------------------------
    clear = (all(g["status"] == "pass" for g in ship.values())
             and all(g["status"] == "pass" for g in guards.values()))
    if not finished or primary is None:
        out["verdict"] = "not run"
    elif not resolved:
        out["verdict"] = "floor"
    elif clear:
        out["verdict"] = "ship-candidate"
    else:
        out["verdict"] = "resolved"

    # ---- reported, gating nothing -------------------------------------------
    rep: dict = dict(reported)
    val = (summary or {}).get("val") or {}
    ref = INCUMBENT_VAL[REFERENCE[run]]
    rep["per_source_bpc"] = {
        s: {"v15": val[s], "reference": ref.get(s),
            "delta": val[s] - ref[s] if s in ref else None,
            "sd_seed": SD_SEED.get(s)} for s in sorted(val) if s != "weighted"}
    rep["weighted_bpc_not_comparable"] = val.get("weighted")
    v2 = art.get("v2")
    rep["strata"] = {}
    for stratum, dists in (("S1", (0,)), ("S2", (0, 1, 2)), ("S3", (0,)), ("S4", (0,)),
                           ("S5", (0,))):
        for d in dists:
            cell = _stratum(v2, stratum, d)
            if cell:
                rep["strata"][f"{stratum}_d{d}"] = {
                    "told": fmt(cell["told"]["hit"]["k"], LATTICE),
                    "control": fmt(cell["control"]["hit"]["k"], LATTICE),
                    **paired(cell["discordant"]["told_only"],
                             cell["discordant"]["control_only"]),
                    "told_k": cell["told"]["hit"]["k"],
                    "told_categories": cell["told"]["category"],
                    "control_categories": cell["control"]["category"],
                    "ack_echo": cell["ack_echo"],
                    "told_hit_by_ack": cell["told_hit_by_ack"]}
    shipped = art.get("v2_shipped") or {}
    rep["n256_pass"] = {
        f"{s}_d0": fmt(c["0"]["told"]["hit"]["k"], c["0"]["n"])
        for s, c in (shipped.get("summary") or {}).items() if "0" in c}
    rep["binding"] = (art.get("binding") or {}).get("pooled")
    if com:
        rep["committed_d0"] = {"told": fmt(com["told"][0], LATTICE),
                               "untold": fmt(com["untold"][0], LATTICE),
                               **paired(com["discordant"][1], com["discordant"][0])}
    rows = (art.get("committed") or {}).get("rows") or []
    item1 = [r for r in rows if r["distance"] == 0 and "elliot" in r["targets"]]
    if item1:
        rep["committed_item1_elliot_seen_value"] = fmt(
            int(sum(r["told_hit"] for r in item1)), len(item1))
    rep["committed_probe_ceiling"] = ("64/80: items 7 and 8 (town, age) are held-out "
                                      "frames snnchat.recall never trains")
    out["reported"] = rep
    return out


# ---------------------------------------------------------------------------
# both seeds
# ---------------------------------------------------------------------------


def overall(verdicts: list[str]) -> str:
    """Both seeds must clear; a straddle or a missing seed is `unresolved`."""
    if len(verdicts) != len(RUNS) or "not run" in verdicts:
        return "unresolved"
    if all(v == "ship-candidate" for v in verdicts):
        return "ship-candidate"
    if all(v in ("ship-candidate", "resolved") for v in verdicts):
        return "resolved"
    if all(v == "floor" for v in verdicts):
        return "floor"
    return "unresolved"


def decision_rule(verdict: str, seeds: list[dict]) -> dict:
    """The rule `PREDICTION_v15.md` carries forward. Fires or says why not."""
    cells = [s["reported"]["strata"] for s in seeds]
    have = all("S2_d0" in c and "S2_d2" in c for c in cells)
    decays = have and all(c["S2_d2"]["told_k"] < 0.5 * c["S2_d0"]["told_k"] for c in cells)
    trained_resolves = any(
        c.get("S2_d0", {}).get("verdict") == "above" for c in cells) or any(
        s["reported"].get("committed_d0", {}).get("verdict") == "above" for s in seeds)
    have = have and all("committed_d0" in s["reported"] for s in seeds)
    if verdict in ("ship-candidate", "resolved"):
        fired = "carried-state training is the next round" if decays else None
        note = ("distance 0 passes; S2 told at distance 2 is below half of distance 0 "
                "on both seeds" if decays else
                "distance 0 passes; S2 does not fall below half by distance 2 on both "
                "seeds, so reach is not shown to be the binding constraint")
    elif verdict == "floor" and have and not trained_resolves:
        fired = ("stop the memory programme: longer windows, carried state and the "
                 "retrieval envelope are moot for recall")
        note = "distance 0 fails on trained values, in trained phrasings too"
    elif verdict == "floor":
        fired = None
        note = ("S1 is at the floor but a TRAINED-phrasing reading resolves or is "
                "missing: recitation without transfer, which the stop rule does not cover")
    else:
        fired, note = None, "unresolved: the rule does not fire on a straddle"
    return {"fired": fired, "note": note}


def load_artifacts(out_root: Path, run: str) -> dict:
    scores, rundir = out_root / "scores" / run, out_root / "runs" / run
    return {
        "committed": _load(scores / "memory_probe.json"),
        "v2": _load(scores / "memory_probe_v2_n8.json"),
        "v2_shipped": _load(scores / "memory_probe_v2_n256.json"),
        "binding": _load(scores / "binding_probe.json"),
        "battery": _load(scores / "battery.json"),
        "heldout": _load(scores / "heldout_n256.json"),
        "summary": _load(rundir / "summary.json"),
        "n_divergences": _n_divergences(rundir / "log.jsonl"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", help="the directory session15_driver.py wrote under")
    ap.add_argument("--out", default=None, help="default <out-root>/score_v15.json")
    ap.add_argument("--check-bands", metavar="CHAT_ROOT", default=None,
                    help="verify the embedded incumbent bpc against the four "
                         "summary.json files under this experiments/chat, and exit")
    args = ap.parse_args(argv)

    if args.check_bands:
        return check_bands(Path(args.check_bands))
    if not args.out_root:
        raise SystemExit("--out-root is required (or --check-bands)")
    transcription = _check_transcription()
    out_root = Path(args.out_root)

    seeds = [score_seed(run, load_artifacts(out_root, run)) for run in RUNS]
    verdict = overall([s["verdict"] for s in seeds])
    rule = decision_rule(verdict, seeds)
    floors = {
        "memory_probe_v2": (_load(out_root / "floors" / "memory_probe_v2_n8.json")
                            or {}).get("summary"),
        "binding": (_load(out_root / "floors" / "binding_probe.json") or {}).get("pooled"),
    }

    print("PREDICTION_v15 -- results\n")
    for s in seeds:
        print(f"[{s['run']}]  verdict: {s['verdict'].upper()}"
              + ("   (ROLLED BACK; reported, not replaced)" if s["rolled_back"] else ""))
        if s["primary_S1"]:
            p = s["primary_S1"]
            print(f"  PRIMARY S1 d0  told {p['told']}\n"
                  f"                 untold {p['untold']}\n"
                  f"                 +{p['told_only']}/-{p['control_only']} "
                  f"McNemar p={p['mcnemar_p']:.4f} -> {p['verdict']}")
        for name, g in {**s["ship_bars"], **s["guards"]}.items():
            print(f"  {g['status']:>7}  {name:<22} {g['detail']}")
        print()
    print(f"OVERALL: {verdict.upper()}  (both seeds must clear; n=2 fixed in advance)")
    print(f"decision rule: {rule['fired'] or 'does not fire'} -- {rule['note']}")

    out = Path(args.out) if args.out else out_root / "score_v15.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "round": "v15", "not_a_research_artifact": True,
        "bars": {"alpha": ALPHA, "committed_told_min": COMMITTED_TOLD_MIN,
                 "committed_untold_max": COMMITTED_UNTOLD_MAX, "s1_told_min": S1_TOLD_MIN,
                 "heldout_floor": list(HELDOUT_FLOOR), "social_min": list(SOCIAL_MIN),
                 "identity_min": list(IDENTITY_MIN), "bpc_sigmas": BPC_SIGMAS,
                 "bpc_band": BPC_BAND, "lattice": LATTICE},
        "bar_transcription": transcription,
        "verdict": verdict, "decision_rule": rule, "seeds": seeds, "floors_shipped": floors,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
