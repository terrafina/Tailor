#!/usr/bin/env python3
"""Extract key metrics from an NVIDIA Garak ``report.jsonl`` file.

Garak emits three artefacts per run (same UUID prefix):

* ``garak.<uuid>.report.jsonl`` — primary, line-delimited JSON record of the
  whole run. Each line carries an ``entry_type`` field; the relevant ones are:

  - ``init`` (run id + start time)
  - ``start_run setup`` (full config dump: target, probe_spec, generations…)
  - ``attempt`` (one row per prompt sent; emitted twice — ``status: 1`` after
    the prompt is sent and ``status: 2`` after detectors have scored it. The
    ``status: 2`` row is the one that carries ``detector_results``.)
  - ``eval`` (one row per ``(probe, detector)`` pair, with passed / fails /
    nones / total_evaluated and bootstrap confidence interval for the rate.
    This is the authoritative summary.)
  - ``digest`` (a final, nested summary keyed by probe-group → probe → detector
    that mirrors the ``eval`` rows but adds severity, tier, descriptions,
    defcon ratings, etc.)
  - ``completion`` (end time)

* ``garak.<uuid>.hitlog.jsonl`` — only the rows that scored as a hit.
* ``garak.<uuid>.report.html`` — pre-rendered HTML view of the digest.

This script reads ``report.jsonl`` (streaming, so it handles >100 MB files
fine) and produces:

* ``overall`` — generations evaluated, breaches, breach rate.
* ``by_probe`` — every (probe, detector) pair with its rate + CI, plus a
  ``primary`` summary per probe using a configurable detector preference.
* ``examples`` — N representative breach attempts per probe (truncated prompt
  + response + the full per-detector scores from the attempt).

The CLI prints JSON to stdout by default; pass ``--md`` for a Markdown summary
on stderr, or ``--out-json PATH`` to write the JSON to a file.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DETECTOR_PRIORITY = ("mitigation.MitigationBypass", "dan.DAN")
HIT_THRESHOLD = 0.5


def stream_rows(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def truncate(text: str | None, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def first_user_text(prompt: dict | None) -> str:
    if not prompt:
        return ""
    for turn in prompt.get("turns") or []:
        if turn.get("role") == "user":
            content = turn.get("content") or {}
            return content.get("text") or ""
    return ""


def output_text(output: Any) -> str:
    """Garak ``outputs`` cells are ``{text, lang, ...}`` dicts; older formats used plain strings."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return output.get("text") or ""
    return str(output)


def _ordered_detectors(present: Iterable[str], priority: tuple[str, ...]) -> list[str]:
    present = list(present)
    primary = [d for d in priority if d in present]
    rest = [d for d in present if d not in priority]
    return primary + rest


def _probe_meta_from_digest(digest: dict | None) -> dict[str, dict]:
    """Flatten ``digest.eval`` into ``{probe_name: {description, tier, ...}}``.

    The digest is nested as ``eval[group][probe][detector]`` with ``_summary``
    siblings carrying group/probe-level descriptions.
    """
    out: dict[str, dict] = {}
    if not digest:
        return out
    for group_name, group in (digest.get("eval") or {}).items():
        if not isinstance(group, dict):
            continue
        for probe_name, probe_block in group.items():
            if probe_name == "_summary" or not isinstance(probe_block, dict):
                continue
            ps = probe_block.get("_summary") or {}
            out[probe_name] = {
                "group": group_name,
                "description": ps.get("probe_descr"),
                "tier": ps.get("probe_tier"),
                "severity": ps.get("probe_severity"),
                "tags": ps.get("probe_tags"),
            }
    return out


def extract(
    report_path: str | Path,
    examples_per_probe: int = 3,
    prompt_chars: int = 400,
    response_chars: int = 400,
    detector_priority: tuple[str, ...] = DEFAULT_DETECTOR_PRIORITY,
) -> dict:
    init = setup = completion = digest = None
    evals: list[dict] = []
    breach_examples: dict[str, list[dict]] = defaultdict(list)

    for r in stream_rows(report_path):
        et = r.get("entry_type")
        if et == "init":
            init = r
        elif et == "start_run setup":
            setup = r
        elif et == "completion":
            completion = r
        elif et == "digest":
            digest = r
        elif et == "eval":
            evals.append(r)
        elif et == "attempt" and r.get("status") == 2:
            probe = r.get("probe_classname")
            if probe is None:
                continue
            if len(breach_examples[probe]) >= examples_per_probe:
                continue  # cheap early exit so we don't carry every breach in memory
            dr = r.get("detector_results") or {}
            outputs = r.get("outputs") or []
            hit_detector, hit_idx = None, None
            for det in _ordered_detectors(dr.keys(), detector_priority):
                for i, score in enumerate(dr[det]):
                    if score is not None and score >= HIT_THRESHOLD:
                        hit_detector, hit_idx = det, i
                        break
                if hit_detector is not None:
                    break
            if hit_detector is None:
                continue
            breach_examples[probe].append({
                "probe": probe,
                "detector": hit_detector,
                "generation_index": hit_idx,
                "all_detector_scores": dict(dr),
                "prompt": truncate(first_user_text(r.get("prompt")), prompt_chars),
                "response": truncate(
                    output_text(outputs[hit_idx]) if hit_idx < len(outputs) else "",
                    response_chars,
                ),
                "attempt_uuid": r.get("uuid"),
                "seq": r.get("seq"),
            })

    probe_meta = _probe_meta_from_digest(digest)

    by_probe_detectors: dict[str, list[dict]] = defaultdict(list)
    for ev in evals:
        passed = ev.get("passed", 0) or 0
        fails = ev.get("fails", 0) or 0
        nones = ev.get("nones", 0) or 0
        total = ev.get("total_evaluated") or (passed + fails + nones)
        by_probe_detectors[ev["probe"]].append({
            "detector": ev["detector"],
            "passed": passed,
            "fails": fails,
            "nones": nones,
            "total": total,
            "breach_rate": (fails / total) if total else 0.0,
            "ci_method": ev.get("confidence_method"),
            "ci_level": ev.get("confidence"),
            "ci_lower": ev.get("confidence_lower"),
            "ci_upper": ev.get("confidence_upper"),
        })

    by_probe: list[dict] = []
    for probe, detectors in by_probe_detectors.items():
        ordered_names = _ordered_detectors((d["detector"] for d in detectors), detector_priority)
        primary = next(d for d in detectors if d["detector"] == ordered_names[0])
        meta = probe_meta.get(probe, {})
        by_probe.append({
            "probe": probe,
            "description": meta.get("description"),
            "group": meta.get("group"),
            "tier": meta.get("tier"),
            "severity": meta.get("severity"),
            "tags": meta.get("tags"),
            "primary": {
                "detector": primary["detector"],
                "fails": primary["fails"],
                "total": primary["total"],
                "breach_rate": primary["breach_rate"],
                "ci_lower": primary["ci_lower"],
                "ci_upper": primary["ci_upper"],
            },
            "detectors": detectors,
        })

    overall_total = sum(p["primary"]["total"] for p in by_probe)
    overall_fails = sum(p["primary"]["fails"] for p in by_probe)
    overall_rate = (overall_fails / overall_total) if overall_total else 0.0

    examples: list[dict] = []
    for probe in by_probe_detectors:
        examples.extend(breach_examples.get(probe, [])[:examples_per_probe])

    setup_d = setup or {}
    return {
        "run": {
            "id": (init or {}).get("run") or setup_d.get("transient.run_id"),
            "garak_version": (init or {}).get("garak_version") or setup_d.get("_config.version"),
            "start_time": (init or {}).get("start_time") or setup_d.get("transient.starttime_iso"),
            "end_time": (completion or {}).get("end_time"),
            "target_type": setup_d.get("plugins.target_type"),
            "target_name": setup_d.get("plugins.target_name"),
            "probe_spec": setup_d.get("plugins.probe_spec"),
            "generations_per_prompt": setup_d.get("run.generations"),
        },
        "overall": {
            "generations_evaluated": overall_total,
            "breaches": overall_fails,
            "breach_rate": overall_rate,
            "primary_detector_priority": list(detector_priority),
        },
        "by_probe": by_probe,
        "examples": examples,
    }


def render_markdown(data: dict) -> str:
    r, o = data["run"], data["overall"]
    lines = [
        "# Garak run summary",
        "",
        f"- **Target:** `{r.get('target_type')}` / `{r.get('target_name')}`",
        f"- **Garak version:** {r.get('garak_version')}",
        f"- **Run ID:** `{r.get('id')}`",
        f"- **Started:** {r.get('start_time')}  ·  **Ended:** {r.get('end_time')}",
        f"- **Generations evaluated:** {o['generations_evaluated']:,}"
        f"  ·  **Breaches:** {o['breaches']:,}"
        f"  ·  **Overall breach rate:** {o['breach_rate']:.1%}",
        "",
        "## Per-probe breach rate (primary detector)",
        "",
        "| Probe | Detector | Breaches / Total | Rate | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for p in data["by_probe"]:
        pr = p["primary"]
        if pr.get("ci_lower") is not None and pr.get("ci_upper") is not None:
            ci = f"{pr['ci_lower']:.1%} – {pr['ci_upper']:.1%}"
        else:
            ci = "—"
        lines.append(
            f"| `{p['probe']}` | `{pr['detector']}` | "
            f"{pr['fails']:,} / {pr['total']:,} | {pr['breach_rate']:.1%} | {ci} |"
        )
    lines += ["", "## Example breaches", ""]
    for ex in data["examples"]:
        lines += [
            f"### `{ex['probe']}` — flagged by `{ex['detector']}` (generation {ex['generation_index']})",
            "",
            "**Attack prompt**",
            "",
            "> " + ex["prompt"].replace("\n", "\n> "),
            "",
            "**Target response**",
            "",
            "> " + ex["response"].replace("\n", "\n> "),
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", help="path to garak.<uuid>.report.jsonl")
    ap.add_argument("--examples-per-probe", type=int, default=3)
    ap.add_argument("--prompt-chars", type=int, default=400)
    ap.add_argument("--response-chars", type=int, default=400)
    ap.add_argument(
        "--detector-priority",
        default=",".join(DEFAULT_DETECTOR_PRIORITY),
        help="comma-separated detector names; first present per probe is treated as primary",
    )
    ap.add_argument("--out-json", help="write extracted JSON here (default: stdout)")
    ap.add_argument("--md", action="store_true", help="also print a Markdown summary to stderr")
    args = ap.parse_args(argv)

    priority = tuple(x.strip() for x in args.detector_priority.split(",") if x.strip())
    data = extract(
        args.report,
        examples_per_probe=args.examples_per_probe,
        prompt_chars=args.prompt_chars,
        response_chars=args.response_chars,
        detector_priority=priority,
    )
    payload = json.dumps(data, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    if args.md:
        sys.stderr.write(render_markdown(data) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
