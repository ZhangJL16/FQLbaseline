#!/usr/bin/env python3
"""Export already completed run comparisons to a standalone XLSX workbook.

The script scans ``exp/`` for run directories shaped like::

    exp/{agent}_OGBench_Env_{env_slug}/date_YYYYMMDD_HHMMSS_seed_{seed}/success.csv

For each ``agent / env / seed`` triplet, the most recent run is selected.
Then two comparisons are produced:

* ``scsfql`` vs ``fql``
* ``scsgfp`` vs ``gfp``

The workbook contains:

* one summary sheet per comparison
* one per-seed detail sheet per comparison
* one note sheet describing why ``Actor`` is the final reported metric

Only the Python standard library is used so the script works in minimal
environments without pandas/openpyxl/xlsxwriter.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable
from xml.sax.saxutils import escape


EXP_DIR_RE = re.compile(r"^(?P<agent>[a-z0-9_-]+)_OGBench_Env_(?P<env>.+)$")
RUN_DIR_RE = re.compile(r"^date_(?P<timestamp>\d{8}_\d{6})_seed_(?P<seed>\d+)$")

PAIR_SPECS = (
    ("scsfql", "fql"),
    ("scsgfp", "gfp"),
)
PREFERRED_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RunRecord:
    agent: str
    env_slug: str
    seed: int
    timestamp: str
    run_dir: Path
    success_csv: Path
    actor_metric: float | None
    flow_metric: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("exp"),
        help="Directory containing experiment outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparisons/scs_method_comparisons.xlsx"),
        help="Path of the generated XLSX workbook.",
    )
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_success_metrics(success_csv: Path) -> tuple[float | None, float | None]:
    with success_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None

    last_row = rows[-1]
    actor_metric = safe_float(last_row.get("final_Actor"))
    if actor_metric is None:
        actor_metric = safe_float(last_row.get("success_Actor"))

    flow_metric = safe_float(last_row.get("final_Flow"))
    if flow_metric is None:
        flow_metric = safe_float(last_row.get("success_Flow"))

    return actor_metric, flow_metric


def collect_latest_runs(exp_root: Path) -> dict[tuple[str, str, int], RunRecord]:
    latest_runs: dict[tuple[str, str, int], RunRecord] = {}

    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")

    for agent_env_dir in sorted(exp_root.iterdir()):
        if not agent_env_dir.is_dir():
            continue
        match = EXP_DIR_RE.match(agent_env_dir.name)
        if not match:
            continue

        agent = match.group("agent")
        env_slug = match.group("env")

        for run_dir in sorted(agent_env_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_match = RUN_DIR_RE.match(run_dir.name)
            if not run_match:
                continue

            success_csv = run_dir / "success.csv"
            if not success_csv.exists():
                continue

            timestamp = run_match.group("timestamp")
            seed = int(run_match.group("seed"))
            actor_metric, flow_metric = read_success_metrics(success_csv)

            record = RunRecord(
                agent=agent,
                env_slug=env_slug,
                seed=seed,
                timestamp=timestamp,
                run_dir=run_dir,
                success_csv=success_csv,
                actor_metric=actor_metric,
                flow_metric=flow_metric,
            )
            key = (agent, env_slug, seed)
            previous = latest_runs.get(key)
            if previous is None or record.timestamp > previous.timestamp:
                latest_runs[key] = record

    return latest_runs


def pretty_env_name(env_slug: str) -> str:
    return env_slug.replace("_", "-")


def metric_mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)


def metric_std(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    if len(numeric) == 1:
        return 0.0
    return pstdev(numeric)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def build_pair_tables(
    latest_runs: dict[tuple[str, str, int], RunRecord],
    method_a: str,
    method_b: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    envs_a = {env for agent, env, _ in latest_runs if agent == method_a}
    envs_b = {env for agent, env, _ in latest_runs if agent == method_b}
    common_envs = sorted(envs_a & envs_b)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    overall_actor_a: list[float] = []
    overall_actor_b: list[float] = []
    overall_flow_a: list[float] = []
    overall_flow_b: list[float] = []

    for env_slug in common_envs:
        seeds_a = {seed for agent, env, seed in latest_runs if agent == method_a and env == env_slug}
        seeds_b = {seed for agent, env, seed in latest_runs if agent == method_b and env == env_slug}
        common_seeds = sorted(seeds_a & seeds_b)
        if all(seed in common_seeds for seed in PREFERRED_SEEDS):
            # Prefer the standardized 3-seed benchmark runs when they exist,
            # instead of mixing in older ad hoc debug seeds.
            common_seeds = list(PREFERRED_SEEDS)
        if not common_seeds:
            continue

        actor_a_values: list[float | None] = []
        actor_b_values: list[float | None] = []
        flow_a_values: list[float | None] = []
        flow_b_values: list[float | None] = []

        for seed in common_seeds:
            run_a = latest_runs[(method_a, env_slug, seed)]
            run_b = latest_runs[(method_b, env_slug, seed)]

            actor_a_values.append(run_a.actor_metric)
            actor_b_values.append(run_b.actor_metric)
            flow_a_values.append(run_a.flow_metric)
            flow_b_values.append(run_b.flow_metric)

            if run_a.actor_metric is not None:
                overall_actor_a.append(run_a.actor_metric)
            if run_b.actor_metric is not None:
                overall_actor_b.append(run_b.actor_metric)
            if run_a.flow_metric is not None:
                overall_flow_a.append(run_a.flow_metric)
            if run_b.flow_metric is not None:
                overall_flow_b.append(run_b.flow_metric)

            detail_rows.append(
                {
                    "env_name": pretty_env_name(env_slug),
                    "seed": seed,
                    f"{method_a}_timestamp": run_a.timestamp,
                    f"{method_b}_timestamp": run_b.timestamp,
                    f"{method_a}_actor": round_or_none(run_a.actor_metric),
                    f"{method_b}_actor": round_or_none(run_b.actor_metric),
                    "actor_delta": round_or_none(
                        None
                        if run_a.actor_metric is None or run_b.actor_metric is None
                        else run_a.actor_metric - run_b.actor_metric
                    ),
                    f"{method_a}_flow": round_or_none(run_a.flow_metric),
                    f"{method_b}_flow": round_or_none(run_b.flow_metric),
                    "flow_delta": round_or_none(
                        None
                        if run_a.flow_metric is None or run_b.flow_metric is None
                        else run_a.flow_metric - run_b.flow_metric
                    ),
                    f"{method_a}_success_csv": run_a.success_csv.as_posix(),
                    f"{method_b}_success_csv": run_b.success_csv.as_posix(),
                }
            )

        actor_a_mean = metric_mean(actor_a_values)
        actor_b_mean = metric_mean(actor_b_values)
        flow_a_mean = metric_mean(flow_a_values)
        flow_b_mean = metric_mean(flow_b_values)

        summary_rows.append(
            {
                "env_name": pretty_env_name(env_slug),
                "seed_count": len(common_seeds),
                f"{method_a}_actor_mean": round_or_none(actor_a_mean),
                f"{method_a}_actor_std": round_or_none(metric_std(actor_a_values)),
                f"{method_b}_actor_mean": round_or_none(actor_b_mean),
                f"{method_b}_actor_std": round_or_none(metric_std(actor_b_values)),
                "actor_delta": round_or_none(
                    None if actor_a_mean is None or actor_b_mean is None else actor_a_mean - actor_b_mean
                ),
                f"{method_a}_flow_mean": round_or_none(flow_a_mean),
                f"{method_a}_flow_std": round_or_none(metric_std(flow_a_values)),
                f"{method_b}_flow_mean": round_or_none(flow_b_mean),
                f"{method_b}_flow_std": round_or_none(metric_std(flow_b_values)),
                "flow_delta": round_or_none(
                    None if flow_a_mean is None or flow_b_mean is None else flow_a_mean - flow_b_mean
                ),
            }
        )

    if summary_rows:
        overall_actor_a_mean = metric_mean(overall_actor_a)
        overall_actor_b_mean = metric_mean(overall_actor_b)
        overall_flow_a_mean = metric_mean(overall_flow_a)
        overall_flow_b_mean = metric_mean(overall_flow_b)

        summary_rows.append(
            {
                "env_name": "OVERALL",
                "seed_count": len(overall_actor_a),
                f"{method_a}_actor_mean": round_or_none(overall_actor_a_mean),
                f"{method_a}_actor_std": round_or_none(metric_std(overall_actor_a)),
                f"{method_b}_actor_mean": round_or_none(overall_actor_b_mean),
                f"{method_b}_actor_std": round_or_none(metric_std(overall_actor_b)),
                "actor_delta": round_or_none(
                    None
                    if overall_actor_a_mean is None or overall_actor_b_mean is None
                    else overall_actor_a_mean - overall_actor_b_mean
                ),
                f"{method_a}_flow_mean": round_or_none(overall_flow_a_mean),
                f"{method_a}_flow_std": round_or_none(metric_std(overall_flow_a)),
                f"{method_b}_flow_mean": round_or_none(overall_flow_b_mean),
                f"{method_b}_flow_std": round_or_none(metric_std(overall_flow_b)),
                "flow_delta": round_or_none(
                    None
                    if overall_flow_a_mean is None or overall_flow_b_mean is None
                    else overall_flow_a_mean - overall_flow_b_mean
                ),
            }
        )

    return summary_rows, detail_rows


def excel_column_name(index: int) -> str:
    index += 1
    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def make_cell(cell_ref: str, value: object, header: bool = False) -> str:
    if value is None:
        return f'<c r="{cell_ref}"/>'

    style = ' s="1"' if header else ""
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"{style}><v>{1 if value else 0}</v></c>'
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"{style}><v>{value}</v></c>'
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f'<c r="{cell_ref}"/>'
        return f'<c r="{cell_ref}"{style}><v>{value}</v></c>'

    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[list[object]]) -> str:
    row_xml_parts: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cell_xml = []
        for col_idx, value in enumerate(row):
            cell_ref = f"{excel_column_name(col_idx)}{row_idx}"
            cell_xml.append(make_cell(cell_ref, value, header=(row_idx == 1)))
        row_xml_parts.append(f'<row r="{row_idx}">{"".join(cell_xml)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        "<sheetData>"
        + "".join(row_xml_parts)
        + "</sheetData></worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        safe_name = escape(name)
        sheets.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" '
            f'r:id="rId{idx}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(sheets)
        + "</sheets></workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        "</fonts>"
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        "</fills>"
        '<borders count="1">'
        "<border><left/><right/><top/><bottom/><diagonal/></border>"
        "</borders>"
        '<cellStyleXfs count="1">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
        "</cellStyleXfs>"
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        "</cellXfs>"
        '<cellStyles count="1">'
        '<cellStyle name="Normal" xfId="0" builtinId="0"/>'
        "</cellStyles></styleSheet>"
    )


def rows_from_dicts(records: list[dict[str, object]]) -> list[list[object]]:
    if not records:
        return [["message"], ["No matching runs found."]]

    header = list(records[0].keys())
    rows = [header]
    for record in records:
        rows.append([record.get(column) for column in header])
    return rows


def sanitize_sheet_name(name: str) -> str:
    invalid_chars = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in name)
    return cleaned[:31]


def build_note_rows(output_path: Path) -> list[list[object]]:
    return [
        ["item", "detail"],
        ["final_metric", "Actor success is the final reported metric for FQL-style runs."],
        [
            "code_evidence_1",
            "main.py returns final_Actor first, otherwise success_Actor. Flow is evaluated but not returned as the final metric.",
        ],
        [
            "code_evidence_2",
            "success.csv stores both Actor and Flow because eval_flow_policy can be enabled, but Actor is always treated as the primary output policy.",
        ],
        [
            "paper_evidence_1",
            "OpenReview abstract: FQL trains an expressive one-step policy with RL and eliminates iterative action generation at test time.",
        ],
        [
            "paper_evidence_2",
            "Project page: the output of FQL is the efficient one-step policy.",
        ],
        ["source_openreview", "https://openreview.net/forum?id=KVf2SFL1pi"],
        ["source_project_page", "https://seohong.me/projects/fql/"],
        ["generated_file", output_path.as_posix()],
    ]


def write_workbook(output_path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_names = [sanitize_sheet_name(name) for name, _ in sheets]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        archive.writestr("xl/styles.xml", styles_xml())

        for idx, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def main() -> None:
    args = parse_args()
    latest_runs = collect_latest_runs(args.exp_root)

    sheets: list[tuple[str, list[list[object]]]] = []
    console_lines: list[str] = []

    for method_a, method_b in PAIR_SPECS:
        summary_rows, detail_rows = build_pair_tables(latest_runs, method_a, method_b)
        pair_name = f"{method_a}_vs_{method_b}"
        sheets.append((f"{pair_name}_summary", rows_from_dicts(summary_rows)))
        sheets.append((f"{pair_name}_details", rows_from_dicts(detail_rows)))

        overall = next((row for row in summary_rows if row.get("env_name") == "OVERALL"), None)
        if overall is not None:
            console_lines.append(
                (
                    f"{pair_name}: actor {method_a}={overall.get(f'{method_a}_actor_mean')} "
                    f"vs {method_b}={overall.get(f'{method_b}_actor_mean')} "
                    f"(delta={overall.get('actor_delta')})"
                )
            )

    sheets.append(("metric_notes", build_note_rows(args.output)))
    write_workbook(args.output, sheets)

    print(f"Wrote workbook: {args.output.as_posix()}")
    for line in console_lines:
        print(line)


if __name__ == "__main__":
    main()
