"""Standalone request report rendering from compute-side analysis results."""
from html import escape
from pathlib import Path

TTFT_COMPONENTS = (
    ("preprocess", "Preprocess", "#4E79A7"),
    ("enginecore_queue", "EngineCore Queue", "#F28E2B"),
    ("cpu_sched_wait", "CPU Sched Wait", "#E15759"),
    ("gpu_dispatch_wait", "GPU Dispatch Wait", "#FF9D4D"),
    ("prefill_exec", "Prefill Exec", "#76B7B2"),
    ("os_sched_wait", "OS Sched Delay (eBPF)", "#B07AA1"),
    ("postprocess_transport", "Postprocess / Transport", "#59A14F"),
    ("other_gap", "Other Gap", "#BAB0AC"),
)
TPOT_COMPONENTS = (
    ("cpu_queue", "CPU Queue", "#E15759"),
    ("gpu_queue", "GPU Queue", "#FF9D4D"),
    ("decode_exec", "Decode Exec", "#4E79A7"),
    ("tail_transport", "Tail Transport", "#86BCB6"),
    ("other_gap", "Other Gap", "#BAB0AC"),
)


def sanitize_filename(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name)).strip("._") or "request"


def render_request_breakdown_svg(svg_path, rid, request_name, ttft_entry, tpot_entry):
    text = lambda value: escape(str(value), quote=True)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="800" viewBox="0 0 1500 800">',
             '<rect width="100%" height="100%" fill="#FAF7F2"/>',
             f'<text x="48" y="40" font-size="24">{text(request_name or rid)}</text>',
             f'<text x="48" y="65" font-size="14">request_id={text(rid)}</text>']
    y = 105
    for name, entry, fields, value_key, total_key in (
        ("TTFT", ttft_entry, TTFT_COMPONENTS, "components_ms", "window_ms"),
        ("TPOT (ms/token)", tpot_entry, TPOT_COMPONENTS, "components_ms_per_token", "avg_ms_per_token"),
    ):
        if entry is None:
            lines.append(f'<text x="48" y="{y}" font-size="18">{name}: unavailable</text>')
            y += 40
            continue
        total = entry[total_key]
        lines.append(f'<text x="48" y="{y}" font-size="20">{name}: {total:.3f}</text>')
        y += 30
        for key, label, color in fields:
            value = entry[value_key].get(key, 0)
            width = min(520, max(0, 520 * value / total)) if total > 0 else 0
            lines.extend([
                f'<rect x="110" y="{y}" width="520" height="12" fill="#E5E7EB"/>',
                f'<rect x="110" y="{y}" width="{width:.2f}" height="12" fill="{color}"/>',
                f'<text x="650" y="{y+12}" font-size="14">{text(label)}: {value:.3f} ms</text>',
            ])
            y += 26
        y += 30
    lines.append('</svg>')
    Path(svg_path).write_text("\n".join(lines), encoding="utf-8")


def render_request_breakdown_index(index_path, request_rows):
    lines = ['<!doctype html><html><head><meta charset="utf-8"><title>Request breakdown</title></head><body>',
             '<h1>Per-Request Compute Breakdown</h1><table><tr><th>Request</th><th>TTFT (ms)</th><th>TPOT (ms/token)</th><th>Prefill Re-entry</th><th>Chart</th></tr>']
    for row in request_rows:
        cells = "".join(f"<td>{escape(str(row[key]))}</td>" for key in ("request_name", "ttft_ms", "tpot_ms", "reentry"))
        lines.append(f'<tr>{cells}<td><a href="{escape(row["file_name"], quote=True)}">open</a></td></tr>')
    lines.append('</table></body></html>')
    Path(index_path).write_text("\n".join(lines), encoding="utf-8")
