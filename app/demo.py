"""Gradio QA UI — calls the same PPEDetector used by the API."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import PPEDetector  # noqa: E402

_detector: PPEDetector | None = None


def get_detector() -> PPEDetector:
    global _detector
    if _detector is None:
        _detector = PPEDetector()
    return _detector


def run(image: np.ndarray):
    if image is None:
        return None, "Upload an image."
    det = get_detector()
    # Gradio gives RGB
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = det.predict(bgr)
    annotated_rgb = cv2.cvtColor(out["annotated_bgr"], cv2.COLOR_BGR2RGB)
    summary = out["summary"]
    lines = [
        f"Compliant: {summary['compliant']}",
        f"Persons: {summary['person_count']}",
        f"Worn PPE: {', '.join(summary['worn_ppe']) or '—'}",
        f"Violations: {', '.join(summary['violation_codes']) or 'none'}",
        "",
        "Details:",
    ]
    for v in summary["violations"]:
        lines.append(
            f"- {v['code']} (conf={v['confidence']:.2f}, severity={v['severity']})"
        )
    if not summary["violations"]:
        lines.append("- No missing-PPE classes above threshold.")
    return annotated_rgb, "\n".join(lines)


def main():
    with gr.Blocks(title="PPE Safety Vision") as demo:
        gr.Markdown(
            "# PPE Safety Vision\n"
            "Upload a construction-site image. The model draws detections and lists PPE violations."
        )
        with gr.Row():
            inp = gr.Image(type="numpy", label="Input")
            out_img = gr.Image(type="numpy", label="Annotated")
        out_txt = gr.Textbox(label="Compliance summary", lines=12)
        btn = gr.Button("Run detection", variant="primary")
        btn.click(fn=run, inputs=inp, outputs=[out_img, out_txt])
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        footer_links=[],  # hide Use via API / Built with Gradio / Settings
    )


if __name__ == "__main__":
    main()
