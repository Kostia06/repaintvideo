"""Gradio web demo for video style transfer."""
import glob
import os
import tempfile

import gradio as gr

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from style_transfer import StyleTransferEngine

MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints")


def _available_models() -> list[str]:
    return sorted(glob.glob(f"{MODEL_DIR}/*.pth"))


def _model_choices() -> list[str]:
    return [os.path.basename(p).replace(".pth", "") for p in _available_models()]


def process_video(
    video_path: str,
    style_name: str,
    strength: float,
) -> str | None:
    models = _available_models()
    names = _model_choices()

    if style_name not in names:
        raise gr.Error(f"Style '{style_name}' not found. Available: {names}")

    model_path = models[names.index(style_name)]
    output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    engine = StyleTransferEngine()
    engine.apply_style_video(
        input_path=video_path,
        style=style_name,
        output_path=output,
    )
    return output


def build_app() -> gr.Blocks:
    style_images = sorted(glob.glob("styles/*.jpg") + glob.glob("styles/*.png"))

    with gr.Blocks(title="Video Style Transfer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Video Neural Style Transfer")
        gr.Markdown("Upload a video, pick a style, and get a temporally consistent styled result.")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
                style_dropdown = gr.Dropdown(
                    choices=_model_choices() or ["(no models — train one first)"],
                    label="Style",
                    value=_model_choices()[0] if _model_choices() else None,
                )
                strength_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    label="Style Strength",
                )
                run_btn = gr.Button("Stylize", variant="primary")

                if style_images:
                    gr.Markdown("### Style References")
                    gr.Gallery(
                        value=[(img, os.path.basename(img)) for img in style_images],
                        columns=3,
                        height=200,
                    )

            with gr.Column(scale=1):
                video_output = gr.Video(label="Styled Output")
                download_btn = gr.File(label="Download")

        def on_run(video: str, style: str, strength: float) -> tuple[str, str]:
            result = process_video(video, style, strength)
            return result, result

        run_btn.click(
            fn=on_run,
            inputs=[video_input, style_dropdown, strength_slider],
            outputs=[video_output, download_btn],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_port=7860)
