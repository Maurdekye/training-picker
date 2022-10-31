import os
import sys
import gradio as gr
import modules.ui
import time
from modules.shared import opts, cmd_opts
from modules import shared, scripts, paths, script_callbacks
from pathlib import Path

try:
    import ffmpeg
except ModuleNotFoundError:
    print("Installing ffmpeg-python")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    import ffmpeg

def on_ui_tabs():
    picker_path = Path(paths.script_path) / "training-picker"
    videos_path = picker_path / "videos"
    framesets_path = picker_path / "extracted-frames"
    for p in [videos_path, framesets_path]:
        os.makedirs(p, exist_ok=True)
    
    videos_list = list(v.name for v in videos_path.iterdir() if v.suffix in [".mp4"])

    with gr.Blocks(analytics_enabled=False) as training_picker:
        with gr.Row():
            with gr.Column():
                input_video = gr.Dropdown(videos_list, elem_id="training_picker_video", label="Video to extract keyframes from")
                extract_frames = gr.Button(value="Extract Keyframes", variant="primary")
                def extract(video_file):
                    print(f"Extracting frames from {video_file}")
                    full_path = videos_path / video_file
                    output_path = framesets_path / Path(video_file).stem
                    os.makedirs(output_path, exist_ok=True)
                    (
                        ffmpeg
                        .input(str(full_path),
                            skip_frame="nokey",
                            vsync="vfr")
                        .output(str((output_path / "%02d.png").resolve()))
                        .run()
                    )
                    print("Extraction complete!")
                extract_frames.click(extract, inputs=[input_video], outputs=[])
            with gr.Column():
                pass
    return (training_picker, "Training Picker", "training_picker"),

def on_ui_settings():
    section = ('training-picker', "Training Picker")
    # shared.opts.add_option("images_history_preload", shared.OptionInfo(False, "Preload images at startup", section=section))
    # shared.opts.add_option("images_history_page_columns", shared.OptionInfo(6, "Number of columns on the page", section=section))
    # shared.opts.add_option("images_history_page_rows", shared.OptionInfo(6, "Number of rows on the page", section=section))
    # shared.opts.add_option("images_history_pages_perload", shared.OptionInfo(20, "Minimum number of pages per load", section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)