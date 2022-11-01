import os
import sys
import gradio as gr
import modules.ui
import time
from modules.shared import opts, cmd_opts
from modules import shared, scripts, paths, script_callbacks
from pathlib import Path
from PIL import Image

try:
    import ffmpeg
except ModuleNotFoundError:
    print("Installing ffmpeg-python")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    import ffmpeg

picker_path = Path(paths.script_path) / "training-picker"
videos_path = picker_path / "videos"
framesets_path = picker_path / "extracted-frames"

current_frame_set = []
current_frame_set_index = 0

for p in [videos_path, framesets_path]:
    os.makedirs(p, exist_ok=True)

def get_videos_list():
    return list(v.name for v in videos_path.iterdir() if v.suffix in [".mp4"])

def get_framesets_list():
    return list(v.name for v in framesets_path.iterdir() if v.is_dir())

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as training_picker:
        videos_list = get_videos_list()
        framesets_list = get_framesets_list()

        # structure
        with gr.Row():
            with gr.Column():
                video_dropdown = gr.Dropdown(choices=videos_list, elem_id="video_dropdown", label="Video to extract keyframes from")
                extract_keyframes_button = gr.Button(value="Extract Keyframes", variant="primary")
                log_output = gr.HTML(value="")
            with gr.Column():
                frameset_dropdown = gr.Dropdown(choices=framesets_list, elem_id="frameset_dropdown", label="Extracted Frame Set", interactive=True)
                frame_browser = gr.Image(interactive=False)
                with gr.Row():
                    prev_button = gr.Button(value="<")
                    frame_counter = gr.HTML(value="")
                    next_button = gr.Button(value=">")

        # events
        def extract_keyframes_button_click(video_file):
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
            return gr.Dropdown.update(choices=get_framesets_list()), f"Successfully created frame set {output_path.name}"
        extract_keyframes_button.click(fn=extract_keyframes_button_click, inputs=[video_dropdown], outputs=[frameset_dropdown, log_output])

        def get_image_update():
            global current_frame_set_index
            global current_frame_set
            return gr.Image.update(value=Image.open(current_frame_set[current_frame_set_index])), f"{current_frame_set_index+1}/{len(current_frame_set)}"

        def frameset_dropdown_change(frameset):
            global current_frame_set_index
            global current_frame_set
            current_frame_set_index = 0
            full_path = framesets_path / frameset
            current_frame_set = [impath for impath in full_path.iterdir() if impath.suffix == ".png"]
            return get_image_update()
        frameset_dropdown.change(fn=frameset_dropdown_change, inputs=[frameset_dropdown], outputs=[frame_browser, frame_counter])

        def prev_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != None:
                current_frame_set_index = (current_frame_set_index - 1) % len(current_frame_set)
            return get_image_update()
        prev_button.click(fn=prev_button_click, inputs=[], outputs=[frame_browser, frame_counter])

        def next_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != None:
                current_frame_set_index = (current_frame_set_index + 1) % len(current_frame_set)
            return get_image_update()
        next_button.click(fn=next_button_click, inputs=[], outputs=[frame_browser, frame_counter])

    return (training_picker, "Training Picker", "training_picker"),

def on_ui_settings():
    section = ('training-picker', "Training Picker")

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)