import subprocess
import platform
import json
import sys
import os
import re
from pathlib import Path

import gradio as gr
from PIL import Image

from modules.ui import create_refresh_button, folder_symbol
from modules import shared, paths, script_callbacks

try:
    import ffmpeg
except ModuleNotFoundError:
    print("Installing ffmpeg-python")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    import ffmpeg

picker_path = Path(paths.script_path) / "training-picker"
videos_path = picker_path / "videos"
framesets_path = picker_path / "extracted-frames"

current_frame_set = []
current_frame_set_index = 0

for p in [videos_path, framesets_path]:
    os.makedirs(p, exist_ok=True)

# copied directly from ui.py
# if it were not defined inside another function, i would import it instead
def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

def create_open_folder_button(path, elem_id):
    button = gr.Button(folder_symbol, elem_id=elem_id)
    if 'gradio.templates' in getattr(path, "__module__", ""):
        button.click(fn=lambda p: open_folder(p), inputs=[path], outputs=[])
    else:
        button.click(fn=lambda: open_folder(path), inputs=[], outputs=[])
    return button

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
                with gr.Row():
                    video_dropdown = gr.Dropdown(choices=videos_list, elem_id="video_dropdown", label="Video to extract frames from:")
                    create_refresh_button(video_dropdown, lambda: None, lambda: {"choices": get_videos_list()}, "refresh_videos_list")
                    create_open_folder_button(videos_path, "open_folder_videos")
                only_keyframes_checkbox = gr.Checkbox(value=True, label="Only extract keyframes (recommended)")
                extract_keyframes_button = gr.Button(value="Extract Keyframes", variant="primary")
                log_output = gr.HTML(value="")
            with gr.Column():
                with gr.Row():
                    frameset_dropdown = gr.Dropdown(choices=framesets_list, elem_id="frameset_dropdown", label="Extracted Frame Set", interactive=True)
                    create_refresh_button(frameset_dropdown, lambda: None, lambda: {"choices": get_framesets_list()}, "refresh_framesets_list")
                    create_open_folder_button(framesets_path, "open_folder_framesets")
                with gr.Row():
                    resize_checkbox = gr.Checkbox(value=True, label="Resize crops to 512x512")
                    gr.Column()
                with gr.Row():
                    output_dir = gr.Text(value=picker_path / "cropped-frames", label="Save crops to:")
                    create_open_folder_button(output_dir, "open_folder_crops")
        with gr.Row():
            with gr.Column():
                    with gr.Row():
                        gr.Column()
                        with gr.Column(scale=2):
                            crop_preview = gr.Image(interactive=False, elem_id="crop_preveiw", show_label=False)
                        gr.Column()
            with gr.Column():
                    frame_browser = gr.Image(interactive=False, elem_id="frame_browser", show_label=False)
                    with gr.Row():
                        prev_button = gr.Button(value="<", elem_id="prev_button")
                        with gr.Row():
                            frame_number = gr.Number(value=0, elem_id="frame_number", live=True, show_label=False)
                            frame_max = gr.HTML(value="", elem_id="frame_max")
                        next_button = gr.Button(value=">", elem_id="next_button")
        
        # invisible elements
        crop_button = gr.Button(elem_id="crop_button", visible=False)
        crop_parameters = gr.Text(elem_id="crop_parameters", visible=False)

        # events
        def extract_keyframes_button_click(video_file, only_keyframes):
            try:
                print(f"Extracting frames from {video_file}")
                full_path = videos_path / video_file
                output_path = framesets_path / Path(video_file).stem
                if output_path.is_dir():
                    print("Directory already exists!")
                    return gr.update(), f"Frame set already exists at {output_path}! Delete the folder first if you would like to recreate it."
                os.makedirs(output_path, exist_ok=True)
                if only_keyframes:
                    stream = ffmpeg.input(
                        str(full_path),
                        skip_frame="nokey",
                        vsync="vfr")
                else:
                    stream = ffmpeg.input(
                        str(full_path)
                    )
                stream.output(str((output_path / "%02d.png").resolve())).run()
                print("Extraction complete!")
                return gr.Dropdown.update(choices=get_framesets_list()), f"Successfully created frame set {output_path.name}"
            except Exception as e:
                print(f"Exception encountered while attempting to extract frames: {e}")
                return gr.update(), f"Error: {e}"
        extract_keyframes_button.click(fn=extract_keyframes_button_click, inputs=[video_dropdown, only_keyframes_checkbox], outputs=[frameset_dropdown, log_output])

        def get_image_update():
            global current_frame_set_index
            global current_frame_set
            return gr.Image.update(value=Image.open(current_frame_set[current_frame_set_index])), current_frame_set_index+1, f"/{len(current_frame_set)}"

        def null_image_update():
            return gr.update(), 0, ""

        def frameset_dropdown_change(frameset):
            global current_frame_set_index
            global current_frame_set
            current_frame_set_index = 0
            full_path = framesets_path / frameset
            current_frame_set = [impath for impath in full_path.iterdir() if impath.suffix == ".png"]
            return get_image_update()
        frameset_dropdown.change(fn=frameset_dropdown_change, inputs=[frameset_dropdown], outputs=[frame_browser, frame_number, frame_max])

        def prev_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = (current_frame_set_index - 1) % len(current_frame_set)
                return get_image_update()
            return null_image_update()
        prev_button.click(fn=prev_button_click, inputs=[], outputs=[frame_browser, frame_number, frame_max])

        def next_button_click():
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = (current_frame_set_index + 1) % len(current_frame_set)
                return get_image_update()
            return null_image_update()
        next_button.click(fn=next_button_click, inputs=[], outputs=[frame_browser, frame_number, frame_max])

        def frame_number_change(frame_number):
            global current_frame_set_index
            global current_frame_set
            if current_frame_set != []:
                current_frame_set_index = int(min(max(0, frame_number - 1), len(current_frame_set) - 1))
                return get_image_update()
            return null_image_update()
        frame_number.change(fn=frame_number_change, inputs=[frame_number], outputs=[frame_browser, frame_number, frame_max])

        def crop_button_click(raw_params, frame_browser, should_resize, output_dir):
            params = json.loads(raw_params)
            im = Image.fromarray(frame_browser)
            cropped = im.crop((params['x1'], params['y1'], params['x2'], params['y2']))
            if should_resize:
                cropped = cropped.resize((512, 512))
            save_path = Path(output_dir)
            os.makedirs(str(save_path.resolve()), exist_ok=True)
            current_images = [r for r in (re.match(r"(\d+).png", f.name) for f in save_path.iterdir()) if r]
            if current_images == []:
                next_image_num = 0
            else:
                next_image_num = 1 + max(int(r.group(1)) for r in current_images)
            filename = save_path / f"{next_image_num}.png"
            cropped.save(filename)
            return gr.Image.update(value=cropped), f"Saved to {filename}"
        crop_button.click(fn=crop_button_click, inputs=[crop_parameters, frame_browser, resize_checkbox, output_dir], outputs=[crop_preview, log_output])

    return (training_picker, "Training Picker", "training_picker"),

def on_ui_settings():
    section = ('training-picker', "Training Picker")

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)