import subprocess
import platform
import json
import sys
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from modules.ui import create_refresh_button, folder_symbol
from modules import shared, paths, script_callbacks

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

def resized_background(im, color):
    dim = max(*im.size)
    w, h = im.size
    background = Image.new(mode="RGBA", size=(dim, dim), color=color)
    background.paste(im, (dim // 2 - w // 2, dim // 2 - h // 2))
    return background

# Outfill methods

def no_outfill(im, **kwargs):
    return im

def stretch(im, **kwargs):
    dim = max(*im.size)
    return im.resize((dim, dim))

def transparent(im, **kwargs):
    return resized_background(im, (0,0,0,0))

def solid(im, **kwargs):
    return resized_background(im, kwargs['color'])

def average(im, **kwargs):
    return resized_background(im, tuple(int(x) for x in np.asarray(im).mean(axis=0).mean(axis=0)))

def dominant(im, **kwargs):
    try:
        import cv2
    except ModuleNotFoundError:
        print("Installing opencv-python")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        import cv2
    _, labels, palette = cv2.kmeans(
        np.float32(np.asarray(im).reshape(-1, 3)), 
        kwargs['n_clusters'], 
        None, 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1), 
        10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    _, counts = np.unique(labels, return_counts=True)
    color = palette[np.argmax(counts)]
    return resized_background(im, tuple(int(x) for x in color))

def border_stretch(im, **kwargs):
    w, h = im.size
    arr = np.asarray(im)
    n = abs(w - h) // 2
    if w > h:
        arr = np.repeat(arr, [n] + [1]*(h-1), axis=0)
        arr = np.repeat(arr, [1]*(h+n-2) + [n], axis=0)
    else:
        arr = np.repeat(arr, [n] + [1]*(w-1), axis=1)
        arr = np.repeat(arr, [1]*(w+n-2) + [n], axis=1)
    return Image.fromarray(arr)

def reflect(im, **kwargs):
    w, h = im.size
    arr = np.asarray(im)
    base = arr.copy()
    if w > h:
        while arr.shape[0] < arr.shape[1]:
            arr = np.concatenate((arr, base[::-1,:]))
            arr = np.concatenate((base[::-1,:], arr))
            base = base[::-1,:]
        n = abs(arr.shape[0] - arr.shape[1]) // 2
        arr = arr[n:-n, :]
    else:
        while arr.shape[1] < arr.shape[0]:
            arr = np.concatenate((arr, base[:,::-1]), axis=1)
            arr = np.concatenate((base[:,::-1], arr), axis=1)
            base = base[:,::-1]
        n = abs(arr.shape[0] - arr.shape[1]) // 2
        arr = arr[:, n:-n]
    return Image.fromarray(arr)

def blur(im, **kwargs):
    dim = max(*im.size)
    w, h = im.size
    background = im.resize((dim, dim))
    background = background.filter(ImageFilter.GaussianBlur(kwargs['blur']))
    background.paste(im, (dim // 2 - w // 2, dim // 2 - h // 2))
    return background

outfill_methods = {
    "Don't outfill": no_outfill,
    "Stretch image": stretch,
    "Transparent": transparent,
    "Solid color": solid,
    "Average image color": average,
    "Dominant image color": dominant,
    "Stretch pixels at border": border_stretch,
    "Reflect image around border": reflect,
    "Blurred & stretched overlay": blur
}

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
                extract_frames_button = gr.Button(value="Extract Frames", variant="primary")
                log_output = gr.HTML(value="")
            with gr.Column():
                with gr.Row():
                    frameset_dropdown = gr.Dropdown(choices=framesets_list, elem_id="frameset_dropdown", label="Extracted Frame Set", interactive=True)
                    create_refresh_button(frameset_dropdown, lambda: None, lambda: {"choices": get_framesets_list()}, "refresh_framesets_list")
                    create_open_folder_button(framesets_path, "open_folder_framesets")
                with gr.Row():
                    resize_checkbox = gr.Checkbox(value=True, label="Resize crops to 512x512")
                    outfill_setting = gr.Dropdown(choices=list(outfill_methods.keys()), value="Don't outfill", label="Outfill method:", interactive=True)
                    with gr.Row():
                        reset_aspect_ratio_button = gr.Button(value="Reset Aspect Ratio")
                        bulk_process_button = gr.Button(value="Bulk process frames with chosen outfill method")
                with gr.Row(visible=False) as outfill_setting_options:
                    outfill_color = gr.ColorPicker(value="#000000", label="Outfill border color:", visible=False, interactive=True)
                    outfill_border_blur = gr.Slider(value=0, min=0, max=20, step=0.01, label="Blur amount:", visible=False, interactive=True)
                    outfill_n_clusters = gr.Slider(value=5, min=1, max=50, step=1, label="Number of clusters:", visible=False, interactive=True)
                with gr.Row():
                    output_dir = gr.Text(value=picker_path / "cropped-frames", label="Save crops to:")
                    create_open_folder_button(output_dir, "open_folder_crops")
        with gr.Row():
            with gr.Column():
                crop_preview = gr.Image(interactive=False, elem_id="crop_preview", show_label=False)
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
        def extract_frames_button_click(video_file, only_keyframes):
            try:
                import ffmpeg
            except ModuleNotFoundError:
                print("Installing ffmpeg-python")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
                import ffmpeg
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
        extract_frames_button.click(fn=extract_frames_button_click, inputs=[video_dropdown, only_keyframes_checkbox], outputs=[frameset_dropdown, log_output])

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

        def process_image(image, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters):
            if should_resize:
                w, h = image.size
                ratio = 512 / max(w, h)
                image = image.resize((int(w / ratio), int(h / ratio)))
            return outfill_methods[outfill_setting](image, color=outfill_color, blur=outfill_border_blur, n_clusters=outfill_n_clusters)

        def crop_button_click(raw_params, frame_browser, should_resize, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters):
            params = json.loads(raw_params)
            im = Image.fromarray(frame_browser)
            cropped = im.crop((params['x1'], params['y1'], params['x2'], params['y2']))
            cropped = process_image(cropped, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters)
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
        crop_button.click(fn=crop_button_click, inputs=[crop_parameters, frame_browser, resize_checkbox, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters], outputs=[crop_preview, log_output])

        def bulk_process_button_click(frameset, should_resize, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters):
            for frame in tqdm((framesets_path / frameset).iterdir()):
                if frame.suffix == ".png":
                    with Image.open(frame) as img:
                        img = process_image(img, should_resize, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters)
                        save_path = Path(output_dir)
                        os.makedirs(str(save_path.resolve()), exist_ok=True)
                        img.save(Path(output_dir) / frame.name)
            return f'Processed images saved to "{output_dir}"!'
        bulk_process_button.click(fn=bulk_process_button_click, inputs=[frameset_dropdown, resize_checkbox, output_dir, outfill_setting, outfill_color, outfill_border_blur, outfill_n_clusters], outputs=[log_output])

        def outfill_setting_change(outfill_setting): 
            outfill_outputs = [
                "outfill_setting_options",
                "outfill_color",
                "outfill_border_blur",
                "outfill_n_clusters"
            ]
            visibility_pairs = {
                "Solid color": [
                    "outfill_setting_options",
                    "outfill_color"
                ],
                "Blurred & stretched overlay" : [
                    "outfill_setting_options",
                    "outfill_border_blur"
                ],
                "Dominant image color": [
                    "outfill_setting_options",
                    "outfill_n_clusters"
                ]
            }
            return [gr.update(visible=(outfill_setting in visibility_pairs and o in visibility_pairs[outfill_setting])) for o in outfill_outputs]
        outfill_setting.change(fn=outfill_setting_change, inputs=[outfill_setting], outputs=[outfill_setting_options, outfill_color, outfill_border_blur, outfill_n_clusters])

        reset_aspect_ratio_button.click(fn=None, _js="resetAspectRatio", inputs=[], outputs=[])

    return (training_picker, "Training Picker", "training_picker"),

def on_ui_settings():
    section = ('training-picker', "Training Picker")

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)