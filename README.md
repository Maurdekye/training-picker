# training-picker

Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.

![image](https://user-images.githubusercontent.com/2313721/199614791-1f573573-a2e2-4358-836d-5655825077e1.png)

## Installation:

1. Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Install [ffmpeg](https://ffmpeg.org/) for your operating system
3. Clone this repository into the `extensions` folder inside the webui
4. Drop videos you want to extract cropped frames from into the `training-picker/videos` folder

## Usage:

1. Open up the Training Picker tab of the webui
2. Select one of the videos you placed in the `training-picker/videos` folder from the dropdown on the left
3. Click 'Extract Keyframes`
4. After the extraction finishes, a new keyframe set for the video should be selectable in the dropdown on the right
5. Select the keyframe set, and the frames will appear in the browser below
6. Scroll up and down to increase or decrease the size of the crop, and click to save a crop at the mouse cursor position
7. Crops will be saved to `training-picker/cropped-frames` by default
