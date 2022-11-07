const intervalStep = 8;

let centerSize = 512;
let aspectRatio = 0;
let mouseOn = false;
let mXPos = 0;
let mYPos = 0;
let xPos = 0;
let yPos = 0;

function cropPreviewRect() {
    var cropPreviewRect = gradioApp().querySelector("#cropPreviewRect");
    if (!cropPreviewRect) {
        cropPreviewRect = document.createElement("div");
        cropPreviewRect.id = "cropPreviewRect";
        gradioApp().getRootNode().appendChild(cropPreviewRect);
    }
    return cropPreviewRect;
}

function getBrushDim() {
    let img = gradioApp().querySelector("#frame_browser img");
    let bound = img.getBoundingClientRect();
    let ar = Math.exp(aspectRatio);
    let brushW = centerSize / ar;
    let brushH = centerSize * ar;
    brushW = Math.min(Math.max(intervalStep, brushW), bound.width);
    brushH = Math.min(Math.max(intervalStep, brushH), bound.height);
    return [brushW, brushH];
}

function updateCpr() {
    let img = gradioApp().querySelector("#frame_browser img");
    let bound = img.getBoundingClientRect();
    let [brushW, brushH] = getBrushDim();
    xPos = Math.min(Math.max(bound.left + brushW / 2 + window.scrollX, mXPos), bound.right - brushW / 2 + window.scrollX);
    yPos = Math.min(Math.max(bound.top + brushH / 2 + window.scrollY, mYPos), bound.bottom - brushH / 2 + window.scrollY);
    let cpr = cropPreviewRect();
    cpr.style.width = brushW + "px";
    cpr.style.height = brushH + "px";
    cpr.style.left = (xPos - brushW / 2) + "px";
    cpr.style.top = (yPos - brushH / 2) + "px";
    cpr.style.display = mouseOn ? 'block' : 'none';
}

function resetAspectRatio() {
    aspectRatio = 0;
    updateCpr();
}

document.addEventListener("DOMContentLoaded", () => {
    (new MutationObserver(e => {
        let img = gradioApp().querySelector("#frame_browser img");
        if (img && (img.getAttribute("listeners") !== "true")) {
            img.addEventListener("mousemove", e => {
                mXPos = e.pageX;
                mYPos = e.pageY;
                updateCpr();
            });
            img.addEventListener("mouseenter", e => {
                mouseOn = true;
                updateCpr();
            });
            img.addEventListener("mouseleave", e => {
                mouseOn = false;
                updateCpr();
            });
            img.addEventListener("wheel", e => {
                e.preventDefault();
                let img = gradioApp().querySelector("#frame_browser img");
                let bound = img.getBoundingClientRect();
                let x = e.deltaY/100;
                if (e.ctrlKey) {
                    aspectRatio -= x * 0.05;
                    aspectRatio = Math.max(-2, Math.min(aspectRatio, 2))
                } else {
                    centerSize -= x * intervalStep;
                    centerSize = Math.max(intervalStep, Math.min(centerSize, Math.max(bound.width, bound.height)))
                }
                updateCpr();
            });
            img.addEventListener("mousedown", e => {
                if (e.button == 0) {
                    updateCpr();
                    let xRatio = img.naturalWidth / img.width;
                    let yRatio = img.naturalHeight / img.height;
                    let bound = img.getBoundingClientRect();
                    let [brushW, brushH] = getBrushDim();
                    let cropData = {
                        x1: Math.floor((xPos - brushW / 2 - bound.left - window.scrollX) * xRatio),
                        y1: Math.floor((yPos - brushH / 2 - bound.top - window.scrollY) * yRatio),
                        x2: Math.floor((xPos + brushW / 2 - bound.left - window.scrollX) * xRatio),
                        y2: Math.floor((yPos + brushH / 2 - bound.top - window.scrollY) * yRatio)
                    };
                    let crop_parameters = gradioApp().querySelector("#crop_parameters textarea");
                    crop_parameters.value = JSON.stringify(cropData);
                    crop_parameters.dispatchEvent(new CustomEvent("input", {})); // necessary to notify gradio that the value has changed
                    gradioApp().querySelector("#crop_button").click();
                } else if (e.button == 1) {
                    resetAspectRatio();
                }
            });
            img.setAttribute("listeners", "true");
        }
    })).observe(gradioApp(), {childList: true, subtree: true});

    document.addEventListener("keydown", e => {
        if (mouseOn) {
            if (e.code == "ArrowRight" || e.code == "KeyD") {
                gradioApp().querySelector("#next_button").click();
            } else if (e.code == "ArrowLeft" || e.code == "KeyA") {
                gradioApp().querySelector("#prev_button").click();
            }
        }
    });
});