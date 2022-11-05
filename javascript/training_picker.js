document.addEventListener("DOMContentLoaded", () => {

    const intervalStep = 8;

    let cropSize = 512;
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

    function updateCpr() {
        let img = gradioApp().querySelector("#frame_browser img");
        let bound = img.getBoundingClientRect();
        xPos = Math.min(Math.max(bound.left + cropSize / 2 + window.scrollX, mXPos), bound.right - cropSize / 2 + window.scrollX);
        yPos = Math.min(Math.max(bound.top + cropSize / 2 + window.scrollY, mYPos), bound.bottom - cropSize / 2 + window.scrollY);
        cropSize = Math.min(Math.max(intervalStep, cropSize), Math.min(bound.width, bound.height));
        let cpr = cropPreviewRect();
        cpr.style.width = cropSize + "px";
        cpr.style.height = cropSize + "px";
        cpr.style.left = (xPos - cropSize / 2) + "px";
        cpr.style.top = (yPos - cropSize / 2) + "px";
        cpr.style.display = mouseOn ? 'block' : 'none';
    }

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
                cropSize -= (e.deltaY/100) * intervalStep;
                updateCpr();
            });
            img.addEventListener("click", e => {
                updateCpr();
                let xRatio = img.naturalWidth / img.width;
                let yRatio = img.naturalHeight / img.height;
                let bound = img.getBoundingClientRect();
                let cropData = {
                    x1: Math.floor((xPos - cropSize / 2 - bound.left - window.scrollX) * xRatio),
                    y1: Math.floor((yPos - cropSize / 2 - bound.top - window.scrollY) * yRatio),
                    x2: Math.floor((xPos + cropSize / 2 - bound.left - window.scrollX) * xRatio),
                    y2: Math.floor((yPos + cropSize / 2 - bound.top - window.scrollY) * yRatio)
                };
                let crop_parameters = gradioApp().querySelector("#crop_parameters textarea");
                crop_parameters.value = JSON.stringify(cropData);
                crop_parameters.dispatchEvent(new CustomEvent("input", {})); // necessary to notify gradio that the value has changed
                gradioApp().querySelector("#crop_button").click();
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