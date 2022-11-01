console.log("ugugug starto!")

function test_func(dropdown) {
    return `This is a test, the value is "${dropdown}".`
}

document.addEventListener("DOMContentLoaded", () => {
    console.log("ugugug dom loaded :)");
    (new MutationObserver(e => {
        console.log("ugugug look 4 img ...");
        var img = gradioApp().querySelector("#frame_browser img");
        if (img) {
            console.log("ugugug found img !!!");
            img.addEventListener("mousemove", e => {
                console.log("ugugug " + e);
            });
        }
    })).observe(gradioApp(), {childList: true, subtree: true})
});