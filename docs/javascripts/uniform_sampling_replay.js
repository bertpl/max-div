// This script steps the solve replay figure of the uniform-sampling case study through its frames.
//
// A figure is an inline SVG (see scripts/uniform_sampling_replay.py) whose `data-points` holds the
// coordinates of every item selected in any frame, and whose `data-frames` holds a JSON list of
// `[elapsed seconds, diversity, [item positions]]` in solve order, positions indexing `data-points`.
// The fragment draws the last frame; showing another frame redraws the data group with that frame's
// items, each as the two rug ticks and the dot that the fragment draws, in data units. Nothing is computed
// here beyond that.
//
// Invariants:
// - one listener set per figure, installed once: the `document$` observable fires on every page
//   navigation;
// - the slider, the caption and the buttons always describe the same frame.
//
// The whole script runs inside one function scope: it shares the page's global scope with the
// explorer script, which declares the same names.

(() => {
const SVG_NS = "http://www.w3.org/2000/svg";

// Append an SVG element with the given attributes to a parent.
function appendElement(parent, tag, attributes) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  parent.appendChild(element);
  return element;
}

// Format a frame's caption the way scripts/uniform_sampling_replay.py formats the last frame's.
function frameCaption(index, frames) {
  const [tSec, diversity] = frames[index];
  return `frame ${index + 1}/${frames.length} · ${tSec.toFixed(1)} s · diversity ${diversity.toFixed(6)}`;
}

// Redraw the data group with one frame's items: per item, its two rug ticks and its dot.
function drawItems(svg, positions) {
  const points = JSON.parse(svg.dataset.points);
  const [near, far] = svg.dataset.rug.split(" ");
  const group = svg.querySelector(".usx-data");
  group.replaceChildren();
  for (const position of positions) {
    const [x, y] = points[position];
    const item = appendElement(group, "g", { class: "usx-item", "data-p": position });
    appendElement(item, "line", { class: "usx-rug usx-rug-x", x1: x, y1: near, x2: x, y2: far, "vector-effect": "non-scaling-stroke" });
    appendElement(item, "line", { class: "usx-rug usx-rug-y", x1: near, y1: y, x2: far, y2: y, "vector-effect": "non-scaling-stroke" });
    appendElement(item, "circle", { class: "usx-dot", cx: x, cy: y, r: svg.dataset.r });
  }
}

// Show frame `index`, clamped to the frames there are, and update the controls to match.
function showFrame(figure, index) {
  const svg = figure.querySelector("svg.usx");
  const frames = JSON.parse(svg.dataset.frames);
  const clamped = Math.min(Math.max(index, 0), frames.length - 1);
  drawItems(svg, frames[clamped][2]);
  figure.querySelector(".usx-slider").value = clamped;
  figure.querySelector(".usx-frame").textContent = frameCaption(clamped, frames);
  figure.querySelector('.usx-step[data-step="-1"]').disabled = clamped === 0;
  figure.querySelector('.usx-step[data-step="1"]').disabled = clamped === frames.length - 1;
}

// Install the listeners on one figure: the slider and the buttons pick a frame, and the arrow keys
// step it while the figure has focus (the slider handles its own arrow keys).
function installReplayListeners(figure) {
  if (figure.dataset.usxReplayReady) return;
  figure.dataset.usxReplayReady = "1";
  const slider = figure.querySelector(".usx-slider");
  slider.addEventListener("input", () => showFrame(figure, Number(slider.value)));
  for (const button of figure.querySelectorAll(".usx-step")) {
    button.addEventListener("click", () => showFrame(figure, Number(slider.value) + Number(button.dataset.step)));
  }
  figure.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
    if (step === undefined || event.target === slider) return;
    event.preventDefault();
    showFrame(figure, Number(slider.value) + step);
  });
  showFrame(figure, Number(slider.value));
}

document$.subscribe(() => {
  for (const figure of document.querySelectorAll(".usx-replay")) installReplayListeners(figure);
});
})();
