// This script adds the interaction to the example figure of the geometric-mean distance guide.
//
// The figure is an inline SVG (see scripts/geomean_distance_explorer.py) whose dots and rug ticks
// carry `data-i`, and whose dots carry the precomputed nearest neighbor under the geometric-mean
// distance (`data-nn`, `data-d`, `data-dx`, `data-dy`) and under the Euclidean one (`data-nne`).
//
// Invariants:
// - one delegated listener set per figure, installed once: the `document$` observable fires on
//   every page navigation;
// - no distance is computed here beyond the level-curve geometry;
// - the hover layer lives in the figure's data coordinates, so the curves are emitted in data units.

// ---- Level curves ----

// Return the four branches of |x - cx| * |y - cy| = d^2 as SVG path strings in data units, sampled
// log-spaced so the branches stay smooth near the asymptotes. A shared coordinate (d = 0) degenerates
// the curve into the two axis-parallel lines through the item.
function hyperbolaPaths(cx, cy, d, samples = 80) {
  const d2 = d * d;
  if (d2 <= 0) {
    return [`M${cx},-1L${cx},2`, `M-1,${cy}L2,${cy}`];
  }
  const paths = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const points = [];
      for (let t = 0; t <= samples; t++) {
        const ax = d2 * Math.pow(1 / d2, t / samples);
        const ay = d2 / ax;
        points.push(`${(cx + sx * ax).toFixed(5)},${(cy + sy * ay).toFixed(5)}`);
      }
      paths.push("M" + points.join("L"));
    }
  }
  return paths;
}

// Append the level curves through the neighbor and at the 1/sqrt(k) reference.
function drawLevels(layer, cx, cy, d, reference) {
  const svgNs = "http://www.w3.org/2000/svg";
  const draw = (level, cls) => {
    for (const path of hyperbolaPaths(cx, cy, level)) {
      const element = document.createElementNS(svgNs, "path");
      element.setAttribute("d", path);
      element.setAttribute("class", cls);
      element.setAttribute("vector-effect", "non-scaling-stroke");
      layer.appendChild(element);
    }
  };
  draw(reference, "gmx-level gmx-level-ref");
  draw(d, "gmx-level gmx-level-nn");
}

// ---- Selection state ----

// Put the figure back to its idle state.
function clearFigure(figure) {
  const svg = figure.querySelector("svg.gmx");
  svg.classList.remove("is-active");
  delete svg.dataset.pinned;
  for (const element of svg.querySelectorAll(".is-picked, .is-neighbor, .is-euclid")) {
    element.classList.remove("is-picked", "is-neighbor", "is-euclid");
    if (element.classList.contains("gmx-dot")) element.setAttribute("r", figure.dataset.dotRadius);
  }
  svg.querySelector(".gmx-hover").replaceChildren();
  figure.querySelector(".gmx-caption").textContent = figure.dataset.hint;
}

// Pick item i:
// - mark the item, its two neighbors and their rug ticks;
// - draw the level curves;
// - write the caption.
function pickItem(figure, i) {
  const svg = figure.querySelector("svg.gmx");
  const dot = svg.querySelector(`.gmx-dot[data-i="${i}"]`);
  if (!dot) return;
  clearFigure(figure);
  const nn = dot.dataset.nn;
  const nne = dot.dataset.nne;
  svg.classList.add("is-active");
  for (const element of svg.querySelectorAll(`.gmx-dot[data-i="${i}"], .gmx-rug[data-i="${i}"]`)) {
    element.classList.add("is-picked");
  }
  // The picked dot's radius is set as an attribute, since not every browser honors `r` from CSS.
  dot.setAttribute("r", (1.4 * parseFloat(figure.dataset.dotRadius)).toFixed(4));
  for (const element of svg.querySelectorAll(`.gmx-dot[data-i="${nn}"], .gmx-rug[data-i="${nn}"]`)) {
    element.classList.add("is-neighbor");
  }
  if (nne !== nn) {
    svg.querySelector(`.gmx-dot[data-i="${nne}"]`).classList.add("is-euclid");
  }
  const cx = parseFloat(dot.getAttribute("cx"));
  const cy = parseFloat(dot.getAttribute("cy"));
  const d = parseFloat(dot.dataset.d);
  const reference = parseFloat(svg.dataset.ref);
  drawLevels(svg.querySelector(".gmx-hover"), cx, cy, d, reference);
  const dx = parseFloat(dot.dataset.dx).toFixed(3);
  const dy = parseFloat(dot.dataset.dy).toFixed(3);
  figure.querySelector(".gmx-caption").textContent =
    `item ${i} → nearest item ${nn}: d = √(|Δx| · |Δy|) = √(${dx} × ${dy}) = ${d.toFixed(3)}` +
    ` (1/√k = ${reference.toFixed(3)})`;
}

// ---- Wiring ----

// Install the listeners on one figure:
// - hover selects, unless an item is pinned;
// - a tap or click pins; a second one on the same item, or one on empty plot area, clears;
// - keyboard focus selects;
// - Escape clears.
function installFigureListeners(figure) {
  if (figure.dataset.gmxReady) return;
  figure.dataset.gmxReady = "1";
  figure.dataset.hint = figure.querySelector(".gmx-caption").textContent;
  const svg = figure.querySelector("svg.gmx");
  figure.dataset.dotRadius = svg.querySelector(".gmx-dot").getAttribute("r");
  const itemOf = (event) => event.target.closest("[data-i]");
  svg.addEventListener("pointerover", (event) => {
    const item = itemOf(event);
    if (item && !svg.dataset.pinned) pickItem(figure, item.dataset.i);
  });
  svg.addEventListener("click", (event) => {
    const item = itemOf(event);
    if (!item || svg.dataset.pinned === item.dataset.i) {
      clearFigure(figure);
      return;
    }
    pickItem(figure, item.dataset.i);
    svg.dataset.pinned = item.dataset.i;
  });
  svg.addEventListener("focusin", (event) => {
    const item = itemOf(event);
    if (item && !svg.dataset.pinned) pickItem(figure, item.dataset.i);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") clearFigure(figure);
  });
}

document$.subscribe(() => {
  for (const figure of document.querySelectorAll(".gmx-figure")) installFigureListeners(figure);
});
