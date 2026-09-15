// This script adds the interaction to the experiment figures of the uniform-sampling case study.
//
// A figure is an inline SVG (see scripts/uniform_sampling_explorer.py) whose dots and rug ticks
// carry `data-i`, and whose dots carry `data-nn`: a JSON object mapping a distance key to the
// precomputed `[nearest neighbor index, distance]` under that distance. The SVG names the objective's
// distance keys in `data-objective` (one for a simple objective, one per term for a hybrid) and every
// key's label in `data-labels`.
//
// Invariants:
// - one delegated listener set per figure, installed once: the `document$` observable fires on
//   every page navigation;
// - no distance is computed here beyond the level-curve geometry;
// - the hover and marks layers live in the figure's data coordinates, so curves and rings are
//   emitted in data units.

const SVG_NS = "http://www.w3.org/2000/svg";
const REFERENCE_KEYS = ["l2", "x", "y"];
// Glyph per reference key; must match REFERENCE_GLYPHS in scripts/uniform_sampling_explorer.py.
const REFERENCE_GLYPHS = { l2: "dot", x: "|", y: "-" };
// The curves extend past the unit square; the hover layer clips them to it.
const FAR = 2;

// Append an SVG element with the given attributes to a layer; strokes keep their pixel width.
function appendElement(layer, tag, attributes) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  element.setAttribute("vector-effect", "non-scaling-stroke");
  layer.appendChild(element);
}

// ---- Level curves ----

// Return the four branches of |x - cx| * |y - cy| = d^2 as SVG path strings in data units, sampled
// log-spaced so the branches stay smooth near the asymptotes. A shared coordinate (d = 0) degenerates
// the curve into the two axis-parallel lines through the item.
function hyperbolaPaths(cx, cy, d, samples = 80) {
  const d2 = d * d;
  if (d2 <= 0) {
    return [`M${cx},${-FAR}L${cx},${FAR}`, `M${-FAR},${cy}L${FAR},${cy}`];
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

// Return the level curve of a distance at value d around (cx, cy) as SVG path strings in data units:
// - L2: a circle;
// - x or y: the two lines at that coordinate offset;
// - L-inf: the edges of the square of half-side d, each extended outward, since min(|dx|, |dy|) = d
//   holds on |dx| = d where |dy| >= d and on |dy| = d where |dx| >= d;
// - geometric mean: the hyperbolas.
function levelPaths(key, cx, cy, d) {
  switch (key) {
    case "l2":
      return [`M${cx - d},${cy}A${d},${d} 0 1 0 ${cx + d},${cy}A${d},${d} 0 1 0 ${cx - d},${cy}`];
    case "x":
      return [`M${cx - d},${-FAR}L${cx - d},${FAR}`, `M${cx + d},${-FAR}L${cx + d},${FAR}`];
    case "y":
      return [`M${-FAR},${cy - d}L${FAR},${cy - d}`, `M${-FAR},${cy + d}L${FAR},${cy + d}`];
    case "linf":
      return [
        `M${cx - d},${cy + d}L${cx - d},${FAR}`,
        `M${cx + d},${cy + d}L${cx + d},${FAR}`,
        `M${cx - d},${cy - d}L${cx - d},${-FAR}`,
        `M${cx + d},${cy - d}L${cx + d},${-FAR}`,
        `M${cx - d},${cy - d}L${-FAR},${cy - d}`,
        `M${cx - d},${cy + d}L${-FAR},${cy + d}`,
        `M${cx + d},${cy - d}L${FAR},${cy - d}`,
        `M${cx + d},${cy + d}L${FAR},${cy + d}`,
      ];
    default:
      return hyperbolaPaths(cx, cy, d);
  }
}

// Append the level curve of one objective distance through the neighbor under it.
function drawLevel(layer, key, cx, cy, d) {
  for (const path of levelPaths(key, cx, cy, d)) {
    appendElement(layer, "path", { d: path, class: `usx-level usx-level-${key}` });
  }
}

// ---- Reference neighbors ----

// Draw a dashed ring around a reference neighbor's dot, with the glyph of its distance inscribed.
// The ring radius and the glyph sizes come from the fragment, which draws the same ring and glyph in
// its legend.
function drawMark(svg, layer, dot, glyph) {
  const cx = parseFloat(dot.getAttribute("cx"));
  const cy = parseFloat(dot.getAttribute("cy"));
  const radius = parseFloat(svg.dataset.ring) * parseFloat(dot.getAttribute("r"));
  appendElement(layer, "circle", { cx, cy, r: radius.toFixed(5), class: "usx-mark" });
  if (glyph === "dot") {
    const r = parseFloat(svg.dataset.center) * radius;
    appendElement(layer, "circle", { cx, cy, r: r.toFixed(5), class: "usx-glyph-dot" });
    return;
  }
  const reach = parseFloat(svg.dataset.reach) * radius;
  const d = glyph === "|" ? `M${cx},${cy - reach}L${cx},${cy + reach}` : `M${cx - reach},${cy}L${cx + reach},${cy}`;
  appendElement(layer, "path", { d, class: "usx-glyph" });
}

// ---- Selection state ----

// Put the figure back to its idle state.
function clearFigure(figure) {
  const svg = figure.querySelector("svg.usx");
  svg.classList.remove("is-active");
  delete svg.dataset.pinned;
  for (const element of svg.querySelectorAll(".is-picked, .is-neighbor, .is-marked")) {
    element.classList.remove("is-picked", "is-neighbor", "is-marked");
    if (element.classList.contains("usx-dot")) element.setAttribute("r", figure.dataset.dotRadius);
  }
  svg.querySelector(".usx-hover").replaceChildren();
  svg.querySelector(".usx-marks").replaceChildren();
  figure.querySelector(".usx-caption").textContent = figure.dataset.hint;
}

// Pick item i:
// - mark the item and its rug ticks, and, for a single-distance objective, its neighbor under that
//   distance in blue;
// - ring each reference neighbor;
// - draw the level curve of every objective distance through the neighbor under it;
// - write the caption: one clause per objective distance.
function pickItem(figure, i) {
  const svg = figure.querySelector("svg.usx");
  const dot = svg.querySelector(`.usx-dot[data-i="${i}"]`);
  if (!dot) return;
  clearFigure(figure);
  const neighbors = JSON.parse(dot.dataset.nn);
  const labels = JSON.parse(svg.dataset.labels);
  const objectiveKeys = svg.dataset.objective.split(" ");
  svg.classList.add("is-active");
  for (const element of svg.querySelectorAll(`.usx-dot[data-i="${i}"], .usx-rug[data-i="${i}"]`)) {
    element.classList.add("is-picked");
  }
  // The picked dot's radius is set as an attribute, since not every browser honors `r` from CSS.
  dot.setAttribute("r", (1.4 * parseFloat(figure.dataset.dotRadius)).toFixed(4));
  if (objectiveKeys.length === 1) {
    const nn = neighbors[objectiveKeys[0]][0];
    for (const element of svg.querySelectorAll(`.usx-dot[data-i="${nn}"], .usx-rug[data-i="${nn}"]`)) {
      element.classList.add("is-neighbor");
    }
  }
  const marks = svg.querySelector(".usx-marks");
  for (const key of REFERENCE_KEYS) {
    const referenceDot = svg.querySelector(`.usx-dot[data-i="${neighbors[key][0]}"]`);
    referenceDot.classList.add("is-marked");
    drawMark(svg, marks, referenceDot, REFERENCE_GLYPHS[key]);
  }
  const cx = parseFloat(dot.getAttribute("cx"));
  const cy = parseFloat(dot.getAttribute("cy"));
  const clauses = [];
  for (const key of objectiveKeys) {
    const [nn, d] = neighbors[key];
    drawLevel(svg.querySelector(".usx-hover"), key, cx, cy, d);
    clauses.push(`${labels[key]}: item ${nn} at d = ${d.toFixed(3)}`);
  }
  figure.querySelector(".usx-caption").textContent = `item ${i} → nearest under the ${clauses.join("; ")}`;
}

// ---- Wiring ----

// Install the listeners on one figure:
// - hover selects, unless an item is pinned;
// - a tap or click pins; a second one on the same item, or one on empty plot area, clears;
// - keyboard focus selects;
// - Escape clears.
function installFigureListeners(figure) {
  if (figure.dataset.usxReady) return;
  figure.dataset.usxReady = "1";
  figure.dataset.hint = figure.querySelector(".usx-caption").textContent;
  const svg = figure.querySelector("svg.usx");
  figure.dataset.dotRadius = svg.querySelector(".usx-dot").getAttribute("r");
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
  for (const figure of document.querySelectorAll(".usx-figure")) installFigureListeners(figure);
});
