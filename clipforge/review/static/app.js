/* ClipForge review — §7.3's keyboard loop.
 *
 * C4's target is 120 candidates in under 8 minutes, so the guiding rule here is
 * that a keypress never waits on the network. Every candidate arrives in the
 * first payload; ratings are fired off without awaiting a response; navigation
 * only moves a highlight and seeks a <video> that is already loaded.
 *
 * review_ms is measured from the moment a candidate gains focus to the moment
 * it is rated (§7.5). That is the honest observation even when it includes a
 * coffee break, which is why `clipforge metrics` reports the median.
 */

const state = {
  streamId: null,
  stream: null,
  all: [],
  view: [],
  cursor: 0,
  markersOnly: false,
  showSignals: false,
  focusedAt: 0,
  sessionStart: 0,
  reviewed: 0,
  playTimer: null,
};

/* Anything that changes something must carry this. The value is never checked —
 * sending it at all is what forces a CORS preflight, which is what stops
 * another page in the browser from posting here. See review/guard.py. */
const POST = {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-ClipForge": "1" },
};

const $ = (id) => document.getElementById(id);
const fmt = (s) => {
  const m = Math.floor(s / 60);
  const rest = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${rest}`;
};

/* ------------------------------------------------------------- bootstrap */

/* `force` skips the auto-open shortcuts.
 *
 * Without it, "back" was unusable: openStream puts ?stream=<id> in the URL so a
 * reload resumes where you were, and boot() honoured that param — so leaving a
 * stream immediately re-entered it. The same applied when only one stream was
 * reviewable, which auto-opens. Going back has to mean the picker, always. */
async function boot(force = false) {
  const res = await fetch("/api/streams");
  const { streams } = await res.json();

  if (force) {
    history.replaceState(null, "", location.pathname);
    return showPicker(streams);
  }

  const fromUrl = new URLSearchParams(location.search).get("stream");
  if (fromUrl && streams.some((s) => s.id === fromUrl)) return openStream(fromUrl);

  const reviewable = streams.filter((s) => s.candidates > 0 && s.has_proxy);
  if (reviewable.length === 1) return openStream(reviewable[0].id);
  showPicker(streams);
}

function showPicker(streams) {
  $("review").hidden = true;
  $("summary").hidden = true;
  $("picker").hidden = false;

  const list = $("stream-list");
  list.innerHTML = "";
  const reviewable = streams.filter((s) => s.candidates > 0 && s.has_proxy);
  $("picker-empty").hidden = reviewable.length > 0;

  for (const s of streams) {
    const ready = s.candidates > 0 && s.has_proxy;
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.disabled = !ready;
    button.onclick = () => openStream(s.id);

    const why = !s.has_proxy ? "no proxy yet" : s.candidates === 0 ? "not scored yet" : "";
    button.innerHTML = `
      <span class="grow">
        <span class="name">${escape(s.title || s.id)}</span><br>
        <span class="muted">${s.date} · ${s.duration_s ? fmt(s.duration_s) : "?"} ·
        ${escape(s.resolution || "")} ${why ? "· " + why : ""}</span>
      </span>
      <span class="count">${s.candidates}</span>
      <span class="muted">${s.rated}/${s.candidates} rated</span>`;
    li.appendChild(button);
    list.appendChild(li);
  }
}

async function openStream(id) {
  const res = await fetch(`/api/streams/${encodeURIComponent(id)}/candidates`);
  if (!res.ok) return showPicker([]);
  const data = await res.json();

  state.streamId = id;
  state.stream = data.stream;
  state.all = data.candidates;
  state.cursor = 0;
  state.markersOnly = false;
  state.sessionStart = performance.now();
  state.reviewed = 0;

  $("picker").hidden = true;
  $("summary").hidden = true;
  $("review").hidden = false;

  $("stream-name").textContent = data.stream.title || id;
  $("stream-meta").textContent =
    `${fmt(data.stream.duration_s || 0)} · ${data.stream.resolution || ""}`;

  const warn = $("warnings");
  if (data.stream.warnings?.length) {
    warn.hidden = false;
    warn.textContent = data.stream.warnings.join("  ");
  } else {
    warn.hidden = true;
  }

  const video = $("video");
  video.src = `/media/${encodeURIComponent(id)}/proxy`;

  applyFilter();
  history.replaceState(null, "", `?stream=${encodeURIComponent(id)}`);
}

/* ------------------------------------------------------------- rendering */

function applyFilter() {
  state.view = state.markersOnly
    ? state.all.filter((c) => c.marker_anchored)
    : state.all;
  $("filter-state").hidden = !state.markersOnly;
  $("filter-toggle").setAttribute("aria-pressed", String(state.markersOnly));
  state.cursor = Math.min(state.cursor, Math.max(state.view.length - 1, 0));
  renderList();
  focusCandidate(state.cursor);
}

function renderList() {
  const list = $("candidates");
  list.innerHTML = "";
  state.view.forEach((c, i) => {
    const li = document.createElement("li");
    li.className = [
      i === state.cursor ? "current" : "",
      c.rating !== null ? `rated-${c.rating}` : "",
      c.marker_anchored ? "marker" : "",
    ].filter(Boolean).join(" ");
    li.innerHTML =
      `<span class="n">${i + 1}</span>` +
      `<span class="when">${fmt(c.t_start)}</span>` +
      `<span class="sc">${c.score.toFixed(2)}</span>`;
    li.onclick = () => focusCandidate(i);
    list.appendChild(li);
  });
  updateProgress();
}

function updateProgress() {
  const rated = state.all.filter((c) => c.rating !== null).length;
  $("progress").textContent = `${rated} / ${state.all.length} rated`;
}

function current() {
  return state.view[state.cursor] || null;
}

function focusCandidate(index) {
  if (!state.view.length) return;
  state.cursor = Math.max(0, Math.min(index, state.view.length - 1));
  const c = current();
  if (!c) return;

  // The clock for review_ms starts the moment the candidate is on screen.
  state.focusedAt = performance.now();

  document.querySelectorAll("#candidates li").forEach((li, i) => {
    li.classList.toggle("current", i === state.cursor);
  });
  document.querySelector("#candidates li.current")
    ?.scrollIntoView({ block: "nearest" });

  $("window").textContent = `${fmt(c.t_start)} – ${fmt(c.t_end)}`;
  $("position").textContent =
    `${c.duration_s.toFixed(1)}s window · peak ${fmt(c.t_peak)}` +
    (c.markers.length ? ` · ${c.markers.length} marker${c.markers.length > 1 ? "s" : ""}` : "");
  $("score").textContent = c.score.toFixed(3);

  drawSpark(c);
  renderSignals(c);
  renderVerdict(c);

  // Seek and hold a frame rather than autoplaying. §7.3 describes a looping
  // silent preview, but that assumes §7.2's pre-rendered 2 s clips; seeking a
  // 700 MB proxy on every keypress and autoplaying would stutter and fight the
  // browser's autoplay policy. `space` plays deliberately.
  stopPlayback();
  const video = $("video");
  if (Number.isFinite(c.t_peak)) {
    try { video.currentTime = c.t_peak; } catch { /* not seekable yet */ }
  }
}

function drawSpark(c) {
  const svg = $("spark");
  svg.innerHTML = "";
  const caption = $("spark-caption");

  if (!c.sparkline?.length) {
    caption.textContent = "no signal series for this window";
    return;
  }

  const [lo, hi] = c.sparkline_range;
  const span = Math.max(hi - lo, 1e-6);
  const W = 600, H = 90, pad = 6;
  const points = c.sparkline.map((v, i) => {
    const x = (i / (c.sparkline.length - 1 || 1)) * W;
    const y = H - pad - ((v - lo) / span) * (H - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const area = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  area.setAttribute("points", `0,${H} ${points.join(" ")} ${W},${H}`);
  area.setAttribute("fill", "rgba(90,169,255,0.22)");
  svg.appendChild(area);

  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", points.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#5aa9ff");
  line.setAttribute("stroke-width", "1.5");
  svg.appendChild(line);

  // Where the peak sits inside the window, and where any markers landed.
  addRule(svg, (c.t_peak - c.t_start) / (c.t_end - c.t_start), W, H, "#5aa9ff", 1.5);
  for (const t of c.markers) {
    addRule(svg, (t - c.t_start) / (c.t_end - c.t_start), W, H, "#b57cff", 1);
  }

  caption.textContent =
    `${c.sparkline_kind} · ${lo.toFixed(1)} to ${hi.toFixed(1)} dBFS` +
    (c.markers.length ? " · purple = marker press" : "");
}

function addRule(svg, fraction, W, H, colour, width) {
  if (!Number.isFinite(fraction)) return;
  const x = Math.max(0, Math.min(1, fraction)) * W;
  const rule = document.createElementNS("http://www.w3.org/2000/svg", "line");
  rule.setAttribute("x1", x); rule.setAttribute("x2", x);
  rule.setAttribute("y1", 0); rule.setAttribute("y2", H);
  rule.setAttribute("stroke", colour);
  rule.setAttribute("stroke-width", width);
  rule.setAttribute("stroke-dasharray", "3 3");
  svg.appendChild(rule);
}

function renderSignals(c) {
  const panel = $("signals");
  panel.hidden = !state.showSignals;
  if (!state.showSignals) return;

  const entries = Object.entries(c.contributions);
  const largest = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-6);

  panel.innerHTML = entries.map(([name, value]) => {
    const width = (Math.abs(value) / largest) * 100;
    return `<div class="sig-row">
      <span>${escape(name)}</span>
      <span class="sig-bar"><i class="${value < 0 ? "neg" : ""}" style="width:${width}%"></i></span>
      <span class="sig-val">${value >= 0 ? "+" : ""}${value.toFixed(3)}</span>
    </div>`;
  }).join("") +
  `<div class="sig-row sig-total">
     <span>sum of contributions</span><span></span>
     <span class="sig-val">${(c.context.total_raw ?? 0).toFixed(3)}</span>
   </div>
   <div class="sig-row sig-total">
     <span>after smoothing (the score)</span><span></span>
     <span class="sig-val">${(c.context.total_smoothed ?? c.score).toFixed(3)}</span>
   </div>` +
  contextLine(c);
}

function contextLine(c) {
  const bits = Object.entries(c.context)
    .filter(([k]) => k.endsWith("_db"))
    .map(([k, v]) => `${k.replace(/_db$/, "")} ${v} dB`);
  return bits.length ? `<div class="sig-context">${escape(bits.join("  ·  "))}</div>` : "";
}

function renderVerdict(c) {
  const el = $("verdict");
  if (c.rating === null) {
    el.className = "verdict";
    el.innerHTML = `<span class="muted">unrated</span>`;
    return;
  }
  const label = ["skip", "maybe", "clip it"][c.rating];
  el.className = `verdict r${c.rating}`;
  el.innerHTML = `<b>${label}</b>` +
    (c.rating_source === "inherited"
      ? ` <span class="muted">— carried from an earlier scoring run</span>` : "");
}

/* --------------------------------------------------------------- actions */

function move(delta) {
  if (!state.view.length) return;
  focusCandidate(state.cursor + delta);
}

function rate(value) {
  const c = current();
  if (!c) return;

  const elapsed = Math.round(performance.now() - state.focusedAt);
  const wasUnrated = c.rating === null;
  c.rating = value;
  c.rating_source = "operator";
  if (wasUnrated) state.reviewed += 1;

  // Fire and forget: the keyboard must never wait on the network.
  fetch(`/api/candidates/${c.id}/rating`, {
    ...POST,
    body: JSON.stringify({ rating: value, review_ms: elapsed }),
  }).catch(() => {});

  renderList();
  // §7.3: "Rating advances automatically to the next candidate."
  if (state.cursor < state.view.length - 1) move(1);
  else focusCandidate(state.cursor);
}

function playWindow() {
  const c = current();
  if (!c) return;
  const video = $("video");

  if (!video.paused) return stopPlayback();

  video.currentTime = c.t_start;
  $("playing").hidden = false;
  video.play().catch(() => { $("playing").hidden = true; });

  // Stop at the window's end rather than running on into the next moment.
  const stopAt = c.t_end;
  const watch = () => {
    if (video.currentTime >= stopAt) stopPlayback();
  };
  video.addEventListener("timeupdate", watch);
  state.playTimer = () => video.removeEventListener("timeupdate", watch);
}

function stopPlayback() {
  const video = $("video");
  video.pause();
  $("playing").hidden = true;
  if (state.playTimer) { state.playTimer(); state.playTimer = null; }
}

async function finish() {
  const seconds = (performance.now() - state.sessionStart) / 1000;
  if (state.reviewed > 0) {
    await fetch(`/api/streams/${encodeURIComponent(state.streamId)}/session`, {
      ...POST,
      body: JSON.stringify({ duration_s: seconds, reviewed: state.reviewed }),
    }).catch(() => {});
  }

  const res = await fetch(`/api/streams/${encodeURIComponent(state.streamId)}/metrics`);
  const m = await res.json();

  // §7.1: 120 candidates in under 8 minutes, i.e. ~4 s each.
  const target = 4000;
  const median = m.median_review_ms;
  const verdict = median === null ? ""
    : median <= target
      ? `<span class="target-hit">within the 4.0 s target (§7.1)</span>`
      : `<span class="target-miss">over the 4.0 s target (§7.1) — fix the UI before adding features</span>`;

  $("summary-body").innerHTML = `
    <table>
      <tr><td>reviewed this session</td><td>${state.reviewed}</td></tr>
      <tr><td>session length</td><td>${fmt(seconds)}</td></tr>
      <tr><td>median per candidate</td><td>${median === null ? "—" : (median / 1000).toFixed(2) + " s"}</td></tr>
      <tr><td>rated overall</td><td>${m.rated} / ${m.candidates}</td></tr>
      <tr><td>clip it</td><td>${m.by_rating["2"] || 0}</td></tr>
      <tr><td>maybe</td><td>${m.by_rating["1"] || 0}</td></tr>
      <tr><td>skip</td><td>${m.by_rating["0"] || 0}</td></tr>
    </table>
    <p>${verdict}</p>`;

  stopPlayback();
  $("review").hidden = true;
  $("summary").hidden = false;
}

/* -------------------------------------------------------------- keyboard */

document.addEventListener("keydown", (event) => {
  // Never swallow keys meant for a text field.
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || event.target.isContentEditable) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if ($("review").hidden) return;

  switch (event.key) {
    case "j": case "ArrowDown": event.preventDefault(); move(1); break;
    case "k": case "ArrowUp":   event.preventDefault(); move(-1); break;
    case "1": rate(0); break;
    case "2": rate(1); break;
    case "3": rate(2); break;
    case " ": event.preventDefault(); playWindow(); break;
    case "?": case "/":
      state.showSignals = !state.showSignals;
      renderSignals(current() || { contributions: {}, context: {} });
      break;
    case "m":
      state.markersOnly = !state.markersOnly;
      applyFilter();
      break;
    case "q": finish(); break;
  }
});

$("filter-toggle").onclick = () => { state.markersOnly = !state.markersOnly; applyFilter(); };
$("back").onclick = () => { stopPlayback(); boot(true); };
$("summary-back").onclick = () => boot(true);

setInterval(() => {
  if ($("review").hidden || !state.sessionStart) return;
  $("clock").textContent = fmt((performance.now() - state.sessionStart) / 1000);
}, 1000);

function escape(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

boot();
