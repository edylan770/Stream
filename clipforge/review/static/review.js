/* ClipForge review — §7.3's keyboard loop.
 *
 * Commit 13's app.js, moved into the shell's module contract. The behaviour is
 * unchanged: it was verified in a browser and this is a relocation, not a
 * rewrite.
 *
 * C4's target is 120 candidates in under 8 minutes, so the guiding rule is that
 * a keypress never waits on the network. Every candidate arrives in the first
 * payload; ratings are fired off without awaiting a response; navigation only
 * moves a highlight and seeks a <video> that is already loaded.
 *
 * review_ms is measured from the moment a candidate gains focus to the moment
 * it is rated (§7.5). That is the honest observation even when it includes a
 * coffee break, which is why `clipforge metrics` reports the median.
 */

import { $, escape, fmt, get, post, postAndForget } from "./api.js";
import * as router from "./router.js";

export const root = "view-review";

const state = {
  streamId: null,
  stream: null,
  all: [],
  view: [],
  cursor: 0,
  markersOnly: false,
  showSignals: false,
  roleColours: {},
  focusedAt: 0,
  sessionStart: 0,
  reviewed: 0,
  nudged: 0,
  nudgeStep: 0.5,
  playTimer: null,
};

/* ---------------------------------------------------------------- windows
 *
 * §7.3's `[`/`]`/`{`/`}`. The operator's boundary, when there is one, beats the
 * detector's everywhere: the readout, `space`, the sparkline rules, the
 * transcript slice, and — through ratings.adjusted_start/_end and
 * render/selection.py — the exported clip.
 *
 * `adj_start`/`adj_end` are undefined until something moves. Falling back to
 * the candidate's own values rather than copying them on load keeps "the
 * operator has an opinion about this window" a fact the code can read, which is
 * what decides whether a rating carries an adjustment at all. */

const startOf = (c) => (c.adjusted_start ?? c.t_start);
const endOf = (c) => (c.adjusted_end ?? c.t_end);
const isAdjusted = (c) => c.adjusted_start !== null && c.adjusted_start !== undefined;

/* ------------------------------------------------------------- lifecycle */

export async function enter(id) {
  const data = await get(`/api/streams/${encodeURIComponent(id)}/candidates`);

  state.streamId = id;
  state.stream = data.stream;
  state.all = data.candidates;
  state.roleColours = data.role_colours || {};
  // From the server, not a literal here: review.nudge_step_s is config, and a
  // second copy of it in the browser would drift silently — the same trap
  // `target_ms` had before commit 25a.
  state.nudgeStep = Number(data.nudge_step_s) || 0.5;
  state.cursor = 0;
  state.markersOnly = false;
  state.sessionStart = performance.now();
  state.reviewed = 0;
  state.nudged = 0;

  $("review-main").hidden = false;
  $("summary").hidden = true;

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

  setUpTranscript(data.stream);
  applyFilter();
  history.replaceState(null, "", `?stream=${encodeURIComponent(id)}`);
}

/* §7.3 wants the transcript beside the window. Phase 2 ships
 * `extract.whisperx.enabled: false`, so a stream processed with defaults has
 * no segments at all — and a third column that is empty on every stream would
 * cost ~320px of video width to show nothing. The column appears only when
 * there is something to put in it; otherwise the detail strip says what
 * turning it on costs. */
function setUpTranscript(stream) {
  const has = Boolean(stream.has_transcript);
  document.querySelector(".layout").classList.toggle("with-transcript", has);
  $("transcript-pane").hidden = !has;

  const note = $("no-transcript");
  note.hidden = has;
  if (!has) {
    note.innerHTML =
      "No transcript. §7.3 shows one beside the window; it needs " +
      "<code>extract.whisperx.enabled: true</code> and a Whisper model " +
      "(a multi-GB download), then a re-run.";
  }
}

export function leave() {
  stopPlayback();
  $("video").removeAttribute("src");
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

  if (!state.view.length) {
    // `clipforge review` no longer refuses to open a stream with nothing in
    // it (commit 14c), so this is reachable — and a rail with nothing in it
    // and no explanation reads as a broken screen.
    const note = document.createElement("li");
    note.className = "note";
    note.style.display = "block";
    note.textContent = state.markersOnly
      ? "No marker-anchored candidates. Press m to see all of them."
      : "No candidates. Run the pipeline for this stream first.";
    list.appendChild(note);
    updateProgress();
    return;
  }

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
  // Leaving a candidate is when its nudges are reported: once per candidate,
  // whatever the operator did next, including nothing.
  reportNudge(current());
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

  renderWindow(c);
  $("score").textContent = c.score.toFixed(3);

  drawSpark(c);
  renderTranscript(c);
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

/* The window readout, and whose window it is.
 *
 * When the operator has moved a boundary the screen has to say so plainly:
 * from here on this moment is theirs, not the detector's, and that is what the
 * FCPXML and the render will use. */
function renderWindow(c) {
  if (!c) return;
  const start = startOf(c);
  const end = endOf(c);
  const adjusted = isAdjusted(c);

  $("window").textContent = `${fmt(start)} – ${fmt(end)}`;
  $("window").classList.toggle("adjusted", adjusted);

  const bits = [`${(end - start).toFixed(1)}s window`, `peak ${fmt(c.t_peak)}`];
  if (c.markers.length) {
    bits.push(`${c.markers.length} marker${c.markers.length > 1 ? "s" : ""}`);
  }
  $("position").textContent = bits.join(" · ");

  const note = $("window-note");
  if (!note || note.classList.contains("at-limit")) return;

  if (!adjusted) {
    note.textContent = "";
    note.className = "window-note";
    return;
  }

  const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
  const parts = [
    `yours: start ${signed(start - c.t_start)}s, end ${signed(end - c.t_end)}s`,
  ];
  // Legal here and forbidden on `candidates` by CHECK — worth saying out loud,
  // because a window that no longer contains the peak it was detected from is
  // either a deliberate trim or a mis-keyed nudge, and only the operator knows.
  if (c.t_peak < start || c.t_peak > end) parts.push("peak is now outside it");
  note.textContent = parts.join(" · ");
  note.className = "window-note is-adjusted";
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
  // `window` rather than `span`: `span` is already the sparkline's dB range a
  // few lines up, and reusing the name here is a SyntaxError that takes the
  // whole module — and therefore the entire review screen — down at load.
  const window_s = c.t_end - c.t_start;
  addRule(svg, (c.t_peak - c.t_start) / window_s, W, H, "#5aa9ff", 1.5);
  for (const t of c.markers) {
    addRule(svg, (t - c.t_start) / window_s, W, H, "#b57cff", 1);
  }

  // The operator's boundaries, drawn over the detector's envelope. The envelope
  // itself still covers the ORIGINAL window — the payload has no samples
  // outside it — so an extension is reported in words rather than drawn as a
  // rule pinned to the edge, which would look like a boundary that is there.
  // Commit 31's nudge_context_s is what turns that into something visible.
  const notes = [];
  if (isAdjusted(c)) {
    const start = startOf(c);
    const end = endOf(c);
    for (const t of [start, end]) {
      if (t >= c.t_start && t <= c.t_end) {
        addRule(svg, (t - c.t_start) / window_s, W, H, "#ffd400", 1.5);
      }
    }
    if (start < c.t_start || end > c.t_end) {
      notes.push("your window extends past what is drawn here");
    } else {
      notes.push("yellow = your boundaries");
    }
  }

  caption.textContent =
    [`${c.sparkline_kind} · ${lo.toFixed(1)} to ${hi.toFixed(1)} dBFS`]
      .concat(c.markers.length ? ["purple = marker press"] : [])
      .concat(notes)
      .join(" · ");
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

/* §7.3: "Transcript text for the window displayed alongside."
 *
 * Lines, not words. The operator is reading to decide whether the moment is
 * worth clipping, and a word-level highlight would multiply the payload for
 * something nobody asked for.
 *
 * A line's colour is its §8.3 speaker colour, sent by the server from
 * `render.captions.styles` — so what is read here and what is burned into the
 * export cannot disagree about who spoke. */
function renderTranscript(c) {
  const list = $("transcript");
  if (!list || $("transcript-pane").hidden) return;

  // Re-sliced against the CURRENT window, so trimming a boundary immediately
  // shows what the clip would no longer contain. The server sliced on the
  // detector's window, so an extension cannot reveal lines that are not in the
  // payload — that is commit 31.
  const start = startOf(c);
  const end = endOf(c);
  const lines = (c.transcript || []).filter((l) => l.t_end >= start && l.t <= end);

  $("transcript-count").textContent = lines.length
    ? `${lines.length} line${lines.length > 1 ? "s" : ""}` : "";

  if (!lines.length) {
    list.innerHTML =
      `<li><span class="empty-note">Nothing said in this window.</span></li>`;
    return;
  }

  list.innerHTML = lines.map((line) => {
    const colour = state.roleColours[line.role] || state.roleColours.default;
    return `<li>` +
      `<span class="at">${escape(fmt(line.t))}</span>` +
      `<span class="said"${colour ? ` style="color:${escape(colour)}"` : ""}>` +
      `${escape(line.text)}</span></li>`;
  }).join("");
  list.scrollTop = 0;
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

// The unit is the key's own suffix, because signals are no longer all in dBFS:
// mic_f0 is hertz, and a context line reading "mic_f0 166.2 dB" would be a
// label stating something false. A null is a signal that had no observation at
// this instant — an unvoiced frame — and is dropped rather than printed.
const CONTEXT_UNITS = { db: "dB", hz: "Hz", st: "semitones" };

function contextLine(c) {
  const bits = Object.entries(c.context)
    .filter(([k, v]) => v !== null && k.slice(k.lastIndexOf("_") + 1) in CONTEXT_UNITS)
    .map(([k, v]) => {
      const cut = k.lastIndexOf("_");
      return `${k.slice(0, cut)} ${v} ${CONTEXT_UNITS[k.slice(cut + 1)]}`;
    });
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

  // §7.3's adjusted window rides along with the rating — ratings.rating is NOT
  // NULL, so there is nowhere to put a boundary before there is a verdict, and
  // inventing one would corrupt §14's tuning input. The keys are omitted when
  // untouched: absent means "no opinion", and the server keeps whatever an
  // earlier session recorded rather than reverting to the detector's window.
  const body = { rating: value, review_ms: elapsed };
  if (isAdjusted(c)) {
    body.adjusted_start = c.adjusted_start;
    body.adjusted_end = c.adjusted_end;
  }

  // Fire and forget: the keyboard must never wait on the network.
  postAndForget(`/api/candidates/${c.id}/rating`, body);

  renderList();
  // §7.3: "Rating advances automatically to the next candidate."
  if (state.cursor < state.view.length - 1) move(1);
  else focusCandidate(state.cursor);
}

/* §7.3: "[ / ] nudge window start earlier / later (0.5 s)", "{ / } end".
 *
 * The clamps here are ONLY the ones arithmetic demands: a window may not
 * invert, start before the recording, or end after it.
 *
 * It is deliberately NOT clamped to score.window.min_window_s / max_window_s.
 * Those are the two numbers §17 tunes *against these very nudges*, so refusing
 * a window outside their range would make the measurement circular — the
 * operator could never record "I wanted this shorter than 8 seconds", which is
 * exactly the observation that has been missing (GUESSES gap 1). A window
 * shorter than min_window_s is a finding, not a mistake.
 *
 * A press that would break a real clamp is a no-op and is NOT counted: a
 * refused keystroke is not a nudge, and counting it would inflate the one
 * number this feature exists to collect. */
function nudge(edge, direction) {
  const c = current();
  if (!c) return;

  const step = state.nudgeStep * direction;
  let start = startOf(c);
  let end = endOf(c);
  if (edge === "start") start += step; else end += step;

  const limit = Number(state.stream?.duration_s);
  if (end - start < state.nudgeStep) return flashLimit("too short to nudge further");
  if (start < 0) return flashLimit("that is the start of the recording");
  if (Number.isFinite(limit) && end > limit) {
    return flashLimit("that is the end of the recording");
  }

  c.adjusted_start = Number(start.toFixed(3));
  c.adjusted_end = Number(end.toFixed(3));
  c.nudge_presses = (c.nudge_presses || 0) + 1;
  // A second visit that adjusts the window again is a second episode and gets
  // its own row; without clearing this, coming back to a candidate would mean
  // the later adjustment was never recorded at all.
  c.nudge_reported = false;
  // ...but the session's denominator counts CANDIDATES nudged, not episodes,
  // because that is what "how often" is a fraction of.
  if (!c.nudge_counted) {
    c.nudge_counted = true;
    state.nudged += 1;
  }

  // Redraw everything that describes the window. The video is deliberately not
  // re-seeked: the operator is usually watching while adjusting, and yanking
  // the playhead on every keypress would make the edit impossible to judge.
  renderWindow(c);
  renderTranscript(c);
  drawSpark(c);
}

function flashLimit(message) {
  const el = $("window-note");
  if (!el) return;
  el.textContent = message;
  el.classList.add("at-limit");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => {
    el.classList.remove("at-limit");
    renderWindow(current());
  }, 1200);
}

/* Posted once per candidate, on leaving it — not on every keypress, which
 * would be a request per 0.5 s of adjustment, and not on rating, which would
 * lose the nudges the operator makes and then walks away from. */
function reportNudge(c) {
  if (!c || !c.nudge_presses || c.nudge_reported) return;
  c.nudge_reported = true;
  postAndForget(`/api/candidates/${c.id}/nudge`, {
    adjusted_start: c.adjusted_start,
    adjusted_end: c.adjusted_end,
    presses: c.nudge_presses,
  });
  // The presses are now accounted for. Anything after this belongs to the next
  // episode, so summing `presses` across rows stays the true keystroke total.
  c.nudge_presses = 0;
}

function playWindow() {
  const c = current();
  if (!c) return;
  const video = $("video");

  if (!video.paused) return stopPlayback();

  video.currentTime = startOf(c);
  $("playing").hidden = false;
  video.play().catch(() => { $("playing").hidden = true; });

  // Stop at the window's end rather than running on into the next moment.
  const stopAt = endOf(c);
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
  // The candidate on screen has not been left yet, so its nudges are still
  // unreported.
  reportNudge(current());
  if (state.reviewed > 0 || state.nudged > 0) {
    await post(`/api/streams/${encodeURIComponent(state.streamId)}/session`, {
      duration_s: seconds, reviewed: state.reviewed, nudged: state.nudged,
    }).catch(() => {});
  }

  const m = await get(`/api/streams/${encodeURIComponent(state.streamId)}/metrics`);

  // §7.1: 120 candidates in under 8 minutes. The target comes from the server
  // now — it used to be the literal 4000 here, so changing
  // `review.target_ms_per_candidate` moved what `clipforge metrics` graded
  // against and left this screen quietly measuring something else.
  const target = m.target_ms;
  const seconds_target = (target / 1000).toFixed(1);
  const median = m.median_review_ms;
  const verdict = median === null
    ? `<span class="muted">nothing rated this session, so there is no pace to report.</span>`
    : median <= target
      ? `<span class="target-hit">within the ${seconds_target} s target (§7.1)</span>`
      : `<span class="target-miss">over the ${seconds_target} s target (§7.1) —
         §7.1 says fix the UI before adding a feature anywhere</span>`;

  const stat = (value, label) =>
    `<div class="stat"><div class="v">${value}</div><span class="k">${label}</span></div>`;

  $("summary-body").innerHTML = `
    <div class="summary-grid">
      ${stat(state.reviewed, "reviewed now")}
      ${stat(fmt(seconds), "session length")}
      ${stat(median === null ? "—" : (median / 1000).toFixed(2) + " s", "median each")}
      ${stat(`${m.rated} / ${m.candidates}`, "rated overall")}
    </div>
    <div class="summary-grid">
      ${stat(`<span class="target-hit">${m.by_rating["2"] || 0}</span>`, "clip it")}
      ${stat(`<span class="target-miss">${m.by_rating["1"] || 0}</span>`, "maybe")}
      ${stat(`<span class="muted">${m.by_rating["0"] || 0}</span>`, "skip")}
    </div>
    <p>${verdict}</p>`;

  stopPlayback();
  $("review-main").hidden = true;
  $("summary").hidden = false;
}

/* -------------------------------------------------------------- keyboard */

export function onKey(event) {
  if (!$("summary").hidden) return;   // the session is over; nothing to drive

  switch (event.key) {
    case "j": case "ArrowDown": event.preventDefault(); move(1); break;
    case "k": case "ArrowUp":   event.preventDefault(); move(-1); break;
    case "1": rate(0); break;
    case "2": rate(1); break;
    case "3": rate(2); break;
    // §7.3: [ / ] move the start earlier / later, { / } the end. On a US
    // layout the braces are shift+brackets, which `event.key` reports directly.
    case "[": event.preventDefault(); nudge("start", -1); break;
    case "]": event.preventDefault(); nudge("start", +1); break;
    case "{": event.preventDefault(); nudge("end", -1); break;
    case "}": event.preventDefault(); nudge("end", +1); break;
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
}

$("filter-toggle").onclick = () => { state.markersOnly = !state.markersOnly; applyFilter(); };
$("back").onclick = () => router.show("library");
$("summary-back").onclick = () => router.show("library");

/* The run view already polls the job and paints the log. Handing off to it
 * beats a second copy of that machinery living in here. */
$("summary-render").onclick = () =>
  router.show("run", { id: state.streamId, render: true });

setInterval(() => {
  if (router.activeName() !== "review" || !state.sessionStart) return;
  $("clock").textContent = fmt((performance.now() - state.sessionStart) / 1000);
}, 1000);
