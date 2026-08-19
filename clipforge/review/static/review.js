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
  // §7.4's section labels, from the server. Not written here: the rail would
  // otherwise keep a header for a section the server had stopped emitting.
  sectionLabels: {},
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

/* `arg` is a stream id, or `{streamId, at}` when §11.6's search deep-links into
 * a moment. Accepting both keeps every existing caller — the library, the run
 * view, boot's `?stream=` resume — unchanged. */
export async function enter(arg) {
  const id = typeof arg === "string" ? arg : arg.streamId;
  const at = typeof arg === "string" ? null : arg.at;
  const data = await get(`/api/streams/${encodeURIComponent(id)}/candidates`);

  state.streamId = id;
  state.stream = data.stream;
  state.all = data.candidates;
  state.roleColours = data.role_colours || {};
  state.sectionLabels = Object.fromEntries(
    (data.sections || []).map((s) => [s.key, s.label])
  );
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
  setUpSections();
  applyFilter();
  if (at !== null && at !== undefined) focusNearest(at);
  history.replaceState(null, "", `?stream=${encodeURIComponent(id)}`);
}

/* Land on the candidate covering `seconds`, or the nearest one.
 *
 * §11.6 returns SEGMENTS, which exist independently of scoring — a memorable
 * line is frequently nowhere near a detected peak. So "nearest" is often not
 * "the one you searched for", and the screen says which of the two happened
 * rather than quietly presenting an unrelated moment as the match. */
function focusNearest(seconds) {
  if (!state.view.length) return;
  const covering = state.view.findIndex(
    (c) => c.t_start <= seconds && c.t_end >= seconds);

  let index = covering;
  if (index < 0) {
    let best = Infinity;
    state.view.forEach((c, i) => {
      const distance = Math.min(Math.abs(c.t_start - seconds),
                                Math.abs(c.t_end - seconds));
      if (distance < best) { best = distance; index = i; }
    });
  }
  if (index < 0) return;
  focusCandidate(index);

  if (covering < 0) {
    const note = $("window-note");
    if (note) {
      note.textContent =
        `nearest candidate to ${fmt(seconds)} — no candidate covers that moment`;
      note.className = "window-note is-adjusted";
    }
  }
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
    // The § belongs in the tooltip: someone reading this needs to know what to
    // turn on, not which section of the spec asked for it.
    note.title = "§7.3 — transcript text for the window, displayed alongside.";
    note.innerHTML =
      "No transcript for this stream. Showing one beside the window needs " +
      "<code>extract.whisperx.enabled: true</code> and a Whisper model " +
      "(a multi-GB download), then a re-run.";
  }
}

/* §7.4's four sections exist only when more than one §6.5 profile ran. A
 * stream scored with one profile has no gameplay ranking and a combined score
 * that mirrors its primary, so four headers over it would be a lie about what
 * the numbers mean — which is what the rail said in prose until this commit.
 * The server decides (`queries.is_sectioned`); this only reads the answer off
 * the candidates it was sent. */
function setUpSections() {
  const sectioned = state.all.some((c) => c.section);
  $("rail-note").hidden = sectioned;
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

  let section = null;
  state.view.forEach((c, i) => {
    // §7.4's headers, emitted on the boundary rather than per section, so the
    // `m` filter can empty a section and its header goes with it.
    if (c.section && c.section !== section) {
      section = c.section;
      const head = document.createElement("li");
      head.className = "cand-section";
      head.textContent = state.sectionLabels[section] || section;
      list.appendChild(head);
    }

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

  // `:not(.cand-section)` because §7.4's headers are list items too, and they
  // scroll with the candidates they head. Indexing every `li` made the header
  // row index 0, so the highlight sat on "Combined winners" and every
  // candidate's selection was one row further down than it looked.
  document.querySelectorAll("#candidates li:not(.cand-section)").forEach((li, i) => {
    li.classList.toggle("current", i === state.cursor);
  });
  document.querySelector("#candidates li.current")
    ?.scrollIntoView({ block: "nearest" });

  renderWindow(c);
  $("score").textContent = c.score.toFixed(3);

  drawSpark(c);
  renderAssets(c);
  renderTranscript(c);
  renderSignals(c);
  renderVerdict(c);

  stopPlayback();
  // The proxy is still seeked to the peak even when a preview is playing over
  // it: `space` then starts the real window immediately instead of buffering
  // from wherever the previous candidate left it.
  const video = $("video");
  if (Number.isFinite(c.t_peak)) {
    try { video.currentTime = c.t_peak; } catch { /* not seekable yet */ }
  }
  startPreview(c);
}

/* §7.2's two file assets.
 *
 * When the `previews` stage has not run, both slots say so and the screen falls
 * back to Phase 1's seek-and-hold — which still works, and is why `previews`
 * is not a dependency of anything. */
function renderAssets(c) {
  const strip = $("thumbstrip");
  const missing = $("assets-missing");
  if (c.thumbstrip_url) {
    strip.src = c.thumbstrip_url;
    strip.hidden = false;
    missing.hidden = true;
  } else {
    strip.hidden = true;
    strip.removeAttribute("src");
    missing.hidden = false;
  }
}

/* §7.3: "Preview autoplays on focus, loops silently."
 *
 * Phase 1 deliberately did not do this, because autoplaying by seeking a 700 MB
 * proxy on every `j` press stutters. A 2 s 250 KB webm does not, which is the
 * whole reason §7.2 calls for the asset.
 *
 * `play()` is allowed to fail and that is not an error: browsers reject
 * autoplay until the page has been interacted with, and the operator's first
 * keypress is exactly that interaction. Failing silently leaves the proxy frame
 * showing underneath, which is the Phase 1 behaviour. */
function startPreview(c) {
  const preview = $("preview");
  if (!c?.preview_url) {
    preview.hidden = true;
    preview.removeAttribute("src");
    return;
  }
  if (preview.getAttribute("src") !== c.preview_url) {
    preview.src = c.preview_url;
  }
  preview.hidden = false;
  preview.currentTime = 0;
  preview.play().catch(() => { /* autoplay policy, or a missing file */ });
}

function hidePreview() {
  const preview = $("preview");
  preview.pause();
  preview.hidden = true;
}

/* Put the loop back after `space` has finished with the window.
 *
 * §7.3's preview autoplays "on focus", and the operator is still focused on
 * this candidate — they have just watched the real thing. Deliberately NOT
 * folded into stopPlayback(), which also runs on the way INTO every candidate:
 * there it would briefly restart the PREVIOUS candidate's clip, because
 * showCandidate sets the new src a few lines later. */
function resumePreview() {
  const preview = $("preview");
  if (!preview.getAttribute("src")) return;
  preview.hidden = false;
  preview.currentTime = 0;
  preview.play().catch(() => {});
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

/* Fallback for a role with no caption style. `game` has none — §8.3 colours
 * speakers, and game audio is not a speaker — so it cannot come from the
 * payload's palette the way mic and party do.
 *
 * Read from the stylesheet rather than written here as a hex. A near-copy of
 * --muted sitting in JS is the same drift the caption colours are sent by the
 * server to avoid, one layer down: the theme would move and this would not. */
function sparkColour(role) {
  const fromPalette = state.roleColours?.[role];
  if (fromPalette) return fromPalette;
  return getComputedStyle(document.documentElement)
    .getPropertyValue("--muted").trim() || "#8b93a3";
}

/* §7.2's "mic + party RMS over the window", drawn rather than fetched.
 *
 * Every track shares ONE dB range, which is the entire reason this is worth
 * drawing: per-track normalisation would put a silent party track at the same
 * height as a mic being shouted into, and the one question the picture answers
 * -- who was making the noise -- would be the one it could not. */
function drawSpark(c) {
  const svg = $("spark");
  svg.innerHTML = "";
  const caption = $("spark-caption");

  const tracks = (c.sparklines || []).filter((t) => t.points?.length);
  if (!tracks.length) {
    caption.textContent = "no signal series for this window";
    return;
  }

  const [lo, hi] = c.sparkline_range;
  const span = Math.max(hi - lo, 1e-6);
  const W = 600, H = 90, pad = 6;

  /* Drawn in reverse so the first-listed role -- mic -- ends up on top. The
   * payload orders them by SPARKLINE_KINDS, which is the order the operator
   * cares about. */
  for (const track of [...tracks].reverse()) {
    const colour = sparkColour(track.role);
    const points = track.points.map((v, i) => {
      const x = (i / (track.points.length - 1 || 1)) * W;
      const y = H - pad - ((v - lo) / span) * (H - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const area = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    area.setAttribute("points", `0,${H} ${points.join(" ")} ${W},${H}`);
    area.setAttribute("fill", colour);
    /* Fill rather than stroke opacity, so two overlapping tracks read as two
     * translucent shapes instead of one muddy one. */
    area.setAttribute("fill-opacity", "0.18");
    svg.appendChild(area);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", points.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", colour);
    line.setAttribute("stroke-width", "1.5");
    svg.appendChild(line);
  }

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

  /* Each track names itself in its own colour, so the legend cannot drift from
   * what was drawn -- it is built from the same array. */
  caption.replaceChildren();
  tracks.forEach((track, i) => {
    if (i) caption.append(" · ");
    const tag = document.createElement("span");
    tag.textContent = track.role;
    tag.style.color = sparkColour(track.role);
    caption.append(tag);
  });
  caption.append(
    ` · ${lo.toFixed(1)} to ${hi.toFixed(1)} dBFS`
    + [""].concat(c.markers.length ? ["purple = marker press"] : [])
          .concat(notes).join(" · ")
  );
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
     <span>after smoothing</span><span></span>
     <span class="sig-val">${(c.context.total_smoothed ?? c.score).toFixed(3)}</span>
   </div>` +
  profileScores(c) +
  contextLine(c);
}

/* The breakdown above explains the PRIMARY profile's composite, because
 * `contributing_signals` is weight x value and a two-profile row would
 * otherwise carry two of them (see score/runner.write_candidates). The score in
 * the box beside the video is §6.5's combined, which is a different number — so
 * all three are printed here rather than leaving the panel explaining one and
 * the screen showing another. */
function profileScores(c) {
  if (!c.section) return "";
  const rows = [
    ["entertainment", c.score_entertainment],
    ["gameplay", c.score_gameplay],
    ["combined", c.score],
  ];
  return rows.map(([name, value]) =>
    `<div class="sig-row sig-total">
       <span>${escape(name)}</span><span></span>
       <span class="sig-val">${Number(value ?? 0).toFixed(3)}</span>
     </div>`).join("");
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

  if (!video.paused) { stopPlayback(); return resumePreview(); }

  // §7.3's `space` is "play full window with audio", so the silent 2 s loop
  // gets out of the way rather than playing over it.
  hidePreview();
  video.currentTime = startOf(c);
  $("playing").hidden = false;
  video.play().catch(() => { $("playing").hidden = true; });

  // Stop at the window's end rather than running on into the next moment.
  const stopAt = endOf(c);
  const watch = () => {
    if (video.currentTime >= stopAt) { stopPlayback(); resumePreview(); }
  };
  video.addEventListener("timeupdate", watch);
  state.playTimer = () => video.removeEventListener("timeupdate", watch);
}

function stopPlayback() {
  const video = $("video");
  video.pause();
  $("playing").hidden = true;
  if (state.playTimer) { state.playTimer(); state.playTimer = null; }
  // Deliberately does NOT restart the preview: stopPlayback runs on the way
  // into every candidate, and showCandidate starts the loop itself once the
  // new candidate's url is known.
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
  // §7.1 rides in the tooltip. The sentence has to say what to DO about the
  // number, and that is the part that stops being read when it arrives wrapped
  // in a section reference.
  const verdict = median === null
    ? `<span class="muted">Nothing rated this session, so there is no pace to report.</span>`
    : median <= target
      ? `<span class="target-hit" title="§7.1">Within the ${seconds_target} s target.</span>`
      : `<span class="target-miss" title="§7.1">Over the ${seconds_target} s target —
         fix the review screen before adding a feature anywhere else.</span>`;

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
