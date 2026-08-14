/* Running the pipeline, watched.
 *
 * The plan is §5.1's decision table — every stage with its marker and the
 * reason it will or will not run. That table is the most useful diagnostic in
 * the pipeline and it costs nothing to show, so it is the whole screen rather
 * than something behind a disclosure triangle.
 *
 * Progress is polled with an absolute cursor rather than streamed. A 4-hour
 * master spends 20-40 minutes in here (§1.3); stage transitions and 10%-step
 * encode progress do not need sub-second latency, and polling survives a reload
 * with no reconnect logic. See review/jobs.py.
 *
 * "Review this stream" appears only once `score` is genuinely done. §7.2 and
 * §7.4 assume review is entered with candidates already in the database, and
 * handing over an empty review screen would read as a bug in the review
 * screen. */

import { $, escape, get, post } from "./api.js";
import * as router from "./router.js";

export const root = "view-run";

/* Fast enough to feel live, slow enough that a multi-hour encode is not
 * thousands of requests. */
const POLL_MS = 700;

const MARKERS = {
  run: ["RUN", "will-run"],
  skip: ["skip", "muted"],
  blocked: ["BLOCK", "blocked"],
  deferred: ["—", "muted"],
  external: ["done", "muted"],
};

let streamId = null;
let plan = null;
let jobId = null;
let cursor = 0;
let timer = null;

/* `arg` is a stream id, or `{id, render: true}` when the review summary hands
 * off — that screen has the Render button, this screen has the log. */
export async function enter(arg) {
  const wantsRender = typeof arg === "object" && arg !== null && arg.render;
  const id = typeof arg === "object" && arg !== null ? arg.id : arg;

  streamId = id;
  jobId = null;
  cursor = 0;
  $("run-log").textContent = "";
  $("run-name").textContent = id;
  history.replaceState(null, "", `?view=run&stream=${encodeURIComponent(id)}`);
  await refreshPlan();
  if (!wantsRender) return;
  // Say why nothing happened rather than landing the operator on a screen
  // that looks like it ignored the button they just pressed.
  if (plan.job_live) {
    append("  something is already running this stream; render not started");
  } else if ($("run-render").disabled) {
    append(`  ${$("run-render-hint").textContent}`);
  } else {
    await start("render");
  }
}

export function leave() {
  clearTimeout(timer);
  timer = null;
}

async function refreshPlan() {
  plan = await get(`/api/streams/${encodeURIComponent(streamId)}/plan`);
  renderPlan();
  if (plan.job && plan.job_live) {
    jobId = plan.job;
    poll();
  }
}

function renderPlan() {
  const list = $("run-stages");
  list.innerHTML = "";
  for (const stage of plan.stages) {
    const [label, cls] = MARKERS[stage.action] || [stage.action, ""];
    const item = document.createElement("li");
    item.className = `stage-row ${cls} ${stage.status === "failed" ? "failed" : ""}`;
    item.innerHTML =
      `<span class="marker">${escape(label)}</span>` +
      `<span class="nm mono">${escape(stage.stage)}</span>` +
      `<span class="muted why">${escape(stage.error || stage.reason)}</span>`;
    list.appendChild(item);
  }

  const pending = plan.will_run.length;
  $("run-summary").textContent = pending
    ? `${pending} stage${pending > 1 ? "s" : ""} to run`
    : "everything up to date";
  $("run-start").disabled = plan.job_live;
  $("run-start").textContent = plan.job_live ? "running…" : pending ? "Run" : "Run anyway";

  const scored = plan.stages.find((s) => s.stage === "score");
  $("run-review").hidden = !(scored && scored.status === "done");

  // Render needs approved moments, and nothing can be approved before `score`
  // has produced candidates to approve. Offering the button first would mean
  // the operator's first render is an error message.
  const renderable = Boolean(scored && scored.status === "done");
  $("run-render").disabled = plan.job_live || !renderable;
  $("run-render-hint").textContent = renderable
    ? "everything rated ‘clip it’, with the configured crop template"
    : "review the stream first — render needs approved moments";
}

/* `kind` is "run" (§5.1's pipeline) or "render" (§8's auto-finish). Both are
 * the same thing from here: something long is happening and the log is how you
 * watch it, so they share the buffer, the cursor and the one-writer guard. */
async function start(kind = "run") {
  $("run-start").disabled = true;
  $("run-render").disabled = true;
  try {
    const job = await post(`/api/streams/${encodeURIComponent(streamId)}/${kind}`);
    jobId = job.id;
    cursor = 0;
    $("run-log").textContent = "";
    poll();
  } catch (error) {
    append(`  ${error.message}`);
    renderPlan();
  }
}

async function poll() {
  let snapshot;
  try {
    snapshot = await get(`/api/jobs/${encodeURIComponent(jobId)}`, { since: cursor });
  } catch {
    // A dropped poll is not a failed run; try again on the next tick.
    timer = setTimeout(poll, POLL_MS);
    return;
  }

  if (snapshot.dropped) append(`  … ${snapshot.dropped} earlier line(s) not shown`);
  for (const line of snapshot.lines) append(line);
  cursor = snapshot.next_cursor;

  $("run-elapsed").textContent = `${snapshot.elapsed_s.toFixed(0)}s`;
  // Both kinds share this screen, so the pill has to say which one is live —
  // otherwise "running" beside a stage table implies the pipeline.
  $("run-state").textContent = snapshot.kind === "render"
    ? `render ${snapshot.state}` : snapshot.state;
  $("run-state").className = `pill ${snapshot.state}`;
  $("run-state").hidden = false;

  if (snapshot.state === "running") {
    timer = setTimeout(poll, POLL_MS);
    return;
  }
  if (snapshot.error) append(`  ${snapshot.error}`);
  await refreshPlan();
}

function append(line) {
  const log = $("run-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.textContent += `${line}\n`;
  if (atBottom) log.scrollTop = log.scrollHeight;
}

export function onKey(event) {
  switch (event.key) {
    case "Enter": if (!$("run-start").disabled) { event.preventDefault(); start(); } break;
    case "Escape": event.preventDefault(); router.show("library"); break;
    case "r":
      if (!$("run-review").hidden) { event.preventDefault(); router.show("review", streamId); }
      break;
  }
}

$("run-back").onclick = () => router.show("library");
$("run-start").onclick = () => start("run");
$("run-render").onclick = () => start("render");
$("run-review").onclick = () => router.show("review", streamId);
