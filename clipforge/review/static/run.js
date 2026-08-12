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

export async function enter(id) {
  streamId = id;
  jobId = null;
  cursor = 0;
  $("run-log").textContent = "";
  $("run-name").textContent = id;
  history.replaceState(null, "", `?view=run&stream=${encodeURIComponent(id)}`);
  await refreshPlan();
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
    item.className = `stage ${cls} ${stage.status === "failed" ? "failed" : ""}`;
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
}

async function start() {
  $("run-start").disabled = true;
  try {
    const job = await post(`/api/streams/${encodeURIComponent(streamId)}/run`);
    jobId = job.id;
    cursor = 0;
    $("run-log").textContent = "";
    poll();
  } catch (error) {
    append(`  ${error.message}`);
    $("run-start").disabled = false;
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
  $("run-state").textContent = snapshot.state;
  $("run-state").className = `pill ${snapshot.state}`;

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
$("run-start").onclick = () => start();
$("run-review").onclick = () => router.show("review", streamId);
