/* The chrome around the views: the bar, and the empty/loading/error blocks.
 *
 * THE BAR. One `#topbar` holds a block per view and this swaps them. The
 * obvious alternative — a global bar stacked above each view — would have cost
 * the review screen about 44px of video height, on the one screen where
 * vertical space is the constraint (C4). Swapping the contents gives the four
 * screens the same chrome for nothing, and keeps every element id exactly
 * where `review.js` and friends already expect to find it.
 *
 * THE STATES. `library.enter`, `review.enter` and `run.refreshPlan` all
 * awaited a fetch with no catch, so any failure after boot was an unhandled
 * promise rejection: a blank pane, no message, nothing to click. `router.show`
 * now catches and shows the error view; these helpers are what everything else
 * uses to say "nothing here yet" or "still loading" in one voice.
 */

import { $, escape } from "./api.js";

/* Which bar belongs to which view. A view with no entry simply gets none —
 * the error view is deliberately bare, because the bar it would show is the
 * one whose data just failed to load. */
const BARS = {
  library: "bar-library",
  add: "bar-add",
  run: "bar-run",
  review: "bar-review",
  search: "bar-search",
};

export function setBar(name) {
  const wanted = BARS[name];
  for (const id of Object.values(BARS)) {
    const element = $(id);
    if (element) element.hidden = id !== wanted;
  }
  const bar = $("topbar");
  if (bar) bar.hidden = wanted === undefined;
}

/* ------------------------------------------------------------------ states */

function block(kind, title, body, extra = "") {
  return `<div class="state state--${kind}"${kind === "error" ? ' role="alert"' : ""}>
    <h2>${escape(title)}</h2>
    ${body ? `<p>${body}</p>` : ""}
    ${extra}
  </div>`;
}

/** "Nothing here yet", with an optional line of what to do about it. */
export function empty(id, title, body = "", extra = "") {
  const host = $(id);
  if (host) host.innerHTML = block("empty", title, body, extra);
}

/** Shown while a fetch is in flight. Cheap insurance against a blank pane. */
export function loading(id, title = "Loading") {
  const host = $(id);
  if (host) host.innerHTML = block("loading", title);
}

export function clear(id) {
  const host = $(id);
  if (host) host.innerHTML = "";
}

/* ------------------------------------------------------------------- error */

/* The error view is a registered view like any other, so `router.show` can
 * fall back to it without importing anything. `retry` reruns whatever failed. */
export const root = "view-error";

let retry = null;

export async function enter(payload) {
  const { error, view, arg } = payload || {};
  $("error-detail").textContent = error?.message || String(error || "unknown error");
  retry = view ? { view, arg } : null;
  $("error-retry").hidden = !retry;
}

export function onKey(event) {
  if (event.key === "Enter" && retry) { event.preventDefault(); doRetry(); }
  if (event.key === "Escape") { event.preventDefault(); goHome(); }
}

let router = null;

/** Wired from main.js, so this module does not import the router that owns it. */
export function useRouter(module) {
  router = module;
}

function doRetry() {
  if (retry && router) router.show(retry.view, retry.arg);
}

function goHome() {
  if (router) router.show("library");
}

$("error-retry").onclick = doRetry;
$("error-home").onclick = goHome;
