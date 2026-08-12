/* Adding a recording: browse, preflight, register.
 *
 * The listing comes from the server because a browser will not hand a page a
 * filesystem path — see review/browse.py for why "drop your recording here"
 * would mean uploading a 50 GB master to localhost.
 *
 * Keyboard-first. §7.3 scopes "the operator should never need the mouse" to the
 * review screen, but choosing a path on a recording drive is the most
 * mouse-dependent thing in the app, so j/k/enter/backspace all work and a path
 * can be pasted straight into the field.
 *
 * Nothing is written until Register. The preflight above the button is the
 * whole point: §4.1 makes record_start_epoch_ms the entire sync solution, and a
 * stream registered without it reads §4.3's epoch marker presses as VOD
 * seconds. Learning that after the row exists is learning it too late. */

import { $, bytes, escape, get, post } from "./api.js";
import * as router from "./router.js";

export const root = "view-add";

let listing = null;
let cursor = 0;
let chosen = null;      // the master path being considered
let preflight = null;

export async function enter() {
  chosen = null;
  preflight = null;
  await browse(null);
}

export function leave() {
  $("add-error").hidden = true;
}

async function browse(path) {
  try {
    listing = await get("/api/browse", { path });
  } catch (error) {
    return fail(error.message);
  }
  cursor = 0;
  chosen = null;
  preflight = null;
  render();
}

function rows() {
  const out = listing.parent ? [{ name: "..", path: listing.parent, is_dir: true, up: true }] : [];
  return out.concat(listing.entries);
}

function render() {
  $("add-path").value = listing.path;
  $("add-error").hidden = !listing.error;
  if (listing.error) $("add-error").textContent = listing.error;

  const roots = $("add-roots");
  roots.innerHTML = "";
  for (const drive of listing.roots) {
    const button = document.createElement("button");
    button.className = "chip";
    button.textContent = drive;
    button.onclick = () => browse(drive);
    roots.appendChild(button);
  }

  const list = $("add-entries");
  list.innerHTML = "";
  rows().forEach((entry, index) => {
    const item = document.createElement("li");
    item.className = [
      index === cursor ? "current" : "",
      entry.is_dir ? "dir" : "",
      entry.is_media ? "media" : "",
    ].filter(Boolean).join(" ");
    item.innerHTML =
      `<span class="what">${entry.is_dir ? "▸" : entry.is_media ? "●" : "·"}</span>` +
      `<span class="nm">${escape(entry.name)}</span>` +
      `<span class="muted sz">${entry.is_dir ? "" : bytes(entry.size)}</span>`;
    item.onclick = () => choose(index);
    list.appendChild(item);
  });

  $("add-truncated").hidden = !listing.truncated;
  $("add-capture").innerHTML =
    `anchor.json ${mark(listing.has_anchor)} · markers.jsonl ${mark(listing.has_markers)}`;

  renderChoice();
}

function mark(present) {
  return present
    ? `<span class="ok">found</span>`
    : `<span class="missing">not here</span>`;
}

function renderChoice() {
  const panel = $("add-choice");
  panel.hidden = !chosen;
  $("add-register").disabled = !preflight;
  if (!chosen) return;

  if (!preflight) {
    panel.innerHTML = `<div class="muted">reading ${escape(chosen)}…</div>`;
    return;
  }

  const anchor = preflight.anchor_ms
    ? `<span class="ok">${preflight.anchor_ms}</span>`
    : `<span class="missing">none — marker times read as VOD seconds</span>`;

  panel.innerHTML = `
    <dl class="facts">
      <dt>stream id</dt><dd class="mono">${escape(preflight.stream_id)}</dd>
      <dt>date</dt><dd>${escape(preflight.date)}</dd>
      <dt>anchor</dt><dd>${anchor}</dd>
      <dt>markers</dt><dd>${preflight.markers
        ? `<span class="ok">${escape(preflight.markers.split(/[\\/]/).pop())}</span>`
        : `<span class="muted">none found</span>`}</dd>
    </dl>` +
    (preflight.already_registered
      ? `<p class="warn-line">that id is already registered — re-registering
         invalidates everything built from the old master, so it stays a
         deliberate <code>clipforge register --force</code></p>` : "") +
    preflight.warnings.map((w) => `<p class="warn-line">${escape(w)}</p>`).join("");
}

async function choose(index) {
  const entry = rows()[index];
  if (!entry) return;
  cursor = index;
  if (entry.is_dir) return browse(entry.path);

  chosen = entry.path;
  preflight = null;
  renderChoice();
  try {
    preflight = await post("/api/register/preflight", { master: chosen, title: title() });
  } catch (error) {
    chosen = null;
    return fail(error.message);
  }
  renderChoice();
}

function title() {
  return $("add-title").value.trim() || null;
}

function fail(message) {
  $("add-error").hidden = false;
  $("add-error").textContent = message;
}

async function register() {
  if (!preflight) return;
  $("add-register").disabled = true;
  try {
    const result = await post("/api/register", {
      master: chosen, title: title(), games: $("add-games").value.trim() || null,
    });
    router.show("run", result.stream_id);
  } catch (error) {
    fail(error.message);
    $("add-register").disabled = false;
  }
}

function move(delta) {
  const all = rows();
  if (!all.length) return;
  cursor = Math.max(0, Math.min(cursor + delta, all.length - 1));
  render();
  document.querySelector("#add-entries li.current")?.scrollIntoView({ block: "nearest" });
}

export function onKey(event) {
  switch (event.key) {
    case "j": case "ArrowDown": event.preventDefault(); move(1); break;
    case "k": case "ArrowUp": event.preventDefault(); move(-1); break;
    case "Enter": event.preventDefault(); chosen ? register() : choose(cursor); break;
    case "Backspace":
      event.preventDefault();
      if (listing.parent) browse(listing.parent);
      break;
    case "Escape": event.preventDefault(); router.show("library"); break;
  }
}

$("add-back").onclick = () => router.show("library");
$("add-register").onclick = () => register();
$("add-path").addEventListener("keydown", (event) => {
  if (event.key === "Enter") browse($("add-path").value.trim());
});
