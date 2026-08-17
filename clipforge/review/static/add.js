/* Adding a recording: drop or browse, preflight, register.
 *
 * The listing comes from the server because a browser will not hand a page a
 * filesystem path — see review/browse.py for why "drop your recording here"
 * would mean uploading a 50 GB master to localhost.
 *
 * DROPPING A FILE STILL WORKS, and still uploads nothing. `File.name` and
 * `File.size` are metadata the browser gives away without reading the file, so
 * they are sent instead and the server finds the master where recordings have
 * come from before. The `File` object itself never leaves this function: no
 * FileReader, no request body carrying it. That is the whole trick, and it is
 * why a 40 GB drop resolves in milliseconds.
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
let filter = "";
let newestFirst = true;

export async function enter() {
  chosen = null;
  preflight = null;
  filter = "";
  $("add-filter").value = "";
  $("add-drop-note").hidden = true;
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

  let entries = listing.entries;
  if (filter) {
    const needle = filter.toLowerCase();
    entries = entries.filter((e) => e.name.toLowerCase().includes(needle));
  }
  if (newestFirst) {
    // Directories keep their alphabetical order — you navigate those. Files are
    // sorted by modification time, because the recording being added is the one
    // just made and OBS names a season's worth of them almost identically.
    const dirs = entries.filter((e) => e.is_dir);
    const files = entries.filter((e) => !e.is_dir)
      .slice().sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
    entries = dirs.concat(files);
  }
  return out.concat(entries);
}

function render() {
  $("add-path").value = listing.path;
  $("add-error").hidden = !listing.error;
  if (listing.error) $("add-error").textContent = listing.error;

  const roots = $("add-roots");
  roots.innerHTML = "";
  for (const drive of listing.roots) {
    const button = document.createElement("button");
    button.className = "btn btn--ghost";
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

  const recent = $("add-recent");
  recent.innerHTML = "";
  if (listing.recent && listing.recent.length) {
    recent.append("recent: ");
    for (const dir of listing.recent.slice(0, 5)) {
      const button = document.createElement("button");
      button.className = "btn btn--ghost btn--tiny";
      button.textContent = dir.split(/[\\/]/).filter(Boolean).pop() || dir;
      button.title = dir;
      button.onclick = () => browse(dir);
      recent.appendChild(button);
    }
  }

  $("add-sort").textContent = newestFirst ? "newest first" : "by name";
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
      <dt>audio</dt><dd>${audioLine(preflight.audio)}</dd>
    </dl>` +
    (preflight.already_registered
      ? `<p class="warn-line">that id is already registered — re-registering
         invalidates everything built from the old master, so it stays a
         deliberate <code>clipforge register --force</code></p>` : "") +
    preflight.warnings.map((w) => `<p class="warn-line">${escape(w)}</p>`).join("");
}

/* §4.2's track map, read from the container before anything is written.
 *
 * This is the line that answers "where does the mic audio go?" — it does not go
 * anywhere, because OBS records all four tracks INTO THIS FILE. Which also means
 * that if OBS was not set to multi-track, everything downstream is wrong in a
 * way §4.2 calls unrecoverable, and that has to be visible while the choice is
 * still the operator's. */
function audioLine(audio) {
  if (!audio) return `<span class="muted">not read</span>`;
  // `detail` carries the raw ffprobe text. It goes in the tooltip, not the
  // line: somebody who has just dragged in the wrong file needs a sentence,
  // not an argv.
  if (audio.error) {
    return `<span class="missing" title="${escape(audio.detail || "")}">` +
           `${escape(audio.error)}</span>`;
  }

  const roles = Object.keys(audio.roles || {});
  const count = audio.track_count || 0;
  const named = roles.length
    ? roles.map((r) => escape(r)).join(", ")
    : "none identified";
  const body = `${count} track${count === 1 ? "" : "s"} — ${named}`;

  const notes = (audio.warnings || [])
    .map((w) => `<p class="warn-line">${escape(w)}</p>`).join("");

  // §4.2 mandates mic, game and party on their own tracks. One track is the
  // contamination case the spec calls unrecoverable; two or three is the shape
  // `audio_split` refuses outright rather than guessing at.
  const ok = roles.includes("mic") && roles.includes("party");
  return `<span class="${ok ? "ok" : "missing"}">${escape(body)}</span>${notes}`;
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

/* Drop-to-locate.
 *
 * `file.name` and `file.size` are read; `file` itself goes no further. There is
 * no FileReader here and no request body carrying it, which is what makes
 * dropping a 40 GB master resolve in milliseconds instead of copying it. */
async function dropped(file) {
  const note = $("add-drop-note");
  note.hidden = false;
  note.className = "muted small";
  note.textContent = `looking for ${file.name}…`;

  let found;
  try {
    found = await post("/api/browse/locate", {
      name: file.name, size: file.size, path: listing ? listing.path : null,
    });
  } catch (error) {
    note.className = "error";
    note.textContent = error.message;
    return;
  }

  if (!found.path) {
    // The size mismatch comes FIRST when there is one: "a file of this name
    // exists but is a different recording" is far more actionable than a list
    // of folders, and burying it under one is how it gets missed.
    const stale = (found.size_mismatch || []).length
      ? `${file.name} exists at ${found.size_mismatch.join(", ")} but is a ` +
        `different size — that is a different recording, not this one. `
      : `Could not find ${file.name} on this machine. `;

    // "Not found" is only actionable if you can see where it looked. Capped,
    // because the list grows with every folder ever registered from.
    const dirs = found.searched || [];
    const shown = dirs.slice(0, 4).join(", ");
    const more = dirs.length > 4 ? ` and ${dirs.length - 4} more` : "";
    const where = dirs.length ? `Looked in: ${shown}${more}. ` : "";

    note.className = "error";
    note.textContent =
      `${stale}${where}Browse to it below, or add its folder to ` +
      `review.browse.search_dirs.`;
    return;
  }

  note.className = "muted small";
  note.textContent = `found at ${found.path}`;
  await browse(found.path);            // a file path lists its own folder
  const index = rows().findIndex((entry) => entry.path === found.path);
  if (index >= 0) await choose(index);
}

const zone = $("add-drop");
for (const name of ["dragenter", "dragover"]) {
  zone.addEventListener(name, (event) => {
    event.preventDefault();
    zone.classList.add("over");
  });
}
for (const name of ["dragleave", "drop"]) {
  zone.addEventListener(name, () => zone.classList.remove("over"));
}
zone.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer?.files?.[0];
  if (file) dropped(file);
});
// Without this the browser NAVIGATES AWAY to the dropped file, losing the page.
for (const name of ["dragover", "drop"]) {
  window.addEventListener(name, (event) => event.preventDefault());
}

$("add-back").onclick = () => router.show("library");
$("add-register").onclick = () => register();
$("add-path").addEventListener("keydown", (event) => {
  if (event.key === "Enter") browse($("add-path").value.trim());
});
$("add-filter").addEventListener("input", (event) => {
  filter = event.target.value.trim();
  cursor = 0;
  render();
});
$("add-sort").onclick = () => {
  newestFirst = !newestFirst;
  cursor = 0;
  render();
};
