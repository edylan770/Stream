/* The stream list — where the app opens.
 *
 * Every stream, not only the reviewable ones. A stream that has been registered
 * but not processed is the single most important row here: it is the one that
 * needs an action, and hiding it until it is ready would mean the operator has
 * no way to see that the thing they just added exists. */

import { $, escape, fmt, get } from "./api.js";
import * as router from "./router.js";

export const root = "view-library";

let streams = [];
let cursor = 0;

export async function enter() {
  history.replaceState(null, "", location.pathname);
  const data = await get("/api/streams");
  streams = data.streams;
  cursor = Math.min(cursor, Math.max(streams.length - 1, 0));
  render();
}

function ready(stream) {
  return stream.candidates > 0 && stream.has_proxy;
}

function status(stream) {
  if (ready(stream)) return `${stream.rated}/${stream.candidates} rated`;
  if (!stream.has_proxy && !stream.stages_done) return "not processed yet";
  if (!stream.has_proxy) return "no proxy yet";
  return "not scored yet";
}

function render() {
  const list = $("stream-list");
  list.innerHTML = "";
  $("library-empty").hidden = streams.length > 0;

  streams.forEach((stream, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = index === cursor ? "current" : "";
    button.onclick = () => open(index);
    button.innerHTML = `
      <span class="grow">
        <span class="name">${escape(stream.title || stream.id)}</span><br>
        <span class="muted">${escape(stream.date)} ·
        ${stream.duration_s ? fmt(stream.duration_s) : "not probed"} ·
        ${escape(stream.resolution || "")}</span>
      </span>
      <span class="count">${ready(stream) ? stream.candidates : ""}</span>
      <span class="muted state">${escape(status(stream))}</span>`;
    if (stream.warnings?.length) {
      const warn = document.createElement("span");
      warn.className = "pill warn";
      warn.textContent = "!";
      warn.title = stream.warnings.join("  ");
      button.appendChild(warn);
    }
    item.appendChild(button);
    list.appendChild(item);
  });
}

/* A ready stream goes straight to review; anything else goes to the run view,
 * which is the thing that can actually get it there. */
function open(index) {
  const stream = streams[index];
  if (!stream) return;
  cursor = index;
  router.show(ready(stream) ? "review" : "run", stream.id);
}

function move(delta) {
  if (!streams.length) return;
  cursor = Math.max(0, Math.min(cursor + delta, streams.length - 1));
  render();
  document.querySelector("#stream-list button.current")?.scrollIntoView({ block: "nearest" });
}

export function onKey(event) {
  switch (event.key) {
    case "j": case "ArrowDown": event.preventDefault(); move(1); break;
    case "k": case "ArrowUp": event.preventDefault(); move(-1); break;
    case "Enter": event.preventDefault(); open(cursor); break;
    case "a": event.preventDefault(); router.show("add"); break;
  }
}

$("add-recording").onclick = () => router.show("add");
