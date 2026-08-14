/* Which view is on screen.
 *
 * A registry rather than direct imports, so a view can navigate to another one
 * without importing it — review.js going back to the library would otherwise
 * form an import cycle with whatever module owns the switching.
 *
 * Each view module exports `root` (an element id), `enter(arg)`, and optionally
 * `leave()` and `onKey(event)`. The keyboard listener lives here so exactly one
 * view can ever be listening: the review screen's j/k and the file browser's
 * j/k are the same keys doing different things, and two handlers both checking
 * whether their own section is hidden is how that goes wrong. */

import { $ } from "./api.js";
import * as shell from "./shell.js";

const views = new Map();
let current = null;

export function register(name, module) {
  views.set(name, module);
}

export function activeName() {
  return current;
}

function swap(name) {
  for (const [key, module] of views) {
    const element = $(module.root);
    if (element) element.hidden = key !== name;
  }
  current = name;
  shell.setBar(name);
}

export async function show(name, arg) {
  const next = views.get(name);
  if (!next) throw new Error(`no view ${name}`);

  if (current && current !== name) {
    const previous = views.get(current);
    if (previous?.leave) previous.leave();
  }

  swap(name);

  // Every view's enter() awaits a fetch. Three of the four had no catch, so a
  // server that had stopped left a blank pane and an unhandled rejection in
  // the console — the operator's only signal that anything had happened.
  try {
    await next.enter(arg);
  } catch (error) {
    if (name === "error") throw error;      // nowhere left to fall back to
    swap("error");
    await views.get("error").enter({ error, view: name, arg });
  }
}

export function startKeyboard() {
  document.addEventListener("keydown", (event) => {
    // Never swallow keys meant for a text field.
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || event.target.isContentEditable) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    const module = views.get(current);
    if (module?.onKey) module.onKey(event);
  });
}
