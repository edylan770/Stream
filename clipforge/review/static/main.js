/* Boot: register the views, then decide which one opens.
 *
 * The URL carries enough to resume — `?stream=<id>` reopens review, and
 * `?view=run&stream=<id>` reopens a run in progress — so a reload during a
 * four-hour proxy encode does not lose the screen you were watching. */

import * as router from "./router.js";
import * as shell from "./shell.js";
import * as library from "./library.js";
import * as add from "./add.js";
import * as run from "./run.js";
import * as review from "./review.js";
import { get } from "./api.js";

router.register("library", library);
router.register("add", add);
router.register("run", run);
router.register("review", review);
router.register("error", shell);
shell.useRouter(router);
router.startKeyboard();

async function boot() {
  const params = new URLSearchParams(location.search);
  const wanted = params.get("stream");

  if (wanted) {
    const { streams } = await get("/api/streams");
    const stream = streams.find((s) => s.id === wanted);
    if (stream) {
      const view = params.get("view") === "run" || !(stream.candidates > 0 && stream.has_proxy)
        ? "run" : "review";
      return router.show(view, wanted);
    }
  }
  return router.show("library");
}

/* A failure here is before any view exists, so it cannot go through the
 * router's own fallback — but it no longer destroys the DOM to say so. */
boot().catch((error) => router.show("error", { error }).catch(() => {
  document.body.innerHTML =
    `<section class="pane"><h1>ClipForge</h1>` +
    `<div class="state state--error"><h2>Could not start</h2>` +
    `<p>${String(error && error.message ? error.message : error)}</p></div></section>`;
}));
