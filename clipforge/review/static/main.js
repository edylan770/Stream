/* Boot: register the views, then decide which one opens.
 *
 * The URL carries enough to resume — `?stream=<id>` reopens review, and
 * `?view=run&stream=<id>` reopens a run in progress — so a reload during a
 * four-hour proxy encode does not lose the screen you were watching. */

import * as router from "./router.js";
import * as library from "./library.js";
import * as add from "./add.js";
import * as run from "./run.js";
import * as review from "./review.js";
import { get } from "./api.js";

router.register("library", library);
router.register("add", add);
router.register("run", run);
router.register("review", review);
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

boot().catch((error) => {
  document.body.innerHTML =
    `<section class="pane"><h1>ClipForge</h1><p class="empty">${error.message}</p></section>`;
});
