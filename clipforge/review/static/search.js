/* §11.6's pull search — free text in, transcript segments out.
 *
 * "Keyword search cannot do this; the operator remembers the vibe, not the
 * words." So the box takes a description, not a quote, and the placeholder says
 * so — a search box that looks like a keyword filter gets used like one, and
 * then appears not to work.
 *
 * SUBMIT ON ENTER, NOT AS YOU TYPE. Every query costs an embedding round trip
 * to Ollama plus a scan of the index — measured at ~0.7 s over 200k segments —
 * so a keystroke-triggered search would fire a dozen of those to answer one
 * question. Debouncing would only hide that; the query was never incremental.
 *
 * AN EMPTY RESULT SAYS WHICH KIND OF EMPTY IT IS. Three different things
 * produce zero rows and they need three different sentences: nothing is
 * indexed, the embedder is unreachable, or the search ran and this library has
 * no such moment. Showing "no results" for all three is how a broken search
 * looks healthy — and on shipped defaults the first is the COMMON case, because
 * §5.7's transcription is off. */

import { $, escape, fmt, get } from "./api.js";
import * as router from "./router.js";
import * as shell from "./shell.js";

export const root = "view-search";

const state = {
  query: "",
  hits: [],
  cursor: 0,
  corpus: null,
  searching: false,
};

export async function enter(initial) {
  history.replaceState(null, "", location.pathname + "?view=search");
  state.hits = [];
  state.cursor = 0;

  // The corpus survey is a separate, cheap call so the screen can say what is
  // searchable BEFORE anything is typed — rather than looking ready and then
  // explaining after the first query that nothing is indexed.
  const info = await get("/api/search/corpus");
  state.corpus = info;
  renderCorpus();

  const input = $("search-input");
  input.value = typeof initial === "string" ? initial : state.query;
  input.focus();
  input.select();

  if (!info.searchable) {
    renderUnsearchable();
  } else {
    shell.empty("search-state", "Describe a moment",
      "Not the words that were said — what was happening. " +
      "&ldquo;the time the game crashed&rdquo;, &ldquo;explaining the tier list&rdquo;.");
  }
}

export function leave() {
  state.query = $("search-input").value;
}

function renderCorpus() {
  const info = state.corpus;
  const el = $("search-corpus");
  if (!info) { el.textContent = ""; return; }
  el.textContent = info.searchable
    ? `${info.embedded} segment(s) · ${info.streams} stream(s) · ${info.model}`
    : "nothing indexed";
}

/* The screen's most important state, because it is the DEFAULT one: a library
 * processed with shipped defaults has no transcripts and therefore nothing to
 * search. Saying so, with the reason and the cost, beats an empty list. */
function renderUnsearchable() {
  const info = state.corpus;
  $("search-results").innerHTML = "";
  if (info.segments > 0) {
    shell.empty(
      "search-state", "Nothing is indexed yet",
      `${info.segments} transcript segment(s) exist but none has an embedding. ` +
      "Run the pipeline again to fill them in — the <code>embeddings</code> " +
      "stage is what this searches.");
  } else {
    shell.empty(
      "search-state", "No transcripts to search",
      "Transcription ships <b>off</b> because it costs a multi-GB model " +
      "download. Turn on <code>extract.whisperx.enabled</code> and re-run a " +
      "stream; this search and the review screen's transcript panel are what " +
      "it buys.");
  }
}

async function submit(event) {
  if (event) event.preventDefault();
  const query = $("search-input").value.trim();
  if (!query || state.searching) return;
  if (!state.corpus?.searchable) return renderUnsearchable();

  state.searching = true;
  state.query = query;
  $("search-results").innerHTML = "";
  shell.loading("search-state", "Searching");

  try {
    const data = await get(`/api/search?q=${encodeURIComponent(query)}`);
    state.hits = data.hits || [];
    state.cursor = 0;
    render();
  } catch (error) {
    // A failed embed is the likely cause and it is not the operator's fault,
    // so it gets its own message rather than the generic error view.
    shell.empty("search-state", "Could not search",
      escape(error?.message || String(error)));
  } finally {
    state.searching = false;
  }
}

function render() {
  const list = $("search-results");
  list.innerHTML = "";

  if (!state.hits.length) {
    shell.empty("search-state", "Nothing matched",
      "The index is fine — this library has no moment like that. " +
      "Try describing it differently.");
    return;
  }
  shell.clear("search-state");

  state.hits.forEach((hit, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = index === state.cursor ? "current" : "";
    button.onclick = () => open(index);

    // "no candidate covers this" is said plainly rather than hidden. §11.6
    // returns SEGMENTS and they exist independently of scoring, so a memorable
    // line is frequently nowhere near a detected peak — and a result that
    // silently opened an unrelated candidate would misreport what was found.
    const covered = hit.candidate_id !== null && hit.candidate_id !== undefined;
    button.innerHTML = `
      <span class="grow">
        <span class="result-text">${escape(hit.text)}</span>
        <span class="sub">${escape(hit.stream_title || hit.stream_id)} ·
          ${escape(hit.stream_date || "")} · ${fmt(hit.t_start)}
          ${hit.speaker ? `· ${escape(hit.speaker)}` : ""}</span>
      </span>
      <span class="result-meta ${covered ? "" : "muted"}">${
        covered ? "candidate" : "no candidate"}</span>`;
    item.appendChild(button);
    list.appendChild(item);
  });
}

/* Opening a result goes to the review screen for that stream, carrying the time
 * so it can focus the right moment. `review.enter` takes a stream id; the time
 * rides in the URL rather than as a second argument so a reload keeps it. */
function open(index) {
  const hit = state.hits[index];
  if (!hit) return;
  state.cursor = index;
  router.show("review", { streamId: hit.stream_id, at: hit.t_start });
}

function move(delta) {
  if (!state.hits.length) return;
  state.cursor = Math.max(0, Math.min(state.cursor + delta, state.hits.length - 1));
  render();
  document.querySelector("#search-results button.current")
    ?.scrollIntoView({ block: "nearest" });
}

/* The router already returns early for INPUT targets, so these only fire once
 * focus has left the box — which is what makes a text field and the library's
 * j/k coexist without a second global handler. */
export function onKey(event) {
  switch (event.key) {
    case "j": case "ArrowDown": event.preventDefault(); move(1); break;
    case "k": case "ArrowUp": event.preventDefault(); move(-1); break;
    case "Enter": event.preventDefault(); open(state.cursor); break;
    case "/": event.preventDefault(); $("search-input").focus(); break;
    case "Escape": event.preventDefault(); router.show("library"); break;
  }
}

$("search-form").addEventListener("submit", submit);
$("search-back").onclick = () => router.show("library");

// Escape inside the box returns focus to the list rather than leaving the
// screen, so the operator can start navigating results without the mouse.
$("search-input").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    $("search-input").blur();
  }
});
