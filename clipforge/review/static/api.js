/* Talking to the server, and the two formatting helpers everything needs.
 *
 * ES modules, no bundler, nothing fetched from anywhere. §2.2 says avoid heavy
 * frameworks, and a dedicated streaming PC may have no internet at all — so
 * there is no build step to forget to run and no CDN to be offline. */

/* Anything that changes something must carry this header. The value is never
 * checked: sending it at all is what forces a CORS preflight, which is what
 * stops another page in the browser from posting here. See review/guard.py. */
export const HEADER = "X-ClipForge";

export async function get(path, params) {
  const url = new URL(path, location.origin);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null) url.searchParams.set(key, value);
  }
  const response = await fetch(url);
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", [HEADER]: "1" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/* Fire and forget: the review keyboard must never wait on the network. */
export function postAndForget(path, body) {
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", [HEADER]: "1" },
    body: JSON.stringify(body || {}),
  }).catch(() => {});
}

async function failure(response) {
  let detail = `${response.status}`;
  try {
    const body = await response.json();
    detail = body.detail || detail;
  } catch {
    try { detail = (await response.text()) || detail; } catch { /* give up */ }
  }
  const error = new Error(detail);
  error.status = response.status;
  return error;
}

export const $ = (id) => document.getElementById(id);

/* m:ss.s — the review screen's window readout depends on the tenths. */
export function fmt(seconds) {
  const total = Number(seconds) || 0;
  const minutes = Math.floor(total / 60);
  return `${minutes}:${(total % 60).toFixed(1).padStart(4, "0")}`;
}

export function bytes(size) {
  if (size === null || size === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Number(size);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function escape(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
