# OBS logs for `scene_events`

**Every `.txt` file in this directory is parsed by `tests/test_scene_events.py`,
automatically.** Adding a real log is the entire integration step: drop it in,
run `pytest`, and it is covered.

## Why this directory exists

`clipforge/extract/obs_log.py` was written without a single real OBS log ever
having been seen — there is no OBS install on the machine it was built on. Its
regexes are transcribed from the documented log format and **none of them is
validated**. A regex that does not match produces zero events, silently, and is
indistinguishable from a stream in which nobody switched scenes.

The parametrized fixture directory is the cheapest possible correction path.

## Adding a real log

1. On the streaming PC, find a log in `%APPDATA%\obs-studio\logs\`. Any session
   in which you recorded will do.

2. **Check it before copying it anywhere:**

   ```
   clipforge scene-events --check "<path to the log>"
   ```

   That prints which patterns fired, the recording spans it found, the scene
   timeline, and every line that looks like it should have matched and did not.
   **Send back that report rather than the log** — OBS logs contain machine
   paths, hardware details and sometimes stream URLs.

3. If patterns are dead, fix them in the one place they live:
   `clipforge/config/clipforge.yaml` → `extract.scene_events.patterns`.

4. Copy the log in here, **redacted if you like** — the parser only reads the
   timestamp prefixes, the scene-switch lines and the recording banners, so
   deleting every other line loses nothing this is testing. Then delete
   `synthetic-UNVALIDATED.txt`.

5. Move the `scene_events` rows in `spec/GUESSES.md` from **arbitrary** to
   **grounded**, and drop the UNVALIDATED banner from `obs_log.py`'s docstring.

## What the tests do and do not assert

They assert the **arithmetic** — that elapsed time converts to
recording-relative seconds, that spans are clipped to the recording, that a
scene already up when recording began runs from t=0, that an ambiguous log is
refused rather than guessed.

They deliberately **do not** assert that any pattern matches real OBS output,
because no file here can settle that. A green suite over a synthetic log means
the parser is self-consistent, not that it is right.
