"""Vision / OCR (spec §5.9). Phase 7 — not built, and explicitly cuttable.

The spec's own assessment: the piece most likely to defeat the build, and the
signal most affordable to lose, because audio signals already fire during
multikills — the operator reacts to them.

If it is ever built: sample at 2-4 fps not 60, use OpenCV `matchTemplate`
against cropped UI regions rather than full-frame OCR, and version templates by
game patch.
"""
