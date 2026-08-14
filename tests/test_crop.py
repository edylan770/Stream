"""§8.4's crop templates: geometry, validation, and the filter graph.

All pure — no ffmpeg. The graph is a string, and the things that go wrong with
it (a region off the edge of the frame, two regions on top of each other, a
23% vertical stretch) are all decidable from the numbers.
"""

from __future__ import annotations

import pytest

from clipforge import config
from clipforge.render import crop
from clipforge.render.crop import CONTAIN, FILL, STRETCH, Rect, Region, Template


def template(*regions, source=(1920, 1080), output=(1080, 1920), name="t"):
    return Template(name=name, source_resolution=source, output=output,
                    regions=tuple(regions))


def region(name, src, dst, fit=None):
    return Region(name=name, src=Rect(*src), dst=Rect(*dst), fit=fit)


# --------------------------------------------------------------------------
# the shipped templates
# --------------------------------------------------------------------------


def test_the_shipped_templates_all_parse():
    templates = crop.load_templates(config.load())

    assert "full_frame" in templates
    assert "marvel_rivals_facecam" in templates


def test_every_shipped_template_validates_against_its_own_source():
    """A template that cannot pass its own declared resolution is a typo."""
    for name, spec in crop.load_templates(config.load()).items():
        width, height = spec.source_resolution
        check = crop.validate(spec, width, height)
        assert isinstance(check, crop.Check), name


def test_the_default_template_covers_the_whole_output():
    """`full_frame` is what renders before any coordinate is measured, so it
    must not be the one that leaves a black band."""
    cfg = config.load()
    spec = crop.get_template(cfg)

    check = crop.validate(spec, *spec.source_resolution)

    assert check.uncovered == 0
    assert not any("not covered" in w for w in check.warnings)


def test_reframing_16x9_to_9x16_does_not_trip_the_upscale_warning():
    """It always enlarges by about 1.78x. A warning that fires on every clip
    is a warning nobody reads."""
    spec = crop.load_templates(config.load())["full_frame"]

    check = crop.validate(spec, *spec.source_resolution)

    assert not check.upscaled


def test_the_spec_example_tiles_the_output_exactly():
    """§8.4's facecam 0..810 and gameplay 810..1920 should meet."""
    spec = crop.load_templates(config.load())["marvel_rivals_facecam"]

    check = crop.validate(spec, *spec.source_resolution)

    assert check.uncovered == 0


def test_the_spec_example_would_distort_without_a_fit_policy():
    """The finding: src 960x800 into dst 1080x1110 is a 1.23x vertical
    stretch. The shipped template says `fill` for exactly this reason."""
    spec = crop.load_templates(config.load())["marvel_rivals_facecam"]
    gameplay = next(r for r in spec.regions if r.name == "gameplay")

    assert gameplay.src.aspect != pytest.approx(gameplay.dst.aspect, rel=0.05)
    assert gameplay.fit == FILL


def test_an_unknown_template_lists_the_ones_that_exist():
    with pytest.raises(crop.CropError, match="full_frame"):
        crop.get_template(config.load(), "no_such_template")


# --------------------------------------------------------------------------
# resolving onto a real source
# --------------------------------------------------------------------------


def test_a_matching_resolution_is_returned_untouched():
    """The common case does no arithmetic and so cannot drift."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    assert crop.resolve(spec, 1920, 1080) is spec


def test_coordinates_scale_to_a_smaller_source_of_the_same_shape():
    """The fixture is 640x360 and the templates are declared on 1920x1080."""
    spec = template(region("a", (960, 540, 480, 270), (0, 0, 1080, 1920)))

    moved = crop.resolve(spec, 640, 360)

    assert moved.source_resolution == (640, 360)
    assert moved.regions[0].src == Rect(320, 180, 160, 90)
    assert moved.regions[0].dst == spec.regions[0].dst   # output never moves


def test_scaling_rounds_outward_so_a_region_never_shrinks():
    """C2: expand rather than contract. A pixel lost off the facecam is worse
    than a pixel of border gained."""
    spec = template(region("a", (11, 11, 101, 101), (0, 0, 1080, 1920)))

    moved = crop.resolve(spec, 640, 360)

    assert moved.regions[0].src.w >= 101 * 640 / 1920
    assert moved.regions[0].src.x <= -(-11 * 640 // 1920)


def test_scaling_clamps_inside_the_frame():
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    moved = crop.resolve(spec, 640, 360)

    assert moved.regions[0].src.inside(640, 360)


def test_a_different_aspect_ratio_is_refused():
    """Scaling x and y by different factors moves every region somewhere the
    operator did not put it, and the result looks plausible."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    with pytest.raises(crop.CropError, match="refused rather than guessed"):
        crop.resolve(spec, 1920, 1200)


def test_a_rounding_sized_aspect_difference_still_passes():
    """1280x720 against 1920x1080 is the same shape."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    assert crop.resolve(spec, 1280, 720).source_resolution == (1280, 720)


# --------------------------------------------------------------------------
# validation §8.4 never mentions
# --------------------------------------------------------------------------


def test_a_source_rectangle_outside_the_frame_is_refused():
    spec = template(region("a", (1800, 0, 400, 1080), (0, 0, 1080, 1920)))

    with pytest.raises(crop.CropError, match="outside the 1920x1080 source"):
        crop.validate(spec, 1920, 1080)


def test_a_destination_outside_the_output_is_refused():
    spec = template(region("a", (0, 0, 100, 100), (0, 1900, 1080, 100)))

    with pytest.raises(crop.CropError, match="outside the 1080x1920 output"):
        crop.validate(spec, 1920, 1080)


def test_overlapping_regions_are_refused():
    """One region drawn over another means part of what the template says to
    show cannot be seen. There is no layout where that is the intent."""
    spec = template(
        region("top", (0, 0, 960, 540), (0, 0, 1080, 1000)),
        region("bottom", (0, 540, 960, 540), (0, 900, 1080, 1020)),
    )

    with pytest.raises(crop.CropError, match="overlap"):
        crop.validate(spec, 1920, 1080)


def test_the_overlap_error_names_both_regions_and_the_rectangle():
    spec = template(
        region("facecam", (0, 0, 960, 540), (0, 0, 1080, 1000)),
        region("gameplay", (0, 540, 960, 540), (0, 900, 1080, 1020)),
    )

    with pytest.raises(crop.CropError) as raised:
        crop.validate(spec, 1920, 1080)

    message = str(raised.value)
    assert "facecam" in message and "gameplay" in message
    assert "[0, 900, 1080, 100]" in message


def test_a_gap_is_warned_and_measured_not_refused():
    """A letterbox band is a legitimate design, and it is also exactly what a
    typo looks like. Reporting it is the only honest answer."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 960)))

    check = crop.validate(spec, 1920, 1080)

    assert check.uncovered == 1080 * 960
    assert any("not covered" in w for w in check.warnings)


def test_a_big_upscale_is_warned_not_refused():
    """§8.4's facecam is 360x270 filling 1080x810 — 3x, and visibly soft."""
    spec = template(region("a", (0, 0, 360, 270), (0, 0, 1080, 810)))

    check = crop.validate(spec, 1920, 1080)

    assert check.upscaled and check.upscaled[0].startswith("a (3.0x)")
    assert any("visibly soft" in w for w in check.warnings)


def test_the_upscale_threshold_is_the_callers_to_set():
    spec = template(region("a", (0, 0, 360, 270), (0, 0, 1080, 810)))

    assert not crop.validate(spec, 1920, 1080, upscale_warn_factor=4.0).upscaled


def test_an_odd_output_dimension_is_refused():
    """libx264 with yuv420p rejects it outright, and the error names neither
    the setting nor the cause."""
    spec = template(region("a", (0, 0, 100, 100), (0, 0, 1081, 1920)),
                    output=(1081, 1920))

    with pytest.raises(crop.CropError, match="odd"):
        crop.validate(spec, 1920, 1080)


def test_a_template_with_no_regions_is_refused():
    with pytest.raises(crop.CropError, match="no regions"):
        crop.parse_template("bad", {"source_resolution": [1920, 1080],
                                    "output": [1080, 1920], "regions": []})


def test_a_malformed_rectangle_names_the_region_and_the_field():
    with pytest.raises(crop.CropError, match="bad.face.src"):
        crop.parse_template("bad", {
            "source_resolution": [1920, 1080], "output": [1080, 1920],
            "regions": [{"name": "face", "src": [0, 0, 10], "dst": [0, 0, 10, 10]}],
        })


def test_an_unknown_fit_is_refused_at_parse_time():
    with pytest.raises(crop.CropError, match="fit must be one of"):
        crop.parse_template("bad", {
            "source_resolution": [1920, 1080], "output": [1080, 1920],
            "regions": [{"name": "f", "src": [0, 0, 10, 10],
                         "dst": [0, 0, 10, 10], "fit": "cover"}],
        })


# --------------------------------------------------------------------------
# the filter graph
# --------------------------------------------------------------------------


def test_one_region_needs_no_split():
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    graph = crop.filter_complex(spec)

    assert "split" not in graph
    assert graph.startswith("[0:v]crop=1920:1080:0:0")


def test_two_regions_split_the_input_once():
    """One decode, not two: the source is read once and forked in the graph."""
    spec = template(
        region("face", (0, 0, 960, 540), (0, 0, 1080, 810)),
        region("game", (0, 540, 960, 540), (0, 810, 1080, 1110)),
    )

    graph = crop.filter_complex(spec)

    assert "[0:v]split=2[s0][s1]" in graph
    assert graph.count("crop=") >= 2


def test_the_canvas_is_made_by_padding_the_first_region_not_by_a_color_source():
    """A `color` source is an infinite stream and would need a duration bound
    to terminate; padding inherits the video's own timing."""
    spec = template(
        region("face", (0, 0, 960, 540), (0, 0, 1080, 810)),
        region("game", (0, 540, 960, 540), (0, 810, 1080, 1110)),
    )

    graph = crop.filter_complex(spec)

    assert "color=" not in graph
    assert "pad=1080:1920:0:0:0x000000[c0]" in graph
    assert "[c0][r1]overlay=0:810[c1]" in graph


def test_regions_land_at_their_destination_offsets():
    spec = template(
        region("a", (0, 0, 960, 540), (0, 0, 540, 1920)),
        region("b", (960, 0, 960, 540), (540, 0, 540, 1920)),
    )

    graph = crop.filter_complex(spec)

    assert "overlay=540:0" in graph


def test_fill_covers_then_centre_crops():
    spec = template(region("a", (0, 0, 960, 800), (0, 0, 1080, 1110)))

    graph = crop.filter_complex(spec, default_fit=FILL)

    assert "force_original_aspect_ratio=increase" in graph
    assert graph.count("crop=") == 2       # the source rect, then the overflow


def test_contain_fits_then_pads():
    spec = template(region("a", (0, 0, 960, 800), (0, 0, 1080, 1110)))

    graph = crop.filter_complex(spec, default_fit=CONTAIN, pad_color="#101216")

    assert "force_original_aspect_ratio=decrease" in graph
    assert "pad=1080:1110:(ow-iw)/2:(oh-ih)/2:0x101216" in graph


def test_stretch_is_the_bare_scale_ss84_asks_for():
    spec = template(region("a", (0, 0, 960, 800), (0, 0, 1080, 1110)))

    graph = crop.filter_complex(spec, default_fit=STRETCH)

    assert "scale=1080:1110" in graph
    assert "force_original_aspect_ratio" not in graph


def test_a_region_overrides_the_default_fit():
    spec = template(
        region("a", (0, 0, 960, 540), (0, 0, 1080, 960), fit=STRETCH),
        region("b", (0, 540, 960, 540), (0, 960, 1080, 960)),
    )

    graph = crop.filter_complex(spec, default_fit=CONTAIN)

    assert "force_original_aspect_ratio=increase" not in graph
    assert "force_original_aspect_ratio=decrease" in graph


def test_captions_come_last_after_every_scale():
    """libass draws onto the frame it is given. Burning before the composition
    would scale the text along with the region."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    graph = crop.filter_complex(spec, captions="captions.ass")

    assert graph.endswith("ass=captions.ass[vout]")
    assert graph.index("scale=") < graph.index("ass=")


def test_without_captions_the_graph_still_ends_at_the_named_label():
    """So the caller maps the same label either way."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    graph = crop.filter_complex(spec, label="vout")

    assert graph.endswith("[vout]")
    assert "ass=" not in graph


def test_the_caption_filename_is_bare():
    """MEASURED in commit 23: an absolute Windows path splits at the drive
    colon inside a filter description, and no escaping survives an apostrophe
    in a parent directory."""
    spec = template(region("a", (0, 0, 1920, 1080), (0, 0, 1080, 1920)))

    graph = crop.filter_complex(spec, captions="captions.ass")

    assert "ass=captions.ass" in graph
    assert ":/" not in graph.split("ass=")[-1]


def test_pad_colour_converts_rgb_to_the_unambiguous_form():
    assert crop.pad_colour("#FF8000") == "0xFF8000"
    assert crop.pad_colour("black") == "black"


def test_a_bad_pad_colour_is_refused():
    with pytest.raises(crop.CropError, match="RRGGBB"):
        crop.pad_colour("#12")


def test_describe_reports_the_warnings_it_found():
    spec = template(region("a", (0, 0, 640, 360), (0, 0, 1080, 960)))
    check = crop.validate(spec, 640, 360)

    lines = crop.describe(spec, check)

    assert any("WARNING" in line for line in lines)
    assert any("1080x1920" in line for line in lines)


# --------------------------------------------------------------------------
# against the fixture
# --------------------------------------------------------------------------


def test_the_default_template_resolves_onto_the_fixture(manifest):
    """The fixture is 640x360 — the same shape as the declared 1920x1080, so
    it scales rather than being refused."""
    width, height = (int(v) for v in manifest["resolution"].split("x"))
    spec = crop.get_template(config.load())

    moved = crop.resolve(spec, width, height)
    check = crop.validate(moved, width, height)

    assert moved.source_resolution == (width, height)
    assert moved.regions[0].src.inside(width, height)
    # 640x360 into 1080x1920 is a 3x enlargement, well past the threshold.
    assert check.upscaled
