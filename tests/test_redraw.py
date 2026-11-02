import math

import cv2
import numpy as np

from whiteout.redraw import (TextItem, combine, coverage, dedupe, fit_circle, frame_region, infer_rotation,
                             merge_arcs, merge_collinear, render_lines, resolve_overlaps, sharpen, side_by_side,
                             snap_hv, split_words, text_anchor, thin, trace, vectorize, write_svg)


def test_infer_rotation_tall_box_is_vertical():
    assert infer_rotation("2.506", (10, 10, 40, 200), 0) == 90
    assert infer_rotation("2.506", (10, 10, 200, 40), 90) == 0
    assert infer_rotation("A", (10, 10, 20, 40), 0) == 0
    assert infer_rotation("NOTE", (0, 0, 100, 20), 270) == 0


def test_dedupe_keeps_central_read():
    a = TextItem("1.250 DIA", (100, 100, 220, 125))
    b = TextItem("1.250 DIA", (102, 101, 221, 126))
    c = TextItem("4X R.06", (400, 100, 500, 125))
    out = dedupe([a, b, c], [5, 300, 300])
    assert len(out) == 2 and b in out and c in out


def test_thin_gives_one_pixel_line():
    m = np.zeros((40, 120), np.uint8)
    m[15:22, 10:110] = 1
    sk = thin(m)
    cols = sk[:, 20:100].sum(0)
    assert (cols == 1).all()


def test_trace_straight_line_and_junction():
    m = np.zeros((60, 120), np.uint8)
    cv2.line(m, (10, 30), (110, 30), 1, 1)
    cv2.line(m, (60, 5), (60, 55), 1, 1)
    paths = trace(m)
    assert len(paths) == 4  # a cross: four arms meeting at the junction
    for p in paths:
        assert np.hypot(*(p[0] - p[-1])) > 20


def test_fit_circle():
    t = np.linspace(0, math.pi, 50)
    pts = np.column_stack([30 + 12 * np.cos(t), 40 + 12 * np.sin(t)])
    cx, cy, r, rms = fit_circle(pts)
    assert abs(cx - 30) < 1e-6 and abs(cy - 40) < 1e-6 and abs(r - 12) < 1e-6 and rms < 1e-6


def test_snap_hv_only_when_small_move():
    assert snap_hv(0, 10, 100, 11) == (0, 10.5, 100, 10.5)
    assert snap_hv(5, 0, 6, 80) == (5.5, 0, 5.5, 80)
    assert snap_hv(0, 0, 100, 30) == (0, 0, 100, 30)


def test_merge_collinear_joins_pieces_keeps_dashes():
    segs = [[0, 10, 50, 10, 3], [51, 10, 120, 10, 3], [200, 10, 220, 10, 3]]
    out = merge_collinear(segs)
    assert len(out) == 2
    xs = sorted((min(s[0], s[2]), max(s[0], s[2])) for s in out)
    assert xs[0] == (0, 120) and xs[1] == (200, 220)


def test_merge_collinear_removes_doubled_trace():
    out = merge_collinear([[0, 0, 100, 0, 3], [20, 0.5, 80, 0.5, 3]])
    assert len(out) == 1


def test_merge_arcs_to_circle():
    arcs = [{"t": "arc", "cx": 50, "cy": 50, "r": 20, "a0": a, "sweep": 88, "w": 2} for a in (0, 90, 180, 270)]
    out = merge_arcs(arcs)
    assert len(out) == 1 and out[0]["t"] == "circle"
    out = merge_arcs(arcs[:2])
    assert len(out) == 1 and out[0]["t"] == "arc" and out[0]["sweep"] >= 170


def test_vectorize_rectangle_and_circle():
    m = np.zeros((300, 400), np.uint8)
    cv2.rectangle(m, (40, 40), (360, 260), 1, 3)
    cv2.circle(m, (200, 150), 60, 1, 3)
    prims, stats = vectorize(m)
    lines = [q for q in prims if q["t"] == "line"]
    rounds = [q for q in prims if q["t"] in ("circle", "arc")]
    assert len(lines) == 4
    assert rounds and all(abs(q["r"] - 60) < 2 for q in rounds)
    drawn = render_lines(prims, m.shape)
    assert coverage(m, drawn) > 0.97 and coverage(drawn, m) > 0.97


def test_coverage():
    a = np.zeros((20, 20), np.uint8)
    a[10, 2:18] = 1
    b = np.zeros_like(a)
    b[12, 2:18] = 1
    assert coverage(a, b, tol=3) == 1.0
    assert coverage(a, b, tol=1) == 0.0
    assert coverage(np.zeros_like(a), b) == 1.0


def test_split_words_by_row_width():
    assert split_words("ONE TWO THREE FOUR".split(), [100, 100]) == ["ONE TWO", "THREE FOUR"]
    assert split_words(["A"], [10, 10]) == ["A", ""]


def test_combine_job_text_and_gap_fill():
    read = [TextItem("1.2S0", (100, 100, 160, 120), snapped=True), TextItem("NOTE 1", (300, 100, 380, 120), snapped=True)]
    jobs = [TextItem("1.250", (100, 100, 160, 120), source="job", snapped=True),  # same line, better reading
            TextItem("4X R.06", (600, 400, 700, 420), source="job", snapped=True),  # a gap the read left
            TextItem("NOTE 1 BREAK EDGES", (300, 100, 380, 140), source="job", snapped=True)]  # covers a read line
    out = combine(read, jobs, gh=20)
    assert [t.text for t in out] == ["1.250", "NOTE 1", "4X R.06"]
    assert out[0].source == "job"


def test_resolve_overlaps_keeps_snapped_read():
    a = TextItem("1.950", (100, 100, 180, 120), snapped=True, source="job")
    b = TextItem("1.947 DIA", (100, 100, 230, 120), snapped=True)
    c = TextItem("ELSEWHERE", (500, 500, 640, 520), snapped=True)
    out = resolve_overlaps([a, b, c])
    assert b in out and c in out and a not in out


def test_sharpen_rounded_corner():
    dp = np.array([[0, 2], [2, 0], [100, 0], [102, 2], [102, 80]], np.float32)
    out = sharpen(dp, 4.0, closed=False)
    assert len(out) == 4 and np.allclose(out[2], [102, 0])


def test_text_anchor_rotations():
    it = TextItem("ABC", (10, 20, 110, 40), rotation=0, base=40)
    assert text_anchor(it) == (10, 40, 100)
    it = TextItem("ABC", (10, 20, 30, 120), rotation=90, base=30)
    assert text_anchor(it) == (30, 120, 100)
    it = TextItem("ABC", (10, 20, 30, 120), rotation=270, base=10)
    assert text_anchor(it) == (10, 20, 100)


def test_frame_region_finds_broken_frame(tmp_path):
    g = np.full((1000, 1400), 255, np.uint8)
    # frame drawn as a dashed, broken line, as on old scans
    for x in range(100, 1300, 12):
        g[150:153, x:x + 9] = 0
        g[850:853, x:x + 9] = 0
    for y in range(150, 853, 12):
        g[y:y + 9, 100:103] = 0
        g[y:y + 9, 1297:1300] = 0
    g[960:990, 0:1400] = 0  # a card edge outside the sheet
    r = frame_region(g, (110, 160, 1290, 840))
    assert r[0] < 100 < r[0] + 60 and r[2] > 1300 and r[3] < 960


def test_write_svg_places_rotated_text(tmp_path):
    p = tmp_path / "a.svg"
    texts = [TextItem("2.506", (100, 100, 120, 180), rotation=90, cap=14, base=120)]
    prims = [{"t": "line", "p": [0, 0, 50, 0], "w": 3.0}, {"t": "circle", "cx": 50, "cy": 50, "r": 10, "w": 3.0}]
    write_svg(p, 400, 300, (0, 0, 400, 300), prims, texts)
    s = p.read_text()
    assert 'rotate(-90 120.0 180.0)' in s and "textLength=\"80.0\"" in s and "<circle" in s


def test_side_by_side_width_limit():
    a = np.zeros((500, 2500), np.uint8)
    out = side_by_side(a, a, max_w=3000)
    assert out.shape[1] <= 3000 and abs(out.shape[0] - 500 * 3000 / 5024) <= 1
