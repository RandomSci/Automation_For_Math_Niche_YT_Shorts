from manim import *
import math
import random as _fm_random
import numpy as _fmnp

class _Tracker:
    """Intercepts scene.add() to collect every mobject for clean FadeOut."""
    def __init__(self, scene):
        self._s = scene
        self._all = VGroup()
    def add(self, *mobs):
        for m in mobs:
            if m is not None:
                try: self._all.add(m)
                except Exception: pass
        self._s.add(*[m for m in mobs if m is not None])
    def play(self, *a, **kw): return self._s.play(*a, **kw)
    def wait(self, t=1.0, **kw): return self._s.wait(t, **kw)
    def remove(self, *a, **kw): return self._s.remove(*a, **kw)
    def __getattr__(self, n): return getattr(self._s, n)
    def collected(self): return self._all


def _fm_style(scene, n_styles):
    return abs(hash(id(scene))) % n_styles


def _fm_smooth_tangents(points):
    pts = [_fmnp.array(p, dtype=float) for p in points]
    n = len(pts)
    tangents = []
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = (pts[i + 1] - pts[i - 1]) * 0.5
        tangents.append(t)
    for i in range(n):
        for dim in range(2):
            td = tangents[i][dim]
            if abs(td) <= 1e-9:
                continue
            if i < n - 1:
                seg_d = pts[i + 1][dim] - pts[i][dim]
                if seg_d * td > 0 and abs(td) / 3.0 > abs(seg_d):
                    tangents[i] = tangents[i] * (3.0 * abs(seg_d) / abs(td))
            td = tangents[i][dim]
            if abs(td) <= 1e-9:
                continue
            if i > 0:
                seg_d = pts[i][dim] - pts[i - 1][dim]
                if seg_d * td > 0 and abs(td) / 3.0 > abs(seg_d):
                    tangents[i] = tangents[i] * (3.0 * abs(seg_d) / abs(td))
    return pts, tangents


def _fm_set_line_smooth(vmobject, points):
    pts, tangents = _fm_smooth_tangents(points)
    n = len(pts)
    if n < 2:
        return vmobject
    vmobject.start_new_path(pts[0])
    for i in range(n - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        h1 = p0 + tangents[i] / 3.0
        h2 = p1 - tangents[i + 1] / 3.0
        vmobject.add_cubic_bezier_curve_to(h1, h2, p1)
    return vmobject


BRAND_WHITE = "#F5F7FA"
BRAND_GREEN = "#38D996"
BRAND_RED   = "#FF4D4D"
BRAND_GOLD  = "#FFD166"
BRAND_GRAY  = "#8A94A6"
BRAND_PANEL = "#0D1B2A"
BRAND_BG    = "#060F1A"
BRAND_NAVY  = "#0B1628"

_ACCENT_POOL = [
    "#38D996",
    "#FFD166",
    "#7B8CFF",
    "#FF8C69",
    "#4DD9FF",
    "#C084FC",
    "#34D399",
]


def _fm_accent(scene, base_color=None):
    if base_color and base_color not in (BRAND_GRAY, BRAND_PANEL, BRAND_BG, BRAND_NAVY):
        return base_color
    return _ACCENT_POOL[abs(hash(id(scene))) % len(_ACCENT_POOL)]


def fm_glow_around(mobject, color=None, n_layers=3):
    if color is None:
        color = BRAND_GOLD
    layers = VGroup()
    for i in range(n_layers, 0, -1):
        c = mobject.copy()
        c.scale(1 + i * 0.022)
        c.set_opacity(max(0.07 - i * 0.018, 0.01))
        try:
            c.set_stroke(color, width=1.2 * i, opacity=max(0.10 - i * 0.025, 0.02))
        except Exception:
            pass
        layers.add(c)
    return VGroup(layers, mobject)


def fm_concept_pills(labels, colors=None, panel_color=BRAND_PANEL, text_color=None,
                      font_size=44, direction=None, spacing=0.4, accent_color=None):
    if colors is None:
        if accent_color is not None:
            colors = [accent_color] * 4
        else:
            colors = [BRAND_GOLD, BRAND_GREEN, BRAND_RED, BRAND_WHITE]
    if text_color is None:
        text_color = BRAND_WHITE
    if direction is None:
        direction = RIGHT if len(labels) <= 3 else DOWN

    safe_w = config.frame_width * 0.88
    safe_h = config.frame_height * 0.78

    pill_groups = []
    for i, label in enumerate(labels):
        c = colors[i % len(colors)]
        txt = Text(label, font_size=font_size, color=text_color, weight=BOLD)
        pad_x, pad_y = 0.32, 0.22
        box_w = txt.width + 2 * pad_x
        box_h = txt.height + 2 * pad_y
        pill = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.16)
        pill.set_fill(panel_color, opacity=1.0)
        pill.set_stroke(c, width=2.0, opacity=0.9)
        txt.move_to(pill.get_center())
        pill_groups.append(VGroup(pill, txt))

    import numpy as _np
    if len(pill_groups) >= 4 and _np.array_equal(direction, RIGHT):
        mid = (len(pill_groups) + 1) // 2
        row1 = VGroup(*pill_groups[:mid])
        row2 = VGroup(*pill_groups[mid:])
        row1.arrange(RIGHT, buff=spacing)
        row2.arrange(RIGHT, buff=spacing)
        pills = VGroup(row1, row2).arrange(DOWN, buff=spacing * 1.1)
    else:
        pills = VGroup(*pill_groups)
        pills.arrange(direction, buff=spacing)

    if pills.width > safe_w:
        pills.scale(safe_w / pills.width)
    if pills.height > safe_h:
        pills.scale(safe_h / pills.height)

    return pills


def fm_card(label_text, value_text, accent_color=BRAND_GOLD,
             panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
             label_size=32, value_size=68, buff=0.38):
    val = Text(value_text, font_size=value_size, color=text_color, weight=BOLD)
    lbl = Text(label_text, font_size=label_size, color=accent_color)
    content = VGroup(lbl, val).arrange(DOWN, buff=0.18)
    pad_x = buff + 0.1
    pad_y = buff
    box_w = content.width + 2 * pad_x
    box_h = content.height + 2 * pad_y
    box = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.18)
    box.set_fill(panel_color, opacity=1.0)
    box.set_stroke(accent_color, width=2.0, opacity=0.85)
    content.move_to(box.get_center())
    return VGroup(box, content)


def fm_two_cards(left_label, left_val, left_color,
                  right_label, right_val, right_color,
                  panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                  label_size=30, value_size=68, spacing=0.7, buff=None,
                  title=None, subtitle=None, header=None):
    left  = fm_card(left_label,  left_val,  left_color,  panel_color, text_color, label_size, value_size)
    right = fm_card(right_label, right_val, right_color, panel_color, text_color, label_size, value_size)
    group = VGroup(left, right).arrange(RIGHT, buff=spacing)
    safe_w = config.frame_width * 0.88
    if group.width > safe_w:
        group.scale(safe_w / group.width)
    return group


def fm_card_row(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                 label_size=26, value_size=44, spacing=0.45, buff=None):
    cards = VGroup()
    for entry in items:
        if isinstance(entry, dict):
            label = entry.get("label", "")
            value = entry.get("value", "")
            color = entry.get("color", BRAND_GOLD)
        else:
            label, value, color = entry
        if not isinstance(value, str):
            value = f"${abs(value):,.0f}" if isinstance(value, (int, float)) else str(value)
        c = fm_card(label, value, color, panel_color, text_color, label_size, value_size, buff=0.24)
        cards.add(c)
    cards.arrange(RIGHT, buff=spacing)
    safe_w = config.frame_width * 0.92
    if cards.width > safe_w:
        cards.scale(safe_w / cards.width)
    return cards


def fm_stacked_cards(items, panel_color=BRAND_PANEL, text_color=BRAND_WHITE,
                      label_size=30, value_size=68, spacing=0.24):
    cards = VGroup()
    for entry in items:
        if isinstance(entry, dict):
            label = entry.get("label", "")
            value = entry.get("value", "")
            color = entry.get("color", BRAND_GOLD)
        else:
            label, value, color = entry
        if not isinstance(value, str):
            value = f"${abs(value):,.0f}" if isinstance(value, (int, float)) else str(value)
        c = fm_card(label, value, color, panel_color, text_color, label_size, value_size, buff=0.32)
        cards.add(c)
    cards.arrange(DOWN, buff=spacing)
    safe_h = config.frame_height * 0.80
    if cards.height > safe_h:
        cards.scale(safe_h / cards.height)
    return cards


def fm_clamp_to_frame(*mobjects, margin_x=0.06, margin_y=0.06):
    valid = [m for m in mobjects if m is not None]
    if not valid:
        return None
    combined = VGroup(*valid)
    safe_w = config.frame_width * (1 - 2 * margin_x)
    safe_h = config.frame_height * (1 - 2 * margin_y)
    width_scale = safe_w / combined.width if combined.width > safe_w else 1.0
    height_scale = safe_h / combined.height if combined.height > safe_h else 1.0
    scale_factor = min(width_scale, height_scale)
    if scale_factor < 1.0:
        combined.scale(scale_factor)
    max_x = config.frame_width / 2 - margin_x * config.frame_width
    max_y = config.frame_height / 2 - margin_y * config.frame_height
    shift_x = 0.0
    shift_y = 0.0
    left = combined.get_left()[0]
    right = combined.get_right()[0]
    top = combined.get_top()[1]
    bottom = combined.get_bottom()[1]
    if left < -max_x:
        shift_x = -max_x - left
    elif right > max_x:
        shift_x = max_x - right
    if bottom < -max_y:
        shift_y = -max_y - bottom
    elif top > max_y:
        shift_y = max_y - top
    if shift_x != 0.0 or shift_y != 0.0:
        combined.shift([shift_x, shift_y, 0])
    return combined


def _fm_collect_play_targets(anim, out):
    sub_animations = getattr(anim, "animations", None)
    if sub_animations:
        for sub in sub_animations:
            _fm_collect_play_targets(sub, out)
        return
    mobj = getattr(anim, "mobject", None)
    if mobj is not None:
        out.append(mobj)


def fm_animate_counter(scene, start_val, end_val, label_text,
                        accent_color=BRAND_GOLD, prefix="", suffix="",
                        duration=3.0, position=None, value_size=130, label_size=38,
                        _style=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    style = _style if _style is not None else _fm_style(scene, 3)

    tracker = ValueTracker(float(start_val))
    end_f   = float(end_val)
    is_whole = float(end_val) == int(float(end_val))
    use_decimal = isinstance(end_val, float) and not is_whole

    def _num():
        v = tracker.get_value()
        if use_decimal:
            s = f"{prefix}{v:,.2f}{suffix}"
        else:
            s = f"{prefix}{int(round(v)):,}{suffix}"
        return Text(s, font_size=value_size, color=BRAND_WHITE, weight=BOLD).move_to(position)

    counter = always_redraw(_num)
    anim_t = max(min(duration * 0.78, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)

    if style == 0:
        lbl = Text(label_text, font_size=label_size, color=accent_color)
        lbl.next_to(position, DOWN, buff=0.85)
        scene.add(counter, lbl)

    elif style == 1:
        accent_bar = Line([position[0] - 0.06, position[1] - 0.55, 0],
                          [position[0] - 0.06, position[1] + 0.55, 0])
        accent_bar.set_stroke(accent_color, width=5, opacity=0.9)
        lbl = Text(label_text, font_size=label_size, color=accent_color)
        lbl.next_to(position, DOWN, buff=0.85)
        scene.add(accent_bar, counter, lbl)

    else:
        bg_rect = RoundedRectangle(width=6.5, height=2.2, corner_radius=0.22)
        bg_rect.set_fill(BRAND_PANEL, opacity=0.9)
        bg_rect.set_stroke(accent_color, width=2.0, opacity=0.5)
        bg_rect.move_to(position)
        lbl = Text(label_text, font_size=label_size, color=accent_color)
        lbl.next_to(position, DOWN, buff=1.2)
        scene.add(bg_rect, counter, lbl)

    scene.play(tracker.animate.set_value(end_f), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), counter


def fm_animate_bar_chart(scene, values, names, colors=None,
                          duration=3.5, title_text="", _style=None, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    if colors is None:
        colors = [BRAND_GREEN, BRAND_GOLD, BRAND_RED, BRAND_WHITE]
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]
    style = _style if _style is not None else _fm_style(scene, 3)
    n     = len(values)

    if style == 1:
        max_v   = max(abs(v) for v in values) if values else 1
        bar_h   = min(0.7, 4.5 / max(n, 1))
        spacing = bar_h * 1.65
        total_h = (n - 1) * spacing
        x_scale = 8.5 / max(max_v * 1.28, 1.0)
        base_x  = -4.0

        baseline = Line([base_x, total_h / 2 + 0.3, 0], [base_x, -total_h / 2 - 0.3, 0])
        baseline.set_stroke(BRAND_GRAY, width=2.0, opacity=0.48)

        bars       = VGroup()
        val_labels = VGroup()
        cat_labels = VGroup()

        for i, (v, name, c) in enumerate(zip(values, names, bar_colors)):
            y     = total_h / 2 - i * spacing
            bw    = max(abs(v) * x_scale, 0.16)
            bar   = RoundedRectangle(width=bw, height=bar_h, corner_radius=0.05)
            bar.set_fill(c, opacity=0.92)
            bar.set_stroke(c, width=1.5, opacity=0.45)
            bar.move_to([base_x + bw / 2, y, 0])
            bars.add(bar)

            val_str = f"{int(v):,}" if isinstance(v, int) else f"{v:.2f}"
            val_lbl = Text(val_str, font_size=22, color=c, weight=BOLD)
            val_lbl.next_to(bar, RIGHT, buff=0.14)
            val_labels.add(val_lbl)

            cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
            cat_lbl.next_to(bar, LEFT, buff=0.18)
            cat_labels.add(cat_lbl)

        chart_group = VGroup(baseline, bars, val_labels, cat_labels)
        chart_group.move_to(pos + np.array([0.5, 0, 0]))

        if title_text:
            ttl = Text(title_text, font_size=28, color=BRAND_GRAY)
            ttl.next_to(chart_group, UP, buff=0.22)
            scene.add(ttl)

        scene.add(baseline, cat_labels)
        grow_t = max(min(duration * 0.65, duration - 0.45), 0.1)
        hold_t = max(duration - grow_t - 0.35, 0.05)
        scene.play(
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.18),
            run_time=grow_t, rate_func=smooth,
        )
        scene.play(
            LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
            run_time=0.35, rate_func=smooth,
        )
        scene.wait(hold_t)
        return _sc.collected(), chart_group

    elif style == 2:
        max_v   = max(abs(v) for v in values) if values else 1
        chart_h = 4.0
        bar_w   = min(1.4, 9.0 / max(n, 1))
        spacing = bar_w * 1.65
        total_w = (n - 1) * spacing
        y_scale = chart_h / max(max_v * 1.28, 1.0)
        base_y  = -chart_h / 2 - 0.15

        baseline = Line([-total_w / 2 - bar_w, base_y, 0], [total_w / 2 + bar_w, base_y, 0])
        baseline.set_stroke(BRAND_GRAY, width=1.5, opacity=0.40)

        dots      = VGroup()
        stems     = VGroup()
        val_labels = VGroup()
        cat_labels = VGroup()

        for i, (v, name, c) in enumerate(zip(values, names, bar_colors)):
            x      = -total_w / 2 + i * spacing
            bar_h2 = max(abs(v) * y_scale, 0.16)
            top_y  = base_y + bar_h2

            stem = Line([x, base_y, 0], [x, top_y, 0])
            stem.set_stroke(c, width=3.5, opacity=0.7)
            stems.add(stem)

            dot = Dot([x, top_y, 0], radius=0.18, color=c)
            dot.set_fill(c, opacity=1.0)
            dots.add(dot)

            val_str = f"{int(v):,}" if isinstance(v, int) else f"{v:.2f}"
            val_lbl = Text(val_str, font_size=24, color=c, weight=BOLD)
            val_lbl.next_to(dot, UP, buff=0.12)
            val_labels.add(val_lbl)

            cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
            cat_lbl.next_to([x, base_y, 0], DOWN, buff=0.12)
            cat_labels.add(cat_lbl)

        chart_group = VGroup(baseline, stems, dots, val_labels, cat_labels)
        chart_cx = chart_group.get_center()[0]
        chart_group.shift(RIGHT * (-chart_cx) + UP * 0.22)

        if title_text:
            ttl = Text(title_text, font_size=28, color=BRAND_GRAY)
            ttl.next_to(chart_group, UP, buff=0.22)
            scene.add(ttl)

        scene.add(baseline, cat_labels)
        grow_t = max(min(duration * 0.62, duration - 0.45), 0.1)
        hold_t = max(duration - grow_t - 0.38, 0.05)
        scene.play(
            LaggedStart(*[Create(s) for s in stems], lag_ratio=0.15),
            run_time=grow_t * 0.6, rate_func=smooth,
        )
        scene.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.15),
            run_time=grow_t * 0.4, rate_func=smooth,
        )
        scene.play(
            LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
            run_time=0.38, rate_func=smooth,
        )
        scene.wait(hold_t)
        return _sc.collected(), chart_group

    else:
        max_v   = max(abs(v) for v in values) if values else 1
        chart_h = 4.2
        bar_w   = min(1.6, 9.5 / max(n, 1))
        spacing = bar_w * 1.62
        total_w = (n - 1) * spacing
        y_scale = chart_h / max(max_v * 1.28, 1.0)
        base_y  = -chart_h / 2 - 0.15

        edge_margin = bar_w / 2 + 0.3
        baseline = Line([-total_w / 2 - edge_margin, base_y, 0], [total_w / 2 + edge_margin, base_y, 0])
        baseline.set_stroke(color=BRAND_GRAY, width=2.0, opacity=0.48)

        bars       = VGroup()
        val_labels = VGroup()
        cat_labels = VGroup()

        for i, (v, name, c) in enumerate(zip(values, names, bar_colors)):
            x     = -total_w / 2 + i * spacing
            bar_h = max(abs(v) * y_scale, 0.16)
            bar   = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
            bar.set_fill(c, opacity=0.92)
            bar.set_stroke(c, width=1.5, opacity=0.55)
            bar.move_to([x, base_y + bar_h / 2, 0])
            bars.add(bar)

            val_str = f"{int(v):,}" if isinstance(v, int) else f"{v:.2f}"
            val_lbl = Text(val_str, font_size=26, color=c, weight=BOLD)
            val_lbl.next_to(bar, UP, buff=0.1)
            val_labels.add(val_lbl)

            cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
            cat_lbl.next_to(bar, DOWN, buff=0.15)
            cat_labels.add(cat_lbl)

        chart_group = VGroup(baseline, bars, val_labels, cat_labels)
        bars_cx = bars.get_center()[0]
        chart_group.shift(RIGHT * (-bars_cx) + UP * 0.22)

        if title_text:
            ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
            ttl.next_to(chart_group, UP, buff=0.22)
            scene.add(ttl)

        scene.add(baseline, cat_labels)
        grow_t = max(min(duration * 0.62, duration - 0.45), 0.1)
        hold_t = max(duration - grow_t - 0.38, 0.05)
        scene.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.18),
            run_time=grow_t, rate_func=smooth,
        )
        scene.play(
            LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
            run_time=0.38, rate_func=smooth,
        )
        scene.wait(hold_t)
        return _sc.collected(), chart_group


def fm_animate_gauge(scene, value, max_val, label_text,
                      accent_color=BRAND_GREEN, duration=3.0,
                      position=None, radius=2.0):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN

    fill_ratio  = max(0.0, min(1.0, float(value) / float(max_val or 1)))
    start_angle = PI + PI * 0.12
    sweep_total = PI - PI * 0.24

    track = Arc(radius=radius, start_angle=start_angle, angle=sweep_total, arc_center=position)
    track.set_stroke(color=BRAND_GRAY, width=16, opacity=0.32)

    tracker = ValueTracker(0.0)

    def _arc():
        frac = tracker.get_value()
        if frac < 1e-6:
            return VMobject()
        a = Arc(radius=radius, start_angle=start_angle, angle=sweep_total * frac, arc_center=position)
        a.set_stroke(color=accent_color, width=16, opacity=1.0)
        return a

    fill_arc = always_redraw(_arc)
    val_str  = f"{int(value)}" if isinstance(value, int) or float(value) == int(value) else f"{value:.1f}"
    val_lbl  = Text(val_str, font_size=100, color=BRAND_WHITE, weight=BOLD)
    val_lbl.move_to(position + UP * 0.72)
    cat_lbl = Text(label_text, font_size=34, color=accent_color)
    cat_lbl.next_to(track, DOWN, buff=0.32)

    scene.add(track, fill_arc, val_lbl, cat_lbl)
    anim_t = max(min(duration * 0.72, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(fill_ratio), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), val_lbl


def fm_animate_donut(scene, percentage, label_text,
                      accent_color=BRAND_GREEN, duration=3.0,
                      position=None, radius=1.85, thickness=0.52):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN

    pct        = max(0.0, min(100.0, float(percentage)))
    fill_angle = (pct / 100.0) * TAU
    inner_r    = max(radius - thickness, 0.05)

    track = Annulus(inner_radius=inner_r, outer_radius=radius,
                     color=BRAND_GRAY, fill_opacity=0.40, stroke_width=0)
    track.move_to(position)

    tracker = ValueTracker(0.0)

    def _fill():
        angle = tracker.get_value()
        if angle < 1e-6:
            return VMobject()
        arc = Arc(
            radius=inner_r + thickness / 2,
            start_angle=PI / 2,
            angle=-angle,
            arc_center=position,
            stroke_width=int(thickness * 105),
        )
        arc.set_stroke(color=accent_color, opacity=1.0)
        return arc

    fill     = always_redraw(_fill)
    pct_lbl  = Text(f"{pct:.0f}%", font_size=90, color=BRAND_WHITE, weight=BOLD)
    pct_lbl.move_to(position)
    cat_lbl  = Text(label_text, font_size=34, color=accent_color)
    cat_lbl.next_to(track, DOWN, buff=0.4)

    scene.add(track, fill, pct_lbl, cat_lbl)
    anim_t = max(min(duration * 0.72, duration - 0.25), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(fill_angle), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), pct_lbl


def fm_animate_line_chart(scene, y_values, end_value_label=None,
                           accent_color=BRAND_GREEN, x_labels=None,
                           duration=3.5, title_text="", _style=None,
                           position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    if y_values and isinstance(y_values[0], (list, tuple)):
        series = [{"y_values": s, "color": accent_color, "label": ""} for s in y_values]
        return fm_animate_line_chart_multi(scene._s, series=series, duration=duration, title_text=title_text)

    n = len(y_values)
    if n < 2:
        return _sc.collected(), None

    end_value_label = end_value_label if end_value_label is not None else ""
    style = _style if _style is not None else _fm_style(scene, 3)

    min_y  = min(y_values)
    max_y  = max(y_values)
    y_span = max(max_y - min_y, 1.0)
    y_pad  = y_span * 0.25
    y_lo   = min_y - y_pad if min_y - y_pad >= 0 else min_y - y_pad * 0.5
    y_hi   = max_y + y_pad
    y_step = max((y_hi - y_lo) / 4, 0.01)
    x_step = max((n - 1) // 5, 1)

    axes = Axes(
        x_range=[0, n - 1, x_step],
        y_range=[y_lo, y_hi, y_step],
        x_length=10.5,
        y_length=5.2,
        axis_config={
            "color": BRAND_GRAY,
            "stroke_opacity": 0.45,
            "include_tip": False,
            "include_numbers": False,
        },
    )
    axes.move_to(pos + DOWN * 0.15)

    pts = [axes.c2p(i, y_values[i]) for i in range(n)]

    grow_t  = max(min(duration * 0.70, duration - 0.55), 0.1)
    label_t = 0.4
    hold_t  = max(duration - grow_t - label_t, 0.05)

    _dot_pt = axes.c2p(n - 1, y_values[-1])
    _dot_x  = _dot_pt[0]
    _dot_y  = _dot_pt[1]
    _frame_right_edge = config.frame_width / 2 - 0.25
    _frame_left_edge  = -config.frame_width / 2 + 0.25
    if _dot_y < -config.frame_height * 0.20:
        _lbl_dir = UP
    else:
        _lbl_dir = UR if _dot_x < config.frame_width * 0.38 else UL

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    if style == 1:
        line = VMobject()
        _fm_set_line_smooth(line, pts)
        line.set_stroke(color=accent_color, width=5.0, opacity=0.95)

        grid_lines = VGroup()
        for k in range(5):
            gy = y_lo + k * (y_hi - y_lo) / 4
            gp = axes.c2p(0, gy)
            gp2 = axes.c2p(n - 1, gy)
            gl = Line([gp[0], gp[1], 0], [gp2[0], gp2[1], 0])
            gl.set_stroke(BRAND_GRAY, width=0.8, opacity=0.18)
            grid_lines.add(gl)

        scene.add(axes, grid_lines)
        scene.play(Create(line), run_time=grow_t, rate_func=smooth)

        last_anchor = line.get_last_point()
        end_dot = Dot(last_anchor, color=accent_color, radius=0.15)
        end_dot.set_fill(accent_color, opacity=1.0)
        if end_value_label:
            end_lbl = Text(end_value_label, font_size=38, color=accent_color, weight=BOLD)
            end_lbl.next_to(end_dot, _lbl_dir, buff=0.18)
            if end_lbl.get_right()[0] > _frame_right_edge:
                end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
            if end_lbl.get_left()[0] < _frame_left_edge:
                end_lbl.shift(RIGHT * (_frame_left_edge - end_lbl.get_left()[0]))
            scene.play(FadeIn(end_dot), Write(end_lbl), run_time=label_t)
        else:
            scene.play(FadeIn(end_dot), run_time=label_t)
        scene.wait(hold_t)
        return _sc.collected(), axes

    elif style == 2:
        step_pts = []
        for i in range(n):
            if i > 0:
                step_pts.append([pts[i][0], pts[i - 1][1], 0])
            step_pts.append(pts[i])

        line = VMobject()
        line.set_points_as_corners(step_pts)
        line.set_stroke(color=accent_color, width=4.0, opacity=0.90)

        baseline_y = y_lo
        fill_step_pts = step_pts + [[pts[-1][0], axes.c2p(0, baseline_y)[1], 0],
                                     [pts[0][0],  axes.c2p(0, baseline_y)[1], 0]]
        fill_region = Polygon(*fill_step_pts, fill_opacity=0.12, stroke_width=0)
        fill_region.set_fill(accent_color)

        scene.add(axes, fill_region)
        scene.play(Create(line), run_time=grow_t, rate_func=smooth)

        last_anchor = line.get_last_point()
        end_dot = Dot(last_anchor, color=accent_color, radius=0.13)
        if end_value_label:
            end_lbl = Text(end_value_label, font_size=38, color=accent_color, weight=BOLD)
            end_lbl.next_to(end_dot, _lbl_dir, buff=0.18)
            if end_lbl.get_right()[0] > _frame_right_edge:
                end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
            if end_lbl.get_left()[0] < _frame_left_edge:
                end_lbl.shift(RIGHT * (_frame_left_edge - end_lbl.get_left()[0]))
            scene.play(FadeIn(end_dot), Write(end_lbl), run_time=label_t)
        else:
            scene.play(FadeIn(end_dot), run_time=label_t)
        scene.wait(hold_t)
        return _sc.collected(), axes

    else:
        line = VMobject()
        _fm_set_line_smooth(line, pts)
        line.set_stroke(color=accent_color, width=4.5, opacity=0.95)

        baseline_y = y_lo
        fill_pts   = pts + [axes.c2p(n - 1, baseline_y), axes.c2p(0, baseline_y)]
        fill_region = Polygon(*fill_pts, fill_opacity=0.20, stroke_width=0)
        fill_region.set_color_by_gradient(accent_color, BRAND_BG)

        scene.add(axes, fill_region)
        scene.play(Create(line), run_time=grow_t, rate_func=smooth)

        last_anchor = line.get_last_point()
        end_dot = Dot(last_anchor, color=accent_color, radius=0.13)

        if end_value_label:
            end_lbl = Text(end_value_label, font_size=38, color=accent_color, weight=BOLD)
            end_lbl.next_to(end_dot, _lbl_dir, buff=0.18)
            if end_lbl.get_right()[0] > _frame_right_edge:
                end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
            if end_lbl.get_left()[0] < _frame_left_edge:
                end_lbl.shift(RIGHT * (_frame_left_edge - end_lbl.get_left()[0]))
            scene.play(FadeIn(end_dot), Write(end_lbl), run_time=label_t)
        else:
            scene.play(FadeIn(end_dot), run_time=label_t)
        scene.wait(hold_t)
        return _sc.collected(), axes


def fm_animate_line_chart_multi(scene, series, duration=4.0, title_text=""):
    _sc = _Tracker(scene)
    scene = _sc
    if not series:
        return _sc.collected(), None

    if series and not isinstance(series[0], dict):
        colors_cycle = [BRAND_GREEN, BRAND_GOLD, BRAND_RED, BRAND_WHITE]
        series = [
            {"y_values": s, "color": colors_cycle[i % len(colors_cycle)], "label": f"Series {i+1}"}
            for i, s in enumerate(series)
        ]

    n = len(series[0]["y_values"])
    if n < 2:
        return _sc.collected(), None

    all_vals = [v for s in series for v in s["y_values"]]
    min_y   = min(all_vals)
    max_y   = max(all_vals)
    y_span  = max(max_y - min_y, 1.0)
    y_pad   = y_span * 0.22
    y_lo    = max(0.0, min_y - y_pad)
    y_hi    = max_y + y_pad
    y_step  = max((y_hi - y_lo) / 4, 0.01)
    x_step  = max((n - 1) // 5, 1)

    axes = Axes(
        x_range=[0, n - 1, x_step],
        y_range=[y_lo, y_hi, y_step],
        x_length=10.5,
        y_length=5.2,
        axis_config={
            "color": BRAND_GRAY,
            "stroke_opacity": 0.45,
            "include_tip": False,
            "include_numbers": False,
        },
    )
    axes.move_to(ORIGIN + DOWN * 0.15)

    lines    = []
    end_dots = []
    end_lbls = []
    _lbl_dirs = []
    for s in series:
        y_values = s["y_values"]
        color    = s.get("color", BRAND_GREEN)
        pts      = [axes.c2p(i, y_values[i]) for i in range(n)]
        line     = VMobject()
        _fm_set_line_smooth(line, pts)
        line.set_stroke(color=color, width=4.5, opacity=0.95)
        lines.append(line)

        end_dot = Dot(pts[-1], color=color, radius=0.11)
        end_lbl = Text(s.get("label", ""), font_size=26, color=color, weight=BOLD)
        _dot_pt2 = axes.c2p(n - 1, y_values[-1])
        _dot_x   = _dot_pt2[0]
        _dot_y2  = _dot_pt2[1]
        if _dot_y2 < -config.frame_height * 0.20:
            _lbl_dir2 = UP
        else:
            _lbl_dir2 = UR if _dot_x < config.frame_width * 0.38 else UL
        end_dots.append(end_dot)
        end_lbls.append(end_lbl)
        _lbl_dirs.append(_lbl_dir2)

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(axes, UP, buff=0.22)
        scene.add(ttl)

    scene.add(axes)
    grow_t  = max(min(duration * 0.65, duration - 0.6), 0.1)
    label_t = 0.45
    hold_t  = max(duration - grow_t - label_t, 0.05)
    scene.play(*[Create(l) for l in lines], run_time=grow_t, rate_func=smooth)

    _frame_right_edge = config.frame_width / 2 - 0.25
    _frame_left_edge2 = -config.frame_width / 2 + 0.25
    for i, (line_obj, end_dot, end_lbl, ldir) in enumerate(zip(lines, end_dots, end_lbls, _lbl_dirs)):
        actual_end = line_obj.get_last_point()
        end_dot.move_to(actual_end)
        end_lbl.next_to(end_dot, ldir, buff=0.12)
        if end_lbl.get_right()[0] > _frame_right_edge:
            end_lbl.shift(LEFT * (end_lbl.get_right()[0] - _frame_right_edge))
        if end_lbl.get_left()[0] < _frame_left_edge2:
            end_lbl.shift(RIGHT * (_frame_left_edge2 - end_lbl.get_left()[0]))

    order   = sorted(range(len(series)), key=lambda i: series[i]["y_values"][-1], reverse=True)
    min_gap = 0.4
    for k in range(1, len(order)):
        prev_i = order[k - 1]
        cur_i  = order[k]
        gap = end_lbls[prev_i].get_bottom()[1] - end_lbls[cur_i].get_top()[1]
        if gap < min_gap:
            end_lbls[cur_i].shift(DOWN * (min_gap - gap))

    scene.play(
        *[FadeIn(d) for d in end_dots],
        *[Write(l) for l in end_lbls],
        run_time=label_t,
    )
    scene.wait(hold_t)
    return _sc.collected(), axes


def fm_animate_waterfall(scene, steps, duration=4.5, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    if steps and isinstance(steps[0], (list, tuple)):
        steps = [{"label": s[0], "value": float(s[1])} for s in steps]
    steps = [dict(s) for s in steps]
    n = len(steps)
    if n < 2:
        return _sc.collected(), None

    bar_w   = min(1.5, 10.5 / n)
    spacing = bar_w * 1.55
    total_w = (n - 1) * spacing
    edge_margin = bar_w / 2 + 0.3

    running = 0.0
    bases   = []
    for s in steps[:-1]:
        bases.append(running)
        running += s["value"]
    bases.append(0.0)
    steps[-1]["value"] = running

    all_tops  = [b + (v["value"] if v["value"] > 0 else 0) for b, v in zip(bases, steps)]
    all_bots  = [b + (v["value"] if v["value"] < 0 else 0) for b, v in zip(bases, steps)]
    min_base  = min(all_bots)
    max_top   = max(all_tops)
    chart_h   = 4.5
    y_scale   = chart_h / max(max_top - min_base, 1.0)
    base_y    = -chart_h / 2 - min_base * y_scale + 0.4
    axis_y    = base_y - 0.45

    baseline  = Line(
        [-total_w / 2 - edge_margin, axis_y, 0],
        [ total_w / 2 + edge_margin, axis_y, 0],
    ).set_stroke(color=BRAND_GRAY, opacity=0.38, width=1.5)

    bars   = VGroup()
    labels = VGroup()

    for i, (step, base) in enumerate(zip(steps, bases)):
        v     = step["value"]
        x_pos = -total_w / 2 + i * spacing
        bar_h = max(abs(v) * y_scale, 0.16)

        if i == n - 1:
            c  = step.get("color", BRAND_GOLD if v >= 0 else BRAND_RED)
            y0 = axis_y if v >= 0 else axis_y - bar_h
        elif v >= 0:
            c  = step.get("color", BRAND_GREEN)
            y0 = axis_y + base * y_scale
        else:
            c  = step.get("color", BRAND_RED)
            y0 = axis_y + (base + v) * y_scale

        bar = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.05)
        bar.set_fill(c, opacity=0.9)
        bar.set_stroke(c, width=1.5, opacity=0.55)
        bar.move_to([x_pos, y0 + bar_h / 2, 0])
        bars.add(bar)

        prefix   = "-" if v < 0 else ""
        val_str  = f"{prefix}{int(abs(v)):,}" if abs(v) >= 1 else f"{prefix}{abs(v):.2f}"
        val_lbl  = Text(val_str, font_size=22, color=c, weight=BOLD)
        val_lbl.next_to(bar, DOWN if (v < 0) else UP, buff=0.08)
        cat_lbl  = Text(step.get("label", ""), font_size=18, color=BRAND_GRAY)
        if v < 0:
            cat_lbl.next_to(val_lbl, DOWN, buff=0.10)
        else:
            cat_lbl.next_to(bar, DOWN, buff=0.08)
        labels.add(VGroup(val_lbl, cat_lbl))

    all_elements = VGroup(baseline, bars, labels)
    safe_bottom = -(config.frame_height * 0.44)
    actual_bottom = all_elements.get_bottom()[1]
    if actual_bottom < safe_bottom:
        shift_up = safe_bottom - actual_bottom
        all_elements.shift(UP * shift_up)

    scene.add(baseline)
    anim_t  = max(min(duration * 0.70, duration - 0.6), 0.1)
    hold_t  = max(duration - anim_t - 0.15, 0.05)
    per_bar = anim_t / n
    for bar, lbl in zip(bars, labels):
        scene.play(GrowFromEdge(bar, DOWN), FadeIn(lbl), run_time=per_bar, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), bars


def fm_animate_text_reveal(scene, lines, colors=None, duration=3.0, sizes=None, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    if colors is None:
        colors = [BRAND_GOLD] + [BRAND_WHITE] * (len(lines) - 1)
    if sizes is None:
        sizes  = [72] + [44] * (len(lines) - 1)

    texts = VGroup(*[
        Text(lines[i], font_size=sizes[i % len(sizes)],
              color=colors[i % len(colors)], weight=BOLD)
        for i in range(len(lines))
    ])
    texts.arrange(DOWN, buff=0.36)
    texts.move_to(pos)

    per_t  = max(min(duration / max(len(lines), 1) * 0.55, 0.85), 0.1)
    hold_t = max(duration - per_t * len(lines), 0.1)
    for t in texts:
        scene.play(FadeIn(t, shift=UP * 0.18), run_time=per_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), texts


def fm_animate_icon_grid(scene, total, filled, label_text,
                          accent_color=BRAND_GREEN, duration=3.0,
                          cols=10, position=None, icon_radius=0.18):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    if label_text is None:
        label_text = ""

    filled  = max(0, min(filled, total))
    rows    = math.ceil(total / max(cols, 1))
    spacing = icon_radius * 2.9
    grid_w  = cols * spacing
    grid_h  = rows * spacing

    safe_w = config.frame_width * 0.82
    safe_h = config.frame_height * 0.62
    scale  = min(safe_w / max(grid_w, 0.1), safe_h / max(grid_h, 0.1), 1.0)
    spacing *= scale
    grid_w  *= scale
    grid_h  *= scale
    r_scaled = icon_radius * scale

    icons = VGroup()
    for i in range(total):
        r_idx = i // cols
        c_idx = i % cols
        x = -grid_w / 2 + c_idx * spacing + spacing / 2
        y =  grid_h / 2 - r_idx * spacing - spacing / 2
        dot = Circle(radius=r_scaled)
        if i < filled:
            dot.set_fill(accent_color, opacity=0.92)
            dot.set_stroke(accent_color, width=1.2, opacity=0.7)
        else:
            dot.set_fill(BRAND_GRAY, opacity=0.15)
            dot.set_stroke(BRAND_GRAY, width=1.0, opacity=0.30)
        dot.move_to([x, y, 0])
        icons.add(dot)

    icons.move_to(ORIGIN + UP * 0.6)

    pct     = filled / max(total, 1) * 100
    pct_lbl = Text(f"{pct:.0f}%", font_size=64, color=BRAND_WHITE, weight=BOLD)
    cat_lbl = Text(label_text, font_size=28, color=accent_color) if label_text else None

    labels = VGroup(pct_lbl) if not cat_lbl else VGroup(pct_lbl, cat_lbl)
    if cat_lbl:
        labels.arrange(RIGHT, buff=0.35)

    full_group = VGroup(icons, labels)
    labels.next_to(icons, DOWN, buff=0.40)

    safe_h_total = config.frame_height * 0.88
    if full_group.height > safe_h_total:
        full_group.scale(safe_h_total / full_group.height)
    full_group.move_to(ORIGIN)

    bottom_edge = full_group.get_bottom()[1]
    if bottom_edge < -config.frame_height * 0.44:
        full_group.shift(UP * (-config.frame_height * 0.44 - bottom_edge))

    anim_t = max(min(duration * 0.68, duration - 0.4), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(
        LaggedStart(*[FadeIn(ic) for ic in icons], lag_ratio=0.04),
        FadeIn(labels),
        run_time=anim_t, rate_func=smooth,
    )
    scene.wait(hold_t)
    return _sc.collected(), icons


def fm_animate_stacked_cards(scene, items, duration=4.0):
    _sc = _Tracker(scene)
    scene = _sc
    cards = fm_stacked_cards(items)
    safe_h = config.frame_height * 0.82
    if cards.height > safe_h:
        cards.scale(safe_h / cards.height)

    per_t  = max(min(duration / max(len(items), 1) * 0.55, 0.72), 0.1)
    hold_t = max(duration - per_t * len(items), 0.15)
    for card in cards:
        scene.play(FadeIn(card, shift=LEFT * 0.45), run_time=per_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), cards


def fm_animate_bullet_chart(scene, actual, target, range_low, range_high,
                              label_text, accent_color=BRAND_GREEN,
                              duration=3.0, position=None, bar_length=8.0):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN

    span   = max(range_high - range_low, 1.0)
    scale  = bar_length / span

    def _x(v):
        return -bar_length / 2 + (v - range_low) * scale

    band_w  = (range_high - range_low) * scale
    band    = RoundedRectangle(width=band_w, height=0.55, corner_radius=0.08)
    band.set_fill(BRAND_GRAY, opacity=0.22)
    band.set_stroke(BRAND_GRAY, width=1.0, opacity=0.3)
    band.move_to([(_x(range_low) + _x(range_high)) / 2, 0, 0])
    band.shift(position)

    tick_x  = _x(target)
    tick    = Line([tick_x, -0.5, 0], [tick_x, 0.5, 0])
    tick.set_stroke(BRAND_WHITE, width=3.5, opacity=0.9)
    tick.shift(position)

    actual_w = max((actual - range_low) * scale, 0.05)
    tracker  = ValueTracker(0.0)

    def _bar():
        w = tracker.get_value()
        if w < 0.01:
            return VMobject()
        b = RoundedRectangle(width=w, height=0.32, corner_radius=0.06)
        b.set_fill(accent_color, opacity=0.95)
        b.set_stroke(accent_color, width=1.0, opacity=0.6)
        b.move_to([_x(range_low) + w / 2, 0, 0])
        b.shift(position)
        return b

    bar = always_redraw(_bar)

    target_lbl = Text(f"Target: {int(target):,}", font_size=26, color=BRAND_WHITE)
    target_lbl.next_to(tick, UP, buff=0.22)
    actual_lbl = Text(f"{int(actual):,}", font_size=42, color=accent_color, weight=BOLD)
    actual_lbl.next_to(band, DOWN, buff=0.32)
    cat_lbl    = Text(label_text, font_size=30, color=BRAND_GRAY)
    cat_lbl.next_to(actual_lbl, DOWN, buff=0.15)

    scene.add(band, tick, bar, target_lbl, actual_lbl, cat_lbl)
    anim_t = max(min(duration * 0.70, duration - 0.3), 0.1)
    hold_t = max(duration - anim_t, 0.05)
    scene.play(tracker.animate.set_value(actual_w), run_time=anim_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), actual_lbl


def fm_animate_glow_reveal(scene, text_str, accent_color=BRAND_WHITE,
                            duration=3.0, font_size=88, subtitle=None,
                            subtitle_color=None, _style=None, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if subtitle_color is None:
        subtitle_color = accent_color
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position

    safe_w = config.frame_width * 0.84
    text = Text(text_str, font_size=font_size, color=BRAND_WHITE, weight=BOLD)
    if text.width > safe_w:
        text.scale(safe_w / text.width)

    sub = None
    if subtitle:
        sub = Text(subtitle, font_size=38, color=subtitle_color)
        VGroup(text, sub).arrange(DOWN, buff=0.42).move_to(pos)
    else:
        text.move_to(pos)

    intro_t = max(min(duration * 0.38, 1.3), 0.15)
    hold_t  = max(duration - intro_t - (0.28 if subtitle else 0), 0.05)

    rings = VGroup()
    for i in range(5):
        r = Circle(radius=0.5 + i * 0.55)
        r.set_stroke(accent_color, width=max(2.5 - i * 0.4, 0.4),
                     opacity=max(0.32 - i * 0.055, 0.03))
        r.move_to(text.get_center())
        rings.add(r)
    scene.play(
        FadeIn(text, scale=0.88),
        LaggedStart(*[Create(r) for r in rings], lag_ratio=0.12),
        run_time=intro_t, rate_func=smooth,
    )
    if subtitle:
        scene.play(FadeIn(sub, shift=UP * 0.12), run_time=0.28, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), text


def fm_animate_timeline(scene, events, accent_color=BRAND_GOLD, duration=4.0,
                         show_index=False, _style=None, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    n = len(events)
    if n < 1:
        return _sc.collected(), VGroup()
    style = _style if _style is not None else _fm_style(scene, 2)

    line_w = min(max(n * 1.75, 4.0), 11.0)
    dots   = VGroup()
    labels = VGroup()

    if style == 1:
        col_h   = config.frame_height * 0.72
        step_h  = col_h / max(n, 1)
        start_y = col_h / 2 - step_h / 2

        spine = Line([0, col_h / 2, 0], [0, -col_h / 2, 0])
        spine.set_stroke(BRAND_GRAY, width=2.0, opacity=0.45)
        scene.add(spine)

        for i, event in enumerate(events):
            y   = start_y - i * step_h
            dot = Dot([0, y, 0], radius=0.14, color=accent_color)
            dot.set_fill(accent_color, opacity=1.0)
            dots.add(dot)

            prefix = f"{i + 1}. " if show_index else ""
            lbl    = Text(f"{prefix}{event}", font_size=22, color=BRAND_WHITE)
            if i % 2 == 0:
                lbl.next_to(dot, RIGHT, buff=0.28)
            else:
                lbl.next_to(dot, LEFT, buff=0.28)
            labels.add(lbl)

        anim_t = max(min(duration * 0.72, duration - 0.4), 0.1)
        hold_t = max(duration - anim_t, 0.05)
        scene.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.14),
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in labels], lag_ratio=0.14),
            run_time=anim_t, rate_func=smooth,
        )
        scene.wait(hold_t)
        return _sc.collected(), dots

    else:
        line = Line([-line_w / 2 - 0.1, 0, 0], [line_w / 2 + 0.1, 0, 0])
        line.set_stroke(BRAND_GRAY, width=2.0, opacity=0.45)
        scene.add(line)

        for i, event in enumerate(events):
            x   = -line_w / 2 + i * (line_w / max(n - 1, 1)) if n > 1 else 0
            dot = Dot([x, 0, 0], radius=0.13, color=accent_color)
            dot.set_stroke(accent_color, width=1.5, opacity=0.7)
            dots.add(dot)

            prefix = f"{i + 1}. " if show_index else ""
            lbl    = Text(f"{prefix}{event}", font_size=22, color=BRAND_WHITE)
            if i % 2 == 0:
                lbl.next_to(dot, UP, buff=0.28)
            else:
                lbl.next_to(dot, DOWN, buff=0.28)
            labels.add(lbl)

        anim_t = max(min(duration * 0.72, duration - 0.4), 0.1)
        hold_t = max(duration - anim_t, 0.05)
        scene.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.14),
            LaggedStart(
                *[FadeIn(l, shift=(UP if i % 2 == 0 else DOWN) * 0.12)
                  for i, l in enumerate(labels)],
                lag_ratio=0.14,
            ),
            run_time=anim_t, rate_func=smooth,
        )
        scene.wait(hold_t)
        return _sc.collected(), dots


def fm_animate_single_value(scene, value_str, label_text,
                             accent_color=BRAND_GOLD, duration=3.0,
                             value_size=140, label_size=38,
                             sublabel=None, sublabel_color=None, _style=None,
                             position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if sublabel_color is None:
        sublabel_color = BRAND_GRAY
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    style = _style if _style is not None else _fm_style(scene, 3)

    val_mob = Text(value_str, font_size=value_size, color=BRAND_WHITE, weight=BOLD)
    lbl_mob = Text(label_text, font_size=label_size, color=accent_color)
    intro_t = max(min(duration * 0.38, 1.1), 0.1)
    hold_t  = max(duration - intro_t, 0.05)

    if style == 0:
        group = VGroup(val_mob, lbl_mob).arrange(DOWN, buff=0.55)
        if sublabel:
            sub = Text(sublabel, font_size=28, color=sublabel_color)
            group = VGroup(val_mob, lbl_mob, sub).arrange(DOWN, buff=0.48)
        group.move_to(pos)
        scene.play(
            FadeIn(val_mob, scale=0.88),
            FadeIn(lbl_mob, shift=UP * 0.15),
            run_time=intro_t, rate_func=smooth,
        )
        if sublabel:
            scene.play(FadeIn(sub, shift=UP * 0.1), run_time=0.25)
            hold_t = max(hold_t - 0.25, 0.05)

    elif style == 1:
        val_mob.move_to(pos)
        accent_line = Line(
            [pos[0] - val_mob.width / 2 - 0.2, pos[1] - val_mob.height / 2 - 0.28, 0],
            [pos[0] + val_mob.width / 2 + 0.2, pos[1] - val_mob.height / 2 - 0.28, 0],
        )
        accent_line.set_stroke(accent_color, width=4.0, opacity=0.85)
        lbl_mob.next_to(val_mob, DOWN, buff=0.55)
        scene.play(
            FadeIn(val_mob, scale=0.90),
            Create(accent_line),
            run_time=intro_t, rate_func=smooth,
        )
        scene.play(FadeIn(lbl_mob, shift=UP * 0.12), run_time=0.28)
        hold_t = max(hold_t - 0.28, 0.05)
        if sublabel:
            sub = Text(sublabel, font_size=28, color=sublabel_color)
            sub.next_to(lbl_mob, DOWN, buff=0.2)
            scene.play(FadeIn(sub), run_time=0.2)
            hold_t = max(hold_t - 0.2, 0.05)

    else:
        bg = RoundedRectangle(width=7.0, height=3.4, corner_radius=0.28)
        bg.set_fill(BRAND_PANEL, opacity=0.88)
        bg.set_stroke(accent_color, width=2.0, opacity=0.45)
        bg.move_to(pos)
        lbl_mob.move_to(pos + UP * 0.8)
        val_mob.move_to(pos + DOWN * 0.3)
        group = VGroup(bg, lbl_mob, val_mob)
        scene.play(
            FadeIn(bg, scale=0.94),
            FadeIn(lbl_mob, shift=DOWN * 0.1),
            FadeIn(val_mob, scale=0.88),
            run_time=intro_t, rate_func=smooth,
        )
        if sublabel:
            sub = Text(sublabel, font_size=28, color=sublabel_color)
            sub.next_to(val_mob, DOWN, buff=0.22)
            scene.play(FadeIn(sub), run_time=0.2)
            hold_t = max(hold_t - 0.2, 0.05)

    scene.wait(hold_t)
    return _sc.collected(), val_mob


def fm_formula(scene, lines="", font_size=60, color=BRAND_WHITE, duration=3.0,
               position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    if not lines:
        scene.wait(duration)
        return _sc.collected(), VGroup()
    if isinstance(lines, str):
        lines = [lines]
    safe_w = config.frame_width * 0.86
    safe_h = config.frame_height * 0.7
    text_mobs = [Text(s, font_size=font_size, color=color, weight=BOLD) for s in lines]
    group = VGroup(*text_mobs).arrange(DOWN, buff=0.28)
    if group.width > safe_w:
        group.scale_to_fit_width(safe_w)
    if group.height > safe_h:
        group.scale_to_fit_height(safe_h)
    group.move_to(position)

    intro_t = max(min(duration * 0.4, 1.2), 0.1)
    hold_t  = max(duration - intro_t, 0.05)
    scene.play(
        LaggedStart(*[FadeIn(t, scale=0.92) for t in text_mobs], lag_ratio=0.15),
        run_time=intro_t, rate_func=smooth,
    )
    scene.wait(hold_t)
    return _sc.collected(), group


def fm_animate_comparison_bars(scene, items, duration=4.0, title_text="",
                                show_net=True, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position

    def _to_float(v):
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().rstrip('%').replace(',', '').replace('$', '')
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def _parse_item(item):
        if isinstance(item, (list, tuple)):
            label = str(item[0]) if len(item) > 0 else ""
            val = _to_float(item[1]) if len(item) > 1 else 0.0
            color = item[2] if len(item) > 2 else BRAND_GREEN
            return (label, val, color)
        return ("", 0.0, BRAND_GREEN)

    items = [_parse_item(i) for i in items]

    if show_net:
        net = sum(float(v) for _, v, _ in items)
        net_color = BRAND_GREEN if net >= 0 else BRAND_RED
        items = list(items) + [("Net", net, net_color)]

    n       = len(items)
    bar_w   = min(2.2, 9.5 / max(n, 1))
    spacing = bar_w * 1.55
    total_w = (n - 1) * spacing
    edge_margin = bar_w / 2 + 0.35

    pos_vals = [v for _, v, _ in items if v > 0]
    neg_vals = [v for _, v, _ in items if v < 0]
    max_pos  = max(pos_vals) if pos_vals else 1
    max_neg  = abs(min(neg_vals)) if neg_vals else 1
    total_h  = max_pos + max_neg
    scale    = 4.2 / max(total_h, 1.0)
    zero_y   = max_neg * scale - 2.1
    cat_row_y = zero_y - 0.32

    bars       = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, (label, value, color) in enumerate(items):
        x      = -total_w / 2 + i * spacing
        bar_h  = max(abs(value) * scale, 0.16)
        is_neg = value < 0
        y_bot  = zero_y - bar_h if is_neg else zero_y
        bar    = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
        bar.set_fill(color, opacity=0.92)
        bar.set_stroke(color, width=1.5, opacity=0.55)
        bar.move_to([x, y_bot + bar_h / 2, 0])
        bars.add(bar)

        v_str   = f"{int(abs(value)):,}" if abs(value) >= 1 else f"{abs(value):.2f}"
        if is_neg:
            v_str = f"-{v_str}"
        val_lbl = Text(v_str, font_size=28, color=color, weight=BOLD)
        val_lbl.next_to(bar, UP if not is_neg else DOWN, buff=0.1)
        val_labels.add(val_lbl)

        cat_lbl = Text(label, font_size=22, color=BRAND_GRAY)
        cat_lbl.move_to([x, cat_row_y, 0])
        if is_neg and val_lbl.get_bottom()[1] < cat_lbl.get_top()[1] + 0.05:
            cat_lbl.next_to(val_lbl, DOWN, buff=0.15)
        cat_labels.add(cat_lbl)

    baseline = Line([-total_w / 2 - edge_margin, zero_y, 0], [total_w / 2 + edge_margin, zero_y, 0])
    baseline.set_stroke(BRAND_GRAY, width=2.2, opacity=0.55)
    scene.add(baseline)

    min_gap = 0.08
    for k in range(1, len(cat_labels)):
        prev = cat_labels[k - 1]
        cur  = cat_labels[k]
        overlap = prev.get_right()[0] - cur.get_left()[0] + min_gap
        if overlap > 0:
            cur.shift(RIGHT * (overlap / 2))
            prev.shift(LEFT * (overlap / 2))

    if title_text:
        ttl = Text(title_text, font_size=30, color=BRAND_GRAY)
        ttl.next_to(VGroup(bars, val_labels), UP, buff=0.28)
        scene.add(ttl)

    scene.add(cat_labels)
    grow_t = max(min(duration * 0.70, duration - 0.4), 0.1)
    hold_t = max(duration - grow_t, 0.05)
    scene.play(
        LaggedStart(
            *[GrowFromEdge(b, DOWN if v >= 0 else UP)
              for (_, v, _), b in zip(items, bars)],
            lag_ratio=0.18,
        ),
        run_time=grow_t * 0.65, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
        run_time=grow_t * 0.35, rate_func=smooth,
    )
    scene.wait(hold_t)
    return _sc.collected(), bars


def fm_animate_data_table(scene, headers, rows, duration=4.0,
                           header_color=BRAND_GOLD, accent_row=None,
                           accent_color=BRAND_RED, label_text=None,
                           title_text=None, position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    """Animated data table with header row and data rows.
    headers: list of column header strings.
    rows: list of lists (each inner list = one row of values as strings).
    accent_row: index of row to highlight (0-based), or None.
    Renders centered and clamped to frame."""
    n_cols = len(headers)
    n_rows = len(rows)

    safe_w    = config.frame_width * 0.80
    safe_h    = config.frame_height * 0.78
    col_w     = min(safe_w / max(n_cols, 1), 3.2)
    row_h     = min(safe_h / max(n_rows + 1, 1), 1.05)
    total_w   = col_w * n_cols
    total_h   = row_h * (n_rows + 1)

    all_cells = VGroup()

    for c_idx, hdr in enumerate(headers):
        x = -total_w / 2 + c_idx * col_w + col_w / 2
        y =  total_h / 2 - row_h / 2
        bg = RoundedRectangle(width=col_w, height=row_h, corner_radius=0.06)
        bg.set_fill(BRAND_PANEL, opacity=0.95)
        bg.set_stroke(header_color, width=1.5, opacity=0.55)
        bg.move_to([x, y, 0])
        lbl = Text(str(hdr), font_size=min(int(row_h * 28), 32), color=header_color, weight=BOLD)
        lbl.move_to([x, y, 0])
        if lbl.width > col_w * 0.88:
            lbl.scale(col_w * 0.88 / lbl.width)
        all_cells.add(bg, lbl)

    row_mobs = []
    for r_idx, row in enumerate(rows):
        is_accent = (accent_row is not None and r_idx == accent_row)
        fill_c    = accent_color if is_accent else BRAND_PANEL
        fill_op   = 0.30 if is_accent else 0.65
        text_c    = accent_color if is_accent else BRAND_WHITE
        row_group = VGroup()
        for c_idx, val in enumerate(row):
            x = -total_w / 2 + c_idx * col_w + col_w / 2
            y =  total_h / 2 - (r_idx + 1) * row_h - row_h / 2
            bg = RoundedRectangle(width=col_w, height=row_h, corner_radius=0.04)
            bg.set_fill(fill_c, opacity=fill_op)
            bg.set_stroke(BRAND_GRAY, width=0.8, opacity=0.30)
            bg.move_to([x, y, 0])
            lbl = Text(str(val), font_size=min(int(row_h * 26), 30), color=text_c)
            lbl.move_to([x, y, 0])
            if lbl.width > col_w * 0.88:
                lbl.scale(col_w * 0.88 / lbl.width)
            row_group.add(bg, lbl)
        all_cells.add(row_group)
        row_mobs.append(row_group)

    all_cells.move_to(pos)
    fm_clamp_to_frame(all_cells, margin_x=0.06, margin_y=0.08)

    intro_t = max(min(duration * 0.30, 1.0), 0.15)
    per_row = max((duration - intro_t) / max(n_rows, 1) * 0.55, 0.12)
    hold_t  = max(duration - intro_t - per_row * n_rows, 0.1)

    header_cells = VGroup(*[all_cells[i] for i in range(n_cols * 2)])
    scene.play(FadeIn(header_cells), run_time=intro_t, rate_func=smooth)
    for row_group in row_mobs:
        scene.play(FadeIn(row_group, shift=UP * 0.08), run_time=per_row, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), all_cells


def fm_icon(name: str, size: float = 1.0, color=None):
    """Return a VGroup icon mobject for the given name. Does not animate — caller positions and adds."""
    if color is None:
        color = BRAND_GOLD

    g = VGroup()
    s = size

    if name == "sigma":
        g.add(Text("Σ", font_size=int(72 * s), color=color, weight=BOLD))
    elif name == "integral":
        g.add(Text("∫", font_size=int(80 * s), color=color, weight=BOLD))
    elif name == "pi_sym":
        g.add(Text("π", font_size=int(72 * s), color=color, weight=BOLD))
    elif name == "infinity":
        g.add(Text("∞", font_size=int(72 * s), color=color, weight=BOLD))
    elif name == "gradient":
        g.add(Text("∇", font_size=int(72 * s), color=color, weight=BOLD))
    elif name == "derivative":
        g.add(Text("d/dx", font_size=int(52 * s), color=color, weight=BOLD))
    elif name == "matrix_sym":
        g.add(Text("[ ]", font_size=int(72 * s), color=color, weight=BOLD))
    elif name == "neuron":
        body = Circle(radius=0.38 * s)
        body.set_fill(color, opacity=0.88).set_stroke(color, width=2.0)
        for angle in [PI * 0.25, PI * 0.75, PI * 1.25, PI * 1.75]:
            dendrite = Line([0, 0, 0], [0.6 * s * math.cos(angle), 0.6 * s * math.sin(angle), 0])
            dendrite.set_stroke(color, width=2.0, opacity=0.7)
            g.add(dendrite)
        g.add(body)
    elif name == "dollar":
        g.add(Text("$", font_size=int(68 * s), color=color, weight=BOLD))
    elif name == "coin":
        outer = Circle(radius=0.50 * s)
        outer.set_fill(color, opacity=0.90).set_stroke(color, width=2.5)
        inner = Circle(radius=0.28 * s)
        inner.set_fill(BRAND_PANEL, opacity=0.80).set_stroke(color, width=1.0, opacity=0.45)
        sign  = Text("$", font_size=int(24 * s), color=color, weight=BOLD)
        g.add(outer, inner, sign)
    elif name == "house":
        roof_pts = [[-0.48*s, 0, 0], [0, 0.48*s, 0], [0.48*s, 0, 0]]
        roof = Polygon(*roof_pts)
        roof.set_fill(color, opacity=0.88).set_stroke(color, width=1.5, opacity=0.6)
        body = RoundedRectangle(width=0.65*s, height=0.42*s, corner_radius=0.04)
        body.set_fill(color, opacity=0.65).set_stroke(color, width=1.5, opacity=0.6)
        body.next_to(roof, DOWN, buff=0)
        g.add(roof, body)
    elif name == "person":
        head = Circle(radius=0.18 * s)
        head.set_fill(color, opacity=0.88).set_stroke(color, width=1.2)
        body = RoundedRectangle(width=0.30*s, height=0.36*s, corner_radius=0.06)
        body.set_fill(color, opacity=0.72).set_stroke(color, width=1.2)
        body.next_to(head, DOWN, buff=0.04 * s)
        g.add(head, body)
    elif name == "clock":
        face = Circle(radius=0.50 * s)
        face.set_fill(BRAND_PANEL, opacity=0.85).set_stroke(color, width=2.5)
        hour   = Line([0, 0, 0], [0, 0.28*s, 0]).set_stroke(color, width=3.0)
        minute = Line([0, 0, 0], [0.22*s, 0, 0]).set_stroke(color, width=2.0)
        g.add(face, hour, minute)
    elif name == "arrow_up":
        shaft = RoundedRectangle(width=0.14*s, height=0.38*s, corner_radius=0.04)
        shaft.set_fill(color, opacity=0.90).shift(DOWN * 0.10 * s)
        tip = Polygon([-0.28*s, 0, 0], [0, 0.32*s, 0], [0.28*s, 0, 0])
        tip.set_fill(color, opacity=0.90).shift(UP * 0.14 * s)
        g.add(shaft, tip)
    elif name == "arrow_down":
        shaft = RoundedRectangle(width=0.14*s, height=0.38*s, corner_radius=0.04)
        shaft.set_fill(color, opacity=0.90).shift(UP * 0.10 * s)
        tip = Polygon([-0.28*s, 0, 0], [0, -0.32*s, 0], [0.28*s, 0, 0])
        tip.set_fill(color, opacity=0.90).shift(DOWN * 0.14 * s)
        g.add(shaft, tip)
    elif name == "warning":
        tri = Polygon([-0.48*s, -0.38*s, 0], [0.48*s, -0.38*s, 0], [0, 0.48*s, 0])
        tri.set_fill(color, opacity=0.88).set_stroke(color, width=1.5)
        excl = Text("!", font_size=int(34 * s), color=BRAND_PANEL, weight=BOLD)
        excl.shift(DOWN * 0.04 * s)
        g.add(tri, excl)
    elif name == "checkmark":
        pts = [[-0.38*s, 0.0, 0], [-0.08*s, -0.32*s, 0], [0.48*s, 0.38*s, 0]]
        mark = VMobject()
        mark.set_points_as_corners(pts)
        mark.set_stroke(color, width=max(3.5*s, 1.5), opacity=0.95)
        g.add(mark)
    elif name == "fire":
        outer = Circle(radius=0.34 * s).stretch(0.44 / 0.68, dim=0)
        outer.set_fill(color, opacity=0.88).set_stroke(color, width=1.0)
        inner = Circle(radius=0.225 * s).stretch(0.24 / 0.45, dim=0)
        inner.set_fill(BRAND_WHITE, opacity=0.45).shift(DOWN * 0.04 * s)
        g.add(outer, inner)
    else:
        c = Circle(radius=0.38 * s)
        c.set_fill(color, opacity=0.88).set_stroke(color, width=2.0)
        g.add(c)

    return g


def fm_animate_vector(scene, direction, label_text, accent_color=BRAND_GOLD,
                       duration=3.5, origin=None, scale=2.5, show_components=False,
                       position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is not None:
        origin = position
    if origin is None:
        origin = ORIGIN

    dx, dy = direction[0], direction[1]
    length = math.sqrt(dx**2 + dy**2)
    if length > 1e-9:
        dx, dy = dx / length * scale, dy / length * scale

    tip = [origin[0] + dx, origin[1] + dy, 0]
    arrow = Arrow(
        start=origin, end=tip,
        buff=0,
        stroke_width=5,
        max_tip_length_to_length_ratio=0.18,
        color=accent_color,
    )

    lbl = Text(label_text, font_size=36, color=accent_color, weight=BOLD)
    mid = [(origin[0] + tip[0]) / 2, (origin[1] + tip[1]) / 2, 0]
    perp_x = -dy / max(scale, 0.01) * 0.45
    perp_y =  dx / max(scale, 0.01) * 0.45
    lbl.move_to([mid[0] + perp_x, mid[1] + perp_y, 0])

    components = VGroup()
    if show_components:
        comp_x = Line(origin, [origin[0] + dx, origin[1], 0])
        comp_x.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)
        comp_y = Line([origin[0] + dx, origin[1], 0], tip)
        comp_y.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)
        comp_x_lbl = Text(f"{dx:.1f}", font_size=22, color=BRAND_GRAY)
        comp_x_lbl.next_to(comp_x, DOWN, buff=0.12)
        comp_y_lbl = Text(f"{dy:.1f}", font_size=22, color=BRAND_GRAY)
        comp_y_lbl.next_to(comp_y, RIGHT, buff=0.12)
        components.add(comp_x, comp_y, comp_x_lbl, comp_y_lbl)
        scene.add(components)

    draw_t = max(min(duration * 0.55, 1.6), 0.1)
    hold_t = max(duration - draw_t - 0.3, 0.05)
    scene.play(Create(arrow), run_time=draw_t, rate_func=smooth)
    scene.play(FadeIn(lbl, shift=UP * 0.12), run_time=0.3, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), arrow


def fm_animate_matrix(scene, rows_data, label_text="", accent_color=BRAND_GOLD,
                       duration=4.0, position=None, cell_size=0.9, font_size=36):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN

    n_rows = len(rows_data)
    n_cols = max(len(r) for r in rows_data) if rows_data else 1

    cells = VGroup()
    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            val_str = str(val)
            cell_txt = Text(val_str, font_size=font_size, color=BRAND_WHITE, weight=BOLD)
            x = (c_idx - (n_cols - 1) / 2) * cell_size
            y = ((n_rows - 1) / 2 - r_idx) * cell_size
            cell_txt.move_to([x, y, 0])
            cells.add(cell_txt)

    total_w = n_cols * cell_size
    total_h = n_rows * cell_size
    bracket_h = total_h + 0.3

    left_top    = [-total_w / 2 - 0.55, bracket_h / 2, 0]
    left_mid_t  = [-total_w / 2 - 0.35, bracket_h / 2, 0]
    left_mid_b  = [-total_w / 2 - 0.35, -bracket_h / 2, 0]
    left_bot    = [-total_w / 2 - 0.55, -bracket_h / 2, 0]
    right_top   = [total_w / 2 + 0.55, bracket_h / 2, 0]
    right_mid_t = [total_w / 2 + 0.35, bracket_h / 2, 0]
    right_mid_b = [total_w / 2 + 0.35, -bracket_h / 2, 0]
    right_bot   = [total_w / 2 + 0.55, -bracket_h / 2, 0]

    left_bracket = VMobject()
    left_bracket.set_points_as_corners([left_top, left_mid_t, left_mid_b, left_bot])
    left_bracket.set_stroke(accent_color, width=3.5, opacity=0.9)

    right_bracket = VMobject()
    right_bracket.set_points_as_corners([right_top, right_mid_t, right_mid_b, right_bot])
    right_bracket.set_stroke(accent_color, width=3.5, opacity=0.9)

    matrix_group = VGroup(left_bracket, right_bracket, cells)
    matrix_group.move_to(position)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=32, color=accent_color)
        lbl_mob.next_to(matrix_group, DOWN, buff=0.4)

    safe_w = config.frame_width * 0.85
    safe_h = config.frame_height * 0.80
    combined = VGroup(matrix_group) if not lbl_mob else VGroup(matrix_group, lbl_mob)
    if combined.width > safe_w:
        combined.scale(safe_w / combined.width)
    if combined.height > safe_h:
        combined.scale(safe_h / combined.height)

    draw_t = max(min(duration * 0.28, 1.0), 0.1)
    per_row = max((duration - draw_t - 0.3) / max(n_rows, 1), 0.12)
    hold_t = max(duration - draw_t - per_row * n_rows - 0.2, 0.05)

    scene.play(
        Create(left_bracket), Create(right_bracket),
        run_time=draw_t, rate_func=smooth,
    )
    for r_idx in range(n_rows):
        row_cells = [cells[r_idx * n_cols + c] for c in range(min(n_cols, len(rows_data[r_idx])))]
        scene.play(
            LaggedStart(*[FadeIn(c, scale=0.85) for c in row_cells], lag_ratio=0.15),
            run_time=per_row, rate_func=smooth,
        )
    if lbl_mob:
        scene.play(FadeIn(lbl_mob), run_time=0.2)
    scene.wait(hold_t)
    return _sc.collected(), matrix_group


def fm_animate_bell_curve(scene, label_text="", accent_color=BRAND_GOLD,
                           duration=4.0, position=None, show_std_regions=False,
                           skew=None, skewed=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN

    n_pts   = 120
    x_range = 3.6
    curve_w = 10.0
    curve_h = 3.4

    xs = [(-x_range + i * 2 * x_range / (n_pts - 1)) for i in range(n_pts)]
    ys = [math.exp(-0.5 * x * x) for x in xs]

    base_y = position[1] - curve_h * 0.12

    pts = [
        [position[0] + x / x_range * curve_w / 2,
         base_y + y * curve_h,
         0]
        for x, y in zip(xs, ys)
    ]

    curve = VMobject()
    _fm_set_line_smooth(curve, pts)
    curve.set_stroke(accent_color, width=4.5, opacity=0.95)

    fill_pts = pts + [[pts[-1][0], base_y, 0], [pts[0][0], base_y, 0]]
    fill_region = Polygon(*fill_pts, fill_opacity=0.07, stroke_width=0)
    fill_region.set_fill(accent_color)

    std_markers = VGroup()
    if show_std_regions:
        for sign in [-1, 1]:
            sx = position[0] + sign * curve_w / 2 / x_range
            tick = Line([sx, base_y - 0.12, 0], [sx, base_y + 0.22, 0])
            tick.set_stroke(accent_color, width=2.0, opacity=0.55)
            std_markers.add(tick)

    baseline = Line([pts[0][0], base_y, 0], [pts[-1][0], base_y, 0])
    baseline.set_stroke(BRAND_GRAY, width=1.5, opacity=0.35)

    draw_t = max(min(duration * 0.55, 2.0), 0.1)
    fade_t = min(0.3, duration * 0.08)
    hold_t = max(duration - draw_t - fade_t, 0.05)

    scene.add(baseline, fill_region)
    if show_std_regions and len(std_markers) > 0:
        scene.add(std_markers)
    scene.play(Create(curve), run_time=draw_t, rate_func=smooth)
    scene.wait(hold_t)
    collected = _sc.collected()
    try:
        scene._s.play(FadeOut(collected), run_time=fade_t)
        scene._s.remove(collected)
    except Exception:
        try:
            scene._s.remove(collected)
        except Exception:
            pass
    return collected, curve


def fm_animate_scatter(scene, points=None, label_text="", accent_color=BRAND_GOLD,
                        duration=4.0, position=None, show_regression=False,
                        x_label="x", y_label="y", highlight_points=None,
                        title_text=""):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    if not points:
        scene.wait(duration)
        return _sc.collected(), VGroup()

    clean_points = []
    for p in points:
        try:
            px, py = float(p[0]), float(p[1])
            clean_points.append((px, py))
        except Exception:
            continue
    points = clean_points
    if not points:
        scene.wait(duration)
        return _sc.collected(), VGroup()

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    plot_w = 8.0
    plot_h = 4.5
    pad    = 0.6

    def _to_screen(px, py):
        sx = position[0] - plot_w / 2 + pad + (px - x_min) / x_span * (plot_w - 2 * pad)
        sy = position[1] - plot_h / 2 + pad + (py - y_min) / y_span * (plot_h - 2 * pad)
        return [sx, sy, 0]

    x_axis = Line([position[0] - plot_w / 2, position[1] - plot_h / 2, 0],
                  [position[0] + plot_w / 2, position[1] - plot_h / 2, 0])
    x_axis.set_stroke(BRAND_GRAY, width=2.0, opacity=0.55)
    y_axis = Line([position[0] - plot_w / 2, position[1] - plot_h / 2, 0],
                  [position[0] - plot_w / 2, position[1] + plot_h / 2, 0])
    y_axis.set_stroke(BRAND_GRAY, width=2.0, opacity=0.55)

    x_lbl_mob = Text(x_label, font_size=24, color=BRAND_GRAY)
    x_lbl_mob.next_to(x_axis, DOWN, buff=0.2)
    y_lbl_mob = Text(y_label, font_size=24, color=BRAND_GRAY)
    y_lbl_mob.next_to(y_axis, LEFT, buff=0.2)

    dots = VGroup()
    for px, py in points:
        sp = _to_screen(px, py)
        dot = Dot(sp, radius=0.10, color=accent_color)
        dot.set_fill(accent_color, opacity=0.85)
        dots.add(dot)

    reg_line = None
    if show_regression and len(points) >= 2:
        n = len(points)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        if abs(den) > 1e-9:
            slope = num / den
            intercept = mean_y - slope * mean_x
            y_at_xmin = slope * x_min + intercept
            y_at_xmax = slope * x_max + intercept
            reg_start = _to_screen(x_min, y_at_xmin)
            reg_end   = _to_screen(x_max, y_at_xmax)
            reg_line  = Line(reg_start, reg_end)
            reg_line.set_stroke(BRAND_RED, width=3.0, opacity=0.85)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=accent_color)
        lbl_mob.move_to([position[0], position[1] + plot_h / 2 + 0.4, 0])

    scene.add(x_axis, y_axis, x_lbl_mob, y_lbl_mob)
    if lbl_mob:
        scene.add(lbl_mob)

    dot_t  = max(min(duration * 0.62, 2.2), 0.1)
    hold_t = max(duration - dot_t - 0.35, 0.05)

    scene.play(
        LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06),
        run_time=dot_t, rate_func=smooth,
    )
    if reg_line is not None:
        scene.play(Create(reg_line), run_time=0.35, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), dots


def fm_animate_probability_bar(scene, outcomes, label_text="",
                                accent_color=BRAND_GOLD, duration=4.0,
                                position=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    if not outcomes:
        return _sc.collected(), VGroup()

    n       = len(outcomes)
    bar_w   = min(1.4, 9.0 / max(n, 1))
    spacing = bar_w * 1.6
    total_w = (n - 1) * spacing
    chart_h = 4.0
    base_y  = position[1] - chart_h / 2

    baseline = Line(
        [position[0] - total_w / 2 - bar_w / 2 - 0.3, base_y, 0],
        [position[0] + total_w / 2 + bar_w / 2 + 0.3, base_y, 0],
    )
    baseline.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)

    bars       = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, outcome in enumerate(outcomes):
        if len(outcome) >= 3:
            name, prob, bar_color = outcome[0], outcome[1], outcome[2]
        else:
            name, prob = outcome[0], outcome[1]
            bar_color = accent_color
        prob = max(0.0, min(1.0, float(prob)))
        x    = position[0] - total_w / 2 + i * spacing
        bar_h = max(prob * chart_h, 0.06)
        bar = RoundedRectangle(width=bar_w, height=bar_h, corner_radius=0.06)
        bar.set_fill(bar_color, opacity=0.88)
        bar.set_stroke(bar_color, width=1.5, opacity=0.55)
        bar.move_to([x, base_y + bar_h / 2, 0])
        bars.add(bar)

        pct_str = f"{prob * 100:.1f}%"
        val_lbl = Text(pct_str, font_size=24, color=accent_color, weight=BOLD)
        val_lbl.next_to(bar, UP, buff=0.1)
        val_labels.add(val_lbl)

        cat_lbl = Text(name, font_size=20, color=BRAND_GRAY)
        cat_lbl.next_to(bar, DOWN, buff=0.15)
        cat_labels.add(cat_lbl)

    chart_group = VGroup(baseline, bars, val_labels, cat_labels)
    chart_group.move_to(position)

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=accent_color)
        lbl_mob.next_to(chart_group, UP, buff=0.28)
        scene.add(lbl_mob)

    scene.add(baseline, cat_labels)
    grow_t = max(min(duration * 0.65, 2.2), 0.1)
    hold_t = max(duration - grow_t - 0.35, 0.05)

    scene.play(
        LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.16),
        run_time=grow_t, rate_func=smooth,
    )
    scene.play(
        LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.12),
        run_time=0.35, rate_func=smooth,
    )
    scene.wait(hold_t)
    return _sc.collected(), bars


def fm_animate_number_line(scene, value, min_val, max_val, label_text="",
                            accent_color=BRAND_GOLD, duration=3.5,
                            position=None, line_length=9.0,
                            tick_labels=None):
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    if not isinstance(tick_labels, (list, tuple)):
        tick_labels = None

    line_mob = Line(
        [position[0] - line_length / 2, position[1], 0],
        [position[0] + line_length / 2, position[1], 0],
    )
    line_mob.set_stroke(BRAND_GRAY, width=3.0, opacity=0.55)

    span = max(max_val - min_val, 1.0)

    n_ticks = 5
    ticks_group = VGroup()
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        tick_val = min_val + frac * span
        tx = position[0] - line_length / 2 + frac * line_length
        tick = Line([tx, position[1] - 0.15, 0], [tx, position[1] + 0.15, 0])
        tick.set_stroke(BRAND_GRAY, width=2.0, opacity=0.4)
        ticks_group.add(tick)
        if tick_labels:
            if i < len(tick_labels):
                tl = Text(str(tick_labels[i]), font_size=20, color=BRAND_GRAY)
            else:
                tl = Text(f"{tick_val:.1f}", font_size=20, color=BRAND_GRAY)
        else:
            tl = Text(f"{tick_val:.1f}" if isinstance(tick_val, float) else str(int(tick_val)),
                      font_size=20, color=BRAND_GRAY)
        tl.next_to(tick, DOWN, buff=0.15)
        ticks_group.add(tl)

    _value_safe = value if value is not None else (min_val + max_val) / 2
    frac_val = max(0.0, min(1.0, (_value_safe - min_val) / span))
    target_x = position[0] - line_length / 2 + frac_val * line_length

    tracker = ValueTracker(position[0] - line_length / 2)

    def _dot():
        cx = tracker.get_value()
        d = Dot([cx, position[1], 0], radius=0.22, color=accent_color)
        d.set_fill(accent_color, opacity=1.0)
        d.set_stroke(accent_color, width=2.5, opacity=0.6)
        return d

    dot = always_redraw(_dot)

    val_str = f"{_value_safe:.2f}" if isinstance(_value_safe, float) else str(int(_value_safe))
    val_lbl = Text(val_str, font_size=44, color=accent_color, weight=BOLD)
    val_lbl.move_to([target_x, position[1] + 0.65, 0])

    lbl_mob = None
    if label_text:
        lbl_mob = Text(label_text, font_size=30, color=BRAND_GRAY)
        lbl_mob.move_to([position[0], position[1] - 0.75, 0])

    scene.add(line_mob, ticks_group, dot)
    if lbl_mob:
        scene.add(lbl_mob)

    move_t = max(min(duration * 0.65, 2.0), 0.1)
    hold_t = max(duration - move_t - 0.3, 0.05)

    scene.play(tracker.animate.set_value(target_x), run_time=move_t, rate_func=smooth)
    scene.play(FadeIn(val_lbl, shift=DOWN * 0.1), run_time=0.3)
    scene.wait(hold_t)
    return _sc.collected(), dot

def fm_animate_histogram(scene, values, bin_count=8, label_text="",
                          accent_color=BRAND_GOLD, duration=4.0, position=None,
                          x_label="", show_curve=False):
    """Frequency histogram with optional overlaid normal curve.
    values: raw data list OR list of (bin_label, count) tuples.
    Returns (_sc.collected(), bars)."""
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = _fmnp.array(position, dtype=float)

    if values and isinstance(values[0], (list, tuple)):
        bin_labels = [str(v[0]) for v in values]
        counts = [float(v[1]) for v in values]
    else:
        raw = [float(v) for v in values]
        if not raw:
            scene.wait(duration)
            return _sc.collected(), VGroup()
        lo, hi = min(raw), max(raw)
        span = max(hi - lo, 1.0)
        bin_w = span / bin_count
        counts = [0] * bin_count
        for v in raw:
            idx = min(int((v - lo) / bin_w), bin_count - 1)
            counts[idx] += 1
        bin_labels = [f"{lo + i * bin_w:.1f}" for i in range(bin_count)]

    n = len(counts)
    max_c = max(counts) if counts else 1
    chart_h = 4.0
    bar_w = min(1.4, 9.5 / max(n, 1))
    total_w = n * bar_w
    base_y = pos[1] - chart_h * 0.5

    baseline = Line(
        [pos[0] - total_w / 2 - 0.1, base_y, 0],
        [pos[0] + total_w / 2 + 0.1, base_y, 0],
    )
    baseline.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)

    bars = VGroup()
    val_labels = VGroup()
    cat_labels = VGroup()

    for i, (cnt, lbl) in enumerate(zip(counts, bin_labels)):
        x = pos[0] - total_w / 2 + i * bar_w + bar_w / 2
        bh = max(cnt / max_c * chart_h, 0.06)
        alpha = 0.55 + 0.45 * (cnt / max_c)
        bar = RoundedRectangle(width=bar_w * 0.92, height=bh, corner_radius=0.04)
        bar.set_fill(accent_color, opacity=alpha)
        bar.set_stroke(accent_color, width=1.2, opacity=0.5)
        bar.move_to([x, base_y + bh / 2, 0])
        bars.add(bar)
        if cnt > 0:
            vl = Text(str(int(cnt)), font_size=20, color=accent_color, weight=BOLD)
            vl.next_to(bar, UP, buff=0.08)
            val_labels.add(vl)
        cl = Text(lbl, font_size=16, color=BRAND_GRAY)
        cl.next_to([x, base_y, 0], DOWN, buff=0.12)
        cat_labels.add(cl)

    overlay_curve = None
    if show_curve:
        n_pts = 80
        xs_norm = [(-3.5 + i * 7.0 / (n_pts - 1)) for i in range(n_pts)]
        ys_norm = [math.exp(-0.5 * x * x) for x in xs_norm]
        peak_bar = max(cnt / max_c * chart_h for cnt in counts)
        curve_pts = [
            [pos[0] - total_w / 2 + (xn + 3.5) / 7.0 * total_w,
             base_y + yn * peak_bar,
             0]
            for xn, yn in zip(xs_norm, ys_norm)
        ]
        overlay_curve = VMobject()
        _fm_set_line_smooth(overlay_curve, curve_pts)
        overlay_curve.set_stroke(BRAND_WHITE, width=3.0, opacity=0.7)

    if label_text:
        ttl = Text(label_text, font_size=30, color=accent_color, weight=BOLD)
        ttl.move_to([pos[0], base_y + chart_h + 0.55, 0])
        scene.add(ttl)

    scene.add(baseline, cat_labels)
    grow_t = max(min(duration * 0.65, duration - 0.4), 0.1)
    hold_t = max(duration - grow_t - 0.3, 0.05)
    scene.play(
        LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08),
        run_time=grow_t, rate_func=smooth,
    )
    if val_labels:
        scene.play(LaggedStart(*[FadeIn(l) for l in val_labels], lag_ratio=0.06),
                   run_time=0.3, rate_func=smooth)
    if overlay_curve:
        scene.play(Create(overlay_curve), run_time=0.4, rate_func=smooth)
        hold_t = max(hold_t - 0.4, 0.05)
    scene.wait(hold_t)
    return _sc.collected(), bars


def fm_animate_transform(scene, matrix_2x2, duration=5.0, position=None,
                          label_text="", accent_color=BRAND_GREEN,
                          show_det=True):
    """2x2 linear transformation: animates a grid of dots and basis vectors
    transforming under the given matrix. Shows determinant if show_det=True.
    matrix_2x2: [[a,b],[c,d]]. Returns (_sc.collected(), arrows)."""
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = _fmnp.array(position, dtype=float)

    a, b, c, d = (float(matrix_2x2[0][0]), float(matrix_2x2[0][1]),
                  float(matrix_2x2[1][0]), float(matrix_2x2[1][1]))
    det = a * d - b * c

    scale = 1.1
    grid_range = 3
    grid_dots = VGroup()
    for gx in range(-grid_range, grid_range + 1):
        for gy in range(-grid_range, grid_range + 1):
            dot = Dot([pos[0] + gx * scale, pos[1] + gy * scale, 0],
                      radius=0.05, color=BRAND_GRAY)
            dot.set_fill(BRAND_GRAY, opacity=0.35)
            grid_dots.add(dot)

    e1_arrow = Arrow(
        start=[pos[0], pos[1], 0],
        end=[pos[0] + scale, pos[1], 0],
        buff=0, stroke_width=5,
        max_tip_length_to_length_ratio=0.2,
        color=BRAND_GREEN,
    )
    e2_arrow = Arrow(
        start=[pos[0], pos[1], 0],
        end=[pos[0], pos[1] + scale, 0],
        buff=0, stroke_width=5,
        max_tip_length_to_length_ratio=0.2,
        color=BRAND_RED,
    )
    e1_lbl = Text("î", font_size=28, color=BRAND_GREEN, weight=BOLD)
    e2_lbl = Text("ĵ", font_size=28, color=BRAND_RED, weight=BOLD)
    e1_lbl.next_to(e1_arrow, DOWN, buff=0.12)
    e2_lbl.next_to(e2_arrow, LEFT, buff=0.12)

    e1_arrow_t = Arrow(
        start=[pos[0], pos[1], 0],
        end=[pos[0] + a * scale, pos[1] + c * scale, 0],
        buff=0, stroke_width=5,
        max_tip_length_to_length_ratio=0.2,
        color=BRAND_GREEN,
    )
    e2_arrow_t = Arrow(
        start=[pos[0], pos[1], 0],
        end=[pos[0] + b * scale, pos[1] + d * scale, 0],
        buff=0, stroke_width=5,
        max_tip_length_to_length_ratio=0.2,
        color=BRAND_RED,
    )

    def _transform_dot(dot):
        ox = (dot.get_center()[0] - pos[0]) / scale
        oy = (dot.get_center()[1] - pos[1]) / scale
        nx = a * ox + b * oy
        ny = c * ox + d * oy
        return [pos[0] + nx * scale, pos[1] + ny * scale, 0]

    mat_lbl = Text(
        f"[{a:.0f}  {b:.0f}]\n[{c:.0f}  {d:.0f}]",
        font_size=32, color=BRAND_WHITE,
    )
    mat_lbl.move_to([pos[0] + config.frame_width * 0.32, pos[1] + 1.5, 0])

    det_lbl = None
    if show_det:
        det_str = f"det = {det:.2f}" if det != int(det) else f"det = {int(det)}"
        det_lbl = Text(det_str, font_size=28,
                       color=BRAND_GREEN if abs(det) > 0.01 else BRAND_RED)
        det_lbl.next_to(mat_lbl, DOWN, buff=0.3)

    arrows = VGroup(e1_arrow, e2_arrow)
    scene.add(grid_dots, arrows, e1_lbl, e2_lbl)
    if label_text:
        ttl = Text(label_text, font_size=30, color=accent_color, weight=BOLD)
        ttl.move_to([pos[0], pos[1] + config.frame_height * 0.38, 0])
        scene.add(ttl)
    scene.add(mat_lbl)
    if det_lbl:
        scene.add(det_lbl)

    setup_t = max(min(duration * 0.25, 1.0), 0.2)
    transform_t = max(min(duration * 0.50, 2.5), 0.3)
    hold_t = max(duration - setup_t - transform_t, 0.1)

    scene.play(
        FadeIn(grid_dots), FadeIn(arrows), FadeIn(e1_lbl), FadeIn(e2_lbl),
        run_time=setup_t, rate_func=smooth,
    )
    anims = []
    for dot in grid_dots:
        anims.append(dot.animate.move_to(_transform_dot(dot)))
    anims.append(Transform(e1_arrow, e1_arrow_t))
    anims.append(Transform(e2_arrow, e2_arrow_t))
    scene.play(*anims, run_time=transform_t, rate_func=smooth)
    scene.wait(hold_t)
    return _sc.collected(), arrows


def fm_animate_derivative(scene, func=None, x_val=1.0, duration=5.0,
                            position=None, label_text="", accent_color=BRAND_GREEN,
                            x_range=(-3.0, 3.0), y_range=None):
    """Plots a curve and animates a tangent line sweeping to x_val,
    showing the derivative as the slope. func: Python callable f(x).
    Defaults to x^2. Returns (_sc.collected(), tangent_line)."""
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = _fmnp.array(position, dtype=float)

    if func is None:
        func = lambda x: x * x

    x_lo, x_hi = x_range
    n_pts = 80
    xs = [x_lo + i * (x_hi - x_lo) / (n_pts - 1) for i in range(n_pts)]
    try:
        ys = [float(func(x)) for x in xs]
    except Exception:
        ys = [x * x for x in xs]

    y_lo_data = min(ys)
    y_hi_data = max(ys)
    if y_range:
        y_lo_data, y_hi_data = y_range
    y_span = max(y_hi_data - y_lo_data, 1.0)

    plot_w = 9.0
    plot_h = 4.5

    def _to_screen(x, y):
        sx = pos[0] + (x - x_lo) / (x_hi - x_lo) * plot_w - plot_w / 2
        sy = pos[1] + (y - y_lo_data) / y_span * plot_h - plot_h / 2
        return [sx, sy, 0]

    x_ax = Line(
        [pos[0] - plot_w / 2, pos[1] - plot_h / 2, 0],
        [pos[0] + plot_w / 2, pos[1] - plot_h / 2, 0],
    )
    x_ax.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)
    y_ax = Line(
        [pos[0] - plot_w / 2, pos[1] - plot_h / 2, 0],
        [pos[0] - plot_w / 2, pos[1] + plot_h / 2, 0],
    )
    y_ax.set_stroke(BRAND_GRAY, width=2.0, opacity=0.5)

    screen_pts = [_to_screen(x, y) for x, y in zip(xs, ys)]
    curve = VMobject()
    _fm_set_line_smooth(curve, screen_pts)
    curve.set_stroke(accent_color, width=4.0, opacity=0.95)

    dx = (x_hi - x_lo) / (n_pts - 1) * 0.5
    try:
        slope = (func(x_val + dx) - func(x_val - dx)) / (2 * dx)
        y_at_x = func(x_val)
    except Exception:
        slope = 2 * x_val
        y_at_x = x_val * x_val

    contact_pt = _to_screen(x_val, y_at_x)
    tang_len = plot_w * 0.35
    norm = math.sqrt(1 + slope * slope * (plot_h / plot_w * (x_hi - x_lo) / y_span) ** 2)
    sx_scale = tang_len / max(norm, 0.01)

    screen_slope = slope * (plot_h / y_span) / (plot_w / (x_hi - x_lo))
    tang_dx = tang_len * 0.5
    tang_dy = screen_slope * tang_dx

    tangent = Line(
        [contact_pt[0] - tang_dx, contact_pt[1] - tang_dy, 0],
        [contact_pt[0] + tang_dx, contact_pt[1] + tang_dy, 0],
    )
    tangent.set_stroke(BRAND_RED, width=3.5, opacity=0.95)

    contact_dot = Dot(contact_pt, radius=0.14, color=BRAND_RED)
    contact_dot.set_fill(BRAND_RED, opacity=1.0)

    slope_str = f"slope = {slope:.2f}" if slope != int(slope) else f"slope = {int(slope)}"
    slope_lbl = Text(slope_str, font_size=30, color=BRAND_RED, weight=BOLD)
    slope_lbl.move_to([contact_pt[0] + 1.8, contact_pt[1] + 0.55, 0])
    if slope_lbl.get_right()[0] > config.frame_width / 2 - 0.3:
        slope_lbl.shift(LEFT * (slope_lbl.get_right()[0] - (config.frame_width / 2 - 0.3)))

    if label_text:
        ttl = Text(label_text, font_size=28, color=accent_color, weight=BOLD)
        ttl.next_to([pos[0], pos[1] + plot_h / 2, 0], UP, buff=0.18)
        scene.add(ttl)

    scene.add(x_ax, y_ax)
    draw_t = max(min(duration * 0.45, 1.8), 0.2)
    tang_t = max(min(duration * 0.25, 1.0), 0.2)
    hold_t = max(duration - draw_t - tang_t - 0.25, 0.05)

    scene.play(Create(curve), run_time=draw_t, rate_func=smooth)
    scene.play(
        GrowFromCenter(contact_dot),
        Create(tangent),
        run_time=tang_t, rate_func=smooth,
    )
    scene.play(FadeIn(slope_lbl, shift=UP * 0.12), run_time=0.25)
    scene.wait(hold_t)
    return _sc.collected(), tangent


def fm_animate_neural_network(scene, layer_sizes=None, duration=5.0,
                                position=None, label_text="",
                                accent_color=BRAND_GREEN,
                                highlight_path=True):
    """Draws a fully-connected neural network diagram and animates
    a forward-pass highlight through it.
    layer_sizes: list of ints e.g. [3,4,4,2]. Returns (_sc.collected(), nodes)."""
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = _fmnp.array(position, dtype=float)
    if layer_sizes is None:
        layer_sizes = [3, 4, 3, 2]

    layer_sizes = [max(1, min(int(s), 8)) for s in layer_sizes[:6]]
    n_layers = len(layer_sizes)

    h_gap = min(2.8, 10.0 / max(n_layers, 1))
    v_gap = min(1.1, 5.5 / max(max(layer_sizes), 1))
    node_r = min(0.22, v_gap * 0.38)

    layer_xs = [pos[0] - (n_layers - 1) * h_gap / 2 + i * h_gap
                for i in range(n_layers)]

    all_nodes = []
    for li, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
        nodes_in_layer = []
        for ni in range(n_nodes):
            ny = pos[1] + (ni - (n_nodes - 1) / 2) * v_gap
            node = Circle(radius=node_r)
            if li == 0:
                node.set_fill(BRAND_GOLD, opacity=0.85)
                node.set_stroke(BRAND_GOLD, width=2.0)
            elif li == n_layers - 1:
                node.set_fill(BRAND_GREEN, opacity=0.85)
                node.set_stroke(BRAND_GREEN, width=2.0)
            else:
                node.set_fill(BRAND_PANEL, opacity=0.95)
                node.set_stroke(accent_color, width=2.0, opacity=0.7)
            node.move_to([lx, ny, 0])
            nodes_in_layer.append(node)
        all_nodes.append(nodes_in_layer)

    edges = VGroup()
    for li in range(n_layers - 1):
        for src in all_nodes[li]:
            for dst in all_nodes[li + 1]:
                e = Line(src.get_center(), dst.get_center())
                e.set_stroke(BRAND_GRAY, width=0.8, opacity=0.22)
                edges.add(e)

    node_group = VGroup(*[n for layer in all_nodes for n in layer])

    layer_names = ["Input"] + ["Hidden"] * (n_layers - 2) + ["Output"]
    lbl_group = VGroup()
    for lx, name in zip(layer_xs, layer_names):
        bottom_y = pos[1] - max(layer_sizes) * v_gap / 2 - 0.38
        lbl = Text(name, font_size=20, color=BRAND_GRAY)
        lbl.move_to([lx, bottom_y, 0])
        lbl_group.add(lbl)

    if label_text:
        ttl = Text(label_text, font_size=30, color=accent_color, weight=BOLD)
        ttl.move_to([pos[0], pos[1] + max(layer_sizes) * v_gap / 2 + 0.55, 0])
        scene.add(ttl)

    scene.add(edges, node_group, lbl_group)

    setup_t = max(min(duration * 0.35, 1.5), 0.2)
    scene.play(
        FadeIn(edges),
        LaggedStart(*[GrowFromCenter(n) for n in node_group], lag_ratio=0.04),
        FadeIn(lbl_group),
        run_time=setup_t, rate_func=smooth,
    )

    if highlight_path and n_layers > 1:
        path_t = max(min(duration * 0.50, 2.5), 0.3)
        hold_t = max(duration - setup_t - path_t, 0.1)
        chosen = [_fm_random.choice(layer) for layer in all_nodes]
        pulses = []
        for node in chosen:
            glow = Circle(radius=node_r * 1.6)
            glow.set_stroke(accent_color, width=3.0, opacity=0.8)
            glow.set_fill(accent_color, opacity=0.18)
            glow.move_to(node.get_center())
            pulses.append(glow)
        edge_highlights = []
        for i in range(len(chosen) - 1):
            he = Line(chosen[i].get_center(), chosen[i + 1].get_center())
            he.set_stroke(accent_color, width=3.0, opacity=0.9)
            edge_highlights.append(he)
        scene.play(
            LaggedStart(
                *[FadeIn(p) for p in pulses],
                *[Create(e) for e in edge_highlights],
                lag_ratio=0.18,
            ),
            run_time=path_t, rate_func=smooth,
        )
        scene.wait(hold_t)
    else:
        scene.wait(max(duration - setup_t, 0.1))

    return _sc.collected(), node_group


def fm_animate_attention_heatmap(scene, matrix=None, row_labels=None,
                                  col_labels=None, duration=4.5,
                                  position=None, label_text="",
                                  accent_color=BRAND_GREEN):
    """Animated attention/correlation heatmap. matrix: 2D list of floats 0-1.
    Cells animate in with color intensity proportional to value.
    Returns (_sc.collected(), cell_group)."""
    _sc = _Tracker(scene)
    scene = _sc
    if position is None:
        position = ORIGIN
    pos = _fmnp.array(position, dtype=float)

    if matrix is None:
        matrix = [
            [0.9, 0.1, 0.05, 0.1],
            [0.2, 0.7, 0.15, 0.2],
            [0.1, 0.2, 0.85, 0.1],
            [0.05, 0.1, 0.1, 0.9],
        ]

    n_rows = len(matrix)
    n_cols = max(len(r) for r in matrix)

    safe_w = config.frame_width * 0.72
    safe_h = config.frame_height * 0.68
    cell_w = min(safe_w / max(n_cols, 1), safe_h / max(n_rows, 1), 1.6)
    cell_h = cell_w
    total_w = cell_w * n_cols
    total_h = cell_h * n_rows

    if row_labels is None:
        row_labels = [f"Q{i+1}" for i in range(n_rows)]
    if col_labels is None:
        col_labels = [f"K{j+1}" for j in range(n_cols)]

    cells = VGroup()
    val_texts = VGroup()
    for ri, row in enumerate(matrix):
        for ci, val in enumerate(row):
            v = max(0.0, min(1.0, float(val)))
            x = pos[0] - total_w / 2 + ci * cell_w + cell_w / 2
            y = pos[1] + total_h / 2 - ri * cell_h - cell_h / 2

            r_int = int(int(accent_color[1:3], 16) * v + 13 * (1 - v))
            g_int = int(int(accent_color[3:5], 16) * v + 27 * (1 - v))
            b_int = int(int(accent_color[5:7], 16) * v + 42 * (1 - v))
            cell_color = f"#{r_int:02x}{g_int:02x}{b_int:02x}"

            cell = RoundedRectangle(width=cell_w * 0.92, height=cell_h * 0.92,
                                    corner_radius=0.06)
            cell.set_fill(cell_color, opacity=min(0.3 + v * 0.7, 1.0))
            cell.set_stroke(BRAND_GRAY, width=0.6, opacity=0.3)
            cell.move_to([x, y, 0])
            cells.add(cell)

            if cell_w > 0.8:
                vt = Text(f"{v:.2f}", font_size=max(int(cell_w * 14), 14),
                          color=BRAND_WHITE if v > 0.4 else BRAND_GRAY)
                vt.move_to([x, y, 0])
                val_texts.add(vt)

    row_lbl_grp = VGroup()
    for ri, rl in enumerate(row_labels):
        y = pos[1] + total_h / 2 - ri * cell_h - cell_h / 2
        lbl = Text(str(rl), font_size=max(int(cell_w * 13), 14), color=BRAND_GRAY)
        lbl.move_to([pos[0] - total_w / 2 - 0.4, y, 0])
        row_lbl_grp.add(lbl)

    col_lbl_grp = VGroup()
    for ci, cl in enumerate(col_labels):
        x = pos[0] - total_w / 2 + ci * cell_w + cell_w / 2
        lbl = Text(str(cl), font_size=max(int(cell_w * 13), 14), color=BRAND_GRAY)
        lbl.move_to([x, pos[1] + total_h / 2 + 0.35, 0])
        col_lbl_grp.add(lbl)

    if label_text:
        ttl = Text(label_text, font_size=30, color=accent_color, weight=BOLD)
        ttl.move_to([pos[0], pos[1] + total_h / 2 + 0.8, 0])
        scene.add(ttl)

    scene.add(row_lbl_grp, col_lbl_grp)

    intro_t = max(min(duration * 0.15, 0.6), 0.1)
    cell_t = max(min(duration * 0.60, 2.5), 0.3)
    hold_t = max(duration - intro_t - cell_t - 0.25, 0.05)

    scene.play(FadeIn(row_lbl_grp), FadeIn(col_lbl_grp), run_time=intro_t)
    scene.play(
        LaggedStart(*[FadeIn(c, scale=0.7) for c in cells], lag_ratio=0.04),
        run_time=cell_t, rate_func=smooth,
    )
    if val_texts:
        scene.play(LaggedStart(*[FadeIn(t) for t in val_texts], lag_ratio=0.03),
                   run_time=0.25)
    scene.wait(hold_t)
    return _sc.collected(), cells