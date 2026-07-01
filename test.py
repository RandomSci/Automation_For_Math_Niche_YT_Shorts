from manim import *
import time


class FinanceBeatsTest(Scene):
    """10 beats back to back, deliberately chosen to mirror real
    finance_explainer.py content: number counters, bar comparisons,
    a donut/percentage fill, a line chart growing, and a shape morph
    (Transform) between beats -- not trivial shapes, so the timing
    result actually reflects the real workload."""

    def construct(self):
        self.camera.background_color = "#071018"
        start_wall_time = time.time()

        title = Text("WARNING SIGNS", font_size=60, color="#F5F7FA")
        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.6)
        self.play(FadeOut(title), run_time=0.4)

        counter_label = Text("$200", font_size=96, color="#F5F7FA")
        self.play(Write(counter_label), run_time=0.8)
        target = Text("$34,000", font_size=96, color="#38D996")
        self.play(Transform(counter_label, target), run_time=1.0)
        self.play(FadeOut(counter_label), run_time=0.4)

        bar1 = Rectangle(width=1.5, height=0.1, color="#FF4D4D", fill_opacity=1).shift(LEFT * 2)
        bar2 = Rectangle(width=1.5, height=0.1, color="#38D996", fill_opacity=1).shift(RIGHT * 2)
        self.play(
            bar1.animate.stretch_to_fit_height(3).align_to(bar1, DOWN),
            bar2.animate.stretch_to_fit_height(1.2).align_to(bar2, DOWN),
            run_time=1.2,
        )
        self.play(FadeOut(bar1), FadeOut(bar2), run_time=0.4)

        donut = Annulus(inner_radius=1.0, outer_radius=1.4, color="#FFD166", fill_opacity=1)
        pct_text = Text("60%", font_size=48, color="#F5F7FA")
        self.play(Create(donut), Write(pct_text), run_time=1.2)
        self.play(FadeOut(donut), FadeOut(pct_text), run_time=0.4)

        points = [
            [-4, -1, 0], [-2.5, -0.4, 0], [-1, 0.2, 0],
            [0.5, 0.6, 0], [2, 1.3, 0], [4, 2.2, 0],
        ]
        line_graph = VMobject(color="#38D996")
        line_graph.set_points_as_corners([points[0]])
        self.add(line_graph)
        for i in range(1, len(points)):
            new_line = VMobject(color="#38D996")
            new_line.set_points_as_corners(points[: i + 1])
            self.play(Transform(line_graph, new_line), run_time=0.3)
        self.play(FadeOut(line_graph), run_time=0.4)

        square = Square(color="#8A94A6", fill_opacity=0.8).shift(LEFT * 2)
        circle = Circle(color="#FFD166", fill_opacity=0.8).shift(RIGHT * 2)
        self.play(Transform(square, circle), run_time=1.2)
        self.play(FadeOut(square), run_time=0.4)

        end_wall_time = time.time()
        print(f"\n>>> CONSTRUCT WALL TIME (excludes final encode): {end_wall_time - start_wall_time:.2f}s\n")