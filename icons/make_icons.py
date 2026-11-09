"""
Generate the application icons: "PyABR3" in white on Carolina Blue, with a black
ABR waveform behind the lettering. Run from anywhere:

    uv run python icons/make_icons.py

Writes pyabr3.png, pyabr3.ico (Windows) and pyabr3.icns (macOS) in this directory.
"""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAROLINA_BLUE = (75, 156, 211)  # #4B9CD3
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SIZE = 1024
SUPERSAMPLE = 4
TEXT = "PyABR3"
TEXT_WIDTH_FRACTION = 0.88
LINE_WIDTH_FRACTION = 0.03
WAVE_POINTS = 6000
HALO_WIDTH_FRACTION = 0.006  # blue edge that keeps the black line off the letters
ICO_SIZES = [(s, s) for s in [16, 24, 32, 48, 64, 128, 256]]

# (latency, height, width): waves I to V with their following troughs, arbitrary units
ABR_COMPONENTS = [
    (1.4, 0.55, 0.22),
    (2.0, -0.35, 0.2),
    (2.7, 0.35, 0.2),
    (3.3, -0.3, 0.2),
    (3.9, 0.7, 0.24),
    (4.6, -0.3, 0.2),
    (5.2, 0.5, 0.2),
    (5.9, 1.0, 0.3),
    (7.0, -1.0, 0.55),
]


def abr_waveform(n_points: int):
    t = np.linspace(0.0, 9.0, n_points)
    y = np.zeros(n_points)
    for latency, height, width in ABR_COMPONENTS:
        y += height * np.exp(-0.5 * ((t - latency) / width) ** 2)
    return t / t[-1], y / np.max(np.abs(y))


def fit_font(font_path: str, text: str, target_width: float):
    """Largest font size at which the text fits the target width."""
    size = 10
    while ImageFont.truetype(font_path, size + 2).getlength(text) < target_width:
        size += 2
    return ImageFont.truetype(font_path, size)


def draw_master() -> Image.Image:
    canvas = SIZE * SUPERSAMPLE
    image = Image.new("RGB", (canvas, canvas), CAROLINA_BLUE)
    draw = ImageDraw.Draw(image)

    # a line of overlapping discs: wide polylines from PIL have gaps at the bends
    x, y = abr_waveform(WAVE_POINTS)
    radius = canvas * LINE_WIDTH_FRACTION / 2.0
    for px, py in zip(canvas * x, canvas * (0.5 - 0.42 * y)):
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=BLACK)

    font_path = str(Path(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    font = fit_font(font_path, TEXT, canvas * TEXT_WIDTH_FRACTION)
    draw.text(
        (canvas / 2, canvas / 2),
        TEXT,
        font=font,
        anchor="mm",
        fill=WHITE,
        stroke_width=int(canvas * HALO_WIDTH_FRACTION),
        stroke_fill=CAROLINA_BLUE,
    )
    return image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def main():
    icon_dir = Path(__file__).parent
    master = draw_master()
    master.save(icon_dir / "pyabr3.png")
    master.save(icon_dir / "pyabr3.ico", sizes=ICO_SIZES)
    master.save(icon_dir / "pyabr3.icns")


if __name__ == "__main__":
    main()
