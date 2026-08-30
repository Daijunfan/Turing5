#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    growth = list(csv.DictReader((ROOT / "results/kernel_growth.csv").open()))
    width, height, margin = 800, 360, 50
    points = []
    for row in growth:
        x = margin + (int(row["horizon"]) - 1) * (width - 2 * margin) / 19
        y = height - margin - (int(row["kernel_size"]) - 1) * 70
        points.append(f"{x:.1f},{y:.1f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
<polyline fill="none" stroke="#0969da" stroke-width="3" points="{' '.join(points)}"/>
<text x="{width/2}" y="{height-10}" text-anchor="middle">Horizon</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">|K_H|</text>
<text x="{width/2}" y="28" text-anchor="middle" font-size="18">MORPH-N finite-horizon executable kernels</text>
</svg>"""
    (ROOT / "results/kernel_growth.svg").write_text(svg)
    print("wrote results/kernel_growth.svg")


if __name__ == "__main__":
    main()
