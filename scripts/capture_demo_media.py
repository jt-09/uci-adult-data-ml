"""Capture Gradio demo screenshots and a walkthrough video for the README.

Requires the demo server running at http://127.0.0.1:7860 (python demo/app.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

DEMO_URL = "http://127.0.0.1:7860"
ASSETS = Path(__file__).resolve().parents[1] / "demo" / "assets"
VIEWPORT = {"width": 1280, "height": 900}


def _fill_number(page, label: str, value: str) -> None:
    page.get_by_role("spinbutton", name=label).fill(value)


def _select_dropdown(page, label: str, value: str) -> None:
    combo = page.get_by_role("combobox", name=label)
    combo.click()
    page.get_by_role("option", name=value, exact=True).click()


def _predict_and_wait(page, wait_ms: int = 4000) -> None:
    page.get_by_role("button", name="Predict").click()
    page.wait_for_timeout(wait_ms)


def capture() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    video_dir = ASSETS / "_video_tmp"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size={"width": VIEWPORT["width"], "height": VIEWPORT["height"]},
        )
        page = context.new_page()
        page.goto(DEMO_URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1500)

        page.screenshot(path=str(ASSETS / "demo-01-home.png"), full_page=True)

        # Trial 1: default median profile
        _predict_and_wait(page)
        page.screenshot(path=str(ASSETS / "demo-02-default-prediction.png"), full_page=True)

        # Trial 2: high-income profile
        _fill_number(page, "age", "52")
        _fill_number(page, "education-num", "14")
        _fill_number(page, "capital-gain", "15000")
        _fill_number(page, "hours-per-week", "60")
        _select_dropdown(page, "education", "Masters")
        _select_dropdown(page, "occupation", "Exec-managerial")
        _select_dropdown(page, "marital-status", "Married-civ-spouse")
        page.wait_for_timeout(500)
        _predict_and_wait(page)
        page.screenshot(path=str(ASSETS / "demo-03-high-income-trial.png"), full_page=True)

        # Trial 3: low-income profile
        _fill_number(page, "age", "19")
        _fill_number(page, "education-num", "9")
        _fill_number(page, "capital-gain", "0")
        _fill_number(page, "hours-per-week", "15")
        _select_dropdown(page, "education", "HS-grad")
        _select_dropdown(page, "occupation", "Handlers-cleaners")
        _select_dropdown(page, "workclass", "Private")
        _select_dropdown(page, "marital-status", "Never-married")
        page.wait_for_timeout(500)
        _predict_and_wait(page)
        page.screenshot(path=str(ASSETS / "demo-04-low-income-trial.png"), full_page=True)

        page.wait_for_timeout(1000)
        video_path = Path(page.video.path()) if page.video else None
        context.close()
        browser.close()

    if video_path and video_path.exists():
        dest = ASSETS / "demo-walkthrough.webm"
        shutil.move(str(video_path), dest)
        shutil.rmtree(video_dir, ignore_errors=True)
        print(f"Video saved: {dest}")
    else:
        print("Warning: no video file produced")

    for name in sorted(ASSETS.glob("demo-*.png")):
        print(f"Screenshot: {name}")


if __name__ == "__main__":
    capture()
