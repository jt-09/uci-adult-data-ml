"""Capture Gradio demo screenshots and a walkthrough video for the README.

Requires the demo server running at http://127.0.0.1:7860 (python demo/app.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

DEMO_URL = "http://127.0.0.1:7860"
ASSETS = Path(__file__).resolve().parents[1] / "demo" / "assets"
VIEWPORT = {"width": 1280, "height": 900}
RESULTS_HOLD_MS = 5500
FORM_HOLD_MS = 2000


def _fill_number(page: Page, label: str, value: str) -> None:
    page.get_by_role("spinbutton", name=label).fill(value)


def _select_dropdown(page: Page, label: str, value: str) -> None:
    combo = page.get_by_role("combobox", name=label)
    combo.click()
    page.get_by_role("option", name=value, exact=True).click()


def _scroll_to_top(page: Page) -> None:
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    page.wait_for_timeout(800)


def _wait_for_results(page: Page, timeout_ms: int = 45_000) -> None:
    page.get_by_text("Predicted income", exact=False).wait_for(state="visible", timeout=timeout_ms)
    page.get_by_text("Top contributing features", exact=False).wait_for(
        state="visible", timeout=timeout_ms
    )
    page.get_by_text("P(>50K)", exact=False).wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_timeout(500)


def _show_results(page: Page, hold_ms: int = RESULTS_HOLD_MS) -> None:
    """Scroll the prediction + SHAP block into view and pause for the recording."""
    heading = page.get_by_role("heading", name="Top contributing features")
    heading.scroll_into_view_if_needed()
    page.wait_for_timeout(400)
    page.evaluate("window.scrollBy({top: -80, behavior: 'smooth'})")
    page.wait_for_timeout(hold_ms)


def _predict_and_show_results(page: Page, screenshot_path: Path | None = None) -> None:
    predict_btn = page.get_by_role("button", name="Predict")
    predict_btn.scroll_into_view_if_needed()
    page.wait_for_timeout(400)
    predict_btn.click()
    _wait_for_results(page)
    _show_results(page)
    if screenshot_path is not None:
        page.screenshot(path=str(screenshot_path), full_page=False)


def _apply_high_income_profile(page: Page) -> None:
    _scroll_to_top(page)
    _fill_number(page, "age", "52")
    _fill_number(page, "education-num", "14")
    _fill_number(page, "capital-gain", "15000")
    _fill_number(page, "hours-per-week", "60")
    _select_dropdown(page, "education", "Masters")
    _select_dropdown(page, "occupation", "Exec-managerial")
    _select_dropdown(page, "marital-status", "Married-civ-spouse")
    page.wait_for_timeout(FORM_HOLD_MS)


def _apply_low_income_profile(page: Page) -> None:
    _scroll_to_top(page)
    _fill_number(page, "age", "19")
    _fill_number(page, "education-num", "9")
    _fill_number(page, "capital-gain", "0")
    _fill_number(page, "hours-per-week", "15")
    _select_dropdown(page, "education", "HS-grad")
    _select_dropdown(page, "occupation", "Handlers-cleaners")
    _select_dropdown(page, "workclass", "Private")
    _select_dropdown(page, "marital-status", "Never-married")
    page.wait_for_timeout(FORM_HOLD_MS)


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
        _scroll_to_top(page)
        page.wait_for_timeout(FORM_HOLD_MS)
        page.screenshot(path=str(ASSETS / "demo-01-home.png"), full_page=True)

        # Trial 1: default median profile -> results + SHAP
        _predict_and_show_results(page, ASSETS / "demo-02-default-prediction.png")

        # Trial 2: high-income profile
        _apply_high_income_profile(page)
        _predict_and_show_results(page, ASSETS / "demo-03-high-income-trial.png")

        # Trial 3: low-income profile
        _apply_low_income_profile(page)
        _predict_and_show_results(page, ASSETS / "demo-04-low-income-trial.png")

        page.wait_for_timeout(1000)
        video_path = Path(page.video.path()) if page.video else None
        context.close()
        browser.close()

    if video_path and video_path.exists():
        dest = ASSETS / "demo-walkthrough.webm"
        if dest.exists():
            dest.unlink()
        shutil.move(str(video_path), dest)
        shutil.rmtree(video_dir, ignore_errors=True)
        print(f"Video saved: {dest}")
    else:
        print("Warning: no video file produced")

    for name in sorted(ASSETS.glob("demo-*.png")):
        print(f"Screenshot: {name}")


if __name__ == "__main__":
    capture()
