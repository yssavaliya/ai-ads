from __future__ import annotations

from scripts.publish_buffer import auto_publish_block_reason


def test_auto_publish_allows_rendered_mp4_reel():
    assert auto_publish_block_reason(
        {
            "asset_type": "reel",
            "asset_path": "assets/aimagix-studio/2026-11-07/reel.mp4",
            "warnings": [],
        }
    ) is None


def test_auto_publish_allows_site_manifest_video_asset():
    assert auto_publish_block_reason(
        {
            "asset": {
                "type": "reel",
                "video": "assets/aimagix-studio/2026-11-07/reel.mp4",
                "thumb": "assets/aimagix-studio/2026-11-07/reel-poster.jpg",
            },
            "warnings": [],
        }
    ) is None


def test_auto_publish_blocks_non_mp4_reels():
    reason = auto_publish_block_reason(
        {"asset_type": "reel", "asset_path": "assets/reel.mov", "warnings": []}
    )
    assert reason == "Buffer Reel publishing requires a rendered MP4 asset"


def test_auto_publish_blocks_known_failed_render_even_for_reel():
    reason = auto_publish_block_reason(
        {
            "asset_type": "reel",
            "asset_path": "assets/reel.mp4",
            "warnings": ["Reel render failed; falling back to single image"],
        }
    )
    assert reason is not None
    assert reason.startswith("render QA failed:")
