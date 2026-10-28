from __future__ import annotations

from unittest.mock import MagicMock, patch
import httpx
from src.delivery.buffer_publisher import BufferPublisher


def test_buffer_publisher_get_public_asset_url():
    pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
    url = pub.get_public_asset_url("assets/aimagix-studio/2026-10-29/card.jpg")
    assert "https://raw.githubusercontent.com" in url
    assert url.endswith("assets/aimagix-studio/2026-10-29/card.jpg")


def test_buffer_publisher_schedule_post_success():
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": {
            "createPost": {
                "__typename": "PostActionSuccess",
                "post": {"id": "up1", "status": "draft", "dueAt": None},
            }
        }
    }

    with patch("httpx.Client.post", return_value=fake_response):
        pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
        res = pub.schedule_post(text="Hello Buffer Ad", image_url="https://example.com/ad.jpg")
        assert res["success"] is True
        assert res["posts"][0]["id"] == "up1"


def test_buffer_publisher_send_daily_prompts():
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": {
            "createPost": {
                "__typename": "PostActionSuccess",
                "post": {"id": "up1", "status": "draft", "dueAt": None},
            }
        }
    }

    with patch("httpx.Client.post", return_value=fake_response):
        pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
        business = {"name": "Aimagix Studio"}
        prompts = [
            {
                "title": "Ad 1",
                "score": 60,
                "caption": "Lower score ad",
                "asset_path": "assets/aimagix-studio/2026-10-29/1.jpg",
            },
            {
                "title": "Ad 2",
                "score": 85,
                "caption": "Winning high score ad",
                "hashtags": ["#marketing", "#viral"],
                "asset_path": "assets/aimagix-studio/2026-10-29/2.jpg",
            },
        ]
        delivered = pub.send_daily_prompts(business, prompts)
        assert delivered is True
