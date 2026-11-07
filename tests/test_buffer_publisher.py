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

    mock_post = MagicMock(return_value=fake_response)
    with patch("httpx.Client.post", mock_post):
        pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
        res = pub.schedule_post(text="Hello Buffer Ad", image_url="https://example.com/ad.jpg")
        assert res["success"] is True
        assert res["posts"][0]["id"] == "up1"

    payload = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert payload["assets"] == [{"image": {"url": "https://example.com/ad.jpg"}}]
    assert payload["metadata"]["instagram"]["type"] == "post"


def test_buffer_publisher_schedule_reel_uses_video_asset_and_reel_metadata():
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": {
            "createPost": {
                "__typename": "PostActionSuccess",
                "post": {"id": "reel1", "status": "scheduled", "dueAt": None},
            }
        }
    }
    mock_post = MagicMock(return_value=fake_response)

    with patch("httpx.Client.post", mock_post):
        pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
        result = pub.schedule_post(text="A short product film", video_url="https://example.com/reel.mp4")

    payload = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert result["posts"][0]["id"] == "reel1"
    assert payload["assets"] == [{"video": {"url": "https://example.com/reel.mp4"}}]
    assert payload["metadata"]["instagram"]["type"] == "reel"


def test_buffer_publisher_rejects_image_and_video_in_same_post():
    pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
    try:
        pub.schedule_post(
            text="Not a supported mixed post",
            image_url="https://example.com/ad.jpg",
            video_url="https://example.com/reel.mp4",
        )
    except ValueError as exc:
        assert "either an image or a video" in str(exc)
    else:
        raise AssertionError("Expected mixed media to be rejected")


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


def test_buffer_publisher_send_daily_reel_uses_site_video_url():
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": {
            "createPost": {
                "__typename": "PostActionSuccess",
                "post": {"id": "reel1", "status": "scheduled", "dueAt": None},
            }
        }
    }
    mock_post = MagicMock(return_value=fake_response)
    prompt = {
        "title": "Product story",
        "score": 90,
        "caption": "A tiny product film.",
        "asset_type": "reel",
        "asset": {
            "type": "reel",
            "video": "assets/aimagix-studio/2026-11-07/reel.mp4",
            "thumb": "assets/aimagix-studio/2026-11-07/reel-poster.jpg",
        },
    }

    with patch("httpx.Client.post", mock_post):
        pub = BufferPublisher(access_token="test_token", profile_ids=["pid1"])
        assert pub.send_daily_prompts({"name": "Aimagix Studio"}, [prompt]) is True

    payload = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert payload["assets"] == [{
        "video": {"url": "https://raw.githubusercontent.com/yssavaliya/ai-ads/gh-pages/"
                         "assets/aimagix-studio/2026-11-07/reel.mp4"}
    }]
    assert payload["metadata"]["instagram"]["type"] == "reel"
