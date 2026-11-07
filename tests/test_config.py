import os

from src.shared.config import Settings


def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "")
    monkeypatch.setenv("SMTP_TLS", "")
    monkeypatch.setenv("DRY_RUN", "")
    s = Settings()
    assert s.smtp_port == 587
    assert s.smtp_tls is True
    assert s.dry_run is False


def test_generation_defaults_use_open_source_model_providers():
    s = Settings(_env_file=None)
    assert s.llm_provider_order == "huggingface"
    assert s.hf_llm_model == "Qwen/Qwen2.5-7B-Instruct"
    assert s.image_provider_order == "huggingface"


def test_real_values_parsed(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_TLS", "false")
    monkeypatch.setenv("DRY_RUN", "true")
    s = Settings()
    assert s.smtp_port == 465
    assert s.smtp_tls is False
    assert s.dry_run is True


def test_env_example_only_uses_real_settings():
    """A typo in .env.example (e.g. a renamed variable) silently does nothing."""
    from pathlib import Path

    from src.shared.config import Settings

    fields = {name.upper() for name in Settings.model_fields}
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    names = {line.split("=", 1)[0].strip() for line in example.splitlines() if "=" in line and not line.lstrip().startswith("#")}
    assert names and not (names - fields), sorted(names - fields)
