import os

from src.shared.config import Settings


def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "")
    monkeypatch.setenv("OPENCODE_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("SMTP_TLS", "")
    s = Settings()
    assert s.smtp_port == 587
    assert s.opencode_timeout_seconds == 120
    assert s.smtp_tls is True


def test_real_values_parsed(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("OPENCODE_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("SMTP_TLS", "false")
    s = Settings()
    assert s.smtp_port == 465
    assert s.opencode_timeout_seconds == 300
    assert s.smtp_tls is False