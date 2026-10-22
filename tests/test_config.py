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


def test_real_values_parsed(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_TLS", "false")
    monkeypatch.setenv("DRY_RUN", "true")
    s = Settings()
    assert s.smtp_port == 465
    assert s.smtp_tls is False
    assert s.dry_run is True
