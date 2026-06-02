"""Shared test fixtures. Sets fake credentials so Settings loads without a .env."""

import os

import pytest

# Must be set before any liberty_quant import triggers Settings()
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("LQ_EXECUTION_MODE", "local_paper")
os.environ.setdefault("LQ_DB_PATH", "/tmp/liberty_quant_test/test.db")


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Each test gets isolated storage paths and fresh settings."""
    monkeypatch.setenv("LQ_DB_PATH", str(tmp_path / "test.db"))
    from liberty_quant.config import reload_settings

    reload_settings()
    yield
    reload_settings()
