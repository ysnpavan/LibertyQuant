"""Tests for the notifications layer."""

from unittest.mock import patch

from liberty_quant.notifications.notifier import (
    ConsoleNotifier,
    MultiNotifier,
    Notifier,
    create_notifier,
    notification_event,
)


class RecordingNotifier(Notifier):
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def send(self, message: str) -> bool:
        if self.fail:
            return False
        self.messages.append(message)
        return True


def test_console_notifier_always_succeeds():
    assert ConsoleNotifier().send("hello") is True


def test_trade_executed_formatting():
    n = RecordingNotifier()
    n.trade_executed(side="buy", qty=10, symbol="NVDA", price=200.5, strategy="momentum", confidence=0.75)
    assert len(n.messages) == 1
    assert "BUY" in n.messages[0]
    assert "NVDA" in n.messages[0]
    assert "$200.50" in n.messages[0]
    assert "75%" in n.messages[0]


def test_position_closed_shows_pnl_direction():
    n = RecordingNotifier()
    n.position_closed(symbol="AMD", qty=5, exit_price=150.0, pnl=-42.5, reason="stop_loss")
    assert "🔴" in n.messages[0]
    assert "-42.50" in n.messages[0]

    n.position_closed(symbol="AMD", qty=5, exit_price=170.0, pnl=57.5, reason="take_profit")
    assert "✅" in n.messages[1]


def test_multi_notifier_succeeds_if_any_channel_works():
    good, bad = RecordingNotifier(), RecordingNotifier(fail=True)
    multi = MultiNotifier([bad, good])
    assert multi.send("test") is True
    assert good.messages == ["test"]


def test_notification_event_dispatch():
    n = RecordingNotifier()
    notification_event(n, "kill_switch_engaged", {"reason": "daily loss limit"})
    assert "KILL SWITCH" in n.messages[0]


def test_notification_event_never_raises():
    n = RecordingNotifier()
    # Unknown event: ignored
    notification_event(n, "nonexistent_event", {})
    # Bad payload: swallowed
    notification_event(n, "trade_executed", {"bad_kwarg": 1})
    assert True  # reaching here is the test


def test_create_notifier_console_only_by_default():
    notifier = create_notifier()
    assert isinstance(notifier, ConsoleNotifier)


def test_whatsapp_notifier_posts_to_twilio(monkeypatch, tmp_path):
    monkeypatch.setenv("LQ_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token_test")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    monkeypatch.setenv("WHATSAPP_TO", "whatsapp:+911234567890")
    monkeypatch.setenv("LQ_DB_PATH", str(tmp_path / "t.db"))

    from liberty_quant.config import reload_settings
    reload_settings()

    from liberty_quant.notifications.notifier import WhatsAppNotifier

    class FakeResponse:
        status_code = 201
        text = "ok"

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs["data"]
        return FakeResponse()

    with patch("liberty_quant.notifications.notifier.httpx.post", side_effect=fake_post):
        notifier = WhatsAppNotifier()
        assert notifier.send("test message") is True

    assert "AC_test" in captured["url"]
    assert captured["data"]["Body"] == "test message"
    assert captured["data"]["To"] == "whatsapp:+911234567890"
