from services import fpl_api


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_bootstrap_is_not_permanently_process_cached(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response({"call": len(calls)})

    monkeypatch.setattr(fpl_api._SESSION, "get", fake_get)

    assert fpl_api.bootstrap_static() == {"call": 1}
    assert fpl_api.bootstrap_static() == {"call": 2}
    assert len(calls) == 2
    assert calls[0][1] == (
        fpl_api.DEFAULT_CONNECT_TIMEOUT_SECONDS,
        fpl_api.DEFAULT_READ_TIMEOUT_SECONDS,
    )


def test_session_retries_transient_get_failures():
    adapter = fpl_api._SESSION.get_adapter("https://")

    assert adapter.max_retries.total == 1
    assert adapter.max_retries.respect_retry_after_header is False
    assert 429 in adapter.max_retries.status_forcelist
    assert 503 in adapter.max_retries.status_forcelist
