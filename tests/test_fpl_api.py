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


def test_session_retries_transient_get_failures():
    adapter = fpl_api._SESSION.get_adapter("https://")

    assert adapter.max_retries.total == 3
    assert 429 in adapter.max_retries.status_forcelist
    assert 503 in adapter.max_retries.status_forcelist
