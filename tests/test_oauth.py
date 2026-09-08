"""Tests for the custom-OAuth (authorization code + PKCE) implementation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import stat
import time
import urllib.error
import urllib.parse
from unittest.mock import patch

import pytest

from ucode import oauth

HOST = "https://dbc-test.cloud.databricks.com"
CLIENT_ID = "abc-client-id"
ACCOUNT_ID = "02945107-4221-4317-9276-5e0e9ed7f194"


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


def _http_error(status: int, payload: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/oidc/v1/token",
        status,
        "error",
        {},  # type: ignore[arg-type]
        io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def _posted_fields(mock_urlopen) -> dict[str, str]:
    """Decode the form body of the request the module sent."""
    request = mock_urlopen.call_args[0][0]
    return {
        key: values[0]
        for key, values in urllib.parse.parse_qs(request.data.decode("utf-8")).items()
    }


class TestGeneratePkcePair:
    def test_challenge_is_unpadded_base64url_sha256_of_verifier(self):
        verifier, challenge = oauth.generate_pkce_pair()
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        assert challenge == expected.rstrip(b"=").decode()

    def test_challenge_carries_no_base64_padding(self):
        # Databricks rejects a padded challenge, and the padding is the easy bug.
        _, challenge = oauth.generate_pkce_pair()
        assert "=" not in challenge

    def test_verifier_length_is_within_rfc7636_range(self):
        verifier, _ = oauth.generate_pkce_pair()
        assert 43 <= len(verifier) <= 128

    def test_each_call_is_unique(self):
        assert oauth.generate_pkce_pair()[0] != oauth.generate_pkce_pair()[0]


class TestEndpoints:
    def test_workspace_endpoints(self):
        assert oauth.token_endpoint(HOST) == f"{HOST}/oidc/v1/token"
        assert oauth.authorize_endpoint(HOST) == f"{HOST}/oidc/v1/authorize"

    def test_account_endpoints_include_account_id(self):
        host = "https://accounts.cloud.databricks.com"
        assert (
            oauth.token_endpoint(host, ACCOUNT_ID) == f"{host}/oidc/accounts/{ACCOUNT_ID}/v1/token"
        )

    def test_trailing_slash_does_not_double_up(self):
        assert oauth.token_endpoint(f"{HOST}/") == f"{HOST}/oidc/v1/token"


class TestBuildAuthorizeUrl:
    def _params(self, **kwargs) -> dict[str, str]:
        url = oauth.build_authorize_url(
            HOST, CLIENT_ID, code_challenge="chal", state="st", **kwargs
        )
        return {
            key: values[0]
            for key, values in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()
        }

    def test_requests_s256_pkce_and_a_code(self):
        params = self._params()
        assert params["code_challenge_method"] == "S256"
        assert params["code_challenge"] == "chal"
        assert params["response_type"] == "code"
        assert params["client_id"] == CLIENT_ID
        assert params["state"] == "st"

    def test_requests_offline_access_so_a_refresh_token_comes_back(self):
        assert "offline_access" in self._params()["scope"].split()

    def test_sends_no_client_secret(self):
        assert "client_secret" not in self._params()

    def test_scopes_are_space_joined(self):
        assert self._params(scopes=("all-apis", "offline_access"))["scope"] == (
            "all-apis offline_access"
        )


class TestExchangeCode:
    def test_posts_authorization_code_grant_without_a_secret(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(
                {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}
            )
            tokens = oauth.exchange_code(
                HOST, CLIENT_ID, code="the-code", code_verifier="the-verifier"
            )
        fields = _posted_fields(mock_urlopen)
        assert fields["grant_type"] == "authorization_code"
        assert fields["code"] == "the-code"
        assert fields["code_verifier"] == "the-verifier"
        assert fields["client_id"] == CLIENT_ID
        assert "client_secret" not in fields
        assert tokens.access_token == "at"
        assert tokens.refresh_token == "rt"

    def test_posts_form_encoded_with_no_authorization_header(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "at", "expires_in": 60})
            oauth.exchange_code(HOST, CLIENT_ID, code="c", code_verifier="v")
        request = mock_urlopen.call_args[0][0]
        assert request.headers["Content-type"] == "application/x-www-form-urlencoded"
        assert "Authorization" not in request.headers

    def test_targets_the_account_endpoint_when_an_account_id_is_given(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "at", "expires_in": 60})
            oauth.exchange_code(HOST, CLIENT_ID, code="c", code_verifier="v", account_id=ACCOUNT_ID)
        assert mock_urlopen.call_args[0][0].full_url == (
            f"{HOST}/oidc/accounts/{ACCOUNT_ID}/v1/token"
        )

    def test_missing_access_token_is_an_error(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"expires_in": 60})
            with pytest.raises(RuntimeError, match="no access_token"):
                oauth.exchange_code(HOST, CLIENT_ID, code="c", code_verifier="v")


class TestRefreshAccessToken:
    def test_posts_refresh_token_grant_without_a_secret(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "new", "expires_in": 3600})
            oauth.refresh_access_token(HOST, CLIENT_ID, "old-rt")
        fields = _posted_fields(mock_urlopen)
        assert fields == {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": "old-rt",
        }

    def test_keeps_the_existing_refresh_token_when_the_response_omits_one(self):
        # A refresh response need not rotate the refresh token; dropping it would
        # lose the session on the next call.
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "new", "expires_in": 3600})
            tokens = oauth.refresh_access_token(HOST, CLIENT_ID, "old-rt")
        assert tokens.refresh_token == "old-rt"

    def test_uses_a_rotated_refresh_token_when_one_is_returned(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(
                {"access_token": "new", "refresh_token": "rotated", "expires_in": 3600}
            )
            tokens = oauth.refresh_access_token(HOST, CLIENT_ID, "old-rt")
        assert tokens.refresh_token == "rotated"


class TestOauthErrorMessages:
    def test_invalid_client_explains_the_public_app_requirement(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = _http_error(
                401,
                {"error": "invalid_client", "error_description": "Client authentication failed"},
            )
            with pytest.raises(RuntimeError) as excinfo:
                oauth.refresh_access_token(HOST, CLIENT_ID, "rt")
        message = str(excinfo.value)
        assert "invalid_client" in message
        assert "Client authentication failed" in message
        assert "non-confidential" in message

    def test_invalid_grant_tells_the_user_how_to_sign_in_again(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = _http_error(
                401, {"error": "invalid_grant", "error_description": "Refresh token is invalid"}
            )
            with pytest.raises(RuntimeError, match="--oauth-client-id"):
                oauth.refresh_access_token(HOST, CLIENT_ID, "rt")

    def test_unreachable_endpoint_is_reported_with_the_url(self):
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("no route to host")
            with pytest.raises(RuntimeError, match="Could not reach"):
                oauth.refresh_access_token(HOST, CLIENT_ID, "rt")


class TestTokenCache:
    def test_round_trips_a_token_set(self):
        tokens = oauth.TokenSet("at", "rt", time.time() + 3600, "all-apis offline_access")
        oauth.store_tokens(HOST, CLIENT_ID, tokens)
        loaded = oauth.load_cached_tokens(HOST, CLIENT_ID)
        assert loaded is not None
        assert (loaded.access_token, loaded.refresh_token) == ("at", "rt")
        assert loaded.scope == "all-apis offline_access"

    def test_cache_file_is_owner_only(self):
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("at", "rt", time.time() + 60))
        mode = stat.S_IMODE(oauth.TOKEN_CACHE_PATH.stat().st_mode)
        assert mode == 0o600, f"refresh token cache is mode {oct(mode)}"

    def test_entries_are_keyed_by_host_and_client_id(self):
        oauth.store_tokens(HOST, "client-a", oauth.TokenSet("at-a", "rt-a", time.time() + 60))
        oauth.store_tokens(HOST, "client-b", oauth.TokenSet("at-b", "rt-b", time.time() + 60))
        a = oauth.load_cached_tokens(HOST, "client-a")
        b = oauth.load_cached_tokens(HOST, "client-b")
        assert a is not None and b is not None
        assert a.access_token == "at-a"
        assert b.access_token == "at-b"

    def test_unknown_client_id_is_a_cache_miss(self):
        oauth.store_tokens(HOST, "client-a", oauth.TokenSet("at-a", "rt-a", time.time() + 60))
        assert oauth.load_cached_tokens(HOST, "other-client") is None

    def test_missing_cache_file_is_a_miss_not_an_error(self):
        assert oauth.load_cached_tokens(HOST, CLIENT_ID) is None

    def test_corrupt_cache_file_is_a_miss_not_an_error(self):
        oauth.TOKEN_CACHE_PATH.write_text("{ not json", encoding="utf-8")
        assert oauth.load_cached_tokens(HOST, CLIENT_ID) is None

    def test_forget_tokens_removes_only_that_entry(self):
        oauth.store_tokens(HOST, "client-a", oauth.TokenSet("at-a", "rt-a", time.time() + 60))
        oauth.store_tokens(HOST, "client-b", oauth.TokenSet("at-b", "rt-b", time.time() + 60))
        oauth.forget_tokens(HOST, "client-a")
        assert oauth.load_cached_tokens(HOST, "client-a") is None
        assert oauth.load_cached_tokens(HOST, "client-b") is not None

    def test_dry_run_writes_nothing(self, monkeypatch):
        monkeypatch.setattr(oauth, "is_dry_run", lambda: True)
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("at", "rt", time.time() + 60))
        assert not oauth.TOKEN_CACHE_PATH.exists()


class TestTokenSetFreshness:
    def test_token_expiring_inside_the_buffer_is_not_fresh(self):
        tokens = oauth.TokenSet("at", "rt", time.time() + oauth.EXPIRY_BUFFER_SECONDS - 5)
        assert not tokens.is_fresh

    def test_token_well_inside_its_lifetime_is_fresh(self):
        assert oauth.TokenSet("at", "rt", time.time() + 3600).is_fresh

    def test_empty_access_token_is_never_fresh(self):
        assert not oauth.TokenSet("", "rt", time.time() + 3600).is_fresh


class TestGetToken:
    def test_fresh_cached_token_is_served_without_any_request(self):
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("cached", "rt", time.time() + 3600))
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            assert oauth.get_token(HOST, CLIENT_ID) == "cached"
        mock_urlopen.assert_not_called()

    def test_expired_token_is_refreshed_and_the_new_one_cached(self):
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("stale", "rt", time.time() - 10))
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(
                {"access_token": "refreshed", "expires_in": 3600}
            )
            assert oauth.get_token(HOST, CLIENT_ID) == "refreshed"
        cached = oauth.load_cached_tokens(HOST, CLIENT_ID)
        assert cached is not None and cached.access_token == "refreshed"

    def test_force_refresh_ignores_a_still_fresh_token(self):
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("cached", "rt", time.time() + 3600))
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "fresh", "expires_in": 3600})
            assert oauth.get_token(HOST, CLIENT_ID, force_refresh=True) == "fresh"

    def test_no_cached_session_raises_an_actionable_error(self):
        with pytest.raises(RuntimeError, match="ug configure --oauth-client-id"):
            oauth.get_token(HOST, CLIENT_ID)

    def test_never_opens_a_browser(self):
        # `ug auth-token` runs non-interactively from an agent's key helper.
        oauth.store_tokens(HOST, CLIENT_ID, oauth.TokenSet("cached", "rt", time.time() + 3600))
        with patch("webbrowser.open") as mock_open:
            oauth.get_token(HOST, CLIENT_ID)
        mock_open.assert_not_called()


class TestLogin:
    def _login(self, captured: dict, monkeypatch):
        monkeypatch.setattr(oauth, "_await_callback", lambda *_a, **_k: captured)
        monkeypatch.setattr("webbrowser.open", lambda *_a, **_k: True)
        return oauth.login(HOST, CLIENT_ID, open_browser=False)

    def test_state_mismatch_is_rejected_and_nothing_is_cached(self, monkeypatch):
        with pytest.raises(RuntimeError, match="state mismatch"):
            self._login({"code": "c", "state": "not-the-state-we-sent"}, monkeypatch)
        assert oauth.load_cached_tokens(HOST, CLIENT_ID) is None

    def test_timeout_with_no_redirect_is_reported(self, monkeypatch):
        with pytest.raises(RuntimeError, match="Timed out"):
            self._login({}, monkeypatch)

    def test_authorization_error_is_surfaced(self, monkeypatch):
        with pytest.raises(RuntimeError, match="access_denied"):
            self._login({"error": "access_denied"}, monkeypatch)

    def test_a_response_without_a_refresh_token_is_rejected(self, monkeypatch):
        # Without offline_access the session could not outlive the access token.
        def fake_await(redirect, timeout):
            return {"code": "c", "state": fake_await.state}

        def capture_state(*args, **kwargs):
            fake_await.state = kwargs["state"]
            return "https://example.invalid/authorize"

        monkeypatch.setattr(oauth, "_await_callback", fake_await)
        monkeypatch.setattr(oauth, "build_authorize_url", capture_state)
        with patch("ucode.oauth.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse({"access_token": "at", "expires_in": 3600})
            with pytest.raises(RuntimeError, match="offline_access"):
                oauth.login(HOST, CLIENT_ID, open_browser=False)
