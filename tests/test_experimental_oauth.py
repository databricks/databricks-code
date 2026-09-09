"""Isolated tests for the hidden custom-client OAuth feature."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import pytest
from databricks.sdk import oauth
from typer.testing import CliRunner

import ucode.databricks as db_mod
from ucode.cli import app
from ucode.experimental_oauth import get_custom_client_token

WS = "https://example.databricks.com"
runner = CliRunner()


class TestCustomClientToken:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(oauth.TokenCache, "BASE_PATH", str(tmp_path / "oauth"))
        monkeypatch.setenv("DATABRICKS_BEARER", "unrelated-bearer")
        monkeypatch.setattr(db_mod, "run", Mock(side_effect=AssertionError("CLI not expected")))
        monkeypatch.setattr(
            db_mod, "find_profile_name_for_host", Mock(side_effect=AssertionError("No profile"))
        )
        self.endpoints = oauth.OidcEndpoints(
            authorization_endpoint=f"{WS}/oidc/v1/authorize",
            token_endpoint=f"{WS}/oidc/v1/token",
        )
        self.discovery = Mock(return_value=self.endpoints)
        monkeypatch.setattr(oauth, "get_workspace_endpoints", self.discovery)
        self.browser = Mock(return_value=self._credentials("browser-token", "browser-refresh"))
        monkeypatch.setattr(oauth.Consent, "launch_external_browser", self.browser)
        self.refresh = Mock(return_value=self._credentials("refreshed", "rotated-refresh").token())
        monkeypatch.setattr(oauth, "retrieve_token", self.refresh)

    def _credentials(self, access_token, refresh_token):
        return oauth.SessionCredentials(
            token=oauth.Token(
                access_token=access_token,
                token_type="Bearer",
                refresh_token=refresh_token,
                expiry=datetime.now(UTC) + timedelta(hours=1),
            ),
            token_endpoint=self.endpoints.token_endpoint,
            client_id="custom-client",
            redirect_url="http://localhost:8020",
        )

    def _cache(self, workspace=WS, client_id="custom-client"):
        return oauth.TokenCache(
            host=workspace,
            oidc_endpoints=self.endpoints,
            client_id=client_id,
            redirect_url="http://localhost:8020",
            scopes=["offline_access", "all-apis"],
        )

    def test_browser_login_uses_custom_client_and_redirect(self, capsys):
        redirect_url = "http://localhost:41735/ai-devtools-workspace-oauth"
        token = get_custom_client_token(WS, client_id="custom-client", redirect_url=redirect_url)
        assert token == "browser-token"
        self.browser.assert_called_once()
        self.refresh.assert_not_called()
        cached = self._cache().load().token()
        assert cached.access_token == token
        assert cached.refresh_token == "browser-refresh"
        output = capsys.readouterr()
        assert output.out == ""
        query = parse_qs(output.err.split("?", 1)[1].strip())
        assert query["client_id"] == ["custom-client"]
        assert query["redirect_uri"] == [redirect_url]
        assert set(query["scope"][0].split()) == {"offline_access", "all-apis"}
        assert query["code_challenge_method"] == ["S256"]

    def test_reuses_cached_token_without_refresh_or_login(self, capsys):
        self._cache().save(self._credentials("cached", "refresh"))
        assert get_custom_client_token(WS + "/", client_id="custom-client") == "cached"
        self.discovery.assert_called_once_with(WS)
        self.browser.assert_not_called()
        self.refresh.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_expired_token_refreshes_with_custom_client_and_saves_rotation(self):
        cached = self._credentials("expired", "old-refresh")
        self._cache().save(cached)
        cache_path = Path(self._cache().filename)
        payload = json.loads(cache_path.read_text())
        payload["token"]["expiry"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        cache_path.write_text(json.dumps(payload))
        assert get_custom_client_token(WS, client_id="custom-client") == "refreshed"
        self.refresh.assert_called_once_with(
            client_id="custom-client",
            client_secret=None,
            token_url=self.endpoints.token_endpoint,
            params={"grant_type": "refresh_token", "refresh_token": "old-refresh"},
            use_params=True,
            headers={},
        )
        self.browser.assert_not_called()
        assert self._cache().load().token().refresh_token == "rotated-refresh"

    def test_force_refresh_bypasses_fresh_access_token(self):
        self._cache().save(self._credentials("cached", "refresh"))
        assert (
            get_custom_client_token(WS, client_id="custom-client", force_refresh=True)
            == "refreshed"
        )
        self.refresh.assert_called_once()
        self.browser.assert_not_called()
        assert self._cache().load().token().access_token == "refreshed"

    def test_refresh_failure_falls_back_to_browser(self, capsys):
        self._cache().save(self._credentials("cached", "revoked-refresh"))
        self.refresh.side_effect = ValueError("sensitive server response")
        assert (
            get_custom_client_token(WS, client_id="custom-client", force_refresh=True)
            == "browser-token"
        )
        self.browser.assert_called_once()
        assert self._cache().load().token().refresh_token == "browser-refresh"
        output = capsys.readouterr()
        assert output.out == ""
        assert "Sign in again" in output.err
        assert "sensitive server response" not in output.err

    def test_missing_refresh_token_falls_back_to_browser(self):
        self._cache().save(self._credentials("cached", None))
        assert (
            get_custom_client_token(WS, client_id="custom-client", force_refresh=True)
            == "browser-token"
        )
        self.browser.assert_called_once()

    @pytest.mark.parametrize(
        ("workspace", "client_id"),
        [(WS, "another-client"), ("https://other.databricks.com", "custom-client")],
    )
    def test_cache_is_isolated_by_workspace_and_client(self, workspace, client_id):
        self._cache().save(self._credentials("cached", "refresh"))
        assert get_custom_client_token(workspace, client_id=client_id) == "browser-token"
        self.browser.assert_called_once()
        assert self._cache().load().token().access_token == "cached"

    def test_corrupt_cache_starts_browser_login(self):
        cache_path = Path(self._cache().filename)
        cache_path.parent.mkdir()
        cache_path.write_text("not JSON")
        assert get_custom_client_token(WS, client_id="custom-client") == "browser-token"
        self.browser.assert_called_once()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
    def test_cache_is_owner_only(self):
        get_custom_client_token(WS, client_id="custom-client")
        assert Path(self._cache().filename).stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize(
        "redirect_url",
        [
            "https://example.com/callback",
            "http://localhost/callback",
            "http://localhost:0/callback",
            "http://localhost:99999/callback",
            "http://localhost:8020/callback?query=1",
            "http://user@localhost:8020/callback",
        ],
    )
    def test_invalid_redirect_fails_before_network(self, redirect_url):
        with pytest.raises(RuntimeError, match="--redirect-url must be"):
            get_custom_client_token(WS, client_id="custom-client", redirect_url=redirect_url)
        self.discovery.assert_not_called()

    def test_empty_client_id_is_rejected(self):
        with pytest.raises(RuntimeError, match="--client-id must not be empty"):
            get_custom_client_token(WS, client_id=" ")
        self.discovery.assert_not_called()

    def test_login_failure_is_actionable_and_does_not_expose_response(self):
        self.browser.side_effect = ValueError("sensitive server response")
        with pytest.raises(RuntimeError, match="registered redirect URL") as error:
            get_custom_client_token(WS, client_id="custom-client")
        assert "sensitive server response" not in str(error.value)


class TestCustomClientCommand:
    @pytest.fixture(autouse=True)
    def _no_production_auth(self, monkeypatch):
        monkeypatch.setattr(
            "ucode.cli.get_databricks_token",
            Mock(side_effect=AssertionError("Experimental auth must not use production auth")),
        )

    def test_experimental_options_are_hidden(self):
        result = runner.invoke(app, ["auth-token", "--help"])
        assert result.exit_code == 0
        assert "--client-id" not in result.output
        assert "--redirect-url" not in result.output

    def test_custom_client_ignores_saved_pat_and_profile(self):
        with (
            patch(
                "ucode.cli.load_state",
                return_value={"workspace": "https://ws", "profile": "saved", "use_pat": True},
            ),
            patch("ucode.cli.ensure_pat_bearer") as pat,
            patch(
                "ucode.experimental_oauth.get_custom_client_token", return_value="custom-token"
            ) as fetch,
        ):
            result = runner.invoke(app, ["auth-token", "--client-id", "my-client"])
        assert result.exit_code == 0, result.output
        assert result.stdout == "custom-token\n"
        pat.assert_not_called()
        fetch.assert_called_once_with(
            "https://ws",
            client_id="my-client",
            redirect_url="http://localhost:8020",
            force_refresh=False,
        )

    def test_custom_client_forwards_redirect_and_force_refresh_without_setup(self):
        with (
            patch("ucode.cli.load_state", return_value={}),
            patch(
                "ucode.experimental_oauth.get_custom_client_token", return_value="custom-token"
            ) as fetch,
        ):
            result = runner.invoke(
                app,
                [
                    "auth-token",
                    "--host",
                    "https://ws",
                    "--client-id",
                    "my-client",
                    "--redirect-url",
                    "http://localhost:41735/callback",
                    "--force-refresh",
                ],
            )
        assert result.exit_code == 0, result.output
        assert result.stdout == "custom-token\n"
        fetch.assert_called_once_with(
            "https://ws",
            client_id="my-client",
            redirect_url="http://localhost:41735/callback",
            force_refresh=True,
        )

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (["--client-id", "my-client", "--use-pat"], "cannot be combined"),
            (["--redirect-url", "http://localhost:8020"], "requires --client-id"),
        ],
    )
    def test_invalid_custom_client_options(self, options, message):
        with patch("ucode.experimental_oauth.get_custom_client_token") as fetch:
            result = runner.invoke(app, ["auth-token", *options])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert message in result.stderr
        fetch.assert_not_called()

    def test_custom_client_failure_prints_no_token(self):
        with (
            patch("ucode.cli.load_state", return_value={}),
            patch(
                "ucode.experimental_oauth.get_custom_client_token",
                side_effect=RuntimeError("OAuth failed"),
            ),
        ):
            result = runner.invoke(
                app, ["auth-token", "--host", "https://ws", "--client-id", "my-client"]
            )
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "OAuth failed" in result.stderr
