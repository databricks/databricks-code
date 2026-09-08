"""OAuth authorization-code flow for public Databricks apps."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ucode.config_io import APP_DIR, is_dry_run

TOKEN_CACHE_PATH = APP_DIR / "oauth-tokens.json"
CACHE_VERSION = 1

DEFAULT_SCOPES = ("all-apis", "offline_access")

# This URI must be registered on the custom app.
DEFAULT_REDIRECT_URI = "http://localhost:8020"
REDIRECT_URI_ENV = "UCODE_OAUTH_REDIRECT_URI"

EXPIRY_BUFFER_SECONDS = 120
_HTTP_TIMEOUT = 30
_LOGIN_TIMEOUT_SECONDS = 300.0

_ERROR_HINTS = {
    "invalid_client": "Check that this is a public app client ID, not an integration ID.",
    "invalid_grant": (
        "The refresh token expired or was revoked. Run "
        "`ug configure --oauth-client-id <client-id>` again."
    ),
    "invalid_scope": "Grant the app `all-apis` and `offline_access`.",
    "unauthorized_client": "The app does not allow authorization-code login.",
}


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None = None
    expires_at: float = 0.0
    scope: str = ""

    @property
    def is_fresh(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - EXPIRY_BUFFER_SECONDS


def authorize_endpoint(host: str) -> str:
    return f"{host.rstrip('/')}/oidc/v1/authorize"


def token_endpoint(host: str) -> str:
    return f"{host.rstrip('/')}/oidc/v1/token"


def generate_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def redirect_uri() -> str:
    return os.environ.get(REDIRECT_URI_ENV, "").strip() or DEFAULT_REDIRECT_URI


def build_authorize_url(
    host: str,
    client_id: str,
    *,
    code_challenge: str,
    state: str,
    redirect: str | None = None,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect or redirect_uri(),
            "scope": " ".join(scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{authorize_endpoint(host)}?{query}"


def _format_oauth_error(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode("utf-8"))
    except (OSError, ValueError):
        detail = {}
    if isinstance(detail, dict) and detail.get("error"):
        code = str(detail.get("error"))
        description = str(detail.get("error_description") or "").strip()
        message = f"Databricks OAuth error `{code}`"
        if description:
            message += f": {description}"
        hint = _ERROR_HINTS.get(code)
        if hint:
            message += f"\n{hint}"
        return message
    return f"Databricks token endpoint returned HTTP {exc.code}."


def _post_form(url: str, fields: dict[str, str]) -> dict:
    body = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_format_oauth_error(exc)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not reach the Databricks token endpoint {url}: {exc}") from None
    except json.JSONDecodeError:
        raise RuntimeError(f"Databricks token endpoint {url} returned invalid JSON.") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Databricks token endpoint {url} returned invalid JSON.")
    return payload


def _token_set_from_payload(payload: dict, *, keep_refresh_token: str | None = None) -> TokenSet:
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise RuntimeError("Databricks token endpoint returned no access_token.")
    try:
        lifetime = float(payload.get("expires_in", 3600))
    except (TypeError, ValueError):
        lifetime = 3600.0
    # A refresh response is not required to rotate the refresh token; when it
    # doesn't, keep the one we already hold or the session is lost on next call.
    refresh_token = payload.get("refresh_token") or keep_refresh_token
    return TokenSet(
        access_token=access_token,
        refresh_token=str(refresh_token) if refresh_token else None,
        expires_at=time.time() + lifetime,
        scope=str(payload.get("scope") or ""),
    )


def exchange_code(
    host: str,
    client_id: str,
    *,
    code: str,
    code_verifier: str,
    redirect: str | None = None,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> TokenSet:
    payload = _post_form(
        token_endpoint(host),
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "scope": " ".join(scopes),
            "code": code,
            "redirect_uri": redirect or redirect_uri(),
            "code_verifier": code_verifier,
        },
    )
    return _token_set_from_payload(payload)


def refresh_access_token(
    host: str,
    client_id: str,
    refresh_token: str,
) -> TokenSet:
    payload = _post_form(
        token_endpoint(host),
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
    )
    return _token_set_from_payload(payload, keep_refresh_token=refresh_token)


def cache_key(host: str, client_id: str) -> str:
    return f"{host.rstrip('/')}::{client_id}"


def _read_entries() -> dict:
    try:
        raw = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    entries = raw.get("entries")
    return entries if isinstance(entries, dict) else {}


def load_cached_tokens(host: str, client_id: str) -> TokenSet | None:
    entry = _read_entries().get(cache_key(host, client_id))
    if not isinstance(entry, dict):
        return None
    access_token = entry.get("access_token")
    refresh_token = entry.get("refresh_token")
    if not isinstance(access_token, str) and not isinstance(refresh_token, str):
        return None
    try:
        expires_at = float(entry.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        expires_at = 0.0
    return TokenSet(
        access_token=access_token if isinstance(access_token, str) else "",
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        expires_at=expires_at,
        scope=str(entry.get("scope") or ""),
    )


def store_tokens(host: str, client_id: str, tokens: TokenSet) -> None:
    if is_dry_run():
        return
    entries = dict(_read_entries())
    entries[cache_key(host, client_id)] = {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": tokens.expires_at,
        "scope": tokens.scope,
    }
    payload = json.dumps({"version": CACHE_VERSION, "entries": entries}, indent=2)
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(TOKEN_CACHE_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.chmod(TOKEN_CACHE_PATH, 0o600)
    except OSError as exc:
        raise RuntimeError(f"Failed to write the OAuth token cache {TOKEN_CACHE_PATH}") from exc


def not_signed_in_message(host: str, client_id: str) -> str:
    return (
        f"No custom-OAuth session for {host} (client id {client_id}). Run "
        f"`ug configure --oauth-client-id {client_id}` once to sign in; after that "
        "tokens refresh without a browser."
    )


def get_token(
    host: str,
    client_id: str,
    *,
    force_refresh: bool = False,
) -> str:
    cached = load_cached_tokens(host, client_id)
    if cached and cached.is_fresh and not force_refresh:
        return cached.access_token
    if not cached or not cached.refresh_token:
        raise RuntimeError(not_signed_in_message(host, client_id))
    tokens = refresh_access_token(host, client_id, cached.refresh_token)
    store_tokens(host, client_id, tokens)
    return tokens.access_token


def _await_callback(redirect: str, timeout: float) -> dict[str, str]:
    parsed = urllib.parse.urlparse(redirect)
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            query = urllib.parse.urlparse(self.path).query
            params = {key: values[0] for key, values in urllib.parse.parse_qs(query).items()}
            if "code" not in params and "error" not in params:
                self.send_response(404)
                self.end_headers()
                return
            captured.update(params)
            succeeded = "code" in params
            note = (
                "Signed in. You can close this tab and return to your terminal."
                if succeeded
                else "Authorization failed: "
                + (params.get("error_description") or params.get("error") or "unknown error")
            )
            body = (
                '<html><body style="font-family:system-ui;padding:2rem">'
                f"<p>{note}</p></body></html>"
            ).encode()
            self.send_response(200 if succeeded else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    address = (parsed.hostname or "localhost", parsed.port or 80)
    try:
        server = ThreadingHTTPServer(address, Handler)
    except OSError as exc:
        raise RuntimeError(
            f"Could not listen on {address[0]}:{address[1]} to receive the OAuth "
            f"redirect ({exc}). Free that port, or set {REDIRECT_URI_ENV} to a "
            "redirect URI registered on the app integration."
        ) from None
    deadline = time.time() + timeout
    with server:
        server.timeout = 1.0
        while not captured and time.time() < deadline:
            server.handle_request()
    return captured


def login(
    host: str,
    client_id: str,
    *,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    open_browser: bool = True,
    timeout: float = _LOGIN_TIMEOUT_SECONDS,
) -> TokenSet:
    import webbrowser

    from ucode.ui import print_note, print_section, print_success

    verifier, challenge = generate_pkce_pair()
    expected_state = secrets.token_urlsafe(16)
    redirect = redirect_uri()
    url = build_authorize_url(
        host,
        client_id,
        code_challenge=challenge,
        state=expected_state,
        redirect=redirect,
        scopes=scopes,
    )

    print_section("Databricks Custom OAuth Login")
    print_note(f"Opening a browser to sign in to {host}.")
    print_note(f"If it doesn't open, visit:\n{url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except webbrowser.Error:
            pass

    captured = _await_callback(redirect, timeout)
    if not captured:
        raise RuntimeError(
            f"Timed out after {int(timeout)}s waiting for the OAuth redirect to {redirect}. "
            "Confirm that URI is registered on the app integration."
        )
    if captured.get("error"):
        detail = captured.get("error_description") or captured["error"]
        raise RuntimeError(f"Databricks refused the authorization request: {detail}")
    if captured.get("state") != expected_state:
        raise RuntimeError(
            "OAuth state mismatch — the redirect did not come from this login attempt. "
            "Nothing was saved; run the command again."
        )
    code = captured.get("code") or ""
    if not code:
        raise RuntimeError("The OAuth redirect carried no authorization code.")

    tokens = exchange_code(
        host,
        client_id,
        code=code,
        code_verifier=verifier,
        redirect=redirect,
        scopes=scopes,
    )
    if not tokens.refresh_token:
        raise RuntimeError(
            "Databricks issued an access token but no refresh token, so the session "
            "cannot outlive it. Add `offline_access` to the app integration's scopes."
        )
    store_tokens(host, client_id, tokens)
    print_success(f"Custom OAuth session saved for {host}")
    return tokens
