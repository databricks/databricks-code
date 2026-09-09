"""Experimental custom-client OAuth, separate from production Databricks CLI auth."""

from __future__ import annotations

from urllib.parse import urlparse

from ucode.ui import err_console, normalize_workspace_url, print_warning_err

DEFAULT_REDIRECT_URL = "http://localhost:8020"


def get_custom_client_token(
    workspace: str,
    client_id: str,
    redirect_url: str = DEFAULT_REDIRECT_URL,
    *,
    force_refresh: bool = False,
) -> str:
    """Reuse the SDK's PKCE flow and per-workspace/client token cache."""
    from databricks.sdk import oauth

    if not client_id.strip():
        raise RuntimeError("--client-id must not be empty.")
    try:
        redirect = urlparse(redirect_url)
        valid_redirect = (
            redirect.scheme == "http"
            and redirect.hostname in {"localhost", "127.0.0.1"}
            and redirect.port is not None
            and redirect.port > 0
            and not (redirect.username or redirect.password or redirect.query or redirect.fragment)
        )
    except ValueError:
        valid_redirect = False
    if not valid_redirect:
        raise RuntimeError(
            "--redirect-url must be a local HTTP callback with a port, "
            "e.g. http://localhost:8020/ai-devtools-workspace-oauth. "
            "Use the exact redirect URL registered for your OAuth app."
        )

    workspace = normalize_workspace_url(workspace)
    scopes = ["offline_access", "all-apis"]
    try:
        endpoints = oauth.get_workspace_endpoints(workspace)
        cache = oauth.TokenCache(
            host=workspace,
            oidc_endpoints=endpoints,
            client_id=client_id,
            redirect_url=redirect_url,
            scopes=scopes,
        )
        credentials = cache.load()
        if credentials is not None:
            try:
                if force_refresh:
                    credentials = oauth.SessionCredentials(
                        token=credentials.refresh(),
                        token_endpoint=endpoints.token_endpoint,
                        client_id=client_id,
                        redirect_url=redirect_url,
                    )
                credentials.token()
            except Exception:
                print_warning_err("Cached OAuth token could not be refreshed. Sign in again.")
                credentials = None
        if credentials is None:
            client = oauth.OAuthClient(
                oidc_endpoints=endpoints,
                client_id=client_id,
                redirect_url=redirect_url,
                scopes=scopes,
            )
            consent = client.initiate_consent()
            err_console.print(
                f"Sign in using your browser: {consent.authorization_url}",
                markup=False,
                soft_wrap=True,
            )
            credentials = consent.launch_external_browser()
        token = credentials.token().access_token
        if not token:
            raise ValueError("OAuth returned no access token")
        cache.save(credentials)
        return token
    except Exception as exc:
        raise RuntimeError(
            "Custom-client OAuth failed. Check the workspace, client ID, and registered "
            f"redirect URL ({redirect_url}); ensure its local port is available and "
            "the SDK token cache is writable, then retry."
        ) from exc
