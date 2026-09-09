# Experimental custom-client OAuth

This opt-in token helper is hidden from CLI help. Its implementation lives in
`src/ucode/experimental_oauth.py`, separate from production Databricks CLI authentication.

```bash
ug auth-token --host https://your-workspace.databricks.com \
  --client-id YOUR_CLIENT_ID \
  --redirect-url http://localhost:8020/ai-devtools-workspace-oauth
```

Use a public OAuth app enabled for your workspace with the exact redirect URL registered
(including its port and path). `--redirect-url` defaults to `http://localhost:8020`.
The first call opens a browser using PKCE and requests `offline_access` and `all-apis`.
Later calls reuse the cached token or refresh it automatically; an unusable refresh token
starts browser login again. Add `--force-refresh` to refresh even before expiry.

Only the access token is printed to stdout; login instructions go to stderr. Tokens,
including refresh tokens, are saved in the SDK's `~/.config/databricks-sdk-py/oauth/`
cache with owner-only file permissions, isolated by workspace and client ID. This path
does not require the Databricks CLI, use its profiles, or reuse `DATABRICKS_BEARER`/PATs.
It does not change the authentication used by `ug configure` or existing agent configurations.

For Python callers:

```python
from ucode.experimental_oauth import get_custom_client_token

token = get_custom_client_token(
    "https://your-workspace.databricks.com",
    client_id="YOUR_CLIENT_ID",
    redirect_url="http://localhost:8020/ai-devtools-workspace-oauth",
)
```
