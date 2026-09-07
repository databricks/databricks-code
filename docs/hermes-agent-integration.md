# Hermes Agent integration

## Status and scope

This document defines the cross-repository contract for configuring Hermes to
use Databricks-governed models. The user-facing provider name is **Databricks
Model Serving**. Runtime traffic uses Unity AI Gateway routes; the display name
does not change that architecture.

The integration supports route-specific providers for OpenAI Responses,
Anthropic Messages, native Gemini, and OpenAI-compatible OSS chat. It also
registers ucode-managed MCP servers through Hermes's native configuration
interface.

## Ownership boundary

ucode owns all Databricks-specific behavior:

- Databricks CLI profiles, OAuth/PAT authentication, and token refresh
- workspace capability and model discovery
- Gateway route and wire-protocol selection
- Databricks-specific request headers
- managed MCP discovery and registration
- validation, diagnostics, managed-key tracking, and cleanup
- rendering the non-secret patch applied to Hermes configuration

Hermes owns its configuration schema and mutation semantics. It exposes a
native provider entry and a thin, argv-based setup handoff to ucode. After
setup, Hermes resolves the generated providers through its existing generic
transports. Hermes must not add a Databricks transport, static model catalog,
runtime token-refresh branch, or Databricks-specific auxiliary/delegation path.

Normal sessions run with `hermes`; no `ucode hermes` wrapper is required.

## Command contract

The stable setup entry point is:

```text
ucode configure hermes [options]
```

The command configures Hermes and exits without launching it. It must:

- use argv semantics throughout and never invoke a shell;
- accept an explicit Hermes home/profile target for automation and isolation;
- prompt for missing workspace, profile, or model inputs in interactive use;
- support fully specified non-interactive use;
- be idempotent and modify only ucode-managed keys;
- support surgical unconfiguration of those keys;
- return nonzero without changing the active provider/model on failure or
  cancellation; and
- never expose access tokens, refresh tokens, PATs, or token-helper output.

Machine-readable completion output is additive/versionable and contains only
non-secret state. The initial shape is:

```json
{
  "status": "configured",
  "agent": "hermes",
  "hermes_home": "/resolved/path",
  "provider_ids": ["ucode-databricks-codex"],
  "default_provider": "ucode-databricks-codex",
  "default_model": "system.ai.<model>",
  "mcp_servers_configured": [],
  "warnings": []
}
```

Hermes relies on the child exit status, then reloads and validates its own
configuration. It does not duplicate Databricks field validation.

## Configuration transaction

ucode does not edit Hermes YAML directly and does not import Hermes Python
modules. It invokes Hermes's public `config get`, `config set`, and
`config unset` commands with an explicit `HERMES_HOME`.

Hermes does not currently expose an atomic multi-key transaction. ucode checks
that new managed paths are absent and that previously managed values still
match their recorded fingerprints before mutation. Provider definitions are
written before switching the active model. Cleanup removes only values whose
fingerprints still match; uncertain ownership preserves configuration rather
than deleting a possible user replacement. A process failure after mutation
starts can therefore leave a partial update, which is surfaced as an error for
explicit reconciliation.

## Provider representation

Each wire protocol has a separate generated custom provider so every model is
coupled to the correct Gateway route and Hermes transport:

- `ucode-databricks-codex` for OpenAI Responses;
- `ucode-databricks-anthropic` for Anthropic Messages;
- `ucode-databricks-gemini` for native Gemini; and
- `ucode-databricks-oss` for OpenAI-compatible OSS chat.

Hermes retains its internal `provider="custom"` identity while preserving the
selected generated provider separately. Each provider obtains credentials from
a refreshable `ucode auth-token` command; no Databricks token is stored in
Hermes configuration.

## Capability contract

| Capability | Release requirement | Owner | Acceptance evidence |
|---|---|---|---|
| Databricks CLI OAuth | Supported | ucode | Refresh succeeds after token cache invalidation/expiry |
| PAT compatibility | Preserve when explicitly selected | ucode | No PAT is written to Hermes config or output |
| Responses route | Supported | ucode config + Hermes generic transport | Streaming text and tool calls pass through `/ai-gateway/codex/v1` |
| Model-service discovery | Supported | ucode | `/model` contains the authorized coding models |
| Profile-scoped config | Supported | ucode + Hermes config commands | Default and named profiles remain isolated |
| Multiplex isolation | Supported | Hermes generic secret/config resolution | A scoped miss cannot borrow another profile's bearer/config |
| Reconfigure/unconfigure | Supported | ucode | Reconfigure is idempotent; cleanup removes only managed keys |
| Model Provider Service header routing | Deferred for Hermes v1 | ucode | No support claim until a production setup path exists |
| Unity Catalog permissions | Required | Gateway | Allowed models succeed and denied models fail clearly |
| Usage/inference tables | Required where enabled | Gateway, validated by ucode/E2E | Request is visible with expected identity and model |
| Rate limits and policies | Required pass-through | Gateway | 429 classification is preserved with no local bypass |
| Anthropic Messages route | Supported | ucode config + Hermes Messages transport | Claude streaming and tools pass |
| Gemini route | Supported | ucode config + Hermes Gemini transport | Gemini streaming and tools pass |
| OSS/OpenAI-compatible route | Supported | ucode config + Hermes chat transport | Supported OSS chat/tools pass |
| Managed MCP servers | Supported | ucode | Configured tools appear in Hermes without stored bearer tokens |
| Smart routing | Explicit decision gate | ucode | Tested support or explicit unsupported documentation |
| MLflow tracing hooks | Explicit decision gate | ucode | No support claim without Hermes-compatible trace evidence |
| Managed workspace manifest | Explicit decision gate | ucode | Decide whether Hermes belongs in the managed agent schema |

## Failure and security behavior

- Missing ucode is handled by Hermes with official installation guidance and no
  partial configuration.
- Missing Databricks CLI/login, authorization failures, and empty discovery are
  diagnosed by ucode.
- Hermes suppresses raw child output on failure; tests use sentinel secrets and
  assert their absence from stdout, stderr, exceptions, fixtures, and config.
- No short-lived token or PAT is persisted in Hermes configuration.
- Provider discovery and import have no subprocess, OAuth, network, install, or
  configuration side effects.

## Support boundaries

The two currently incompatible models, `gpt-oss-20b` and `gpt-oss-120b`, are
excluded from Hermes provider generation only. Their availability to unrelated
ucode consumers is unchanged. Smart routing, MLflow tracing hooks, and managed
workspace manifests remain separate decisions and are not implied by this
integration.

Databricks workspace authentication, discovery, Gateway routing, policies, and
managed MCP behavior are supported in ucode/Databricks. Generic Hermes session,
transport, and tool behavior remain supported in Hermes.

Primary references:

- [Databricks coding-agent integrations](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-model-services)
- [Databricks Model Provider Services](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-model-provider-services)
- [ucode](https://github.com/databricks/ucode)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
