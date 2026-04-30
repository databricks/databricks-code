# Databricks Coding Gateway

`coding-gateway` is a lightweight launcher for running Codex, Claude Code, and Gemini CLI through Databricks.

## Requirements

- Python 3.12+ — install with `uv` ([uv.astral.sh](https://docs.astral.sh/uv/getting-started/installation/))
- `npm` if tool CLIs need to be installed automatically

## Installation

```bash
uv tool install git+https://github.com/databricks/coding-gateway
```

---

## Setup

### 1. Configure the workspace

```bash
coding-gateway configure
```

Enter your Databricks workspace URL. `coding-gateway` automatically detects whether Databricks AI Gateway is available and configures tool endpoints accordingly.

This writes managed config files for all three tools (`~/.codex/config.toml`, `~/.claude/settings.json`, `~/.gemini/.env`).

### 2. Configure MCP servers (optional)

```bash
coding-gateway configure mcp
```

Add Databricks MCP servers to Claude Code. Supported server types:

- **External** — e.g. confluence-mcp, jira-mcp
- **UC Functions** — Unity Catalog AI functions
- **Genie** — AI/BI dashboards
- **Custom** — any MCP server URL

You will be prompted for OAuth credentials (client ID and secret) that are reused for all servers added in the session.

### 3. Launch a tool

```bash
coding-gateway                    # launches Codex (default)
coding-gateway --tool claude
coding-gateway --tool gemini
```

You can override the model at launch time:

```bash
coding-gateway --tool claude --model databricks-claude-opus-4-7
coding-gateway --tool gemini --model databricks-gemini-2-5-pro
```

---

## Other Commands

| Command | Description |
|---------|-------------|
| `coding-gateway status` | Show current workspace, base URLs, managed config files, and selected models |
| `coding-gateway usage` | Show AI Gateway usage summary |
| `coding-gateway logout` | Clear saved state and restore backed-up config files |
| `coding-gateway configure --dry-run` | Preview config files without writing them |

## Usage Reporting

```bash
coding-gateway usage
```

Requires Databricks AI Gateway. Queries `system.ai_gateway.usage` and shows:

- Token totals for today, last 7 days, and last 30 days
- Active tools and top models this week
- 7-day breakdown per tool (Codex, Claude Code, Gemini CLI)

## Managed Local Files

`coding-gateway` manages these files:

| File | Tool |
|------|------|
| `~/.codex/config.toml` | Codex |
| `~/.claude/settings.json` | Claude Code |
| `~/.gemini/.env` | Gemini CLI |

Existing files are backed up before being overwritten. `coding-gateway logout` restores backups.

## Authentication

- Databricks authentication uses OAuth via `databricks auth login`
- Codex and Claude use a Databricks token helper (no fixed token stored)
- Gemini refreshes its bearer token automatically while running through `coding-gateway`

## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request against `main`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
