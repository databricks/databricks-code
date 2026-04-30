# Databricks Coding Gateway

`coding-gateway` is a lightweight launcher for running Codex, Claude Code, and Gemini CLI through Databricks.

## Requirements

- Python 3.12+
- `uv` (recommended) or `pipx`
- `npm` if tool CLIs need to be installed automatically

## Installation

```bash
uv tool install git+https://github.com/databricks/coding-gateway
```

---

## For Admins

Admins configure a workspace once and export a portable config bundle that users can import.

### 1. Configure the workspace

```bash
coding-gateway configure
```

The interactive flow lets you:

- Set the Databricks workspace URL
- Enable Databricks AI Gateway V2 (and provide the org ID)
- Choose saved launch models for Claude Code and Gemini CLI

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

### 3. Export a config bundle

```bash
coding-gateway export config.json
```

This saves your workspace URL, AI Gateway settings, selected models, and MCP servers into a single JSON file. Distribute this file to your users.

---

## For Users

Users import a config bundle provided by their admin to get set up in one step.

### 1. Install

```bash
uv tool install git+https://github.com/databricks/coding-gateway
```

### 2. Import the config bundle

```bash
coding-gateway import config.json
```

This will:

1. Configure your Databricks workspace and authenticate via `databricks auth login`
2. Write config files for Codex, Claude Code, and Gemini CLI
3. Set up any MCP servers included in the bundle

### 3. Launch a tool

```bash
coding-gateway                    # launches Codex (default)
coding-gateway --tool claude
coding-gateway --tool gemini
```

You can override the model at launch time:

```bash
coding-gateway --tool claude --model databricks-claude-opus-4-6
coding-gateway --tool gemini --model databricks-gemini-2-5-pro
```

---

## Other Commands

| Command | Description |
|---------|-------------|
| `coding-gateway status` | Show current workspace, base URLs, managed config files, and selected models |
| `coding-gateway usage` | Show AI Gateway usage summary (requires AI Gateway V2) |
| `coding-gateway logout` | Clear saved state and restore backed-up config files |

## Usage Reporting

```bash
coding-gateway usage
```

Requires Databricks AI Gateway V2. Queries `system.ai_gateway.usage` and shows:

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
- When AI Gateway V2 is enabled, tool base URLs point to the AI Gateway hostname while auth still uses the original workspace URL

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

