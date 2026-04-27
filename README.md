# Coding Tool Gateway

`coding-tool-gateway` is a lightweight launcher for running Codex, Claude Code, and Gemini CLI through Databricks.

It is designed for a simple setup flow:

```bash
git clone https://github.com/databricks/coding-tool-gateway.git
cd coding-tool-gateway
pipx install .
coding-tool-gateway
```

On first run, `coding-tool-gateway` handles local bootstrap, prompts for your Databricks workspace, configures the selected coding tool, and launches it.

## Why Coding Tool Gateway

- Minimal setup for Databricks-backed coding tools
- One workspace configuration shared across Codex, Claude Code, and Gemini CLI
- Oauth Databricks authentication support
- Managed local config files with restore support through `coding-tool-gateway logout`
- Built-in AI Gateway usage reporting with `coding-tool-gateway usage`

## Supported Tools

- Codex
- Gemini CLI
- Claude Code

## Quick Start

Install and launch:

```bash
git clone https://github.com/databricks/coding-tool-gateway.git
cd coding-tool-gateway
pipx install .
coding-tool-gateway
```

On first launch, `coding-tool-gateway` will:

1. Install missing local dependencies it manages:
   `databricks`, `codex`, `claude`, `gemini`
2. Prompt for your Databricks workspace URL
3. Ask whether you want to use Databricks AI Gateway V2
4. Run Databricks login
5. Write managed tool configuration files
6. Launch the selected tool

After setup, normal usage is simple:

```bash
coding-tool-gateway
coding-tool-gateway --tool codex
coding-tool-gateway --tool claude
coding-tool-gateway --tool gemini
```

## Configuration

Use `configure` to update workspace settings or saved models:

```bash
coding-tool-gateway configure
```

The interactive flow supports:

- `Workspace`
  Reconfigure the Databricks workspace for all tools
- `Models`
  Choose the saved launch model for Claude Code or Gemini CLI

Examples:

```bash
coding-tool-gateway --tool claude --model databricks-claude-opus-4-6
coding-tool-gateway --tool gemini --model databricks-gemini-2-5-pro
```

Notes:

- Codex model selection remains inside Codex via `/model`
- Claude and Gemini can use saved models or an explicit `--model`
- Normal launches do not query the workspace model list unless you are configuring models

## Commands

```bash
coding-tool-gateway configure
coding-tool-gateway status
coding-tool-gateway usage
coding-tool-gateway logout
```

- `configure`
  Configure workspace settings or saved Claude/Gemini models
- `status`
  Show the current workspace, base URLs, managed config files, and selected models
- `usage`
  Show a fixed AI Gateway usage summary for the current user
- `logout`
  Clear saved state and restore any backed-up local config files

## Usage Reporting

```bash
coding-tool-gateway usage
```

`usage` requires Databricks AI Gateway V2. When enabled, it queries `system.ai_gateway.usage` and shows:

- token totals for today, the last 7 days, and the last 30 days
- active tools this week
- top models this week
- a 7-day breakdown for Codex, Claude Code, and Gemini CLI

If the workspace is not configured for AI Gateway V2, `coding-tool-gateway usage` will stop early and tell you to re-run `coding-tool-gateway configure`.

## Managed Local Files

`coding-tool-gateway` manages these local files:

- `~/.codex/config.toml`
- `~/.claude/settings.json`
- `~/.gemini/.env`

If one of these files already exists, `coding-tool-gateway` creates a backup before writing its managed version. `coding-tool-gateway logout` restores the backup.

## Authentication

- Databricks authentication always uses the workspace URL you configured
- If AI Gateway V2 is enabled, tool base URLs point to the AI Gateway hostname while authentication still uses the original workspace URL
- Codex and Claude use a Databricks token helper instead of storing a fixed token
- Gemini refreshes its Databricks bearer token automatically while launched through `coding-tool-gateway`
- `coding-tool-gateway usage` fetches a fresh Databricks token on each run

## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage with `system.ai_gateway.usage`](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Requirements

- Python 3.12+
- `pipx` (recommended) or `pip`
- `npm` if tool CLIs need to be installed automatically

## Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request against `main`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
