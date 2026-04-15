# dbx-code

`dbx-code` is a lightweight launcher for running Codex, Claude Code, and Gemini CLI through Databricks.

It is designed for a simple setup flow:

```bash
git clone <repo-url>
cd dbx-code
pip install .
dbx-code
```

On first run, `dbx-code` handles local bootstrap, prompts for your Databricks workspace, configures the selected coding tool, and launches it.

## Why dbx-code

- Minimal setup for Databricks-backed coding tools
- One workspace configuration shared across Codex, Claude Code, and Gemini CLI
- Automatic Databricks authentication handoff
- Managed local config files with restore support through `dbx-code logout`
- Built-in AI Gateway usage reporting with `dbx-code usage`

## Supported Tools

- Codex
- Claude Code
- Gemini CLI

## Quick Start

Install and launch:

```bash
git clone <repo-url>
cd dbx-code
pip install .
dbx-code
```

On first launch, `dbx-code` will:

1. Install missing local dependencies it manages:
   `databricks`, `codex`, `claude`, `gemini`
2. Prompt for your Databricks workspace URL
3. Ask whether you want to use Databricks AI Gateway V2
4. Run Databricks login
5. Write managed tool configuration files
6. Launch the selected tool

After setup, normal usage is simple:

```bash
dbx-code
dbx-code --tool codex
dbx-code --tool claude
dbx-code --tool gemini
```

## Configuration

Use `configure` to update workspace settings or saved models:

```bash
dbx-code configure
```

The interactive flow supports:

- `Workspace`
  Reconfigure the Databricks workspace for all tools
- `Models`
  Choose the saved launch model for Claude Code or Gemini CLI

Examples:

```bash
dbx-code --tool claude --model databricks-claude-opus-4-6
dbx-code --tool gemini --model databricks-gemini-2-5-pro
```

Notes:

- Codex model selection remains inside Codex via `/model`
- Claude and Gemini can use saved models or an explicit `--model`
- Normal launches do not query the workspace model list unless you are configuring models

## Commands

```bash
dbx-code configure
dbx-code status
dbx-code usage
dbx-code logout
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
dbx-code usage
```

`usage` requires Databricks AI Gateway V2. When enabled, it queries `system.ai_gateway.usage` and shows:

- token totals for today, the last 7 days, and the last 30 days
- active tools this week
- top models this week
- a 7-day breakdown for Codex, Claude Code, and Gemini CLI

If the workspace is not configured for AI Gateway V2, `dbx-code usage` will stop early and tell you to re-run `dbx-code configure`.

## Managed Local Files

`dbx-code` manages these local files:

- `~/.codex/config.toml`
- `~/.claude/settings.json`
- `~/.gemini/.env`

If one of these files already exists, `dbx-code` creates a backup before writing its managed version. `dbx-code logout` restores the backup.

## Authentication

- Databricks authentication always uses the workspace URL you configured
- If AI Gateway V2 is enabled, tool base URLs point to the AI Gateway hostname while authentication still uses the original workspace URL
- Codex and Claude use a Databricks token helper instead of storing a fixed token
- Gemini refreshes its Databricks bearer token automatically while launched through `dbx-code`
- `dbx-code usage` fetches a fresh Databricks token on each run

## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage with `system.ai_gateway.usage`](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Requirements

- Python 3.9+
- `pip`
- `npm` if tool CLIs need to be installed automatically

