# Unity AI Gateway Coding CLI (ucode)

`ucode` is a lightweight launcher for running Codex, Claude Code, Gemini CLI, OpenCode, GitHub Copilot CLI, and Pi through Databricks.

## Requirements

- Python 3.12+ — install with `uv` ([uv.astral.sh](https://docs.astral.sh/uv/getting-started/installation/))
- `npm` if tool CLIs need to be installed automatically

## Installation

```bash
uv tool install git+https://github.com/databricks/ucode
```

Check your version with `ucode --version`. Between releases this looks like
`0.1.0+14.g93986a8` — the trailing `g<hash>` is the exact commit the build came
from, so include it when reporting a bug.

---

## Usage

Just run the tool you want:

```bash
ucode codex      # OpenAI Codex
ucode claude     # Claude Code
ucode gemini     # Gemini CLI
ucode opencode   # OpenCode
ucode copilot    # GitHub Copilot CLI
ucode pi         # Pi
ucode cursor     # Cursor Agent (MCP only — see below)
```

On first launch, `ucode` will prompt for your Databricks workspace URL, authenticate, and configure that tool automatically. Subsequent launches go straight to the agent.

Pass flags directly to the underlying tool:

```bash
ucode claude -r          # resume last session
ucode codex --full-auto
```

All agents route through Databricks AI Gateway using your workspace credentials — no API keys required.

Codex intelligent routing is opt-in. Enabling it asks the AI Gateway router to select the
root-session model before launch and installs profile-scoped hooks that route future
`spawn_agent` calls. Codex may require one-time review of the installed hooks through `/hooks`.

```bash
ucode codex --enable-intelligent-routing
```

The setting persists for the current workspace. Disable it and remove only ucode's routing
hooks with:

```bash
ucode codex --disable-intelligent-routing
```

To configure all tools at once:

```bash
ucode configure
```

To configure specific tools without the picker, pass a comma-separated list:

```bash
ucode configure --agents claude,codex
```

Available agent names are `codex`, `claude`, `gemini`, `opencode`, `copilot`, and `pi`. `cursor` is also accepted (MCP-only — it registers Databricks MCP servers but configures no models).

To configure without the workspace picker, pass a comma-separated list of workspaces:

```bash
ucode configure --workspaces https://first.databricks.com,https://second.databricks.com
```

When multiple workspaces are provided, `ucode` logs into and saves state for each workspace. Launch commands such as `ucode codex` use the first workspace in the list.

Alternatively, pass existing Databricks CLI profiles (from `~/.databrickscfg`) instead of workspace URLs — each profile's host supplies the workspace URL:

```bash
ucode configure --profiles DEFAULT --agents claude,codex
```

Auth behaves the same as `--workspaces`: an OAuth `databricks auth login` is forced by default.

For CI or headless environments where the profile holds a personal access token (`auth_type = pat` in `~/.databrickscfg`), add `--use-pat`. It must be combined with `--profiles` — ucode never picks up a PAT implicitly — and runs no interactive login: the profile's token is used for the whole setup (and by launched agents afterwards), with workspace access verified against the AI Gateway. `--skip-validate` additionally skips the post-configure test message sent through each agent, so configure only writes config files with the freshly discovered models. Together these make setup fully non-interactive:

```bash
ucode configure --profiles DEFAULT --agents claude,codex --use-pat --skip-validate --skip-upgrade
```

### MCP servers (optional)

```bash
ucode configure mcp
```

Add Databricks MCP servers to installed MCP-capable tools: Codex, Claude Code, Gemini CLI, OpenCode, GitHub Copilot CLI, and Cursor Agent.
Options are shown in this order:

- Discovered external MCP connections
- Databricks SQL
- Managed Databricks MCPs (Vector Search, UC Functions, etc.)
- Custom MCP server URL

Discovered external MCP connections are listed directly.

Every Databricks MCP server is registered as a local **stdio** server that runs `ucode mcp-proxy`
— a small bridge (shipped with `ucode`) between the coding tool and the Databricks
streamable-HTTP MCP endpoint. The proxy mints a fresh OAuth token from your Databricks CLI profile
on every request, so MCP auth is handled uniformly for every client and never expires mid-session.
The coding tool starts and stops the proxy as a child process; there's nothing extra to run.

**Cursor** is MCP-only: `cursor-agent` runs models on your own Cursor account, so `ucode`
configures no models for it — it only registers Databricks MCP servers in `~/.cursor/mcp.json`
(via the same proxy). Include it with `ucode configure --agents cursor` or pick it in
`ucode configure mcp`, then launch with `ucode cursor`.

To set up an agent and its MCP server(s) in one command, pass `--mcp` with fully-qualified
service name(s) to `ucode configure`:

```bash
ucode configure --agents claude --mcp system.ai.slack
```

`--mcp` also works without `--agents` for MCP-only clients (it configures just the workspace,
then registers the servers); pass a comma-separated list to register several at once.

### Skills (optional)

Configure Unity Catalog Skills for your coding tools with `ucode configure skills`:

```bash
# Utility tools only: register the schema-less skills MCP connection, no download.
ucode configure skills

# Download mode: fetch every skill in the schema to disk (and register the connection).
ucode configure skills --location main.default --path /abs/project/dir

# Download a named subset of the schema's skills instead of all of them.
ucode configure skills --location main.default --skill my-skill

# MCP mode: expose the schema's skills as MCP tools instead of downloading.
ucode configure skills --location main.default,ml.prod --mcp
```

- **Bare command** (no `--location`) registers the schema-less skills MCP connection — the
  cross-schema utility tools only — and downloads nothing. `--mcp` with no `--location` does the
  same.
- **Download mode** (with `--location`, no `--mcp`) writes each skill flat as `<leaf>/SKILL.md`
  (plus its bundled files) into both `.claude/skills/` and `.agents/skills/`. `--path` (an existing
  absolute directory) is optional; when omitted, skills are written under your home directory. Any
  pre-existing skill dir prompts before it's overwritten. It then registers a schema-less skills
  MCP connection, leaving any prior `--mcp` scope untouched. `--skill <name>[,<name>…]` narrows the
  download to the named skills (by leaf name) from the schema instead of all of them; requested
  names not found in the schema warn and are skipped. `--skill` requires a single `--location`, is
  download-only, and is rejected with `--mcp`.
- **MCP mode** (`--location … --mcp`) sets the connection's location set to exactly `<list>`
  (override-only) and rebuilds its `?schema=` URL; no files are downloaded and `--path` is rejected.

Each run prints the registered server, its URL, the configured agents, and its tools, and reminds
you to run `ucode <agent>` (existing agent sessions need a restart before the MCP tools load).

---

## Other Commands

| Command | Description |
|---------|-------------|
| `ucode status` | Show current workspace, base URLs, managed config files, and selected models |
| `ucode usage` | Show AI Gateway usage summary |
| `ucode revert` | Clear saved state and restore backed-up config files |
| `ucode configure --dry-run` | Preview config files without writing them |
| `ucode configure --agents claude,codex` | Configure specific agents without the interactive picker |
| `ucode configure --workspaces https://first.databricks.com,https://second.databricks.com` | Configure workspaces without the interactive picker |
| `ucode configure --profiles DEFAULT` | Configure using existing Databricks CLI profiles (hosts come from `~/.databrickscfg`) |
| `ucode configure --profiles DEFAULT --use-pat` | Authenticate with the profile's personal access token — no browser login |
| `ucode codex --enable-intelligent-routing` | Enable AI Gateway routing for Codex sessions and subagents |
| `ucode codex --disable-intelligent-routing` | Disable routing and remove ucode's Codex routing hooks |
| `ucode configure --skip-validate` | Write configs without sending a test message through each agent |
| `ucode configure --agents claude --mcp system.ai.slack` | Configure an agent and register its Databricks MCP server(s) in one command |
| `ucode configure skills` | Register the skills MCP connection (utility tools only); no skills download |
| `ucode configure skills --location main.default [--path <dir>]` | Download a schema's skills to disk (under `<dir>`, or your home dir) and register a schema-less skills MCP connection |
| `ucode configure skills --location main.default --skill my-skill` | Download only the named skill(s) from a schema (comma-separated for several) |
| `ucode configure skills --location main.default --mcp` | Expose a schema's skills as MCP tools (override-only) instead of downloading |

## Managed Local Files

`ucode` manages these files:

| File | Tool |
|------|------|
| `~/.codex/config.toml` | Codex |
| `~/.claude/settings.json` | Claude Code |
| `~/.gemini/.env` | Gemini CLI |
| `~/.config/opencode/opencode.json` | OpenCode |
| `~/.copilot/.env` | GitHub Copilot CLI |
| `~/.pi/agent/models.json` | Pi |
| `~/.cursor/mcp.json` | Cursor Agent (MCP servers only) |

Existing files are backed up before being overwritten. `ucode revert` restores backups.


## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Contributing

Contributions are welcome.

### Getting started

```bash
git clone https://github.com/databricks/ucode
cd ucode
uv sync
```

### Development workflow

1. Create a feature branch off `main`.
2. Make your changes — keep them scoped to the requested behavior.
3. Run the test suite before pushing:

   ```bash
   uv run pytest          # unit tests
   uv run ruff check .    # lint
   ```

4. For end-to-end testing against a real workspace:

   ```bash
   UCODE_TEST_WORKSPACE=<db_workspace_url> uv run pytest tests/test_e2e.py -v
   ```

5. Open a pull request against `main`.

### Adding a new agent

- Add `src/ucode/agents/<name>.py` with at least `write_tool_config`, `launch`, `default_model`, and `validate_cmd`.
- Register it in `src/ucode/agents/__init__.py`.
- Add focused tests under `tests/`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
