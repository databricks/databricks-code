# Unity AI Gateway Coding CLI (ucode)

`ucode` is a lightweight launcher for running Codex, Claude Code, Gemini CLI, OpenCode, GitHub Copilot CLI, Pi, and Continue through Databricks.

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
ucode continue   # Continue (cn CLI + VS Code/JetBrains extensions)
ucode cursor     # Cursor Agent (MCP only — see below)
```

On first launch, `ucode` will prompt for your Databricks workspace URL, authenticate, and configure that tool automatically. Subsequent Claude and Codex launches use the generated local settings directly. Use `ucode claude --refresh` or `ucode codex --refresh` when you want to re-check Databricks and update the model/configuration.

Pass flags directly to the underlying tool:

```bash
ucode claude -r          # resume last session
ucode codex --full-auto
```

All agents route through Databricks AI Gateway using your workspace credentials — no API keys required.

Smart routing is opt-in for Codex and Claude Code. Enabling it asks the AI Gateway router
to select the root-session model before launch and installs profile-scoped hooks that route
future subagent calls. Codex may require one-time review of the installed hooks through `/hooks`.

```bash
ucode codex --enable-smart-routing
ucode claude --enable-smart-routing
```

The setting persists per workspace for each agent. Disable and remove only ucode's routing
hooks with:

```bash
ucode codex --disable-smart-routing
ucode claude --disable-smart-routing
```

To configure all tools at once:

```bash
ucode configure
```

To configure specific tools without the picker, pass a comma-separated list:

```bash
ucode configure --agents claude,codex
```

Available agent names are `codex`, `claude`, `gemini`, `opencode`, `copilot`, `pi`, and `continue`. `cursor` is also accepted (MCP-only — it registers Databricks MCP servers but configures no models).

`continue` writes `~/.continue/config.yaml`, which is shared by the `cn` CLI and the Continue VS Code/JetBrains extensions — so configuring it once wires up both. Its chat/edit/apply roles route to your Databricks Claude/GPT models through the Unity AI Gateway.

Naming agents explicitly is treated as a request for all of them: if any one isn't available on the workspace, the run fails without configuring the others. Add `--skip-unavailable` to configure the available subset instead and skip the rest with a warning:

```bash
ucode configure --agents claude,codex,pi --skip-unavailable
```

This is useful in CI against a mix of workspaces — on a workspace whose AI Gateway exposes no OpenAI models, the command above still configures `claude` and `pi`, and reports Codex as skipped. It exits non-zero only when none of the requested agents are available.

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

Add Databricks MCP servers to installed MCP-capable tools: Codex, Claude Code, Gemini CLI, OpenCode, GitHub Copilot CLI, Continue, and Cursor Agent.
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

#### Add servers without replacing existing ones

`ucode configure mcp` **replaces** the registered MCP servers with your selection — anything
outside a `--location`/`--services` scope (or left unchecked in the picker) is removed. To
**add** servers while leaving everything already configured in place, use `ucode mcp add`:

```bash
# Register a whole schema's services, keeping any servers already configured.
ucode mcp add --location system.ai

# Register just a subset (same name rules as `configure mcp --services`).
ucode mcp add --services system.ai.slack,system.ai.github

# No arguments launches the same interactive picker, but never removes servers.
ucode mcp add
```

`ucode mcp add` takes the same `--location` and `--services` options as `ucode configure mcp`;
the only difference is that it never removes servers outside the selection. In the interactive
picker, servers you already have configured are shown as `(already configured)` and can't be
toggled off — you only pick new ones to add.

Pass `--agents` to target specific coding agents. Any named agent that isn't set up yet is
configured first (workspace + models), so this doubles as one-command setup:

```bash
# Set up Claude Code (if needed) and register the server for it, in one command.
ucode mcp add --agents claude --services system.ai.slack

# Target several agents at once.
ucode mcp add --agents claude,codex --location system.ai
```

Without `--agents`, the server is registered for every already-configured agent.

#### Remove configured servers

To unregister servers you've already configured, use `ucode mcp remove`:

```bash
ucode mcp remove

# Remove only from specific agents. A server registered on several agents is
# unregistered from the named ones and kept on the rest.
ucode mcp remove --agents codex
```

It shows the servers you currently have configured — each with the coding tools it's registered
on — and removes the ones you select from those tools. It needs no Databricks login.

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

### Managed config for a workspace (admins)

Author the coding config your developers pick up automatically, instead of asking each of them to
run `ucode configure` by hand. Restricted to workspace admins. `ucode setup help` prints the whole
sequence; the short version is one command for the agents and models, then a command per optional
section, then publish:

```bash
ucode setup                 # agents and models (start here)
ucode setup mcps            # managed MCP servers
ucode setup skills          # managed skills
ucode setup spend-tiers     # spend-based routing
ucode publish                 # publish it to the workspace
```

`ucode setup` walks through the agents to enable and which one bare `ucode` launches, then per agent:
Databricks-hosted models or an external Model Provider Service, the models to expose, and (for Codex)
whether the config writes the agent's own OS-level settings file or a ucode-only one. Interactive
Claude Code configuration installs its gateway configuration in the OS-managed settings scope so
enterprise settings cannot silently override ucode. Non-interactive and CI runs use the local file
without invoking `sudo`, and stop with an actionable error if an existing managed value conflicts.
Claude subscription relay is local-only because its loopback proxy exists only for that session.
Claude Code is asked one model per family (opus/sonnet/haiku/fable), since it selects models by family
alias; any family can be skipped.

The optional sections each edit their own part of the same config, so you can add an MCP server or
change a spend tier later without walking the whole flow. `ucode setup skills --location
main.default,other.schema` skips the prompt. `ucode setup spend-tiers` sets a tiered spend policy
that switches the default agent and model as the workspace burns through a budget. Each section
command also offers to publish right away, so you can apply changes incrementally; answering the
section prompts also runs the matching `ucode configure` step, which does configure this machine.

Everything is written to `~/.ucode/managed-state.json` — the one local managed-config file — which
`ucode publish` publishes. Re-running `ucode setup` keeps the MCP servers, skills, tracing table, and
tiered spend policy already authored, rather than clearing them; to drop one, edit the file and reload
it with `ucode setup --from-file`.

```bash
# Review the manifest and the exact payload `ucode publish` would publish.
ucode setup show

# Skip the prompts and load a hand-written config instead (validated before saving).
ucode setup --from-file ./managed-config.json
```

Once the manifest looks right, publish it:

```bash
# Validate, show a diff against what's live, and ask before publishing.
ucode publish

# Publish without the confirmation prompt (for CI).
ucode publish --yes

# Publish a config file exported with `ucode export` instead of the locally authored one.
ucode publish -f ./managed-config.json
ucode publish --file ./managed-config.json --yes
```

`publish` updates the workspace's existing config in place rather than replacing it, so a failed
publish leaves the current config intact. It shows a diff of exactly what changes against the
published config before asking to confirm, and does nothing when the two already match. It is a
whole-manifest write — every field ucode authors is sent — but because `ucode setup` carries the
other sections forward, a re-run no longer silently drops them. Developers pick the new config up on
their next ucode run.

With `-f`/`--file`, `publish` reads a config file produced by `ucode export` and publishes it through
the same validation, diff, and confirmation flow. The file's `workspace` must match the configured
workspace (it can never redirect publication elsewhere) and its `spec_version` must be a supported
integer; server-owned fields (resource name, workspace ids, timestamps, user ids) and unknown fields
are rejected rather than silently dropped.

### Exporting the config

Any user (not only admins) can print the workspace's managed config as portable JSON with `ucode
export`. The output leads with the source `workspace` URL and a `spec_version` (the export format
version), followed by the canonical external config; credentials and server-assigned fields (the
resource name, timestamps, user ids) are excluded. Without `--file` the JSON is written to stdout;
with `--file`/`-f` the same bytes are written to a file (atomically, and the destination's parent
directory must already exist) while stdout stays empty. The exported file is exactly what `ucode
publish -f <file>` consumes.

```bash
# Print the managed config as JSON.
ucode export

# Write it to a file; stdout stays empty.
ucode export --file ./managed-config.json
```

The output looks like:

```json
{
  "workspace": "https://<workspace-host>",
  "spec_version": 1,
  "default_agent": "CODING_AGENT_CLAUDE_CODE",
  "enabled_agents": [ ... ]
}
```

---

## Other Commands

| Command | Description |
|---------|-------------|
| `ucode status` | Show current workspace, base URLs, managed config files, and selected models |
| `ucode export` | Print the workspace's managed config as portable JSON (`--file <file>` / `-f` to write a file) |
| `ucode usage` | Show AI Gateway usage summary, plus your budget spend against its alert threshold when the workspace reports one |
| `ucode usage --warehouse-id <id>` | Query a specific SQL warehouse instead of discovering one |
| `ucode revert` | Clear saved state and restore backed-up config files |
| `ucode configure --dry-run` | Preview config files without writing them |
| `ucode configure --agents claude,codex` | Configure specific agents without the interactive picker |
| `ucode configure --workspaces https://first.databricks.com,https://second.databricks.com` | Configure workspaces without the interactive picker |
| `ucode configure --profiles DEFAULT` | Configure using existing Databricks CLI profiles (hosts come from `~/.databrickscfg`) |
| `ucode configure --profiles DEFAULT --use-pat` | Authenticate with the profile's personal access token — no browser login |
| `ucode codex --enable-smart-routing` | Enable AI Gateway routing for Codex sessions and subagents |
| `ucode codex --disable-smart-routing` | Disable routing and remove ucode's Codex routing hooks |
| `ucode codex --refresh` | Re-check Databricks, refresh models/configuration, and launch Codex |
| `ucode claude --enable-smart-routing` | Enable AI Gateway routing for Claude Code sessions and subagents |
| `ucode claude --disable-smart-routing` | Disable routing and remove ucode's Claude Code routing hooks |
| `ucode claude --refresh` | Re-check Databricks, refresh models/configuration, and launch Claude Code |
| `ucode configure --skip-validate` | Write configs without sending a test message through each agent |
| `ucode configure --agents claude,codex,pi --skip-unavailable` | Configure the requested agents that are available; skip the rest with a warning |
| `ucode configure --agents claude --mcp system.ai.slack` | Configure an agent and register its Databricks MCP server(s) in one command |
| `ucode mcp add --location system.ai` | Register a schema's MCP servers, keeping any already configured (additive; never removes) |
| `ucode mcp add --services system.ai.slack` | Register specific MCP server(s) without removing existing ones |
| `ucode mcp add --agents claude --services system.ai.slack` | Set up the agent(s) if needed and register the server for them |
| `ucode mcp remove` | Interactively unregister configured MCP servers from your coding tools |
| `ucode mcp remove --agents codex` | Unregister selected servers from specific agents only |
| `ucode configure skills` | Register the skills MCP connection (utility tools only); no skills download |
| `ucode configure skills --location main.default [--path <dir>]` | Download a schema's skills to disk (under `<dir>`, or your home dir) and register a schema-less skills MCP connection |
| `ucode configure skills --location main.default --skill my-skill` | Download only the named skill(s) from a schema (comma-separated for several) |
| `ucode configure skills --location main.default --mcp` | Expose a schema's skills as MCP tools (override-only) instead of downloading |
| `ucode setup` | Author the managed config's agents and models (workspace admins only) |
| `ucode setup mcps` | Add or change the managed config's MCP servers |
| `ucode setup skills [--location a.b,c.d]` | Add or change the managed config's skills |
| `ucode setup spend-tiers` | Set the managed config's tiered spend routing policy |
| `ucode setup help` | Walk through the whole setup sequence, marking what's already configured |
| `ucode setup show` | Print the authored config and the payload `ucode publish` would publish |
| `ucode setup --from-file <file>` | Load a hand-written managed config instead of running the prompts |
| `ucode publish` | Publish the authored managed config to the workspace, after a diff and confirmation (admins only) |
| `ucode publish -f <file>` | Publish a config file exported with `ucode export` instead of the locally authored one |
| `ucode publish --yes` | Publish without the confirmation prompt |

Databricks AI Tools are installed only by `ucode configure`, never by `ucode <agent>` launches.
Use `--enable-databricks-ai-tools` or `--disable-databricks-ai-tools` with `ucode configure` to
control the installation.

## Managed Local Files

`ucode` manages these files:

| File | Tool |
|------|------|
| `~/.codex/ucode.config.toml` (or legacy `~/.codex/config.toml`) | Codex |
| `~/.claude/ucode-settings.json` | Claude Code settings generated by ucode |
| `/etc/claude-code/managed-settings.json` (Linux) or `/Library/Application Support/ClaudeCode/managed-settings.json` (macOS) | Claude Code OS-managed settings |
| `~/.gemini/.env` | Gemini CLI |
| `~/.config/opencode/opencode.json` | OpenCode |
| `~/.copilot/.env` | GitHub Copilot CLI |
| `~/.pi/agent/models.json` | Pi |
| `~/.continue/config.yaml` | Continue (`cn` CLI + IDE extensions) |
| `~/.cursor/mcp.json` | Cursor Agent (MCP servers only) |
| `~/.ucode/managed-state.json` | The managed config — authored by `ucode setup` (admins) and refreshed from the workspace on launch |
| `~/.ucode/managed-backups/` | Baseline backups for OS-managed files changed by ucode |

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
