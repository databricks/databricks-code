# OS-Managed Agent Settings

## Summary

Claude Code and Codex give OS-managed settings higher precedence than user settings. Previously,
ucode could write only its local configuration while an existing machine-managed file silently
overrode the gateway endpoint, authentication helper, provider headers, or model.

The two stacked PRs make precedence handling deterministic:

1. The Claude PR adds the shared managed-file lifecycle and applies it to Claude Code.
2. The Codex PR reuses that lifecycle for TOML, applies it to Codex, and removes the old optional
   managed-settings path.

Claude and Codex configuration is local-only by default. OS-managed synchronization requires the
explicit `--sync-managed-settings` flag and applies to the supported agents selected by the normal
configure flow. Launches never write managed files or elevate privileges; they only verify that
existing higher-precedence values are compatible.

## Configuration Files

| Agent | Local ucode configuration | OS-managed configuration |
| --- | --- | --- |
| Claude Code | `~/.claude/ucode-settings.json` | Linux: `/etc/claude-code/managed-settings.json`; macOS: `/Library/Application Support/ClaudeCode/managed-settings.json` |
| Codex | `~/.codex/ucode.config.toml` | `/etc/codex/managed_config.toml` |

The local file is always written. OS-managed files are changed only by explicit synchronization or
restoration, except for Claude subscription relay, which is always local-only.

## Explicit Synchronization

`ucode configure` uses local settings. `--sync-managed-settings` is the user's explicit consent to
configure selected Claude Code and Codex agents machine-wide. The normal picker still runs when no
agents are named, and the command fails before agent installation or configuration when none of the
final selections support synchronization. Because a write may require administrator approval, an
actual change still requires standard input to be a TTY. An already-identical or compatible external
file needs no elevation.

TTY state is only a privilege boundary; it does not select configuration scope and never changes
whether a cached launch is valid.

## Behavior Matrix

| Invocation | Managed file | Behavior |
| --- | --- | --- |
| Plain configure or launch | Absent | Use the local ucode file. Do not create the managed file. |
| Plain configure or launch | Compatible | Use the local ucode file. Do not modify the managed file. |
| Plain configure or launch | Conflicting | Stop because the higher-precedence value would override ucode. |
| Explicit sync | Absent | Create it after recording an absent baseline. |
| Explicit sync | Last version written by ucode | Update and verify it. |
| Explicit sync | Compatible external file | Leave it untouched. |
| Explicit sync | Conflicting or externally modified file | Refuse to overwrite it. |
| Any | Invalid, unreadable, or symlinked | Stop without modifying the file because precedence cannot be established safely. |

First-time and later `ucode claude` and `ucode codex` launches use only the read-only compatibility
path. They never request administrator permission.

## Explicit Reconciliation

For each agent, ucode:

1. Strictly parses the existing managed JSON or TOML document.
2. Produces the desired document by applying the same gateway overlay used for the local ucode file.
3. Preserves settings outside the paths owned by ucode.
4. Preserves enterprise Claude permission-deny entries while adding ucode-required entries.
5. Records the original baseline before the first change.
6. Requests administrator permission and performs an atomic privileged replacement.
7. Reads the installed file back and verifies its exact contents.
8. Records the last-applied snapshot, owned paths, and a launch fingerprint.

An existing file is updated only when it exactly matches ucode's last-applied snapshot. A compatible
external file remains untouched, and a conflicting or externally modified file is preserved and
reported.

## Privileged Write Transaction

The shared writer handles ordinary MDM-installed and root-owned files rather than treating them as
errors:

- refuses symlink destinations;
- creates a root-owned destination directory when it is absent;
- stages the new file in the destination directory for a same-filesystem atomic rename;
- clones an existing file before replacing its contents, preserving ownership, mode, ACLs, and
  extended attributes;
- creates new files as root-owned and world-readable;
- detects, clears, and restores macOS `schg`, `uchg`, `sappnd`, and `uappnd` flags;
- detects, clears, and restores Linux immutable and append-only attributes;
- verifies the resulting contents after replacement;
- retries once only when device management restored the exact pre-write contents;
- preserves a concurrently changed policy instead of overwriting it.

The privilege boundary is interactive. Non-interactive paths do not invoke either normal `sudo` or
`sudo -n`.

Root access cannot sustainably override an actively enforced policy. If an MDM process immediately
restores the original file twice, ucode stops with an error instead of repeatedly fighting the
management agent. A different concurrent update is also preserved and reported.

## Backup Model

Backups live under `~/.ucode/managed-backups/` with directory mode `0700` and file mode `0600`.
The manifest records, per agent:

- whether the managed file originally existed;
- its absolute path;
- the baseline snapshot and SHA-256 digest;
- the last ucode-applied snapshot and SHA-256 digest;
- the setting paths owned by ucode.

The baseline is created before the first managed change and is not replaced by subsequent
configurations. Later configurations update only the last-applied snapshot. Snapshot filenames are
validated, digests are checked before use, and symlinked backup directories or manifests are
rejected.

If ucode cannot complete a write or revert, the backup remains available for a later retry.

## `ucode revert`

`ucode revert` extends the existing local-file restoration with managed-file restoration:

- If the current file exactly matches ucode's last-applied snapshot, restore the exact baseline.
- If ucode created the file, delete it.
- If an administrator or MDM changed the file later, perform a three-way revert.
- Restore original values only where the current value still matches ucode's last-applied value.
- Remove only list entries added by ucode while preserving externally added entries.
- Preserve externally changed values rather than replacing them with stale baseline values.
- Refuse an unsafe or unparsable merge and retain the backup.

A plain `ucode configure --agent claude` performs the same restoration before writing new local
settings, making it the opt-out path after explicit synchronization. A restoration that needs to
change an OS-managed file must run interactively. If restoration fails, local configuration can
continue, the backup remains available, and launches do not retry elevation. A successful
restoration removes that agent's backup record.

## Cached Launches

After a successful managed reconciliation or compatibility check, ucode stores:

- the managed path;
- the verification scope;
- device and inode numbers;
- size;
- nanosecond modification and change times.

A missing managed file is valid without a fingerprint. For an existing file, a cached launch
compares its fingerprint. When it matches, ucode does not parse the file. When it changes, ucode
runs the read-only compatibility path again. Launches never back up, write, restore, or invoke
`sudo`.

The verification scopes distinguish:

- an interactively reconciled managed file;
- a managed file verified as compatible with local settings;
- a managed file verified as compatible with Claude relay.

Compatibility fingerprints are valid independently of TTY state. Only the explicit synchronization
flag selects the write path.

## Claude Subscription Relay

Claude relay is intentionally local-only. It routes through a loopback refresh proxy whose address
and lifetime belong to one `ucode claude` session. Persisting that address in OS-managed settings
would break bare `claude` launches and future sessions.

Relay therefore never creates or updates the managed file. It allows an absent file or unrelated
enterprise settings, but blocks these higher-precedence conflicts:

- `apiKeyHelper`;
- `env.ANTHROPIC_BASE_URL`;
- `env.ANTHROPIC_CUSTOM_HEADERS`.

If ucode wrote those values during an earlier standard configuration, the user runs `ucode revert`
interactively before switching to relay. External conflicting values require administrator action or
standard Databricks authentication.

## Status and Messages

`ucode status` reports, for Claude and Codex:

- managed settings path;
- state: not configured, current, compatible local settings, compatible relay settings, drifted,
  invalid, unreadable, missing, or unsupported;
- whether a managed baseline backup is available.

Explicit updates announce the backup location, administrator-permission request, and verified
result. An identical or compatible external file produces no elevation message.

Representative blockers are:

```text
Claude Code OS-managed settings at <path> override ucode values: env.ANTHROPIC_BASE_URL. ucode did
not modify the file during this launch. Contact your administrator.
```

```text
Claude Code managed settings at <path> were updated but immediately restored by device management.
Contact your administrator.
```

```text
Cannot safely update Codex managed settings at <path>: <parse error>. ucode did not modify the file.
Repair it or contact your administrator.
```

Parse, symlink, unreadable-file, and managed-conflict failures block launch. Write and restoration
failures occur only during explicit configure or revert operations. Recovery information identifies
the file and recommends configure, revert, or administrator help.

## PR Boundaries

### PR 1: Claude Code

- Add shared strict reads, fingerprints, compatibility checks, secure backups, atomic privileged
  writes, immutable-flag handling, verification, status, and three-way revert helpers.
- Make Claude configuration local-only by default and add explicit managed synchronization.
- Make every launch read-only with conflict detection.
- Restore prior ucode-managed changes during plain configure and revert.
- Add Claude relay-specific safety checks.
- Add Claude managed status and revert output.
- Remove Claude's old managed-settings scope choice.

### PR 2: Codex

- Stack on the Claude PR and reuse the shared lifecycle with strict TOML parsing and serialization.
- Make Codex configuration local-only by default and add explicit managed synchronization.
- Make every launch read-only with conflict detection.
- Restore prior ucode-managed changes during plain configure and revert.
- Add Codex fingerprinted cached launches, status, and revert behavior.
- Keep opt-in smart-routing hooks in the local Codex config rather than adding them by default to
  machine-managed policy.
- Remove the remaining managed-settings scope schema, resolution, setup prompt, summary, and tests.
