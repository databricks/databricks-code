"""Best-effort runtime/bootstrap installer for databricks-code dependencies."""

from __future__ import annotations

from databricks_code.cli import TOOL_SPECS, ensure_bootstrap_dependencies, print_err


def main() -> int:
    try:
        for tool in TOOL_SPECS:
            ensure_bootstrap_dependencies(tool)
    except RuntimeError as exc:
        print_err(f"databricks-code bootstrap failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
