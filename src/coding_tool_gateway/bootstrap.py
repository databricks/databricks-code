"""Best-effort runtime/bootstrap installer for coding-gateway dependencies."""

from __future__ import annotations

from coding_tool_gateway.cli import TOOL_SPECS, ensure_bootstrap_dependencies, print_err


def main() -> int:
    try:
        for tool in TOOL_SPECS:
            ensure_bootstrap_dependencies(tool)
    except RuntimeError as exc:
        print_err(f"coding-gateway bootstrap failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
