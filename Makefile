SANDBOX_HOME := /tmp/coding-gateway-sandbox

.PHONY: run configure status logout usage mcp clean test lint

# Development commands — all run in sandbox to protect real configs
run: ## Launch tool in sandbox (use TOOL=claude, ARGS="--model x")
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway $(if $(TOOL),--tool $(TOOL)) $(ARGS)

configure: ## Run configure in sandbox
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway configure $(ARGS)

mcp: ## Configure MCP servers in sandbox
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway configure mcp $(ARGS)

status: ## Show status from sandbox state
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway status

logout: ## Clear sandbox state
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway logout

usage: ## Show usage stats from sandbox
	CODING_GATEWAY_CONFIG_HOME=$(SANDBOX_HOME) uv run coding-gateway usage $(ARGS)

inspect: ## Show what the sandbox wrote
	@echo "==> Sandbox configs:"
	@ls -la $(SANDBOX_HOME)/.claude/ 2>/dev/null || echo "  .claude/ not created"
	@ls -la $(SANDBOX_HOME)/.codex/ 2>/dev/null || echo "  .codex/ not created"
	@ls -la $(SANDBOX_HOME)/.gemini/ 2>/dev/null || echo "  .gemini/ not created"
	@ls -la $(SANDBOX_HOME)/.coding-gateway/ 2>/dev/null || echo "  .coding-gateway/ not created"

clean: ## Nuke sandbox and start fresh
	rm -rf $(SANDBOX_HOME)
	@echo "Sandbox cleaned"

test: ## Run tests
	uv run pytest $(ARGS)

lint: ## Run linter
	uv run ruff check src/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
