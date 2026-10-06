"""Reuse the established browser interaction/security boundary."""
from backend.app.tools.registry import build_default_tool_registry


def register_browser(registry):
    existing = build_default_tool_registry()
    for tool in existing.list_tools():
        if tool.name.startswith('browser_'):
            registry.register(tool.name, tool.description, tool.parameters, tool.func)
            registry.categories[tool.name] = 'browser'
