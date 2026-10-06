"""
Tool registry for Operon.
Registers tools, validates inputs against JSON schemas, and dispatches execution safely.
"""

import inspect
from jsonschema import Draft202012Validator, ValidationError as SchemaError
from typing import Any, Callable, Optional
from backend.app.tools.base import ToolDefinition, ToolResult
from backend.app.tools.document_tools import document_search, document_read
from backend.app.tools.browser_tools import browser_manager

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict, func: Callable) -> None:
        parameters = {"type": "object", "properties": {}, **parameters, "additionalProperties": False}
        Draft202012Validator.check_schema(parameters)
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func
        )

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Returns tool declarations formatted for model prompts and structured schemas."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        return schemas

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """
        Validates arguments and executes the requested tool.
        Returns a ToolResult envelope under all circumstances.
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                ok=False,
                error=f"Unknown tool '{tool_name}'. Available tools: {list(self._tools.keys())}",
                error_code="UNKNOWN_TOOL"
            )

        args = dict(args)
        aliases = [key for key in args if key.startswith("@")]
        if aliases:
            supported = tool_name in ("browser_type", "browser_select", "browser_click")
            if not supported or len(aliases) != 1 or "element_id" in args:
                return ToolResult(ok=False, error="Ambiguous or unsupported semantic alias", error_code="INVALID_ARGUMENTS")
            alias = aliases[0]
            value = args.pop(alias)
            args["element_id"] = alias
            value_key = {"browser_type": "text", "browser_select": "value"}.get(tool_name)
            if value_key:
                if value_key in args or not isinstance(value, str):
                    return ToolResult(ok=False, error="Semantic alias requires one string value", error_code="INVALID_ARGUMENTS")
                args[value_key] = value
            elif value not in (None, True):
                return ToolResult(ok=False, error="Click alias value must be null or true", error_code="INVALID_ARGUMENTS")
        try:
            Draft202012Validator(tool.parameters).validate(args)
        except SchemaError as error:
            code = "MISSING_ARGUMENTS" if error.validator == "required" else "INVALID_ARGUMENTS"
            return ToolResult(ok=False, error=f"Invalid arguments for {tool_name}: {error.message}", error_code=code)

        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**args)
            else:
                result = tool.func(**args)

            if isinstance(result, ToolResult):
                return result
            return ToolResult(ok=True, data=result)
        except TypeError as e:
            return ToolResult(
                ok=False,
                error=f"Argument mismatch executing '{tool_name}': {str(e)}",
                error_code="ARGUMENT_MISMATCH"
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                error=f"Error executing '{tool_name}': {str(e)}",
                error_code="EXECUTION_ERROR"
            )

def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # 1. document_search
    registry.register(
        name="document_search",
        description="Search company documents (invoices, complaints, contracts) by keyword query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term, vendor name, or document type"}
            },
            "required": ["query"]
        },
        func=document_search
    )

    # 2. document_read
    registry.register(
        name="document_read",
        description="Read the complete text of a document from the company repository by its filename.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Exact filename, e.g. invoice_document.pdf"}
            },
            "required": ["filename"]
        },
        func=document_read
    )

    # 3. browser_open
    registry.register(
        name="browser_open",
        description="Open a workspace URL in the browser, e.g. '/workspace/finance', '/workspace/crm', '/workspace/support'.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target path, e.g. /workspace/finance"}
            },
            "required": ["url"]
        },
        func=browser_manager.browser_open
    )

    # 4. browser_inspect
    registry.register(
        name="browser_inspect",
        description="Inspect the current browser page to get visible text, interactive elements (@id, labels, values), and feedback banners.",
        parameters={"type": "object", "properties": {}},
        func=browser_manager.browser_inspect
    )

    # 5. browser_type
    registry.register(
        name="browser_type",
        description="Type text into an input or textarea element by its semantic ID (e.g. '@company', '@amount'). Clears field first by default.",
        parameters={
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "Semantic element ID, e.g. @company or @amount"},
                "text": {"type": "string", "description": "Text to enter into the field"},
                "clear": {"type": "boolean", "default": True, "description": "Clear before typing"}
            },
            "required": ["element_id", "text"]
        },
        func=browser_manager.browser_type
    )

    # 6. browser_select
    registry.register(
        name="browser_select",
        description="Select an option from a dropdown element by its semantic ID.",
        parameters={
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "Semantic select element ID, e.g. @currency"},
                "value": {"type": "string", "description": "Option value to select, e.g. 'INR'"}
            },
            "required": ["element_id", "value"]
        },
        func=browser_manager.browser_select
    )

    # 7. browser_click
    registry.register(
        name="browser_click",
        description="Click an interactive button, link, or tab by its semantic ID (e.g. '@submit_invoice', '@tab_crm'). Observes server feedback and errors.",
        parameters={
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "Semantic element ID, e.g. @submit_invoice"}
            },
            "required": ["element_id"]
        },
        func=browser_manager.browser_click
    )

    # 8. browser_back
    registry.register(
        name="browser_back",
        description="Navigate back to the previous page in browser history.",
        parameters={"type": "object", "properties": {}},
        func=browser_manager.browser_back
    )

    return registry
