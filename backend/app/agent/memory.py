"""Bounded source-scoped evidence cache; cached content remains untrusted."""

import re
from typing import Any

from backend.app.agent.schemas import WorkingMemory
from backend.app.tools.base import ToolResult


class MemoryManager:
    def __init__(self, initial_memory: WorkingMemory | None = None):
        self.memory = initial_memory or WorkingMemory()

    def update_from_tool_result(self, tool_name: str, tool_args: dict[str, Any], result: ToolResult) -> None:
        if not result.ok:
            self.add_note(f"{tool_name}: {result.error}")
            return
        data = result.data
        if not isinstance(data, dict):
            return
        if tool_name == "document_read":
            source_id = data.get("filename")
            if not source_id:
                return
            content = data.get("content", "")[:12000]
            facts = {}
            for line in content.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = re.sub(r"\W+", "_", key.strip().lower()).strip("_")
                    if key and value.strip():
                        facts[key] = value.strip()
            self.memory.sources[source_id] = {"source_id": source_id, "metadata": {k: v for k, v in data.items() if k != "content"}, "facts": facts, "content": content}
            self.memory.selected_source = source_id
            self.memory.extracted_facts = facts.copy()
            self.memory.source_references = {key: source_id for key in facts}
            self.add_note(f"Read source {source_id}; facts belong only to this source.")
            while len(self.memory.sources) > 8:
                del self.memory.sources[next(iter(self.memory.sources))]
        elif tool_name.startswith("browser_"):
            inspection = data.get("inspection") or data
            url = inspection.get("url") or data.get("current_url") or data.get("page_url")
            if url:
                page = self.memory.pages.get(url, {})
                page.update({k: v for k, v in inspection.items() if k in ("title", "page_text_summary", "interactive_elements", "tables", "feedback")})
                page["source_id"] = url
                self.memory.pages[url] = page
                while len(self.memory.pages) > 4:
                    del self.memory.pages[next(iter(self.memory.pages))]
            if data.get("success_message"):
                self.add_note(str(data["success_message"]))

    def add_note(self, note: str) -> None:
        self.memory.recent_notes = (self.memory.recent_notes + [note[:500]])[-10:]

    def get_snapshot(self) -> dict[str, Any]:
        return self.memory.model_dump()
