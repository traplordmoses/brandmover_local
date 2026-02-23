"""
Per-generation resource tracker.
Records what brand assets, Figma nodes, scripts, and APIs were consulted during a single agent run.
"""

from dataclasses import dataclass, field


@dataclass
class ResourceTracker:
    """Tracks resources consulted during a single agent generation."""

    files_loaded: list[str] = field(default_factory=list)
    figma_nodes_checked: list[str] = field(default_factory=list)
    scripts_executed: list[str] = field(default_factory=list)
    apis_called: list[str] = field(default_factory=list)

    def log_file(self, name: str) -> None:
        if name not in self.files_loaded:
            self.files_loaded.append(name)

    def log_figma(self, node_id: str) -> None:
        if node_id not in self.figma_nodes_checked:
            self.figma_nodes_checked.append(node_id)

    def log_script(self, name: str) -> None:
        if name not in self.scripts_executed:
            self.scripts_executed.append(name)

    def log_api(self, name: str) -> None:
        if name not in self.apis_called:
            self.apis_called.append(name)

    def to_list(self) -> list[str]:
        """Flat list of all resources for feedback logging."""
        items = []
        for f in self.files_loaded:
            items.append(f"file:{f}")
        for n in self.figma_nodes_checked:
            items.append(f"figma:{n}")
        for s in self.scripts_executed:
            items.append(f"script:{s}")
        for a in self.apis_called:
            items.append(f"api:{a}")
        return items

    def to_summary(self) -> str:
        """Human-readable summary for Telegram display."""
        parts = []
        if self.files_loaded:
            parts.append(f"Files: {', '.join(self.files_loaded)}")
        if self.figma_nodes_checked:
            parts.append(f"Figma nodes: {', '.join(self.figma_nodes_checked)}")
        if self.scripts_executed:
            parts.append(f"Scripts: {', '.join(self.scripts_executed)}")
        if self.apis_called:
            parts.append(f"APIs: {', '.join(self.apis_called)}")
        return " | ".join(parts) if parts else "No external resources used"
