"""Emit the search index format expected by Material for file:// pages."""

from pathlib import Path


def on_post_build(config, **kwargs):
    site_dir = Path(config["site_dir"])
    json_index = site_dir / "search" / "search_index.json"
    javascript_index = site_dir / "search" / "search_index.js"

    if json_index.exists():
        javascript_index.write_text(
            "var __index = " + json_index.read_text(encoding="utf-8") + ";\n",
            encoding="utf-8",
        )
