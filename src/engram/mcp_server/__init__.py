"""MCP server — exposes engram as an MCP tool for AI clients.

Usage:
    python -m engram.mcp_server.server

Or via Claude Code:
    "C:/Users/26525/AppData/Local/Temp/gh_cli/bin/gh.exe" mcp setup
"""

from engram.mcp_server.server import main

__all__ = ["main"]
