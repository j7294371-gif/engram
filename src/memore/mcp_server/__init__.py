"""MCP server — exposes memore as an MCP tool for AI clients.

Usage:
    python -m memore.mcp_server.server

Or via Claude Code:
    "C:/Users/26525/AppData/Local/Temp/gh_cli/bin/gh.exe" mcp setup
"""

from memore.mcp_server.server import main

__all__ = ["main"]
