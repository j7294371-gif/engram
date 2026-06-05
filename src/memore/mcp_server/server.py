"""MCP stdio server — exposes memore as tools for any MCP client.

Run:
    python -m memore.mcp_server.server

Or configure Claude Code to use via settings.json:
    {
        "mcpServers": {
            "memore": {
                "command": "python",
                "args": ["-m", "memore.mcp_server.server"]
            }
        }
    }
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from memore import AgentMemory

# ── JSON-RPC over stdio ────────────────────────────────────────

memory: AgentMemory | None = None


def log(msg: str) -> None:
    """Log to stderr (stdout is MCP transport)."""
    print(f"[memore-mcp] {msg}", file=sys.stderr, flush=True)


def send(message: dict[str, Any]) -> None:
    """Send a JSON-RPC message over stdout."""
    body = json.dumps(message).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def read() -> dict[str, Any] | None:
    """Read a JSON-RPC message from stdin."""
    content_length = 0
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("utf-8").strip()
        if line.startswith("Content-Length:"):
            content_length = int(line.split(":")[1].strip())
        elif line == "" and content_length > 0:
            raw = sys.stdin.buffer.read(content_length)
            return json.loads(raw.decode("utf-8"))
        elif line == "":
            continue


# ── Tool definitions ───────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "remember",
        "description": "Store a memory for the current user/session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content"},
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "sensory", "working"],
                    "default": "episodic",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0-1",
                    "default": 0.5,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": "Recall memories relevant to a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "number", "default": 10},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by types: episodic, semantic, etc.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search",
        "description": "Powerful search across all memories with type filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by types",
                },
                "limit": {"type": "number", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Archive a specific memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to archive"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "consolidate",
        "description": "Run memory consolidation (forgetting curve + compression + archival).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sleep": {
                    "type": "boolean",
                    "description": "Full sleep consolidation (slower, more thorough)",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "stats",
        "description": "Get memory system statistics.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "focus",
        "description": "Set the current working memory context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Current task context"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_context",
        "description": "Get current working memory context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_size": {"type": "number", "default": 7},
            },
        },
    },
]


def _ensure_memory() -> AgentMemory:
    global memory
    if memory is None:
        memory = AgentMemory()
        log("AgentMemory initialized")
    return memory


def handle_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Execute a tool and return the result."""
    m = _ensure_memory()

    if name == "remember":
        mid = m.remember(
            arguments["content"],
            memory_type=arguments.get("memory_type", "episodic"),
            importance=arguments.get("importance", 0.5),
            tags=arguments.get("tags"),
        )
        return {"memory_id": mid, "status": "stored"}

    elif name == "recall":
        results = m.recall(
            arguments.get("query", ""),
            limit=arguments.get("limit", 10),
            memory_types=arguments.get("memory_types"),
        )
        return {
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "memory_type": r.memory_type.value,
                    "importance": r.importance,
                    "retrieval_probability": r.retrieval_probability(),
                    "tags": r.tags,
                    "created_at": r.created_at.isoformat(),
                }
                for r in results
            ],
            "count": len(results),
        }

    elif name == "search":
        results = m.search(
            arguments["query"],
            memory_types=arguments.get("memory_types"),
            limit=arguments.get("limit", 20),
        )
        return {
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "memory_type": r.memory_type.value,
                    "importance": r.importance,
                    "tags": r.tags,
                }
                for r in results
            ],
            "count": len(results),
        }

    elif name == "forget":
        m.forget(arguments["memory_id"])
        return {"status": "archived", "memory_id": arguments["memory_id"]}

    elif name == "consolidate":
        if arguments.get("sleep", False):
            report = m.consolidate_sleep()
            return {
                "status": "sleep_consolidation_complete",
                "promotions": report.promotions,
                "abstractions": report.abstractions,
                "archived": report.archived,
                "merged": report.merged,
                "purged": report.purged,
                "compressed": report.compressed,
            }
        else:
            report = m.consolidate()
            return {
                "status": "consolidation_complete",
                "promotions": report.get("promotions", 0),
                "archived": report.get("archived", 0),
            }

    elif name == "stats":
        return m.stats()

    elif name == "focus":
        mem = m._working.focus(arguments["content"])
        return {"memory_id": mem.id, "content": mem.content, "status": "focused"}

    elif name == "get_context":
        ctx = m.get_context(window_size=arguments.get("window_size", 7))
        return {
            "context": [
                {"content": c.content, "attention_weight": c.attention_weight}
                for c in ctx
            ],
            "count": len(ctx),
        }

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Main loop ──────────────────────────────────────────────────

def main() -> None:
    log("Engram MCP server starting...")
    capabilities = {"tools": {}}
    server_info = {"name": "memore-mcp", "version": "0.1.0"}

    while True:
        message = read()
        if message is None:
            break

        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        try:
            if method == "initialize":
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": capabilities,
                        "serverInfo": server_info,
                    },
                })

            elif method == "tools/list":
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": TOOLS},
                })

            elif method == "tools/call":
                result = handle_tool_call(params["name"], params.get("arguments", {}))
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]},
                })

            elif method == "notifications/initialized":
                pass  # Notification — no response

            elif method == "shutdown":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
                break

            else:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })

        except Exception as e:
            log(f"Error handling {method}: {e}")
            traceback.print_exc(file=sys.stderr)
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)},
            })

    log("Engram MCP server stopped.")


if __name__ == "__main__":
    main()
