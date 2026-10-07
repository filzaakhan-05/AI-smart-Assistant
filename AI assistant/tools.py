"""Feature 4: Tools the agent can call (file search)."""
import os

from config import WORKSPACE_DIR
from document_intel import DocumentStore, chunk_text, load_document, rank_chunks

TOOL_SCHEMAS = [
    {
        "name": "search_files",
        "description": (
            "Search the user's uploaded documents and local workspace files for relevant "
            "passages. Use ONLY when the question depends on the user's own files or "
            "documents (e.g. policies, reports, notes). Do NOT use for general knowledge, "
            "casual chat, or writing tasks that need no file content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or question to search for."},
                "scope": {
                    "type": "string",
                    "enum": ["uploaded", "workspace", "all"],
                    "description": "Where to search. Default: all.",
                },
            },
            "required": ["query"],
        },
    }
]


# OpenAI-style tool definitions (derived from the schema above)
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOL_SCHEMAS
]


def _workspace_chunks() -> list[tuple[str, str]]:
    out = []
    if not os.path.isdir(WORKSPACE_DIR):
        return out
    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        path = os.path.join(WORKSPACE_DIR, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                text = load_document(fname, f.read())
        except Exception:
            continue  # skip unsupported/unreadable files
        out += [(f"workspace/{fname}", c) for c in chunk_text(text)]
    return out


def search_files(store: DocumentStore, query: str, scope: str = "all") -> str:
    chunks = []
    if scope in ("uploaded", "all"):
        chunks += store.all_chunks()
    if scope in ("workspace", "all"):
        chunks += _workspace_chunks()
    hits = rank_chunks(query, chunks, k=5)
    if not hits:
        return "No matching content found."
    return "\n\n".join(f"[Source: {s}]\n{t}" for s, t in hits)


def run_tool(name: str, args: dict, store: DocumentStore) -> str:
    if name == "search_files":
        return search_files(store, str(args.get("query", "")), args.get("scope", "all"))
    return f"Unknown tool: {name}"
