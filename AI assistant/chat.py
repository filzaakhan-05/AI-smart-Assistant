"""Feature 1: AI chat with memory + Feature 4: agentic tool use."""
import json

from config import get_client, get_model
from document_intel import DocumentStore
from tools import OPENAI_TOOLS, run_tool

SYSTEM = (
    "You are a helpful AI productivity assistant. Answer clearly and concisely. "
    "You have a file-search tool. Decide yourself whether it is needed: use it when "
    "the user asks about their uploaded documents or files; answer directly otherwise. "
    "When you use file content, base your answer on it and cite the source name. "
    "Use the conversation history to handle follow-up questions."
)


class ChatAssistant:
    def __init__(self, store: DocumentStore, max_tool_rounds: int = 4):
        self.store = store
        self.history: list[dict] = []   # full conversation context
        self.max_tool_rounds = max_tool_rounds

    def reset(self):
        self.history = []

    def send(self, user_message: str) -> tuple[str, list[str]]:
        """Returns (reply_text, tools_used). On error, history is rolled back and the error re-raised."""
        start = len(self.history)
        self.history.append({"role": "user", "content": user_message})
        tools_used: list[str] = []
        try:
            client = get_client()
            for _ in range(self.max_tool_rounds + 1):
                resp = client.chat.completions.create(
                    model=get_model(),
                    max_tokens=1500,
                    tools=OPENAI_TOOLS,
                    messages=[{"role": "system", "content": SYSTEM}] + self.history,
                )
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    text = (msg.content or "").strip() or "(No response.)"
                    self.history.append({"role": "assistant", "content": text})
                    return text, tools_used

                # model_dump keeps provider-specific extras some APIs need echoed back
                self.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tools_used.append(f"{tc.function.name}({args})")
                    try:
                        out = run_tool(tc.function.name, args, self.store)
                    except Exception as e:  # report tool failure to the model
                        out = f"Tool error: {e}"
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": out,
                    })

            msg = "Sorry, I couldn't finish that request."
            self.history.append({"role": "assistant", "content": msg})
            return msg, tools_used
        except Exception:
            del self.history[start:]
            raise
