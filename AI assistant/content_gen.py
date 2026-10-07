"""Feature 3: AI content generation."""
from config import ask

CONTENT_TYPES = ["Email", "Summary", "Social media post", "Report", "Meeting notes"]
TONES = ["Professional", "Friendly", "Formal", "Casual", "Persuasive", "Concise"]

FORMAT_HINTS = {
    "Email": "Include a subject line, greeting, body, and sign-off.",
    "Summary": "Be brief and structured; lead with the main point.",
    "Social media post": "Keep it engaging, with a hook and 2-4 relevant hashtags.",
    "Report": "Use headings: Overview, Findings, Recommendations.",
    "Meeting notes": "Use sections: Attendees, Discussion, Decisions, Action Items (owner + due date).",
}


def generate_content(content_type: str, request: str, tone: str = "Professional",
                     context: str = "") -> str:
    system = (
        f"You are an expert writer. Produce a {content_type.lower()} in a {tone.lower()} tone. "
        f"{FORMAT_HINTS.get(content_type, '')} Output only the finished content."
    )
    user = request if not context else f"{request}\n\nReference material:\n{context}"
    return ask(system, user, max_tokens=2000)
