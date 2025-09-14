import json
from groq import Groq
from models import TicketClassification
from pydantic import ValidationError

TOPIC_TAG_OPTIONS = [
    "How-to", "Product", "Connector", "Lineage", "API/SDK",
    "SSO", "Glossary", "Best practices", "Sensitive data"
]
PRIORITY_OPTIONS = ["P0", "P1", "P2"]
SENTIMENT_OPTIONS = ["Frustrated", "Curious", "Angry", "Neutral"]

SYSTEM_INSTRUCTIONS = f"""
You are an expert support ticket classifier.
Given a support ticket, classify it in this JSON schema:
{{
  "topic_tags": [list, choose relevant from {TOPIC_TAG_OPTIONS}],
  "topic_tag_confidence": {{tag: score, ...}},  // confidence score (0-1) for each topic_tag
  "core_problem": "short topic string",
  "priority": "P0, P1, or P2",
  "sentiment": "Frustrated, Curious, Angry, Neutral"
}}
Only output valid JSON. No explanation.
For topic_tag_confidence, provide the model's confidence (probability, 0 to 1) for each tag in topic_tags.
"""

def _parse_json_strict(raw: str):
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return json.loads(raw)

def classify_ticket(client: Groq, subject: str, body: str, model: str = "llama-3.1-8b-instant"):
    ticket_text = f"Subject: {subject}\nBody: {body}"
    system_message = SYSTEM_INSTRUCTIONS
    user_message = ticket_text
    raw_output = None
    error = None

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        if hasattr(completion.choices[0].message, "content"):
            raw_output = completion.choices[0].message.content
        elif isinstance(completion.choices[0].message, dict) and "content" in completion.choices[0].message:
            raw_output = completion.choices[0].message["content"]
        else:
            raw_output = None
        if not raw_output:
            error = "Model did not return any output."
    except Exception as e:
        raw_output = None
        error = str(e)

    analysis = {
        "prompt": {"system": system_message, "user": user_message},
        "raw_output": raw_output,
        "error": error
    }

    try:
        if raw_output:
            data = _parse_json_strict(raw_output)

            # apply filtering on topic_tags
            conf = data.get("topic_tag_confidence", {})
            filtered_tags = [t for t, score in conf.items() if score is not None and score >= 0.6]

            data["topic_tags"] = filtered_tags
            data["topic_tag_confidence"] = {t: score for t, score in conf.items() if score is not None and score >= 0.6}

            classification = TicketClassification(**data)
        else:
            classification = TicketClassification(
                topic_tags=[],
                topic_tag_confidence={},
                core_problem="",
                priority="",
                sentiment=""
            )
    except Exception as e:
        classification = TicketClassification(
            topic_tags=[],
            topic_tag_confidence={},
            core_problem="",
            priority="",
            sentiment=""
        )
        analysis["error"] = str(e)

    return analysis, classification
