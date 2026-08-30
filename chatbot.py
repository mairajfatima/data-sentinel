"""
Chatbot that answers questions about a validation report using a real LLM API (Groq, free tier).
Only the report (already-computed summary) is sent as context — never the raw uploaded file.
"""

import json
from groq import Groq

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a data quality assistant helping a data analyst understand a validation
report for a CSV they uploaded. You have the validation report JSON below - this contains computed
checks (nulls, duplicates, anomalies, statistics) but NOT the raw dataset itself.

Ground rules:
- Never invent specific numbers, counts, or row values that aren't in the report JSON.
- You MAY and SHOULD reason about likely context: infer the probable domain/subject of the dataset
  from its column names (e.g. columns like "Store", "Weekly_Sales", "Holiday_Flag" suggest retail
  sales data) and say so, clearly flagging it as an inference, not a fact from the report.
- When asked to "explain" a check (e.g. rule checks, nulls, anomalies), don't just repeat the raw
  numbers back as a table - explain in plain English WHY that check matters, what a null/duplicate/
  anomaly in that specific column would mean for someone using this data, and what the analyst
  should actually do about it.
- When asked how to improve the data or the pipeline, give concrete, specific suggestions grounded
  in what the report actually shows (e.g. "Store has 0% nulls so it's reliable as a join key, but
  Fuel_Price has 3% missing - worth checking why before using it in aggregate calculations").
- If asked something the report genuinely has no basis for (e.g. exact values in unflagged rows),
  say so plainly instead of guessing.
Keep answers focused and specific to this dataset, not generic data-quality advice."""


VIZ_KEYWORDS = ["chart", "plot", "graph", "visuali", "show me a", "histogram"]


def wants_visualization(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in VIZ_KEYWORDS)


def ask_chatbot(question: str, report: dict, api_key: str, chat_history: list[dict] | None = None) -> str:
    """
    Send the user's question + the real validation report as context to Groq's API.
    chat_history: list of {"role": "user"/"assistant", "content": str} from earlier turns.
    """
    client = Groq(api_key=api_key)

    report_context = json.dumps(report, default=str)[:12000]  # keep prompt size sane

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nVALIDATION REPORT:\n{report_context}"}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content