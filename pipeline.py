import os
import csv
import json
import re
import uuid
import time
import logging
from datetime import datetime
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENERATED_TICKETS_CSV = os.path.join(OUTPUT_DIR, "generated_tickets.csv")
PROCESSING_LOG_CSV = os.path.join(OUTPUT_DIR, "processing_log.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

TICKET_FIELDS = [
    "ticket_id", "title", "description", "category", "priority",
    "technical_details", "source_id", "source_type", "status", "created_at",
    "quality_score", "quality_issues",
]
LOG_FIELDS = [
    "log_id", "timestamp", "source_id", "source_type", "raw_text",
    "category", "confidence_score", "priority", "ticket_id", "processing_time_s",
]
METRICS_FIELDS = [
    "run_id", "timestamp", "total_processed", "bugs", "feature_requests",
    "praise", "complaints", "spam", "avg_quality_score", "avg_confidence_score",
    "tickets_approved", "tickets_flagged",
]


# ── LLM call with retry ────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_retries: int = 5) -> str:
    """Single Gemini API call with exponential backoff on 429."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # Throttle: 4s between calls keeps well under 15 RPM free-tier limit
    time.sleep(int(os.getenv("LLM_CALL_DELAY", "4")))
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite"),
                contents=prompt,
            )
            return response.text
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                # Extract suggested retry delay from error message, else exponential backoff
                retry_match = re.search(r"retryDelay.*?(\d+)s", msg)
                if retry_match:
                    wait = min(int(retry_match.group(1)) + 5, 120)
                else:
                    wait = min(30 * (2 ** attempt), 120)
                logger.warning("Rate limit (attempt %d/%d). Waiting %ds...", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit: max retries exceeded. Try again in a minute.")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("Could not extract JSON from: %s", text[:200])
    return {}


# ── Agent prompts (single LLM call each) ──────────────────────────────────────

def agent_classify(feedback_text: str) -> dict:
    """Feedback Classifier Agent — one LLM call."""
    prompt = f"""You are an NLP expert classifying user feedback for a mobile productivity app.

Classify the following feedback into exactly ONE category:
- Bug: app crashes, errors, broken features, data loss
- Feature Request: asking for new functionality
- Praise: positive feedback, compliments
- Complaint: negative experience not caused by a bug (pricing, support, slowness)
- Spam: promotional, random, unrelated content

Feedback:
{feedback_text}

Respond ONLY with valid JSON:
{{"category": "Bug|Feature Request|Praise|Complaint|Spam", "confidence_score": 0-100, "justification": "one sentence"}}"""
    result = _call_llm(prompt)
    return _extract_json(result)


def agent_analyze_bug(feedback_text: str) -> dict:
    """Bug Analysis Agent — one LLM call."""
    prompt = f"""You are a senior engineer triaging bug reports.

Extract technical details from this bug report:
{feedback_text}

Respond ONLY with valid JSON:
{{"severity": "Critical|High|Medium|Low", "device": "...", "os_version": "...", "app_version": "...", "steps_to_reproduce": "...", "actual_behavior": "...", "affected_component": "..."}}

Use "Unknown" for any field not mentioned."""
    result = _call_llm(prompt)
    return _extract_json(result)


def agent_analyze_feature(feedback_text: str) -> dict:
    """Feature Extractor Agent — one LLM call."""
    prompt = f"""You are a product manager analyzing feature requests.

Extract details from this feature request:
{feedback_text}

Respond ONLY with valid JSON:
{{"feature_name": "...", "description": "...", "user_impact": "High|Medium|Low", "priority": "High|Medium|Low", "implementation_summary": "..."}}"""
    result = _call_llm(prompt)
    return _extract_json(result)


def agent_create_ticket(feedback_text: str, classification: dict, analysis: dict, source_id: str, source_type: str, priority_overrides: dict) -> dict:
    """Ticket Creator Agent — one LLM call."""
    category = classification.get("category", "Complaint")
    default_priority = priority_overrides.get(category, "Medium")

    prompt = f"""You are a project manager creating an engineering ticket.

Source ID: {source_id} ({source_type})
Category: {category}
Analysis: {json.dumps(analysis)}
Original feedback: {feedback_text}

Create a structured ticket. Respond ONLY with valid JSON:
{{"ticket_id": "TKT-{str(uuid.uuid4())[:8].upper()}", "title": "concise actionable title", "description": "clear description for an engineer", "category": "{category}", "priority": "{default_priority}", "technical_details": "key technical info", "source_id": "{source_id}", "source_type": "{source_type}", "status": "Open", "created_at": "{datetime.now().isoformat()}"}}"""
    result = _call_llm(prompt)
    ticket = _extract_json(result)
    if not ticket.get("ticket_id"):
        ticket["ticket_id"] = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    ticket.setdefault("source_id", source_id)
    ticket.setdefault("source_type", source_type)
    ticket.setdefault("category", category)
    ticket.setdefault("priority", default_priority)
    ticket.setdefault("status", "Open")
    ticket.setdefault("created_at", datetime.now().isoformat())
    return ticket


def agent_quality_review(ticket: dict) -> dict:
    """Quality Critic Agent — one LLM call."""
    prompt = f"""You are a QA lead reviewing an engineering ticket for completeness.

Ticket:
{json.dumps(ticket, indent=2)}

Check: (1) all fields present and non-empty, (2) priority matches severity, (3) title is clear and actionable, (4) description is sufficient.

Respond ONLY with valid JSON:
{{"approved": true|false, "quality_score": 0-100, "issues": ["issue1", "issue2"]}}"""
    result = _call_llm(prompt)
    return _extract_json(result)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _ensure_csv(path: str, fieldnames: list):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_row(path: str, fieldnames: list, row: dict):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)


def load_app_store_reviews() -> list[dict]:
    with open(os.path.join(DATA_DIR, "app_store_reviews.csv"), newline="") as f:
        return list(csv.DictReader(f))


def load_support_emails() -> list[dict]:
    with open(os.path.join(DATA_DIR, "support_emails.csv"), newline="") as f:
        return list(csv.DictReader(f))


def format_review_text(row: dict) -> str:
    return (
        f"[Review ID: {row['review_id']}]\n"
        f"Platform: {row['platform']} | Rating: {row['rating']}/5 | "
        f"Version: {row['app_version']} | Date: {row['date']}\n"
        f"User: {row['user_name']}\n"
        f"Review: {row['review_text']}"
    )


def format_email_text(row: dict) -> str:
    return (
        f"[Email ID: {row['email_id']}]\n"
        f"Subject: {row['subject']}\n"
        f"From: {row['sender_email']} | Timestamp: {row['timestamp']} | "
        f"Priority hint: {row.get('priority', 'N/A')}\n"
        f"Body: {row['body']}"
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_single_feedback(
    feedback_text: str,
    source_id: str,
    source_type: str,
    thresholds: dict,
    progress_callback=None,
) -> dict:
    start_time = datetime.now()
    priority_overrides = thresholds.get("priority_overrides", {})
    min_confidence = thresholds.get("min_confidence", 0)

    # Step 1: Classify
    if progress_callback:
        progress_callback(source_id, "Classifying")
    classification = agent_classify(feedback_text)
    category = classification.get("category", "Complaint")
    confidence = classification.get("confidence_score", 50)
    if isinstance(confidence, str):
        confidence = int(confidence) if confidence.isdigit() else 50
    if confidence < min_confidence:
        category = "Unclassified"
    classification["category"] = category

    # Step 2: Specialized analysis (only for Bug/Feature)
    analysis = {}
    if category == "Bug":
        if progress_callback:
            progress_callback(source_id, "Analyzing bug")
        analysis = agent_analyze_bug(feedback_text)
    elif category == "Feature Request":
        if progress_callback:
            progress_callback(source_id, "Extracting feature")
        analysis = agent_analyze_feature(feedback_text)

    # Step 3: Create ticket
    if progress_callback:
        progress_callback(source_id, "Creating ticket")
    ticket = agent_create_ticket(feedback_text, classification, analysis, source_id, source_type, priority_overrides)

    # Step 4: Quality review
    if progress_callback:
        progress_callback(source_id, "Quality review")
    qc = agent_quality_review(ticket)

    ticket["quality_score"] = qc.get("quality_score", 75)
    ticket["quality_issues"] = "; ".join(qc.get("issues", []))

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        "ticket": ticket,
        "category": category,
        "confidence_score": confidence,
        "quality_score": qc.get("quality_score", 75),
        "approved": qc.get("approved", True),
        "processing_time_s": round(elapsed, 2),
        "source_id": source_id,
        "source_type": source_type,
    }


def run_pipeline(
    max_reviews: int = 5,
    max_emails: int = 3,
    thresholds: dict = None,
    inter_item_delay: int = 10,
    progress_callback=None,
) -> dict:
    if thresholds is None:
        thresholds = {"min_confidence": 0}

    _ensure_csv(GENERATED_TICKETS_CSV, TICKET_FIELDS)
    _ensure_csv(PROCESSING_LOG_CSV, LOG_FIELDS)

    run_id = str(uuid.uuid4())[:8]
    stats = {
        "run_id": run_id, "total": 0,
        "Bug": 0, "Feature Request": 0, "Praise": 0,
        "Complaint": 0, "Spam": 0, "Unclassified": 0,
        "quality_scores": [], "confidence_scores": [],
        "approved": 0, "flagged": 0,
    }

    reviews = load_app_store_reviews()[:max_reviews]
    emails = load_support_emails()[:max_emails]
    all_items = (
        [(format_review_text(r), r["review_id"], "app_store_review") for r in reviews] +
        [(format_email_text(e), e["email_id"], "support_email") for e in emails]
    )

    results = []
    for i, (feedback_text, source_id, source_type) in enumerate(all_items):
        try:
            logger.info("Processing %s (%s)", source_id, source_type)
            result = process_single_feedback(feedback_text, source_id, source_type, thresholds, progress_callback)
            ticket = result["ticket"]
            cat = result["category"]

            _append_row(GENERATED_TICKETS_CSV, TICKET_FIELDS, ticket)
            _append_row(PROCESSING_LOG_CSV, LOG_FIELDS, {
                "log_id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "source_id": source_id,
                "source_type": source_type,
                "raw_text": feedback_text[:300],
                "category": cat,
                "confidence_score": result["confidence_score"],
                "priority": ticket.get("priority", ""),
                "ticket_id": ticket.get("ticket_id", ""),
                "processing_time_s": result["processing_time_s"],
            })

            stats["total"] += 1
            stats[cat] = stats.get(cat, 0) + 1
            stats["quality_scores"].append(result["quality_score"])
            stats["confidence_scores"].append(result["confidence_score"])
            stats["approved" if result["approved"] else "flagged"] += 1
            results.append(result)

            if progress_callback:
                progress_callback(source_id, f"Done ({result['processing_time_s']:.1f}s)")

        except Exception as e:
            logger.error("Error processing %s: %s", source_id, str(e))
            if progress_callback:
                progress_callback(source_id, f"Error: {str(e)[:80]}")

        if i < len(all_items) - 1 and inter_item_delay > 0:
            time.sleep(inter_item_delay)

    _ensure_csv(METRICS_CSV, METRICS_FIELDS)
    qs = stats["quality_scores"]
    cs = stats["confidence_scores"]
    metrics_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_processed": stats["total"],
        "bugs": stats.get("Bug", 0),
        "feature_requests": stats.get("Feature Request", 0),
        "praise": stats.get("Praise", 0),
        "complaints": stats.get("Complaint", 0),
        "spam": stats.get("Spam", 0),
        "avg_quality_score": round(sum(qs) / len(qs), 1) if qs else 0,
        "avg_confidence_score": round(sum(cs) / len(cs), 1) if cs else 0,
        "tickets_approved": stats["approved"],
        "tickets_flagged": stats["flagged"],
    }
    _append_row(METRICS_CSV, METRICS_FIELDS, metrics_row)

    return {
        "run_id": run_id,
        "stats": stats,
        "metrics_row": metrics_row,
        "results": results,
        "output_files": {
            "tickets": GENERATED_TICKETS_CSV,
            "log": PROCESSING_LOG_CSV,
            "metrics": METRICS_CSV,
        },
    }
