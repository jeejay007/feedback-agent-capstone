import os
import csv
import json
import re
import uuid
import logging
from datetime import datetime
from crewai import Crew, Process

from agents import (
    create_csv_reader_agent,
    create_classifier_agent,
    create_bug_analysis_agent,
    create_feature_extractor_agent,
    create_ticket_creator_agent,
    create_quality_critic_agent,
)
from tasks import (
    create_read_task,
    create_classification_task,
    create_bug_analysis_task,
    create_feature_extraction_task,
    create_ticket_creation_task,
    create_quality_review_task,
)

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


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response string."""
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Look for bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("Could not extract JSON from: %s", text[:200])
    return {}


def _ensure_csv(path: str, fieldnames: list):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_row(path: str, fieldnames: list, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def load_app_store_reviews() -> list[dict]:
    path = os.path.join(DATA_DIR, "app_store_reviews.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_support_emails() -> list[dict]:
    path = os.path.join(DATA_DIR, "support_emails.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


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


def process_single_feedback(
    feedback_text: str,
    source_id: str,
    source_type: str,
    thresholds: dict,
    progress_callback=None,
) -> dict:
    """Run the full multi-agent pipeline for one feedback item."""
    start_time = datetime.now()

    reader_agent = create_csv_reader_agent()
    classifier_agent = create_classifier_agent()
    bug_agent = create_bug_analysis_agent()
    feature_agent = create_feature_extractor_agent()
    ticket_agent = create_ticket_creator_agent()
    critic_agent = create_quality_critic_agent()

    # --- Step 1: Read / Parse ---
    if progress_callback:
        progress_callback(source_id, "Reading")
    read_task = create_read_task(reader_agent, feedback_text, source_type)
    read_crew = Crew(agents=[reader_agent], tasks=[read_task], process=Process.sequential, verbose=False)
    read_result = read_crew.kickoff()
    parsed = _extract_json(str(read_result))

    # --- Step 2: Classify ---
    if progress_callback:
        progress_callback(source_id, "Classifying")
    clf_task = create_classification_task(classifier_agent, feedback_text)
    clf_crew = Crew(agents=[classifier_agent], tasks=[clf_task], process=Process.sequential, verbose=False)
    clf_result = clf_crew.kickoff()
    classification = _extract_json(str(clf_result))

    category = classification.get("category", "Complaint")
    confidence = classification.get("confidence_score", 50)

    # Apply confidence threshold
    min_confidence = thresholds.get("min_confidence", 0)
    if isinstance(confidence, str):
        try:
            confidence = int(confidence)
        except ValueError:
            confidence = 50
    if confidence < min_confidence:
        category = "Unclassified"

    # --- Step 3: Specialized analysis ---
    if progress_callback:
        progress_callback(source_id, "Analyzing")
    analysis = {}
    if category == "Bug":
        bug_task = create_bug_analysis_task(bug_agent, feedback_text)
        bug_crew = Crew(agents=[bug_agent], tasks=[bug_task], process=Process.sequential, verbose=False)
        bug_result = bug_crew.kickoff()
        analysis = _extract_json(str(bug_result))
    elif category == "Feature Request":
        feat_task = create_feature_extraction_task(feature_agent, feedback_text)
        feat_crew = Crew(agents=[feature_agent], tasks=[feat_task], process=Process.sequential, verbose=False)
        feat_result = feat_crew.kickoff()
        analysis = _extract_json(str(feat_result))

    # --- Step 4: Create ticket ---
    if progress_callback:
        progress_callback(source_id, "Creating ticket")
    ticket_task = create_ticket_creation_task(
        ticket_agent,
        json.dumps(classification),
        json.dumps(analysis),
        feedback_text,
    )
    ticket_crew = Crew(agents=[ticket_agent], tasks=[ticket_task], process=Process.sequential, verbose=False)
    ticket_result = ticket_crew.kickoff()
    ticket = _extract_json(str(ticket_result))

    if not ticket.get("ticket_id"):
        ticket["ticket_id"] = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    if not ticket.get("source_id"):
        ticket["source_id"] = source_id
    if not ticket.get("source_type"):
        ticket["source_type"] = source_type
    if not ticket.get("category"):
        ticket["category"] = category
    if not ticket.get("status"):
        ticket["status"] = "Open"
    if not ticket.get("created_at"):
        ticket["created_at"] = datetime.now().isoformat()

    # Apply priority override from thresholds
    priority_map = thresholds.get("priority_overrides", {})
    if category in priority_map and not ticket.get("priority"):
        ticket["priority"] = priority_map[category]

    # --- Step 5: Quality review ---
    if progress_callback:
        progress_callback(source_id, "Quality review")
    qc_task = create_quality_review_task(critic_agent, json.dumps(ticket))
    qc_crew = Crew(agents=[critic_agent], tasks=[qc_task], process=Process.sequential, verbose=False)
    qc_result = qc_crew.kickoff()
    qc = _extract_json(str(qc_result))

    final_ticket = qc.get("corrected_ticket", ticket)
    if isinstance(final_ticket, str):
        final_ticket = _extract_json(final_ticket)
    if not final_ticket:
        final_ticket = ticket

    final_ticket["quality_score"] = qc.get("quality_score", 0)
    final_ticket["quality_issues"] = "; ".join(qc.get("issues", []))
    final_ticket["source_id"] = source_id
    final_ticket["source_type"] = source_type

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        "ticket": final_ticket,
        "category": category,
        "confidence_score": confidence,
        "quality_score": qc.get("quality_score", 0),
        "approved": qc.get("approved", False),
        "processing_time_s": round(elapsed, 2),
        "source_id": source_id,
        "source_type": source_type,
    }


def run_pipeline(
    max_reviews: int = 20,
    max_emails: int = 10,
    thresholds: dict = None,
    progress_callback=None,
) -> dict:
    """
    Run the full pipeline over the CSV datasets.

    Returns a summary dict with counts and paths to output files.
    """
    if thresholds is None:
        thresholds = {"min_confidence": 0}

    _ensure_csv(GENERATED_TICKETS_CSV, TICKET_FIELDS)
    _ensure_csv(PROCESSING_LOG_CSV, LOG_FIELDS)

    run_id = str(uuid.uuid4())[:8]
    stats = {
        "run_id": run_id,
        "total": 0,
        "Bug": 0,
        "Feature Request": 0,
        "Praise": 0,
        "Complaint": 0,
        "Spam": 0,
        "Unclassified": 0,
        "quality_scores": [],
        "confidence_scores": [],
        "approved": 0,
        "flagged": 0,
    }

    reviews = load_app_store_reviews()[:max_reviews]
    emails = load_support_emails()[:max_emails]

    all_items = []
    for row in reviews:
        all_items.append((format_review_text(row), row["review_id"], "app_store_review"))
    for row in emails:
        all_items.append((format_email_text(row), row["email_id"], "support_email"))

    results = []
    for feedback_text, source_id, source_type in all_items:
        try:
            logger.info("Processing %s (%s)", source_id, source_type)
            result = process_single_feedback(
                feedback_text, source_id, source_type, thresholds, progress_callback
            )
            ticket = result["ticket"]
            cat = result["category"]

            # Write ticket
            _append_row(GENERATED_TICKETS_CSV, TICKET_FIELDS, ticket)

            # Write log
            log_row = {
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
            }
            _append_row(PROCESSING_LOG_CSV, LOG_FIELDS, log_row)

            # Update stats
            stats["total"] += 1
            stats[cat] = stats.get(cat, 0) + 1
            stats["quality_scores"].append(result["quality_score"])
            stats["confidence_scores"].append(result["confidence_score"])
            if result["approved"]:
                stats["approved"] += 1
            else:
                stats["flagged"] += 1

            results.append(result)

        except Exception as e:
            logger.error("Error processing %s: %s", source_id, str(e))
            if progress_callback:
                progress_callback(source_id, f"Error: {str(e)}")

    # Write metrics
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
