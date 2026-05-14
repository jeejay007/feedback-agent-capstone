from crewai import Task


def create_read_task(agent, feedback_text: str, source_type: str):
    return Task(
        description=(
            f"Parse the following {source_type} feedback entry and return a structured summary.\n\n"
            f"Raw feedback:\n{feedback_text}\n\n"
            "Return a JSON object with keys: source_id, source_type, raw_text, metadata (platform/rating/date etc)."
        ),
        expected_output="A JSON object with source_id, source_type, raw_text, and metadata fields.",
        agent=agent,
    )


def create_classification_task(agent, parsed_data: str):
    return Task(
        description=(
            "Classify the following parsed feedback into exactly one category.\n\n"
            f"Parsed feedback:\n{parsed_data}\n\n"
            "Categories: Bug, Feature Request, Praise, Complaint, Spam.\n"
            "Return JSON with keys: category, confidence_score (0-100), justification."
        ),
        expected_output="JSON with category, confidence_score, and justification.",
        agent=agent,
    )


def create_bug_analysis_task(agent, feedback_text: str):
    return Task(
        description=(
            "Analyze the following bug report and extract all technical details.\n\n"
            f"Feedback:\n{feedback_text}\n\n"
            "Return JSON with keys: severity (Critical/High/Medium/Low), device, os_version, "
            "app_version, steps_to_reproduce, expected_behavior, actual_behavior, affected_component."
        ),
        expected_output="JSON with severity, device, os_version, app_version, steps_to_reproduce, expected_behavior, actual_behavior, affected_component.",
        agent=agent,
    )


def create_feature_extraction_task(agent, feedback_text: str):
    return Task(
        description=(
            "Analyze the following feature request and extract structured information.\n\n"
            f"Feedback:\n{feedback_text}\n\n"
            "Return JSON with keys: feature_name, description, user_impact (High/Medium/Low), "
            "priority (High/Medium/Low), implementation_summary, is_duplicate_likely (true/false)."
        ),
        expected_output="JSON with feature_name, description, user_impact, priority, implementation_summary, is_duplicate_likely.",
        agent=agent,
    )


def create_ticket_creation_task(agent, classification: str, analysis: str, source_data: str):
    return Task(
        description=(
            "Create a structured engineering ticket based on the analysis below.\n\n"
            f"Classification:\n{classification}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Source data:\n{source_data}\n\n"
            "Return JSON with keys: ticket_id, title, description, category, priority, "
            "technical_details, source_id, source_type, status (Open), created_at."
        ),
        expected_output="JSON ticket with all required fields.",
        agent=agent,
    )


def create_quality_review_task(agent, ticket: str):
    return Task(
        description=(
            "Review the following generated ticket for quality and completeness.\n\n"
            f"Ticket:\n{ticket}\n\n"
            "Check: (1) all required fields present, (2) priority is appropriate for category/severity, "
            "(3) title is clear and actionable, (4) description is sufficient for an engineer to act on.\n"
            "Return JSON with keys: approved (true/false), issues (list of strings), "
            "corrected_ticket (the improved ticket JSON or same if approved), quality_score (0-100)."
        ),
        expected_output="JSON with approved, issues, corrected_ticket, and quality_score.",
        agent=agent,
    )
