import os
from crewai import Agent, LLM


def get_llm():
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def create_csv_reader_agent():
    return Agent(
        role="CSV Reader Agent",
        goal="Read and parse feedback data from CSV files accurately",
        backstory=(
            "You are a data ingestion specialist who reads structured CSV files "
            "containing user feedback from app stores and support emails. You ensure "
            "data is clean, complete, and ready for downstream processing."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_classifier_agent():
    return Agent(
        role="Feedback Classifier Agent",
        goal="Classify each piece of user feedback into exactly one category with a confidence score",
        backstory=(
            "You are an NLP expert specializing in user feedback analysis. You categorize "
            "feedback into: Bug, Feature Request, Praise, Complaint, or Spam. "
            "You assign a confidence score (0-100) to each classification and provide "
            "a brief justification."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_bug_analysis_agent():
    return Agent(
        role="Bug Analysis Agent",
        goal="Extract technical details from bug reports including severity, steps to reproduce, and platform info",
        backstory=(
            "You are a senior software engineer who specializes in triage and bug analysis. "
            "Given a bug report you extract: device model, OS version, app version, "
            "steps to reproduce, expected vs actual behavior, and assign a severity "
            "level (Critical / High / Medium / Low)."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_feature_extractor_agent():
    return Agent(
        role="Feature Extractor Agent",
        goal="Identify and structure feature requests with user impact estimation",
        backstory=(
            "You are a product manager assistant who analyzes feature requests from users. "
            "You identify the core feature being requested, estimate its user impact "
            "(High / Medium / Low), and suggest a concise implementation summary. "
            "You also flag duplicate or commonly requested features."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_ticket_creator_agent():
    return Agent(
        role="Ticket Creator Agent",
        goal="Generate well-structured, actionable tickets from analyzed feedback and log them to CSV",
        backstory=(
            "You are a project management specialist who converts analyzed user feedback "
            "into perfectly formatted engineering tickets. Each ticket includes a clear title, "
            "description, priority, category, technical details, and source traceability. "
            "You output clean, consistent JSON that can be saved to CSV."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_quality_critic_agent():
    return Agent(
        role="Quality Critic Agent",
        goal="Review generated tickets for completeness, accuracy, and consistency",
        backstory=(
            "You are a quality assurance lead who reviews engineering tickets before they "
            "are filed. You check that each ticket has all required fields, correct priority "
            "assignment, actionable titles, and consistent formatting. You flag issues and "
            "suggest corrections where needed."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
