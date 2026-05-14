import os
import csv
import json
import threading
import queue
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKETS_CSV = os.path.join(OUTPUT_DIR, "generated_tickets.csv")
LOG_CSV = os.path.join(OUTPUT_DIR, "processing_log.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

CATEGORY_COLORS = {
    "Bug": "🔴",
    "Feature Request": "🟡",
    "Praise": "🟢",
    "Complaint": "🟠",
    "Spam": "⚪",
    "Unclassified": "🔵",
}

PRIORITY_COLORS = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_csv_safe(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def validate_api_key() -> bool:
    key = os.getenv("GOOGLE_API_KEY", "")
    return bool(key and key.strip())


# ── Session state defaults ─────────────────────────────────────────────────────
def init_session():
    defaults = {
        "processing": False,
        "log_messages": [],
        "last_run_stats": None,
        "override_edits": {},
        "pipeline_result": None,
        "pipeline_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("API Key")
    api_key_input = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Your Google Gemini API key",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    st.divider()

    st.subheader("Model")
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash-latest"],
        index=0,
        help="gemini-2.0-flash-lite has the most generous free quota.",
    )
    os.environ["GEMINI_MODEL"] = model_name

    st.divider()

    st.subheader("Rate Limit Settings")
    inter_item_delay = st.slider(
        "Delay between items (seconds)",
        min_value=2, max_value=60, value=3,
        help="Small delay between items. Retries handle actual rate limits automatically.",
    )

    st.subheader("Processing Settings")
    max_reviews = st.slider("Max App Store Reviews", 1, 20, 5)
    max_emails = st.slider("Max Support Emails", 1, 10, 3)
    min_confidence = st.slider("Min Classification Confidence (%)", 0, 100, 0)

    st.subheader("Priority Overrides")
    bug_priority = st.selectbox("Default Bug Priority", ["Critical", "High", "Medium", "Low"], index=1)
    feature_priority = st.selectbox("Default Feature Priority", ["High", "Medium", "Low"], index=1)
    complaint_priority = st.selectbox("Default Complaint Priority", ["High", "Medium", "Low"], index=2)

    thresholds = {
        "min_confidence": min_confidence,
        "priority_overrides": {
            "Bug": bug_priority,
            "Feature Request": feature_priority,
            "Complaint": complaint_priority,
        },
    }

    st.divider()
    st.caption("Capstone Project — Agentic AI Certification")

# ── Main Title ─────────────────────────────────────────────────────────────────
st.title("🤖 Intelligent User Feedback Analysis System")
st.markdown(
    "Multi-agent AI pipeline powered by **CrewAI + Google Gemini** — "
    "classifies feedback, extracts insights, and generates tickets automatically."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_dashboard, tab_run, tab_tickets, tab_override, tab_analytics = st.tabs(
    ["📊 Dashboard", "▶️ Run Pipeline", "🎫 Tickets", "✏️ Manual Override", "📈 Analytics"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.header("Overview")

    tickets_df = load_csv_safe(TICKETS_CSV)
    logs_df = load_csv_safe(LOG_CSV)
    metrics_df = load_csv_safe(METRICS_CSV)

    col1, col2, col3, col4 = st.columns(4)
    total = len(tickets_df)
    bugs = len(tickets_df[tickets_df["category"] == "Bug"]) if total else 0
    features = len(tickets_df[tickets_df["category"] == "Feature Request"]) if total else 0
    critical = (
        len(tickets_df[tickets_df["priority"] == "Critical"])
        if total and "priority" in tickets_df.columns
        else 0
    )

    col1.metric("Total Tickets", total)
    col2.metric("Bugs", bugs)
    col3.metric("Feature Requests", features)
    col4.metric("Critical Issues", critical)

    if not tickets_df.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Category Breakdown")
            cat_counts = tickets_df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.bar_chart(cat_counts.set_index("Category"))

        with col_b:
            st.subheader("Priority Breakdown")
            if "priority" in tickets_df.columns:
                pri_counts = tickets_df["priority"].value_counts().reset_index()
                pri_counts.columns = ["Priority", "Count"]
                st.bar_chart(pri_counts.set_index("Priority"))

        st.subheader("Recent Tickets")
        display_cols = [c for c in ["ticket_id", "title", "category", "priority", "status", "created_at"] if c in tickets_df.columns]
        st.dataframe(tickets_df[display_cols].tail(10), width="stretch")
    else:
        st.info("No tickets yet. Run the pipeline from the **Run Pipeline** tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.header("Run Multi-Agent Pipeline")

    # Preview input data
    with st.expander("Preview Input Data", expanded=False):
        rev_df = load_csv_safe(os.path.join(DATA_DIR, "app_store_reviews.csv"))
        email_df = load_csv_safe(os.path.join(DATA_DIR, "support_emails.csv"))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"App Store Reviews ({len(rev_df)} rows)")
            st.dataframe(rev_df.head(5), width="stretch")
        with c2:
            st.subheader(f"Support Emails ({len(email_df)} rows)")
            st.dataframe(email_df.head(5), width="stretch")

    if not validate_api_key():
        st.warning("Please enter your Google API Key in the sidebar before running.")
    else:
        st.success("Google API Key detected.")

    log_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.button(
        "🚀 Start Pipeline",
        type="primary",
        disabled=st.session_state.processing or not validate_api_key(),
    ):
        st.session_state.processing = True
        st.session_state.log_messages = []
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = None

        log_queue = queue.Queue()
        result_queue = queue.Queue()

        def update_log(source_id: str, stage: str):
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] {source_id} → {stage}"
            log_queue.put(msg)

        def run_in_thread():
            try:
                from pipeline import run_pipeline
                result = run_pipeline(
                    max_reviews=max_reviews,
                    max_emails=max_emails,
                    thresholds=thresholds,
                    inter_item_delay=inter_item_delay,
                    progress_callback=update_log,
                )
                result_queue.put(("ok", result))
            except Exception as e:
                result_queue.put(("error", str(e)))

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        # Poll until done, draining the log queue every 2s
        progress_bar = st.progress(0, text="Starting pipeline...")
        log_box = st.empty()
        total_items = max_reviews + max_emails
        logs = []

        while thread.is_alive():
            time.sleep(2)
            while not log_queue.empty():
                logs.append(log_queue.get_nowait())
            done = len([m for m in logs if "Done" in m or "Error" in m])
            pct = min(int(done / max(total_items, 1) * 100), 99)
            progress_bar.progress(pct, text=f"Processing... {done}/{total_items} items done")
            log_box.code("\n".join(logs[-15:]))

        # Drain any remaining log messages
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())
        st.session_state.log_messages = logs

        progress_bar.progress(100, text="Done!")
        log_box.code("\n".join(logs[-15:]))

        kind, payload = result_queue.get()
        st.session_state.processing = False

        if kind == "ok":
            result = payload
            st.session_state.last_run_stats = result
            st.success(
                f"✅ Pipeline complete! Processed {result['stats']['total']} items. "
                f"Run ID: {result['run_id']}"
            )
            m = result["metrics_row"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Bugs", m["bugs"])
            c2.metric("Features", m["feature_requests"])
            c3.metric("Praise", m["praise"])
            c4.metric("Complaints", m["complaints"])
            c5.metric("Spam", m["spam"])
        else:
            st.error(f"Pipeline error: {payload}")

    # Show log
    if st.session_state.log_messages:
        with st.expander("Processing Log", expanded=True):
            for msg in st.session_state.log_messages[-30:]:
                st.text(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tickets
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tickets:
    st.header("Generated Tickets")

    tickets_df = load_csv_safe(TICKETS_CSV)

    if tickets_df.empty:
        st.info("No tickets generated yet.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        cats = ["All"] + sorted(tickets_df["category"].dropna().unique().tolist())
        pris = ["All"] + sorted(tickets_df["priority"].dropna().unique().tolist()) if "priority" in tickets_df.columns else ["All"]
        stats_opts = ["All"] + sorted(tickets_df["status"].dropna().unique().tolist()) if "status" in tickets_df.columns else ["All"]

        sel_cat = col1.selectbox("Filter by Category", cats)
        sel_pri = col2.selectbox("Filter by Priority", pris)
        sel_stat = col3.selectbox("Filter by Status", stats_opts)

        filtered = tickets_df.copy()
        if sel_cat != "All":
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_pri != "All" and "priority" in filtered.columns:
            filtered = filtered[filtered["priority"] == sel_pri]
        if sel_stat != "All" and "status" in filtered.columns:
            filtered = filtered[filtered["status"] == sel_stat]

        st.caption(f"Showing {len(filtered)} of {len(tickets_df)} tickets")
        st.dataframe(filtered, width="stretch")

        # Download
        st.download_button(
            "📥 Download Tickets CSV",
            tickets_df.to_csv(index=False),
            file_name="generated_tickets.csv",
            mime="text/csv",
        )

        # Detail view
        st.subheader("Ticket Detail")
        if "ticket_id" in tickets_df.columns:
            selected_id = st.selectbox("Select Ticket ID", tickets_df["ticket_id"].tolist())
            if selected_id:
                row = tickets_df[tickets_df["ticket_id"] == selected_id].iloc[0]
                for col_name in tickets_df.columns:
                    val = row[col_name]
                    if pd.notna(val) and str(val).strip():
                        st.markdown(f"**{col_name}:** {val}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Manual Override
# ═══════════════════════════════════════════════════════════════════════════════
with tab_override:
    st.header("Manual Override")
    st.markdown("Edit or approve generated tickets before finalizing.")

    tickets_df = load_csv_safe(TICKETS_CSV)

    if tickets_df.empty:
        st.info("No tickets to edit yet.")
    else:
        ticket_ids = tickets_df["ticket_id"].dropna().tolist() if "ticket_id" in tickets_df.columns else []
        if ticket_ids:
            sel_ticket = st.selectbox("Select ticket to edit", ticket_ids)
            idx = tickets_df[tickets_df["ticket_id"] == sel_ticket].index[0]
            row = tickets_df.loc[idx].to_dict()

            with st.form("override_form"):
                new_title = st.text_input("Title", value=str(row.get("title", "")))
                new_desc = st.text_area("Description", value=str(row.get("description", "")), height=150)
                new_cat = st.selectbox(
                    "Category",
                    ["Bug", "Feature Request", "Praise", "Complaint", "Spam", "Unclassified"],
                    index=["Bug", "Feature Request", "Praise", "Complaint", "Spam", "Unclassified"].index(
                        row.get("category", "Bug")
                    ) if row.get("category") in ["Bug", "Feature Request", "Praise", "Complaint", "Spam", "Unclassified"] else 0,
                )
                new_priority = st.selectbox(
                    "Priority",
                    ["Critical", "High", "Medium", "Low"],
                    index=["Critical", "High", "Medium", "Low"].index(row.get("priority", "Medium"))
                    if row.get("priority") in ["Critical", "High", "Medium", "Low"] else 1,
                )
                new_status = st.selectbox(
                    "Status",
                    ["Open", "In Progress", "Resolved", "Rejected"],
                    index=["Open", "In Progress", "Resolved", "Rejected"].index(row.get("status", "Open"))
                    if row.get("status") in ["Open", "In Progress", "Resolved", "Rejected"] else 0,
                )
                submitted = st.form_submit_button("💾 Save Changes")

            if submitted:
                tickets_df.at[idx, "title"] = new_title
                tickets_df.at[idx, "description"] = new_desc
                tickets_df.at[idx, "category"] = new_cat
                tickets_df.at[idx, "priority"] = new_priority
                tickets_df.at[idx, "status"] = new_status
                tickets_df.to_csv(TICKETS_CSV, index=False)
                st.success(f"Ticket {sel_ticket} updated successfully.")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("Analytics & Metrics")

    tickets_df = load_csv_safe(TICKETS_CSV)
    logs_df = load_csv_safe(LOG_CSV)
    metrics_df = load_csv_safe(METRICS_CSV)

    if metrics_df.empty and tickets_df.empty:
        st.info("No data yet. Run the pipeline first.")
    else:
        if not metrics_df.empty:
            st.subheader("Pipeline Run History")
            display_metrics = [
                c for c in [
                    "run_id", "timestamp", "total_processed", "bugs",
                    "feature_requests", "praise", "complaints", "spam",
                    "avg_quality_score", "avg_confidence_score",
                    "tickets_approved", "tickets_flagged",
                ]
                if c in metrics_df.columns
            ]
            st.dataframe(metrics_df[display_metrics], width="stretch")

        if not tickets_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Category Distribution")
                cat_data = tickets_df["category"].value_counts()
                st.bar_chart(cat_data)

            with col2:
                st.subheader("Priority Distribution")
                if "priority" in tickets_df.columns:
                    pri_data = tickets_df["priority"].value_counts()
                    st.bar_chart(pri_data)

            if "quality_score" in tickets_df.columns:
                st.subheader("Quality Score Distribution")
                qs = pd.to_numeric(tickets_df["quality_score"], errors="coerce").dropna()
                if not qs.empty:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Quality Score", f"{qs.mean():.1f}")
                    col2.metric("Min Quality Score", f"{qs.min():.0f}")
                    col3.metric("Max Quality Score", f"{qs.max():.0f}")
                    st.bar_chart(qs.value_counts().sort_index())

        if not logs_df.empty:
            st.subheader("Processing Log")
            st.dataframe(logs_df.tail(20), width="stretch")
            st.download_button(
                "📥 Download Processing Log",
                logs_df.to_csv(index=False),
                file_name="processing_log.csv",
                mime="text/csv",
            )

        # Classification accuracy vs expected
        expected_df = load_csv_safe(os.path.join(DATA_DIR, "expected_classifications.csv"))
        if not expected_df.empty and not tickets_df.empty and "source_id" in tickets_df.columns:
            st.subheader("Classification Accuracy vs Expected")
            merged = expected_df.merge(
                tickets_df[["source_id", "category"]].rename(columns={"category": "predicted_category"}),
                left_on="source_id",
                right_on="source_id",
                how="inner",
            )
            if not merged.empty and "category" in merged.columns and "predicted_category" in merged.columns:
                merged["correct"] = merged["category"].str.lower() == merged["predicted_category"].str.lower()
                accuracy = merged["correct"].mean() * 100
                st.metric("Classification Accuracy", f"{accuracy:.1f}%")
                st.dataframe(
                    merged[["source_id", "category", "predicted_category", "correct"]],
                    width="stretch",
                )
