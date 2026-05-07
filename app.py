"""Streamlit web UI for CV Ranking Agent."""

import asyncio
import logging
from datetime import datetime, timezone

import streamlit as st

from models import JobDescription
from utils.ui_components import (
    apply_custom_styles,
    render_header,
    render_file_upload_section,
    render_job_details_section,
    render_action_button,
    render_history_section
)
from utils.pipeline import (
    get_kernel,
    save_uploads,
    build_cvs,
    run_pipeline
)
from utils.results_handler import show_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sulzer CV Ranking Agent",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom styles
apply_custom_styles()

# Render header
render_header()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Input section
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    uploaded = render_file_upload_section(st.session_state.uploaded_files)

with col2:
    role, jd_text = render_job_details_section()

# Action button
run = render_action_button(uploaded, role, jd_text)

# Process analysis
if run:
    st.session_state.uploaded_files = uploaded
    status_placeholder = st.empty()
    try:
        kernel = get_kernel()
        tmp_dir, filenames = save_uploads(uploaded)
        cvs, skipped = build_cvs(tmp_dir, filenames)

        if skipped:
            for fname, reason in skipped:
                st.warning(f"⚠️ Skipped **{fname}**: {reason}")

        if not cvs:
            st.error("❌ No readable resumes found. Ensure files are text-based PDFs or DOCX.")
            st.stop()

        jd = JobDescription(role=role.strip(), requirements=jd_text.strip())
        results = asyncio.run(run_pipeline(kernel, cvs, jd, status_placeholder))

        st.session_state.history.append({
            "timestamp": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
            "role": role.strip(),
            "candidates": len(results),
            "top_candidate": results[0].candidate_name if results else "-",
            "top_score": results[0].score if results else 0,
            "results": results,
            "cvs": cvs,
        })

        st.session_state.current_results = {
            "results": results,
            "role": role.strip(),
            "cvs": cvs
        }
        st.session_state.show_results = True

        st.success("✅ Analysis complete!")

    except Exception as e:
        logger.exception("Pipeline failed")
        status_placeholder.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 **Troubleshooting:**\n- Verify your API key in `.env`\n- Ensure PDFs are text-based (not scanned images)\n- Try with fewer resumes if hitting rate limits\n- Wait a moment and retry if service is busy")

# Display current results if available
if st.session_state.show_results and st.session_state.current_results:
    col_new, col_space = st.columns([1, 3])
    with col_new:
        if st.button("🔄 Analyze New Resumes", type="secondary", width='stretch'):
            st.session_state.show_results = False
            st.session_state.current_results = None
            st.session_state.uploaded_files = []
            st.rerun()

    st.markdown('<div class="section-title">🏆 Ranking Results</div>', unsafe_allow_html=True)
    show_results(
        st.session_state.current_results["results"],
        st.session_state.current_results["role"],
        st.session_state.current_results["cvs"]
    )

# History section
history_idx, history_entry = render_history_section(st.session_state.history)
if history_entry:
    show_results(history_entry["results"], history_entry["role"], history_entry["cvs"])
