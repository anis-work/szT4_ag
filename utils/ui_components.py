"""UI components for CV Ranking Agent.

Rendering layer only — all styles live in frontend/styles/main.css
and all markup templates live in frontend/templates/.
"""

import base64
import html
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Asset paths ──────────────────────────────────────────────────────────────
_FRONTEND = Path(__file__).parent.parent / "frontend"
_STYLES_DIR = _FRONTEND / "styles"
_TEMPLATES_DIR = _FRONTEND / "templates"
_ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    """Read a text file and return its contents, or empty string on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_bytes_b64(path: Path) -> str:
    """Read a binary file and return base64-encoded string, or empty on failure."""
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""


def _render_template(name: str, **tokens) -> str:
    """Load an HTML template and replace {{token}} placeholders."""
    template = _read_file(_TEMPLATES_DIR / name)
    for key, value in tokens.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
    return template


# ── Public API ────────────────────────────────────────────────────────────────

def apply_custom_styles() -> None:
    """Inject the Sulzer stylesheet into the Streamlit page."""
    css = _read_file(_STYLES_DIR / "main.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the Sulzer-branded application header."""
    svg_b64 = _read_bytes_b64(_ASSETS_DIR / "SulzerLogo.svg")
    logo_html = (
        f'<img src="data:image/svg+xml;base64,{svg_b64}" height="26" alt="Sulzer"'
        f' style="filter:brightness(0) invert(1); display:block;">'
        if svg_b64
        else '<span style="color:#fff;font-weight:700;font-size:1.1rem;letter-spacing:0.05em;">SULZER</span>'
    )
    st.markdown(_render_template("header.html", logo=logo_html), unsafe_allow_html=True)


def render_file_upload_section(uploader_key) -> list:
    """Render the resume upload section and return uploaded files."""
    st.markdown('<div class="section-title">📁 Upload Resumes</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "uploader",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload candidate resumes in PDF or DOCX format",
        key=f"file_uploader_{uploader_key}"
    )

    if uploaded:
        st.markdown(
            f'<div style="margin-top:0.5rem;font-size:0.875rem;color:#5A7FA8;">✓ {len(uploaded)} file(s) ready</div>',
            unsafe_allow_html=True,
        )
        for f in uploaded:
            size_kb = round(f.size / 1024, 1)
            st.markdown(
                _render_template("file_item.html", filename=html.escape(f.name), size_kb=size_kb),
                unsafe_allow_html=True,
            )

    return uploaded


def render_job_details_section() -> tuple[str, str]:
    """Render the job title and description inputs. Returns (role, jd_text)."""
    st.markdown('<div class="section-title">💼 Job Details</div>', unsafe_allow_html=True)

    role = st.text_input(
        "job_title",
        placeholder="e.g., Senior Data Engineer",
        label_visibility="collapsed",
        key="job_title_input",
    )

    jd_text = st.text_area(
        "job_description",
        height=200,
        placeholder="Paste the complete job description including required skills, experience, and responsibilities...",
        label_visibility="collapsed",
        key="job_desc_input",
    )

    return role, jd_text


def render_action_button(uploaded, role: str, jd_text: str) -> bool:
    """Render the analyze button and readiness status. Returns True when clicked."""
    ready = bool(uploaded and role.strip() and jd_text.strip())

    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        run = st.button("🚀 Analyze Candidates", type="primary", disabled=not ready, width='stretch')
    with col_status:
        if not uploaded:
            st.caption("⚠️ Please upload at least one resume")
        elif not role.strip() or not jd_text.strip():
            st.caption("⚠️ Please provide job title and description")
        else:
            st.caption(f"✅ Ready to analyze {len(uploaded)} candidate(s)")

    return run


def render_stats_bar(results: list) -> None:
    """Render the summary statistics bar above the results table."""
    avg_score = round(sum(r.score for r in results) / len(results))
    strong_matches = sum(1 for r in results if r.score >= 75)
    st.markdown(
        _render_template(
            "stats_bar.html",
            total=len(results),
            top_score=results[0].score,
            avg_score=avg_score,
            strong_matches=strong_matches,
        ),
        unsafe_allow_html=True,
    )


def render_results_table(results: list) -> None:
    """Render the ranked candidates dataframe with interactive toolbar."""
    import html as html_lib
    df = pd.DataFrame([{
        "Rank": r.rank,
        "Candidate Name": r.candidate_name,
        "Score": r.score,
        "Experience (yrs)": r.experience_years,
        "Skills Missing": html_lib.unescape(r.skills_missing or ""),
        "Key Strengths": html_lib.unescape(r.key_strengths or ""),
        "Assessment": html_lib.unescape(r.reason or ""),
    } for r in results])

    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Candidate Name": st.column_config.TextColumn("Candidate Name", width="medium"),
            "Score": st.column_config.NumberColumn("Score", width="small"),
            "Experience (yrs)": st.column_config.NumberColumn("Experience", width="small"),
            "Skills Missing": st.column_config.TextColumn("Missing Skills", width="medium"),
            "Key Strengths": st.column_config.TextColumn("Key Strengths", width="large"),
            "Assessment": st.column_config.TextColumn("Assessment", width="large"),
        },
    )


def render_export_section(results: list, role: str) -> None:
    """Render the CSV download button and footer caption below the results table."""
    import io
    df = pd.DataFrame([{
        "Rank": r.rank,
        "Candidate Name": r.candidate_name,
        "Score": r.score,
        "Experience (yrs)": r.experience_years,
        "Skills Missing": r.skills_missing,
        "Key Strengths": r.key_strengths,
        "Assessment": r.reason,
    } for r in results])

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    st.markdown("---")
    col_dl, col_caption = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"cv_ranking_{role.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            width='stretch',
        )
    with col_caption:
        st.caption(f"📊 Results for: **{role}** | {len(results)} candidates shown")


def render_history_section(history: list) -> tuple:
    """Render the analysis history expander list. Returns (index, entry) or (None, None)."""
    if not history:
        return None, None

    st.markdown('<div class="section-title">📜 Analysis History</div>', unsafe_allow_html=True)

    for i, entry in enumerate(reversed(history)):
        with st.expander(f"🕒 {entry['timestamp']} — {entry['role']} ({entry['candidates']} candidates)"):
            st.caption(f"Top candidate: **{entry['top_candidate']}** (Score: {entry['top_score']})")
            if st.button("👁️ View Results", key=f"history_view_{i}"):
                return i, entry

    return None, None
