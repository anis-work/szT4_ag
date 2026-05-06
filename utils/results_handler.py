"""Results handling and display logic for CV Ranking Agent."""

import streamlit as st
from utils.ui_components import (
    render_stats_bar,
    render_results_table,
    render_export_section
)


def show_results(results: list, role: str, cvs: list):
    """Display the ranking results with export options."""
    # Show statistics
    render_stats_bar(results)
    
    # Display results table (all results, no filtering)
    st.caption(f"Showing {len(results)} candidates")
    st.markdown("---")
    
    render_results_table(results)
    
    # Export section
    render_export_section(results, role)
