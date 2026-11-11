"""
Session analysis history manager and data export component.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from app.utils.export_utils import export_to_csv, export_to_json


def render_history_section():
    """Renders the analysis history table and export action buttons."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    history = st.session_state.analysis_history

    st.markdown("### 📜 Session Analysis History")

    if not history:
        st.info("No recordings have been analyzed yet in this session. Upload or select an audio sample above.")
        return

    # Build summary dataframe for display
    summary_data = []
    for idx, rec in enumerate(reversed(history), start=1):
        top1 = rec.get("top_prediction") or {}
        hw = rec.get("hardware_metrics") or {}
        audio = rec.get("audio_metadata") or {}

        summary_data.append({
            "Seq": len(history) - idx + 1,
            "Timestamp": rec.get("timestamp", "N/A"),
            "Audio File": rec.get("filename", "Audio"),
            "Top-1 Prediction": f"{top1.get('common_name')} ({top1.get('scientific_name')})",
            "Category": top1.get("category", "General"),
            "Similarity": f"{top1.get('similarity_score', 0):.4f}",
            "Duration": f"{audio.get('original_duration_seconds', 0)}s",
            "Latency": f"{hw.get('inference_time_sec', 0)}s",
            "Device": hw.get("device", "N/A")
        })

    df_display = pd.DataFrame(summary_data)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Export Buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        csv_data = export_to_csv(history)
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_data,
            file_name="wildlife_acoustic_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        json_data = export_to_json(history)
        st.download_button(
            label="📥 Export History (JSON)",
            data=json_data,
            file_name="wildlife_acoustic_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:
        if st.button("🗑️ Clear Session History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()

