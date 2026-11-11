"""
Dashboard status badges, hardware telemetry, and model overview cards.
"""

import streamlit as st
from typing import Dict, Any
from inference.model_loader import AnimalCLAPManager


def render_dashboard_header():
    """Renders the top application header and description."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); padding: 22px; border-radius: 12px; margin-bottom: 25px; border: 1px solid #10b981;">
            <h1 style="color: #ecfdf5; margin: 0; font-size: 2.2rem; letter-spacing: -0.5px;">
                🌲 WILDLIFE ACOUSTIC INTELLIGENCE
            </h1>
            <p style="color: #a7f3d0; margin-top: 6px; margin-bottom: 0; font-size: 1.05rem; font-weight: 400;">
                <i>AI-powered wildlife species recognition using bioacoustic signals and official AnimalCLAP (ICASSP 2026).</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hardware_telemetry(gpu_status: Dict[str, Any], load_info: Dict[str, Any]):
    """Renders hardware and runtime telemetry cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if gpu_status.get("cuda_available", False):
            st.metric(
                label="🖥️ Compute Device",
                value="CUDA:0",
                delta="RTX 4050 (6 GB)"
            )
        else:
            st.metric(
                label="🖥️ Compute Device",
                value="CPU Mode",
                delta="Fallback"
            )

    with col2:
        alloc_vram = gpu_status.get("allocated_vram_mb", 0.0)
        total_vram = gpu_status.get("total_vram_gb", 0.0) * 1024
        vram_pct = round((alloc_vram / total_vram) * 100, 1) if total_vram > 0 else 0
        st.metric(
            label="⚡ VRAM Allocated",
            value=f"{alloc_vram:.1f} MB",
            delta=f"{vram_pct}% of 6 GB"
        )

    with col3:
        load_time = load_info.get("load_time_seconds", 1.76)
        st.metric(
            label="⏱️ Model Load Time",
            value=f"{load_time:.2f}s",
            delta="Offline Cached"
        )

    with col4:
        st.metric(
            label="🎵 Target Sample Rate",
            value="48,000 Hz",
            delta="10.0s Windows"
        )

