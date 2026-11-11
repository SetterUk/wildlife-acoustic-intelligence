"""
Audio visualization and signal processing utilities for Streamlit UI.
"""

import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from scipy import signal


def generate_waveform_plot(audio_data: np.ndarray, sr: int, title: str = "Acoustic Waveform") -> go.Figure:
    """Generates an interactive Plotly waveform plot."""
    # Downsample for fast responsive UI rendering
    max_points = 4000
    if len(audio_data) > max_points:
        step = len(audio_data) // max_points
        sub_data = audio_data[::step]
        time_axis = np.linspace(0, len(audio_data) / sr, len(sub_data))
    else:
        sub_data = audio_data
        time_axis = np.linspace(0, len(audio_data) / sr, len(sub_data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=sub_data,
        mode="lines",
        line=dict(color="#2ca02c", width=1.5),
        name="Amplitude"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="#333333"),
        yaxis=dict(title="Normalized Amplitude", showgrid=True, gridcolor="#333333", range=[-1.05, 1.05]),
        margin=dict(l=40, r=20, t=35, b=35),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)"
    )
    return fig


def generate_spectrogram_plot(audio_data: np.ndarray, sr: int, title: str = "Bioacoustic Spectrogram") -> go.Figure:
    """Generates an interactive Plotly Mel/STFT spectrogram plot."""
    # Compute STFT
    nperseg = min(1024, len(audio_data))
    if nperseg < 64:
        return go.Figure()
    
    frequencies, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density"
    )

    # Convert to log scale (dB)
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=times,
        y=frequencies,
        colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8)
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(title="Time (seconds)", showgrid=False),
        yaxis=dict(title="Frequency (Hz)", showgrid=False, range=[0, min(16000, sr // 2)]),
        margin=dict(l=40, r=20, t=35, b=35),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)"
    )
    return fig

