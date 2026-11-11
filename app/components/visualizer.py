"""
Interactive visualization components for prediction rankings and confidence metrics.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


CATEGORY_COLORS = {
    "Birds": "#38bdf8",          # Sky blue
    "Mammals": "#f59e0b",        # Amber
    "Amphibians": "#10b981",     # Emerald
    "Insects": "#a855f7",        # Purple
    "Reptiles": "#ef4444",       # Crimson
    "Marine animals": "#06b6d4", # Cyan
    "General": "#94a3b8"         # Slate
}


def build_topk_ranking_chart(predictions: List[Dict[str, Any]], title: str = "Top Predicted Species (AnimalCLAP Cosine Similarity)") -> go.Figure:
    """Creates a stylized horizontal bar chart for Top-K predictions."""
    if not predictions:
        return go.Figure()

    # Reverse order so #1 is at top of vertical axis
    rev_preds = list(reversed(predictions))

    labels = [f"#{p['rank']} {p['common_name']}<br><i>{p['scientific_name']}</i>" for p in rev_preds]
    scores = [p['similarity_score'] for p in rev_preds]
    categories = [p.get('category', 'General') for p in rev_preds]
    bar_colors = [CATEGORY_COLORS.get(cat, "#3b82f6") for cat in categories]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=[f"<b>{s:.4f}</b> ({cat})" for s, cat in zip(scores, categories)],
        textposition="auto",
        textfont=dict(size=12, color="white"),
        hoverinfo="text",
        hovertext=[
            f"<b>{p['common_name']}</b> (<i>{p['scientific_name']}</i>)<br>"
            f"Taxonomy: {p.get('class')}, {p.get('order')}, {p.get('family')}<br>"
            f"Cosine Similarity: {p['similarity_score']:.4f}<br>"
            f"Typical Habitat: {p.get('habitat', 'N/A')}<br>"
            f"Vocalization: {p.get('vocalization_type', 'N/A')}"
            for p in rev_preds
        ]
    ))

    # Min range adjustment for negative or positive cosine similarities
    min_x = min(0.0, min(scores) - 0.05)
    max_x = max(1.0, max(scores) + 0.1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f1f5f9")),
        xaxis=dict(
            title="Cosine Similarity Score [-1.0 to 1.0]",
            range=[min_x, max_x],
            showgrid=True,
            gridcolor="#334155",
            zerolinecolor="#64748b"
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#e2e8f0")
        ),
        margin=dict(l=160, r=30, t=40, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)"
    )

    return fig

