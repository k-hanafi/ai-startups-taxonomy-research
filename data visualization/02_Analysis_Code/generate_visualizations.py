#!/usr/bin/env python3
"""
Visualization Generator - McKinsey Style
=========================================

Generates the suite of 6 strategic charts for the GPT-5 model comparison dashboard.
Uses Plotly for interactive visualizations with a custom "McKinsey-style" design theme.

Charts Generated:
1. Classification Rates (Bar): Comparison of AI-native identification rates.
2. Agreement Rate (Pie): Consensus vs disagreement breakdown.
3. Confidence Distributions (Histogram): How confident each model is.
4. Confidence Correlation (Scatter): Relationship between model confidence scores.
5. Disagreement Breakdown (Bar): Analysis of disagreement types.
6. Agreement by Confidence (Line): Correlation between confidence and accuracy/agreement.

Outputs:
- Individual HTML files for each chart (saved to insights/01_Presentation_Materials/)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# Define paths
DATA_DIR = Path("../03_Data_Files")
OUTPUT_DIR = Path("../01_Presentation_Materials/charts")

print("Loading data...")
mini_df = pd.read_csv(DATA_DIR / "mini_aligned.csv")
nano_df = pd.read_csv(DATA_DIR / "nano_aligned.csv")
disagreement_df = pd.read_csv(DATA_DIR / "disagreement_dataset.csv")
agreement_by_conf_df = pd.read_csv(DATA_DIR / "agreement_by_confidence.csv")

with open(DATA_DIR / "summary_metrics.json", 'r') as f:
    metrics = json.load(f)

# Professional Color Palette (McKinsey-inspired)
COLORS = {
    'mini': '#3B8FC4',      # Professional Blue
    'nano': '#1ABC9C',      # Teal/Green
    'agreement': '#2E8B57', # Success Green
    'disagreement': '#E74C3C', # Alert Red
    'navy': '#003F5C',
    'teal': '#00A3A1',
    'gray': '#808285'
}

def mckinsey_layout(title=""):
    """
    Applies a clean, professional layout theme to Plotly figures.
    Features minimal gridlines, clear typography, and whitespace.
    """
    return dict(
        font=dict(family='Helvetica Neue, Arial', size=13, color='#58595B'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(t=40, r=20, b=60, l=70),
        height=400,
        title=dict(text=title, font=dict(size=16, color='#003F5C')),
        xaxis=dict(
            gridcolor='#F0F0F0',
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color='#808285')
        ),
        yaxis=dict(
            gridcolor='#F0F0F0',
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color='#808285')
        )
    )

print("Generating visualizations...")

# ============================================================================
# CHART GENERATION
# ============================================================================

# Chart 1: Classification Rates
# Compares the percentage of startups identified as AI-native by each model.
print("Chart 1: Classification Rates...")
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=['GPT-5-mini', 'GPT-5-nano'],
    y=[metrics['mini_ai_rate'], metrics['nano_ai_rate']],
    text=[f"{metrics['mini_ai_rate']}%", f"{metrics['nano_ai_rate']}%"],
    textposition='outside',
    textfont=dict(size=16, color='#003F5C', family='SF Pro Display, Arial'),
    marker=dict(color=[COLORS['mini'], COLORS['nano']], line=dict(width=0)),
    hovertemplate='%{x}<br>%{y:.2f}%<br>Count: %{customdata:,}<extra></extra>',
    customdata=[metrics['mini_ai_count'], metrics['nano_ai_count']]
))

layout1 = mckinsey_layout()
layout1['yaxis'].update(dict(title='AI-native Classification Rate (%)', range=[0, max(metrics['mini_ai_rate'], metrics['nano_ai_rate']) * 1.3]))
layout1['xaxis'].update(dict(title=''))
fig1.update_layout(layout1)
fig1.write_html(OUTPUT_DIR / "chart1_classification_rates.html")


# Chart 2: Agreement Rate
# Donut chart showing the high level of consensus between models.
print("Chart 2: Agreement Rate...")
fig2 = go.Figure()
fig2.add_trace(go.Pie(
    labels=['Agree', 'Disagree'],
    values=[metrics['agreements'], metrics['disagreements']],
    hole=0.5,
    marker=dict(colors=[COLORS['agreement'], COLORS['disagreement']], line=dict(width=0)),
    textinfo='label+percent',
    textfont=dict(size=14, color='white'),
    hovertemplate='%{label}<br>%{value:,} startups<br>%{percent}<extra></extra>'
))

fig2.update_layout(
    **mckinsey_layout(),
    annotations=[dict(
        text=f"{metrics['agreement_rate']}%<br><span style='font-size:12px; color:#808285'>Agreement</span>",
        x=0.5, y=0.5,
        font=dict(size=32, color='#003F5C', family='SF Pro Display, Arial'),
        showarrow=False
    )]
)
fig2.write_html(OUTPUT_DIR / "chart2_agreement_rate.html")


# Chart 3: Confidence Distributions
# Overlaid histograms to compare how confident each model is in its predictions.
print("Chart 3: Confidence Distributions...")
fig3 = go.Figure()

fig3.add_trace(go.Histogram(
    x=mini_df['Confidence_1to5'],
    name='GPT-5-mini',
    opacity=0.75,
    marker=dict(color=COLORS['mini'], line=dict(width=0)),
    xbins=dict(start=0.5, end=5.5, size=1),
    hovertemplate='Confidence: %{x}<br>Count: %{y:,}<extra></extra>'
))

fig3.add_trace(go.Histogram(
    x=nano_df['Confidence_1to5'],
    name='GPT-5-nano',
    opacity=0.75,
    marker=dict(color=COLORS['nano'], line=dict(width=0)),
    xbins=dict(start=0.5, end=5.5, size=1),
    hovertemplate='Confidence: %{x}<br>Count: %{y:,}<extra></extra>'
))

layout3 = mckinsey_layout()
layout3['barmode'] = 'overlay'
layout3['showlegend'] = True
layout3['legend'] = dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
layout3['xaxis'].update(dict(title='Confidence Score (1-5)', tickvals=[1,2,3,4,5]))
layout3['yaxis'].update(dict(title='Number of Startups'))
fig3.update_layout(layout3)

# Add vertical lines for mean confidence
fig3.add_vline(x=metrics['mini_conf_mean'], line_dash="dash", line_color=COLORS['mini'], line_width=2,
               annotation_text=f"Mini μ={metrics['mini_conf_mean']}", annotation_position="top")
fig3.add_vline(x=metrics['nano_conf_mean'], line_dash="dash", line_color=COLORS['nano'], line_width=2,
               annotation_text=f"Nano μ={metrics['nano_conf_mean']}", annotation_position="top right")

fig3.write_html(OUTPUT_DIR / "chart3_confidence_distributions.html")


# Chart 4: Confidence Correlation
# Scatter plot to see if models are confident about the same startups.
print("Chart 4: Confidence Correlation...")
fig4 = go.Figure()
sample_size = min(10000, len(mini_df))
sample_indices = np.random.choice(len(mini_df), sample_size, replace=False)

fig4.add_trace(go.Scatter(
    x=mini_df.iloc[sample_indices]['Confidence_1to5'],
    y=nano_df.iloc[sample_indices]['Confidence_1to5'],
    mode='markers',
    marker=dict(color='#3B8FC4', size=4, opacity=0.3, line=dict(width=0)),
    hovertemplate='Mini: %{x}<br>Nano: %{y}<extra></extra>'
))

# Add regression line
valid_mask = mini_df['Confidence_1to5'].notna() & nano_df['Confidence_1to5'].notna()
mini_clean = mini_df[valid_mask]['Confidence_1to5'].values
nano_clean = nano_df[valid_mask]['Confidence_1to5'].values

if len(mini_clean) > 0:
    z = np.polyfit(mini_clean, nano_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 5, 100)
    y_line = p(x_line)
    
    fig4.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        name='Regression', hoverinfo='skip'
    ))

layout4 = mckinsey_layout()
layout4['xaxis'].update(dict(title='Mini Confidence', range=[0.5, 5.5], tickvals=[1,2,3,4,5]))
layout4['yaxis'].update(dict(title='Nano Confidence', range=[0.5, 5.5], tickvals=[1,2,3,4,5]))
layout4['annotations'] = [dict(
    text=f"r = {metrics['correlation']:.3f}",
    x=4.5, y=1.5,
    font=dict(size=18, color='#003F5C'),
    showarrow=False,
    bgcolor='rgba(255,255,255,0.9)',
    bordercolor='#E0E0E0',
    borderwidth=1,
    borderpad=8
)]
fig4.update_layout(layout4)
fig4.write_html(OUTPUT_DIR / "chart4_confidence_correlation.html")


# Chart 5: Disagreement Breakdown
# Analyzes which model is more "optimistic" when they disagree.
print("Chart 5: Disagreement Breakdown...")
fig5 = go.Figure()
disagreement_counts = disagreement_df['Disagreement_Type'].value_counts()

fig5.add_trace(go.Bar(
    x=disagreement_counts.index,
    y=disagreement_counts.values,
    text=[f"{v:,}<br>({v/disagreement_counts.sum()*100:.1f}%)" for v in disagreement_counts.values],
    textposition='outside',
    marker=dict(color=[COLORS['mini'], COLORS['nano']], line=dict(width=0)),
    hovertemplate='%{x}<br>%{y:,} startups<br>%{customdata:.1f}%<extra></extra>',
    customdata=[v/disagreement_counts.sum()*100 for v in disagreement_counts.values]
))

layout5 = mckinsey_layout()
layout5['xaxis'].update(dict(title=''))
layout5['yaxis'].update(dict(title='Number of Disagreements'))
layout5['showlegend'] = False
fig5.update_layout(layout5)
fig5.write_html(OUTPUT_DIR / "chart5_disagreement_breakdown.html")


# Chart 6: Agreement by Confidence
# Validates that higher confidence scores actually correlate with higher agreement (proxy for accuracy).
print("Chart 6: Agreement by Confidence...")
fig6 = go.Figure()

mini_data = agreement_by_conf_df[agreement_by_conf_df['Model'] == 'Mini']
fig6.add_trace(go.Scatter(
    x=mini_data['Confidence_Level'],
    y=mini_data['Agreement_Rate'],
    mode='lines+markers',
    name='GPT-5-mini',
    line=dict(color=COLORS['mini'], width=3),
    marker=dict(size=10, color=COLORS['mini']),
    hovertemplate='Confidence: %{x}<br>Agreement: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>',
    customdata=mini_data['Count']
))

nano_data = agreement_by_conf_df[agreement_by_conf_df['Model'] == 'Nano']
fig6.add_trace(go.Scatter(
    x=nano_data['Confidence_Level'],
    y=nano_data['Agreement_Rate'],
    mode='lines+markers',
    name='GPT-5-nano',
    line=dict(color=COLORS['nano'], width=3),
    marker=dict(size=10, color=COLORS['nano']),
    hovertemplate='Confidence: %{x}<br>Agreement: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>',
    customdata=nano_data['Count']
))

layout6 = mckinsey_layout()
layout6['showlegend'] = True
layout6['legend'] = dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
layout6['xaxis'].update(dict(title='Confidence Level', tickvals=[1,2,3,4,5]))
layout6['yaxis'].update(dict(title='Agreement Rate (%)', range=[90, 100]))
fig6.update_layout(layout6)
fig6.write_html(OUTPUT_DIR / "chart6_agreement_by_confidence.html")

print("All visualizations generated.")
