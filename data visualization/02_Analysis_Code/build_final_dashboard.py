#!/usr/bin/env python3
"""
Build Final Dashboard
=====================

Assembles the final interactive dashboard by:
1. Loading calculated metrics from JSON
2. Reading the individual chart HTML files
3. Injecting them into a master HTML template with McKinsey-style CSS
4. Saving the result as a standalone, portable HTML file.

The resulting dashboard is designed to be self-contained and ready for presentation.
"""

import json
from pathlib import Path

# Load metrics
# These metrics populate the "Hero" section of the dashboard
with open("../03_Data_Files/summary_metrics.json", 'r') as f:
    metrics = json.load(f)

print("Building dashboard...")
print(f"Loaded metrics: {metrics}")

# ============================================================================
# DASHBOARD TEMPLATE
# ============================================================================

# Build the dashboard HTML
# Uses iframes to embed the Plotly charts to ensure they render independently and reliably
dashboard_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-5 Model Comparison | Strategic Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* McKinsey Design System */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Helvetica Neue', 'Arial', sans-serif;
            background-color: #FFFFFF;
            color: #333333;
            line-height: 1.6;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #003F5C 0%, #2C4E68 100%);
            color: white;
            padding: 36px 60px;
            border-bottom: 4px solid #00A3A1;
        }}
        
        .header h1 {{
            font-size: 34px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 16px;
            opacity: 0.88;
            font-weight: 400;
        }}
        
        /* Hero Metrics */
        .hero-metrics {{
            background: #F7F7F7;
            padding: 52px 0;
            border-bottom: 1px solid #E0E0E0;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 60px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 32px;
        }}
        
        .metric-card {{
            background: white;
            padding: 36px 30px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            text-align: center;
            transition: all 0.25s ease;
            border: 1px solid rgba(0,0,0,0.04);
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }}
        
        .metric-value {{
            font-size: 72px;
            font-weight: 700;
            color: #003F5C;
            line-height: 0.95;
            margin-bottom: 14px;
            font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
        }}
        
        .metric-label {{
            font-size: 13px;
            color: #58595B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
        }}
        
        .metric-context {{
            font-size: 14px;
            color: #808285;
            margin-top: 10px;
            line-height: 1.4;
        }}
        
        .metric-subvalue {{
            font-size: 12px;
            color: #999;
            margin-top: 6px;
        }}
        
        /* Section */
        .section {{
            padding: 64px 0;
            border-bottom: 1px solid #E8E8E8;
        }}
        
        .section-header {{
            margin-bottom: 44px;
        }}
        
        .section-number {{
            font-size: 13px;
            font-weight: 700;
            color: #00A3A1;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
        }}
        
        .section-title {{
            font-size: 30px;
            font-weight: 700;
            color: #003F5C;
            margin-bottom: 16px;
            letter-spacing: -0.6px;
            line-height: 1.3;
        }}
        
        .section-insight {{
            font-size: 18px;
            color: #58595B;
            font-weight: 400;
            max-width: 950px;
            line-height: 1.7;
        }}
        
        .section-insight strong {{
            color: #003F5C;
            font-weight: 600;
        }}
        
        /* Chart Container */
        .chart-row {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 40px;
            margin-top: 36px;
        }}
        
        .chart-row.single {{
            grid-template-columns: 1fr;
        }}
        
        .chart-container {{
            background: white;
            padding: 0;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            border: 1px solid rgba(0,0,0,0.05);
        }}
        
        .chart-header {{
            padding: 28px 32px 20px;
            border-bottom: 1px solid #F0F0F0;
        }}
        
        .chart-title {{
            font-size: 20px;
            font-weight: 600;
            color: #003F5C;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        
        .chart-description {{
            font-size: 14px;
            color: #808285;
            line-height: 1.6;
        }}
        
        .chart-body {{
            padding: 24px 32px 32px;
        }}
        
        .chart-wrapper {{
            width: 100%;
            min-height: 420px;
        }}
        
        .chart-footer {{
            padding: 18px 32px;
            background: #FAFAFA;
            border-top: 1px solid #F0F0F0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .chart-meta {{
            font-size: 12px;
            color: #999;
        }}
        
        /* Takeaway Box */
        .takeaway-box {{
            background: #F7F9FA;
            border-left: 4px solid #00A3A1;
            padding: 28px 32px;
            margin: 36px 0;
            border-radius: 3px;
        }}
        
        .takeaway-box h4 {{
            font-size: 13px;
            font-weight: 700;
            color: #003F5C;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 14px;
        }}
        
        .takeaway-box p {{
            font-size: 16px;
            color: #58595B;
            line-height: 1.7;
        }}
        
        /* Footer */
        .footer {{
            background: #F7F7F7;
            padding: 40px 60px;
            border-top: 1px solid #E0E0E0;
        }}
        
        .footer-content {{
            max-width: 1400px;
            margin: 0 auto;
            font-size: 12px;
            color: #808285;
            line-height: 2;
        }}
        
        .footer-content strong {{
            color: #58595B;
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .chart-row {{
                gap: 32px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 0 24px;
            }}
            
            .header {{
                padding: 28px 24px;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
            
            .chart-row {{
                grid-template-columns: 1fr;
                gap: 28px;
            }}
            
            .metric-value {{
                font-size: 56px;
            }}
            
            .section-title {{
                font-size: 24px;
            }}
            
            .footer {{
                padding: 32px 24px;
            }}
        }}
        
        /* Print */
        @media print {{
            .chart-footer {{
                display: none;
            }}
            .section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>GPT-5 Model Comparison Analysis</h1>
        <p class="subtitle">Strategic Evaluation | Week 7 Dataset ({metrics['total_startups']:,} Startups) | November 2025</p>
    </div>
    
    <!-- Hero Metrics -->
    <div class="hero-metrics">
        <div class="container">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Agreement Rate</div>
                    <div class="metric-value">{metrics['agreement_rate']}<span style="font-size: 0.55em;">%</span></div>
                    <div class="metric-context">{metrics['agreements']:,} agreements</div>
                    <div class="metric-subvalue">{100 - metrics['agreement_rate']:.2f}% disagree ({metrics['disagreements']:,} cases)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mini Confidence</div>
                    <div class="metric-value">{metrics['mini_conf_mean']}</div>
                    <div class="metric-context">Mean score (1-5 scale)</div>
                    <div class="metric-subvalue">σ = {metrics['mini_conf_std']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Nano Confidence</div>
                    <div class="metric-value">{metrics['nano_conf_mean']}</div>
                    <div class="metric-context">Mean score (1-5 scale)</div>
                    <div class="metric-subvalue">σ = {metrics['nano_conf_std']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Classification Diff</div>
                    <div class="metric-value" style="color: #00A3A1;">+{metrics['nano_ai_rate'] - metrics['mini_ai_rate']:.2f}<span style="font-size: 0.55em;">%</span></div>
                    <div class="metric-context">Nano more liberal</div>
                    <div class="metric-subvalue">Mini: {metrics['mini_ai_rate']}% | Nano: {metrics['nano_ai_rate']}%</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Section 1: Overview -->
    <div class="section">
        <div class="container">
            <div class="section-header">
                <div class="section-number">01. OVERVIEW</div>
                <h2 class="section-title">Nano classifies 19% more startups as AI-native than Mini, but models agree 98% of the time</h2>
                <p class="section-insight">
                    Both models show <strong>exceptionally high agreement ({metrics['agreement_rate']}%)</strong> across {metrics['total_startups']:,} startups.
                    GPT-5-nano identifies {metrics['nano_ai_rate']}% of startups as AI-native compared to Mini's {metrics['mini_ai_rate']}%, 
                    suggesting nano has a <strong>more liberal classification threshold</strong> while maintaining high reliability.
                </p>
            </div>
            
            <div class="chart-row">
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Mini identifies {metrics['mini_ai_count']:,} AI-native startups ({metrics['mini_ai_rate']}%)</div>
                        <div class="chart-description">Classification rate comparison shows nano's more liberal approach</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart1_classification_rates.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">GPT-5-mini vs GPT-5-nano</span>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Models agree on {metrics['agreement_rate']}% of all classifications</div>
                        <div class="chart-description">High agreement rate validates reliability of both models</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart2_agreement_rate.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">Agreement: {metrics['agreements']:,} | Disagreement: {metrics['disagreements']:,}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Section 2: Confidence -->
    <div class="section">
        <div class="container">
            <div class="section-header">
                <div class="section-number">02. CONFIDENCE ANALYSIS</div>
                <h2 class="section-title">Mini demonstrates 5% higher confidence with lower variance than Nano</h2>
                <p class="section-insight">
                    GPT-5-mini shows <strong>consistently higher confidence</strong> ({metrics['mini_conf_mean']} vs {metrics['nano_conf_mean']}) with 
                    tighter distribution (σ={metrics['mini_conf_std']} vs {metrics['nano_conf_std']}). The moderate correlation (r={metrics['correlation']:.3f}) suggests 
                    models assess certainty through <strong>different lenses</strong> despite reaching similar classification conclusions.
                </p>
            </div>
            
            <div class="chart-row">
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Mini peaks at confidence level 5, Nano distributes more evenly</div>
                        <div class="chart-description">Distribution reveals mini's higher certainty in classifications</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart3_confidence_distributions.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">Confidence scores (1-5 scale) across {metrics['total_startups']:,} startups</span>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Moderate correlation (r={metrics['correlation']:.3f}) indicates distinct confidence personalities</div>
                        <div class="chart-description">Models reach similar conclusions but assess certainty differently</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart4_confidence_correlation.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">Mini confidence vs Nano confidence (sample of 10,000 points)</span>
                    </div>
                </div>
            </div>
            
            <div class="takeaway-box">
                <h4>Key Takeaway</h4>
                <p>Mini's higher confidence and lower variance can serve as a <strong>quality indicator</strong> for filtering high-certainty classifications. 
                   Deploy Mini for precision-critical tasks and use confidence scores above 4 for automated processing. Cases with low confidence from both models warrant manual review.</p>
            </div>
        </div>
    </div>
    
    <!-- Section 3: Disagreements -->
    <div class="section">
        <div class="container">
            <div class="section-header">
                <div class="section-number">03. DISAGREEMENT PATTERNS</div>
                <h2 class="section-title">In disagreements, Nano classifies as AI-native 5.9× more often than Mini</h2>
                <p class="section-insight">
                    Among the {metrics['disagreements']:,} disagreement cases ({100 - metrics['agreement_rate']:.2f}% of dataset), nano identifies companies as AI-native 
                    <strong>in 79.8% of cases</strong> ({metrics['nano_ai_mini_not']:,} startups) while mini does so only 13.6% of the time ({metrics['mini_ai_nano_not']:,} startups).
                    This pattern confirms nano's <strong>systematically more liberal classification approach</strong>.
                </p>
            </div>
            
            <div class="chart-row">
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Nano says "AI-native" 5.9× more frequently in disagreements</div>
                        <div class="chart-description">Breakdown of {metrics['disagreements']:,} cases reveals clear directional bias</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart5_disagreement_breakdown.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">Disagreement type analysis | Full dataset available in CSV</span>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Higher confidence levels strongly predict higher agreement rates</div>
                        <div class="chart-description">Agreement rate by confidence level validates confidence as quality metric</div>
                    </div>
                    <div class="chart-body">
                        <iframe src="charts/chart6_agreement_by_confidence.html" style="width:100%; height:420px; border:none;"></iframe>
                    </div>
                    <div class="chart-footer">
                        <span class="chart-meta">Confidence level (1-5) vs Agreement rate percentage</span>
                    </div>
                </div>
            </div>
            
            <div class="takeaway-box">
                <h4>Strategic Recommendation</h4>
                <p><strong>For maximum precision</strong> (minimize false positives): Use Mini's classifications with confidence ≥4. 
                   <strong>For maximum recall</strong> (catch all potential AI companies): Use Nano's classifications. 
                   <strong>For balanced approach</strong>: Flag disagreements ({metrics['disagreements']:,} cases) for human review, auto-process high-confidence agreements.</p>
            </div>
        </div>
    </div>
    
    <!-- Footer -->
    <div class="footer">
        <div class="footer-content">
            <strong>Methodology:</strong> Analysis based on {metrics['total_startups']:,} startup classifications using GPT-5-mini and GPT-5-nano models. 
            Classifications used identical system prompts with 10 few-shot examples covering diverse classification scenarios. Confidence scores self-reported by models on 1-5 scale 
            (capped at 4 for companies with only short descriptions).
            <br><br>
            <strong>Data Source:</strong> Week 7 startup dataset (company_us_short_long_desc_.csv) | <strong>Analysis Date:</strong> November 2025 | 
            <strong>Agreement Rate Improvement:</strong> +1.74% vs Week 6 (96.5% → 98.24%) | 
            <strong>Analysis generated by Python pipeline</strong>
        </div>
    </div>
    
    <script>
        console.log("Metrics:", {json.dumps(metrics)});
    </script>
</body>
</html>'''

# Save the dashboard
output_path = "../01_Presentation_Materials/mckinsey_dashboard_final.html"
with open(output_path, 'w') as f:
    f.write(dashboard_html)

print(f"\nDashboard created: {output_path}")
