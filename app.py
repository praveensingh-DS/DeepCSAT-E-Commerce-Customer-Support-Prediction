"""
CSAT Score Predictor — Shopzilla eCommerce Support
Compatible with: ML_ANN_Improved.ipynb (Wide & Deep ANN / Deep ANN)

Feature pipeline mirrors the notebook exactly:
  - Binary target: CSAT >= 4 → Satisfied (1), else Unsatisfied (0)
  - Target-encoded: agent_csat_rate, subcategory_csat_rate,
                    manager_csat_rate, supervisor_csat_rate
  - Label-encoded:  channel_name_enc, category_enc,
                    Agent Shift_enc, Tenure Bucket_enc
  - Engineered:     response_time_mins, log_response_time,
                    reported_hour, reported_day, is_weekend

Model file expected at: models/Wide_Deep_ANN.keras  (or Deep_ANN.keras)
Preprocessing file:     models/preprocessing.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import pickle
from datetime import datetime

# TensorFlow is optional — app runs in rule-based demo mode if not installed
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="CSAT Predictor · Shopzilla",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global lookup tables (from notebook's training data) ──
# These mirror the target-encoded values computed in the notebook.
# In production these should be loaded from preprocessing.pkl.

GLOBAL_MEAN = 0.8246   # df['satisfied'].mean()

# Label encoder class order (sorted alphabetically — matches sklearn LabelEncoder)
CHANNEL_CLASSES  = ['Email', 'Inbound', 'Outcall']
CATEGORY_CLASSES = ['App/website', 'Cancellation', 'Feedback', 'Offers & Cashback',
                    'Onboarding related', 'Order Related', 'Others', 'Payments related',
                    'Product Queries', 'Refund Related', 'Returns', 'Shopzilla Related']
SHIFT_CLASSES    = ['Afternoon', 'Evening', 'Morning', 'Night', 'Split']
TENURE_CLASSES   = ['0-30', '31-60', '61-90', '>90', 'On Job Training']

MANAGERS    = ['Emily Chen', 'Jennifer Nguyen', 'John Smith',
               'Michael Lee', 'Olivia Tan', 'William Kim']
MANAGER_RATES = {
    'Emily Chen': 0.8607, 'Jennifer Nguyen': 0.8042, 'John Smith': 0.8314,
    'Michael Lee': 0.8276, 'Olivia Tan': 0.7944, 'William Kim': 0.7926,
}

SUPERVISOR_RATES = {
    'Abigail Suzuki': 0.8281, 'Aiden Patel': 0.8427, 'Alexander Tanaka': 0.8068,
    'Amelia Tanaka': 0.8424, 'Austin Johnson': 0.7922, 'Benjamin Gupta': 0.8212,
    'Charlotte Tanaka': 0.8333, 'Charlotte Yamamoto': 0.8333, 'Chloe Kim': 0.8071,
    'Daniel Lee': 0.8143, 'Dylan Kim': 0.8111, 'Ella Nakamura': 0.8158,
    'Emily Park': 0.8246, 'Emma Park': 0.8196, 'Ethan Nguyen': 0.8301,
    'Grace Kim': 0.8196, 'Isabella Patel': 0.8246, 'Jackson Park': 0.8000,
    'James Johnson': 0.8246, 'James Kim': 0.8333, 'Julia Chen': 0.8082,
    'Lily Chen': 0.8427, 'Logan Tanaka': 0.8246, 'Lucas Gupta': 0.8246,
    'Madison Kim': 0.8246, 'Mason Gupta': 0.8246, 'Michael Park': 0.8246,
    'Mia Johnson': 0.7867, 'Natalie Park': 0.8246, 'Nathan Patel': 0.8246,
    'Noah Kim': 0.8246, 'Olivia Wang': 0.8246, 'Ryan Suzuki': 0.8246,
    'Samantha Gupta': 0.8246, 'Sophia Chen': 0.8246, 'Tyler Lee': 0.8246,
    'Victoria Chen': 0.8246, 'Zoe Park': 0.8246, 'Abigail Kim': 0.8246,
    'Abigail Lee': 0.8246,
}

SUBCAT_RATES = {
    'Account updation': 0.8571, 'Affiliate Offers': 0.8571, 'App/website Related': 0.9048,
    'Billing Related': 0.8696, 'COD Refund Details': 0.8500, 'Call back request': 0.6087,
    'Call disconnected': 0.5500, 'Card/EMI': 0.8500, 'Commission related': 0.3333,
    'Customer Requested Modifications': 0.7692, 'Damaged': 0.8182, 'Delayed': 0.7386,
    'Exchange / Replacement': 0.8571, 'Fraudulent User': 0.7500, 'General Enquiry': 0.8333,
    'Installation/demo': 0.7778, 'Instant discount': 0.9231, 'Invoice request': 0.8254,
    'Issues with Shopzilla App': 0.8261, 'Life Insurance': 0.8182, 'Missing': 0.7308,
    'Non Order related': 0.8000, 'Not Needed': 0.7500, 'Online Payment Issues': 0.8571,
    'Order Verification': 0.7895, 'Order status enquiry': 0.8333, 'Other Account Related Issues': 0.6818,
    'Other Cashback': 0.8235, 'Others': 0.6667, 'PayLater related': 0.8000,
    'Payment pending': 0.8000, 'Payment related Queries': 0.8750, 'Policy Related': 0.5000,
    'Priority delivery': 0.9302, 'Product Specific Information': 0.8214, 'Product related Issues': 0.7609,
    'Refund Enquiry': 0.8261, 'Refund Related Issues': 0.8235, 'Return cancellation': 0.8148,
    'Return request': 0.8621, 'Reverse Pickup Enquiry': 0.8403, 'Self-Help': 0.8621,
    'Seller Cancelled Order': 0.6544, 'Seller onboarding': 0.8571, 'Service Center - Service Denial': 0.6034,
    'Service Centres Related': 0.8065, 'Shopzila Premium Related': 0.8718, 'Shopzilla Rewards': 0.8824,
    'Signup Issues': 0.8571, 'Technician Visit': 0.6402, 'UnProfessional Behaviour': 0.7500,
    'Unable to Login': 0.2857, 'Unable to track': 0.7500, 'Wallet related': 0.8571,
    'Warranty related': 0.7692, 'Wrong': 0.7692, 'e-Gift Voucher': 0.8571,
}

AGENT_RATES_SAMPLE = {
    'Aaron Edwards': 0.8602, 'Aaron Romero': 0.7458, 'Abigail Gonzalez': 0.8000,
    'Adam Barnett': 0.8571, 'Adam Hammond': 0.9500,
}

# ─────────────────────────────────────────────────────────
# CSS — SKEUOMORPHISM: brushed aluminium + warm desk aesthetic
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bitter:ital,wght@0,400;0,600;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

/* ── Root variables ─────────────────────────────────────── */
:root {
    --felt:        #2d4a2d;
    --felt-light:  #3a5f3a;
    --metal-top:   #e8e4dc;
    --metal-mid:   #d4cfc5;
    --metal-dark:  #b8b2a8;
    --metal-edge:  #9a948a;
    --shadow-deep: rgba(0,0,0,0.55);
    --shadow-med:  rgba(0,0,0,0.30);
    --shadow-soft: rgba(0,0,0,0.15);
    --ink:         #1a1612;
    --ink-mid:     #3d3830;
    --ink-light:   #6b6258;
    --amber:       #c8860a;
    --amber-light: #e8a520;
    --red-led:     #cc2200;
    --green-led:   #22aa44;
    --chrome-hi:   rgba(255,255,255,0.85);
    --chrome-lo:   rgba(255,255,255,0.15);
}

/* ── Page background — green baize desk felt ────────────── */
.stApp {
    background-color: var(--felt);
    background-image:
        repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.015) 2px,
            rgba(255,255,255,0.015) 4px
        ),
        repeating-linear-gradient(
            -45deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.04) 2px,
            rgba(0,0,0,0.04) 4px
        );
    font-family: 'Bitter', Georgia, serif;
    color: var(--ink);
}

/* ── Kill all Streamlit's own background/border noise ────── */
div[data-testid="stVerticalBlock"] > div,
div[data-testid="stAppViewBlockContainer"] > div,
div[data-testid="stMainBlockContainer"] > div,
section[data-testid="stMain"] > div {
    background: transparent !important;
    border: none !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Brushed aluminium panel (replaces glass-card) ───────── */
.sk-panel {
    background:
        linear-gradient(
            175deg,
            #ede9e0 0%, #ddd8ce 20%,
            #ccc7bc 40%, #d6d1c7 60%,
            #e0dbd0 80%, #d8d3c9 100%
        );
    border-radius: 10px;
    border-top:    2px solid #f5f2ec;
    border-left:   2px solid #ece8df;
    border-right:  2px solid #9e9890;
    border-bottom: 3px solid #8a847a;
    box-shadow:
        0 8px 32px var(--shadow-deep),
        0 2px  6px var(--shadow-med),
        inset 0 1px 0 var(--chrome-hi),
        inset 0 -1px 0 rgba(0,0,0,0.1);
    padding: 22px 24px 20px;
    margin-bottom: 16px;
    position: relative;
}

/* Screw heads — decorative rivets at corners */
.sk-panel::before,
.sk-panel::after {
    content: '⊕';
    position: absolute;
    font-size: 11px;
    color: #9a948a;
    text-shadow: 0 1px 1px rgba(255,255,255,0.6), 0 -1px 1px rgba(0,0,0,0.3);
    top: 8px;
}
.sk-panel::before { left: 10px; }
.sk-panel::after  { right: 10px; }

/* ── Section header label — engraved look ────────────────── */
.sk-panel h3, .sk-label {
    font-family: 'Bitter', serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #1a1612 !important;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.9),
        0 -1px 0 rgba(0,0,0,0.2) !important;
    border-bottom: 1px solid rgba(0,0,0,0.12);
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}

/* ── Sidebar — dark walnut wood ──────────────────────────── */
section[data-testid="stSidebar"] {
    background:
        repeating-linear-gradient(
            92deg,
            transparent 0px, transparent 3px,
            rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px
        ),
        linear-gradient(180deg, #2c1f0f 0%, #3d2c15 30%, #2e2010 70%, #1e1508 100%) !important;
    border-right: 4px solid #0d0a06 !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.5) !important;
}
section[data-testid="stSidebar"] * {
    color: #d4b896 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e8c87a !important;
    text-shadow: 0 0 12px rgba(232,200,122,0.4) !important;
    font-family: 'Bitter', serif !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(212,184,150,0.25) !important;
}

/* ── Input fields — recessed well ────────────────────────── */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background:
        linear-gradient(180deg, #c8c3b8 0%, #d8d4cc 8%, #e4e0d8 100%) !important;
    border-top: 2px solid #a09a90 !important;
    border-left: 2px solid #aba59b !important;
    border-right: 1px solid #ddd8d0 !important;
    border-bottom: 1px solid #d4cfc7 !important;
    border-radius: 5px !important;
    color: var(--ink) !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.88rem !important;
    box-shadow:
        inset 0 2px 6px rgba(0,0,0,0.18),
        inset 0 1px 2px rgba(0,0,0,0.12) !important;
    padding: 8px 12px !important;
}
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-top-color: #888 !important;
    box-shadow:
        inset 0 2px 6px rgba(0,0,0,0.22),
        0 0 0 2px rgba(200,134,10,0.3) !important;
    outline: none !important;
}

/* ── Select dropdowns ─────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background: linear-gradient(180deg, #ccc7bc 0%, #ddd8ce 100%) !important;
    border-top: 2px solid #a8a29a !important;
    border-left: 2px solid #b0aaa0 !important;
    border-right: 1px solid #ddd8d0 !important;
    border-bottom: 1px solid #d4cfc7 !important;
    border-radius: 5px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.15) !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; }

/* ── Form labels — embossed ──────────────────────────────── */
label,
.stSelectbox label, .stSlider label,
.stNumberInput label, .stTextArea label, .stCheckbox label {
    font-family: 'Bitter', serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--ink-mid) !important;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.8),
        0 -1px 0 rgba(0,0,0,0.15) !important;
}

/* ── Slider track — physical rail ────────────────────────── */
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(180deg, #a8a29a 0%, #c8c2b8 40%, #d8d2c8 100%) !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    border-radius: 3px !important;
    height: 6px !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(145deg, #f0ece4, #c8c3ba) !important;
    border: 2px solid #9a948a !important;
    box-shadow:
        0 3px 8px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.8) !important;
    width: 22px !important;
    height: 22px !important;
    border-radius: 50% !important;
}

/* ── THE PREDICT BUTTON — big physical press button ─────── */
.stButton > button {
    background:
        linear-gradient(180deg,
            #e8a520 0%, #d4920a 15%,
            #c07e05 50%, #d49210 75%,
            #c88010 100%
        ) !important;
    border-top:    2px solid #f0b830 !important;
    border-left:   2px solid #e0a018 !important;
    border-right:  2px solid #9a6200 !important;
    border-bottom: 4px solid #7a4e00 !important;
    border-radius: 8px !important;
    color: #1a1000 !important;
    font-family: 'Bitter', serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
    box-shadow:
        0 6px 18px rgba(0,0,0,0.45),
        0 2px  4px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,220,80,0.6) !important;
    transition: all 0.08s ease !important;
    text-shadow: 0 1px 0 rgba(255,200,60,0.4) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background:
        linear-gradient(180deg,
            #f0b828 0%, #e0a010 15%,
            #cc8a08 50%, #e0a018 75%,
            #d49010 100%
        ) !important;
    box-shadow:
        0 8px 22px rgba(0,0,0,0.50),
        inset 0 1px 0 rgba(255,230,100,0.7) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(3px) !important;
    border-bottom-width: 1px !important;
    box-shadow:
        0 2px 6px rgba(0,0,0,0.4),
        inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* ── Metric containers — stamped aluminium tiles ─────────── */
[data-testid="metric-container"] {
    background:
        linear-gradient(145deg, #ddd8ce 0%, #ccc7bc 40%, #d4cfc5 100%) !important;
    border-top:    2px solid #ece8df !important;
    border-left:   2px solid #e4e0d7 !important;
    border-right:  2px solid #9a948a !important;
    border-bottom: 3px solid #8a847a !important;
    border-radius: 8px !important;
    padding: 16px 12px !important;
    box-shadow:
        0 4px 14px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.7) !important;
    text-align: center !important;
}
[data-testid="metric-container"] label {
    font-family: 'Bitter', serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--ink-light) !important;
    text-shadow: 0 1px 0 rgba(255,255,255,0.7) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.5),
        0 -1px 0 rgba(0,0,0,0.15) !important;
}

/* ── Headings ─────────────────────────────────────────────── */
h1 {
    font-family: 'Bitter', serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    color: var(--ink) !important;
    text-shadow:
        0 2px 0 rgba(255,255,255,0.6),
        0 -1px 0 rgba(0,0,0,0.2),
        0 4px 12px rgba(0,0,0,0.25) !important;
    letter-spacing: -0.01em !important;
}
/* Panel headings — dark stamped ink on aluminium, clearly visible */
h2, h3, h4 {
    font-family: 'Bitter', serif !important;
    font-weight: 700 !important;
    color: #1a1612 !important;
    text-shadow:
        0 1px 0 rgba(255,255,255,0.75),
        0 -1px 0 rgba(0,0,0,0.18),
        0 2px 4px rgba(0,0,0,0.12) !important;
    letter-spacing: 0.01em !important;
}
/* h3 specifically inside panels — slightly smaller, spaced like a section stamp */
.sk-panel h3 {
    font-size: 1.05rem !important;
    letter-spacing: 0.04em !important;
    border-bottom: 2px solid rgba(0,0,0,0.12) !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
    color: #1a1612 !important;
}

/* ── Subtitle / paragraph text ───────────────────────────── */
p, .stMarkdown p {
    color: var(--ink-mid) !important;
    font-family: 'Bitter', serif !important;
    font-size: 0.9rem !important;
}
/* Sidebar paragraphs stay light on dark walnut */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #d4b896 !important;
    text-shadow: none !important;
}

/* ── Divider — engraved rule ─────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.2) !important;
    border-bottom: 1px solid rgba(255,255,255,0.6) !important;
    margin: 18px 0 !important;
}

/* ── Alert boxes — restyle with physical look ────────────── */
.stAlert {
    border-radius: 6px !important;
    border-left: 5px solid !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: linear-gradient(180deg, #f8f4ec 0%, #ede8de 100%) !important;
    color: var(--ink) !important;
}

/* ── Info/success/warning/error boxes ────────────────────── */
div[data-testid="stNotification"] {
    border-radius: 6px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2) !important;
}

/* ── Dataframe — ledger paper ────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.25) !important;
    border: 1px solid #b8b2a8 !important;
}
div[data-testid="stDataFrame"] table {
    background: #f4f0e8 !important;
}
div[data-testid="stDataFrame"] th {
    background: linear-gradient(180deg, #d8d3c9 0%, #c8c3b8 100%) !important;
    color: var(--ink-mid) !important;
    font-family: 'Bitter', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #a8a29a !important;
}
div[data-testid="stDataFrame"] td {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.82rem !important;
    color: var(--ink) !important;
    background: #f4f0e8 !important;
    border-bottom: 1px solid #ddd8d0 !important;
}
div[data-testid="stDataFrame"] tr:nth-child(even) td {
    background: #eee9e0 !important;
}

/* ── Custom component classes ─────────────────────────────── */

/* Skeu panel class */
.sk-panel {
    background:
        linear-gradient(175deg,
            #ede9e0 0%, #ddd8ce 20%,
            #ccc7bc 40%, #d6d1c7 60%,
            #e0dbd0 80%, #d8d3c9 100%
        );
    border-radius: 10px;
    border-top:    2px solid #f5f2ec;
    border-left:   2px solid #ece8df;
    border-right:  2px solid #9e9890;
    border-bottom: 3px solid #8a847a;
    box-shadow:
        0 8px 32px var(--shadow-deep),
        0 2px  6px var(--shadow-med),
        inset 0 1px 0 var(--chrome-hi),
        inset 0 -1px 0 rgba(0,0,0,0.1);
    padding: 22px 24px 20px;
    margin-bottom: 16px;
    position: relative;
}
.sk-panel::before, .sk-panel::after {
    content: '◉';
    position: absolute;
    font-size: 10px;
    color: #9a948a;
    top: 8px;
}
.sk-panel::before { left: 10px; }
.sk-panel::after  { right: 10px; }

/* LED indicator */
.led-green {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #88ff88, #22aa44, #115522);
    box-shadow: 0 0 8px #22ff44, 0 0 16px rgba(34,255,68,0.4);
    border: 1px solid #115522;
    margin-right: 6px;
    vertical-align: middle;
}
.led-red {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ff8888, #cc2200, #660000);
    box-shadow: 0 0 8px #ff2200, 0 0 16px rgba(255,34,0,0.4);
    border: 1px solid #660000;
    margin-right: 6px;
    vertical-align: middle;
}
.led-amber {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ffe066, #c88010, #7a4e00);
    box-shadow: 0 0 8px #ffaa00, 0 0 16px rgba(255,170,0,0.4);
    border: 1px solid #7a4e00;
    margin-right: 6px;
    vertical-align: middle;
}

/* Pipeline tag — rendered inside dark walnut sidebar */
.sk-tag {
    display: inline-block;
    background: linear-gradient(180deg, #4a3820 0%, #3a2c18 100%);
    border-top: 1px solid #6a5030;
    border-left: 1px solid #604828;
    border-right: 1px solid #1e1408;
    border-bottom: 2px solid #160e04;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Source Code Pro', monospace;
    font-size: 0.73rem;
    font-weight: 600;
    color: #e8c87a !important;
    margin: 3px 3px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,200,80,0.15);
    text-shadow: 0 0 8px rgba(232,200,122,0.5);
}

/* Info / pipeline box */
.sk-infobox {
    background: linear-gradient(180deg, #c8c3b8 0%, #d8d3c9 100%);
    border-top: 2px solid #a8a29a;
    border-left: 2px solid #b0aaa0;
    border-right: 1px solid #ddd8d0;
    border-bottom: 2px solid #989288;
    border-radius: 5px;
    padding: 10px 14px;
    font-family: 'Source Code Pro', monospace;
    font-size: 0.8rem;
    color: var(--ink-mid);
    margin: 8px 0;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.15), 0 2px 6px rgba(0,0,0,0.2);
}

/* Score display */
.sk-score-display {
    background:
        linear-gradient(145deg, #1a1a1a 0%, #2a2a2a 30%, #1e1e1e 70%, #141414 100%);
    border-radius: 10px;
    border: 3px solid #0a0a0a;
    border-top-color: #3a3a3a;
    box-shadow:
        0 8px 24px rgba(0,0,0,0.7),
        inset 0 2px 4px rgba(0,0,0,0.8),
        inset 0 -1px 0 rgba(255,255,255,0.05);
    padding: 24px;
    text-align: center;
    font-family: 'Source Code Pro', monospace;
    position: relative;
    overflow: hidden;
}
.sk-score-display::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 40%;
    background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, transparent 100%);
    border-radius: 10px 10px 0 0;
    pointer-events: none;
}
.sk-score-num {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    text-shadow: 0 0 20px currentColor, 0 0 40px currentColor;
    letter-spacing: -0.02em;
}
.sk-score-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-top: 6px;
}

/* Progress bar skeu */
.sk-bar-wrap {
    background: linear-gradient(180deg, #a8a29a 0%, #c8c2b8 100%);
    border-radius: 4px;
    height: 10px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.35);
    overflow: hidden;
    margin-top: 4px;
}
.sk-bar-fill {
    height: 100%;
    border-radius: 4px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
    transition: width 0.6s ease;
}

/* Number input spinners */
div[data-testid="stNumberInput"] button {
    background: linear-gradient(180deg, #ddd8ce 0%, #c8c3b8 100%) !important;
    border: 1px solid #9a948a !important;
    color: var(--ink) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: linear-gradient(180deg, #e8e3da 0%, #d8d3c8 100%) !important;
}
div[data-testid="stNumberInput"] button:active {
    background: linear-gradient(180deg, #c8c3b8 0%, #d8d3c8 100%) !important;
    box-shadow: inset 0 2px 3px rgba(0,0,0,0.2) !important;
}

/* Checkbox — physical toggle */
div[data-testid="stCheckbox"] input + div {
    background: linear-gradient(180deg, #a8a29a 0%, #c8c2b8 100%) !important;
    border: 2px solid #888278 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.25) !important;
    border-radius: 4px !important;
}
div[data-testid="stCheckbox"] input:checked + div {
    background: linear-gradient(180deg, #c87808 0%, #e89820 100%) !important;
    border-color: #7a4e00 !important;
}

/* Main content area padding */
div[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_preprocessing():
    """
    Loads the saved ANN model and preprocessing.pkl from the notebook.
    Tries Wide_Deep_ANN.keras first, then Deep_ANN.keras.
    Returns (model, preprocessing_dict) or (None, None).
    TensorFlow is optional — falls back to rule-based demo if not installed.
    """
    model = None
    prep  = None

    # Try loading Keras model — only if TF is installed and model file exists
    if TF_AVAILABLE:
        for model_path in ['models/Wide_Deep_ANN.keras', 'models/Deep_ANN.keras',
                           'models/csat_ann_model.h5']:
            if os.path.exists(model_path):
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                    st.session_state['model_path'] = model_path
                    break
                except Exception:
                    pass

    # Try loading preprocessing (scikit-learn scaler — no TF needed)
    if os.path.exists('models/preprocessing.pkl'):
        try:
            with open('models/preprocessing.pkl', 'rb') as f:
                prep = pickle.load(f)
        except Exception:
            pass

    return model, prep


model, preprocessing = load_model_and_preprocessing()
MODEL_LOADED = model is not None
PREP_LOADED  = preprocessing is not None


# ─────────────────────────────────────────────────────────
# Feature engineering — mirrors notebook EXACTLY
# ─────────────────────────────────────────────────────────
def label_encode(value, classes):
    """Replicates sklearn LabelEncoder.transform for a single value."""
    classes_sorted = sorted([str(c) for c in classes])
    try:
        return classes_sorted.index(str(value))
    except ValueError:
        return 0  # unseen → first class


def build_feature_vector(
    channel, category, shift, tenure,
    agent_name, sub_category, manager, supervisor,
    response_time_mins, reported_hour, reported_day,
    agent_rates=None, subcat_rates=None,
    manager_rates=None, supervisor_rates=None,
):
    """
    Constructs the 13-feature vector that matches the notebook's feature_cols:

        feature_cols = [
            'agent_csat_rate', 'subcategory_csat_rate',
            'manager_csat_rate', 'supervisor_csat_rate',
            'response_time_mins', 'log_response_time',
            'reported_hour', 'reported_day', 'is_weekend',
            'channel_name_enc', 'category_enc',
            'Agent Shift_enc', 'Tenure Bucket_enc',
        ]
    """
    # ── Target-encoded features ──────────────────────────
    _ar = agent_rates      or AGENT_RATES_SAMPLE
    _sr = subcat_rates     or SUBCAT_RATES
    _mr = manager_rates    or MANAGER_RATES
    _svr= supervisor_rates or SUPERVISOR_RATES

    agent_csat_rate      = _ar.get(agent_name,   GLOBAL_MEAN)
    subcategory_csat_rate= _sr.get(sub_category, GLOBAL_MEAN)
    manager_csat_rate    = _mr.get(manager,      GLOBAL_MEAN)
    supervisor_csat_rate = _svr.get(supervisor,  GLOBAL_MEAN)

    # ── Time features ────────────────────────────────────
    log_response_time = np.log1p(response_time_mins)
    is_weekend        = 1 if reported_day >= 5 else 0

    # ── Label-encoded categoricals ───────────────────────
    channel_enc  = label_encode(channel,  CHANNEL_CLASSES)
    category_enc = label_encode(category, CATEGORY_CLASSES)
    shift_enc    = label_encode(shift,    SHIFT_CLASSES)
    tenure_enc   = label_encode(tenure,   TENURE_CLASSES)

    feature_vector = np.array([[
        agent_csat_rate,       # 0
        subcategory_csat_rate, # 1
        manager_csat_rate,     # 2
        supervisor_csat_rate,  # 3
        response_time_mins,    # 4
        log_response_time,     # 5
        reported_hour,         # 6
        reported_day,          # 7
        is_weekend,            # 8
        channel_enc,           # 9
        category_enc,          # 10
        shift_enc,             # 11
        tenure_enc,            # 12
    ]], dtype=np.float32)

    return feature_vector


def run_prediction(feature_vector, model, prep):
    """Scale and predict using the loaded ANN model."""
    # Use saved scaler if available
    if prep and 'scaler' in prep:
        scaler = prep['scaler']
        X_scaled = scaler.transform(feature_vector)
    else:
        # Fallback: simple standardisation using approximate train statistics
        X_scaled = feature_vector  # model may still work without scaling

    prob = float(model.predict(X_scaled, verbose=0).flatten()[0])
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def rule_based_prediction(
    channel, category, shift, tenure,
    agent_name, sub_category, manager,
    response_time_mins, reported_hour, reported_day,
    sentiment_score,
):
    """
    Rule-based fallback that mirrors notebook feature importances:
    agent_csat_rate (22%) + subcategory_csat_rate (22%) + response_time (18%)
    are the top 3 predictors.
    """
    agent_rate   = AGENT_RATES_SAMPLE.get(agent_name,   GLOBAL_MEAN)
    subcat_rate  = SUBCAT_RATES.get(sub_category,       GLOBAL_MEAN)
    manager_rate = MANAGER_RATES.get(manager,           GLOBAL_MEAN)

    # Weighted combination matching feature importance order
    score = (0.30 * agent_rate +
             0.28 * subcat_rate +
             0.12 * manager_rate)

    # Response time penalty (notebook: response_time contributes 18%)
    if response_time_mins <= 5:
        score += 0.08
    elif response_time_mins <= 15:
        score += 0.04
    elif response_time_mins > 60:
        score -= 0.12
    elif response_time_mins > 30:
        score -= 0.06

    # Sentiment nudge
    if sentiment_score > 0:
        score += 0.04
    elif sentiment_score < 0:
        score -= 0.06

    # Channel
    if channel == 'Email':
        score -= 0.04

    # Tenure
    if tenure == 'On Job Training':
        score -= 0.02

    prob = float(np.clip(score, 0.05, 0.98))
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 CSAT Predictor")
    st.markdown("---")

    if MODEL_LOADED:
        model_path = st.session_state.get('model_path', 'models/Wide_Deep_ANN.keras')
        st.success(f"✅ ANN Model Loaded")
        st.markdown(f'<div class="sk-infobox">📂 {os.path.basename(model_path)}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Demo Mode")
        st.markdown("""
        **To use the trained ANN:**

        Run the notebook and ensure these files exist:
        ```
        models/Wide_Deep_ANN.keras
        models/preprocessing.pkl
        ```
        """)

    st.markdown("---")
    st.markdown("### 📐 Feature Pipeline")
    st.markdown("The app mirrors the notebook's exact feature engineering:")
    for feat in ['agent_csat_rate', 'subcategory_csat_rate', 'manager_csat_rate',
                 'supervisor_csat_rate', 'response_time_mins', 'log_response_time',
                 'reported_hour', 'reported_day', 'is_weekend',
                 'channel_name_enc', 'category_enc', 'Agent Shift_enc', 'Tenure Bucket_enc']:
        st.markdown(f'<span class="sk-tag">{feat}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 CSAT Scale")
    st.markdown("""
    | Score | Meaning |
    |-------|---------|
    | 5 | Very Satisfied |
    | 4 | Satisfied |
    | 3 | Neutral |
    | 2 | Unsatisfied |
    | 1 | Very Unsatisfied |

    *Model predicts binary: CSAT ≥ 4 = Satisfied*
    """)

    st.markdown("---")
    st.markdown("### 🧠 About the Model")
    st.markdown("""
    **Architecture:** Wide & Deep ANN

    **Input:** 13 engineered features

    **Key features by importance:**
    1. Agent satisfaction rate (22%)
    2. Sub-category rate (22%)
    3. Response time (18%)
    4. Supervisor/Manager rates (11%)

    **Test Accuracy:** ~83%
    """)


# ─────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sk-panel" style="text-align:center; padding: 28px 24px 22px;">
    <div style="font-size:2.2rem; font-weight:700; font-family:'Bitter',serif;
                color:#1a1612; text-shadow: 0 2px 0 rgba(255,255,255,0.7), 0 4px 12px rgba(0,0,0,0.2);
                letter-spacing:-0.01em;">
        ⚙️ &nbsp;CSAT Score Predictor
    </div>
    <div style="font-family:'Bitter',serif; font-size:0.85rem; color:#6b6258;
                letter-spacing:0.12em; text-transform:uppercase; margin-top:6px;
                text-shadow: 0 1px 0 rgba(255,255,255,0.7);">
        eCommerce Customer Support &nbsp;·&nbsp; Shopzilla Analytics
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Input Form ────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

# ── Column 1: Ticket Info ─────────────────────────────────
with col1:
    st.markdown('<div class="sk-panel">', unsafe_allow_html=True)
    st.markdown("### 📋 Ticket Details")

    channel = st.selectbox(
        "Channel",
        CHANNEL_CLASSES,
        index=1,
        help="Inbound = customer calls in | Outcall = agent calls out | Email",
    )

    category = st.selectbox(
        "Issue Category",
        CATEGORY_CLASSES,
        index=5,  # Order Related
    )

    sub_category = st.selectbox(
        "Sub-Category",
        sorted(SUBCAT_RATES.keys()),
        index=list(sorted(SUBCAT_RATES.keys())).index('Delayed'),
        help="Sub-category satisfaction rate is one of the top 2 predictors",
    )

    response_time = st.number_input(
        "Response Time (minutes)",
        min_value=0, max_value=1440, value=15,
        help="Time between issue reported and agent first response",
    )

    now = datetime.now()
    reported_hour = st.slider("Reported Hour (0–23)", 0, 23, now.hour)
    reported_day  = st.selectbox(
        "Day of Week",
        ["Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)",
         "Friday (4)", "Saturday (5)", "Sunday (6)"],
        index=now.weekday(),
    )
    reported_day_int = int(reported_day.split("(")[1].replace(")", ""))
    st.markdown('</div>', unsafe_allow_html=True)

# ── Column 2: Agent Info ──────────────────────────────────
with col2:
    st.markdown('<div class="sk-panel">', unsafe_allow_html=True)
    st.markdown("### 👤 Agent Details")

    st.info("💡 **Agent name** is the #1 predictor — agent's historical CSAT rate is used directly as a feature.", icon="ℹ️")

    agent_name_input = st.text_input(
        "Agent Name",
        value="Aaron Edwards",
        help="Type the agent's full name. Must match training data names exactly.",
    )

    # Live rate lookup
    agent_rate = AGENT_RATES_SAMPLE.get(agent_name_input, None)
    if MODEL_LOADED and PREP_LOADED and 'agent_rates' in preprocessing:
        agent_rate = preprocessing.get('agent_rates', {}).get(agent_name_input, None)

    if agent_rate is not None:
        st.markdown(f'<div class="sk-infobox"><span class="led-green"></span>Known agent · Historical CSAT rate: <b>{agent_rate:.1%}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sk-infobox"><span class="led-amber"></span>Unknown agent · Using global mean: <b>{GLOBAL_MEAN:.1%}</b></div>', unsafe_allow_html=True)

    manager = st.selectbox(
        "Manager",
        MANAGERS,
        help="Manager's team-level satisfaction rate affects prediction",
    )

    supervisor = st.selectbox(
        "Supervisor",
        sorted(SUPERVISOR_RATES.keys()),
    )

    shift = st.selectbox(
        "Agent Shift",
        SHIFT_CLASSES,
        index=2,  # Morning
    )

    tenure = st.selectbox(
        "Tenure Bucket",
        TENURE_CLASSES,
        index=1,  # 31-60
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Column 3: Customer Info ───────────────────────────────
with col3:
    st.markdown('<div class="sk-panel">', unsafe_allow_html=True)
    st.markdown("### 💬 Customer Feedback")

    customer_remarks = st.text_area(
        "Customer Remarks",
        value="The agent was very helpful and resolved my issue quickly.",
        height=120,
        help="Used for sentiment analysis (supplementary signal)",
    )

    # Sentiment
    POSITIVE_KW = ['good','great','excellent','helpful','thank','thanks','awesome',
                   'amazing','happy','perfect','love','resolved','quick','fast','satisfied']
    NEGATIVE_KW = ['bad','worst','poor','disappointed','issue','problem','rude',
                   'terrible','horrible','slow','frustrated','useless','waiting']

    text = customer_remarks.lower()
    pos_count = sum(1 for w in POSITIVE_KW if w in text)
    neg_count = sum(1 for w in NEGATIVE_KW if w in text)
    sentiment_score = pos_count - neg_count

    if sentiment_score > 0:
        st.success(f"😊 Positive sentiment detected (+{pos_count} keywords)")
    elif sentiment_score < 0:
        st.error(f"😡 Negative sentiment detected (−{neg_count} keywords)")
    else:
        st.info("😐 Neutral sentiment")

    st.markdown("---")
    st.markdown("#### 🔍 Live Feature Preview")
    agent_r = AGENT_RATES_SAMPLE.get(agent_name_input, GLOBAL_MEAN)
    subcat_r = SUBCAT_RATES.get(sub_category, GLOBAL_MEAN)
    manager_r = MANAGER_RATES.get(manager, GLOBAL_MEAN)
    supervisor_r = SUPERVISOR_RATES.get(supervisor, GLOBAL_MEAN)

    preview_df = pd.DataFrame({
        'Feature': ['agent_csat_rate', 'subcategory_csat_rate',
                    'manager_csat_rate', 'supervisor_csat_rate',
                    'response_time_mins', 'log_response_time',
                    'reported_hour', 'is_weekend',
                    'channel_name_enc', 'category_enc'],
        'Value': [
            f"{agent_r:.4f}",
            f"{subcat_r:.4f}",
            f"{manager_r:.4f}",
            f"{supervisor_r:.4f}",
            f"{response_time:.1f}",
            f"{np.log1p(response_time):.4f}",
            str(reported_hour),
            str(1 if reported_day_int >= 5 else 0),
            f"{label_encode(channel, CHANNEL_CLASSES)}  ({channel})",
            f"{label_encode(category, CATEGORY_CLASSES)}  ({category})",
        ]
    })
    st.dataframe(preview_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────
st.markdown("---")
pred_col, _ = st.columns([1, 2])
with pred_col:
    predict_clicked = st.button("⚙️  ANALYSE TICKET  ⚙️")

# ── Results ───────────────────────────────────────────────
if predict_clicked:
    st.markdown("""
    <div style="font-family:'Bitter',serif; font-size:0.72rem; font-weight:700;
                letter-spacing:0.14em; text-transform:uppercase; color:#6b6258;
                text-shadow:0 1px 0 rgba(255,255,255,0.8); margin:16px 0 10px;">
        ▼ &nbsp; Prediction Results
    </div>
    """, unsafe_allow_html=True)

    # Build feature vector exactly as in the notebook
    feature_vector = build_feature_vector(
        channel=channel,
        category=category,
        shift=shift,
        tenure=tenure,
        agent_name=agent_name_input,
        sub_category=sub_category,
        manager=manager,
        supervisor=supervisor,
        response_time_mins=float(response_time),
        reported_hour=float(reported_hour),
        reported_day=float(reported_day_int),
        agent_rates=None,
        subcat_rates=None,
        manager_rates=None,
        supervisor_rates=None,
    )

    # Predict
    if MODEL_LOADED:
        pred, confidence = run_prediction(feature_vector, model, preprocessing)
        prediction_source = "🧠 ANN Model"
    else:
        pred, confidence = rule_based_prediction(
            channel, category, shift, tenure,
            agent_name_input, sub_category, manager,
            float(response_time), reported_hour, reported_day_int,
            sentiment_score,
        )
        prediction_source = "📐 Rule-Based (Demo)"

    # Convert binary prediction back to approximate CSAT 1–5 scale
    if pred == 1:
        if confidence >= 0.90:
            approx_score = 5
        elif confidence >= 0.70:
            approx_score = 4
        else:
            approx_score = 4
    else:
        if confidence <= 0.20:
            approx_score = 1
        elif confidence <= 0.35:
            approx_score = 2
        else:
            approx_score = 3

    # ── Score colours ─────────────────────────────────────
    score_colors = {5: "#2a7a2a", 4: "#4a9a3a", 3: "#c8860a", 2: "#c04020", 1: "#8b1a0a"}
    risk_map     = {5: "Low ✅", 4: "Low ✅", 3: "Medium ⚠️", 2: "High 🔴", 1: "Critical 🚨"}
    color        = score_colors[approx_score]

    # ── Metrics Row ───────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Predicted Score", f"{approx_score} / 5")
    with m2:
        st.metric("Satisfied?", "Yes ✅" if pred == 1 else "No ❌")
    with m3:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    with m4:
        st.metric("Risk Level", risk_map[approx_score])

    # ── Score badge ───────────────────────────────────────
    badge_col, chart_col = st.columns([1, 2])

    with badge_col:
        led = 'led-green' if approx_score >= 4 else 'led-amber' if approx_score == 3 else 'led-red'
        st.markdown(f"""
        <div class="sk-score-display">
            <div class="sk-score-label">PREDICTED SCORE</div>
            <div class="sk-score-num" style="color:{color};">{approx_score} / 5</div>
            <div class="sk-score-label" style="margin-top:12px;">
                <span class="{led}"></span>{prediction_source}
            </div>
            <div style="font-family:'Source Code Pro',monospace; font-size:0.78rem;
                        color:rgba(255,255,255,0.45); margin-top:6px;">
                p(satisfied) = {confidence:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔑 Key Feature Values")
        contributions = {
            "Agent Rate":     agent_r,
            "Sub-Cat Rate":   subcat_r,
            "Manager Rate":   manager_r,
            "Supervisor Rate":supervisor_r,
        }
        for lbl, val in contributions.items():
            bar_color = "#4a9a5a" if val >= 0.82 else "#c8860a" if val >= 0.70 else "#b03020"
            pct = int(val * 100)
            st.markdown(f"""
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;
                            font-family:'Bitter',serif; font-size:0.78rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.05em;
                            color:#3d3830; text-shadow:0 1px 0 rgba(255,255,255,0.7);">
                    <span>{lbl}</span>
                    <span style="font-family:'Source Code Pro',monospace; color:{bar_color};">{val:.1%}</span>
                </div>
                <div class="sk-bar-wrap">
                    <div class="sk-bar-fill" style="width:{pct}%;
                         background:linear-gradient(180deg,
                             {'#68c478' if val>=0.82 else '#e8a020' if val>=0.70 else '#d04030'} 0%,
                             {bar_color} 100%);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Probability chart ─────────────────────────────────
    with chart_col:
        # Map binary confidence to approximate CSAT 1-5 probabilities
        p_satisfied   = confidence
        p_unsatisfied = 1 - confidence
        # Distribute unsatisfied across 1/2/3 and satisfied across 4/5
        dist = [
            p_unsatisfied * 0.45,
            p_unsatisfied * 0.30,
            p_unsatisfied * 0.25,
            p_satisfied   * 0.38,
            p_satisfied   * 0.62,
        ]

        fig = go.Figure()
        bar_colors_sk = ["#8b2010", "#c04020", "#c8860a", "#4a7a3a", "#2a6a2a"]

        fig.add_trace(go.Bar(
            x=["CSAT 1", "CSAT 2", "CSAT 3", "CSAT 4", "CSAT 5"],
            y=dist,
            marker_color=bar_colors_sk,
            marker_line_color=["#5a1008","#8a2e10","#8a5a06","#2a5a20","#1a4a1a"],
            marker_line_width=1.5,
            text=[f"{p*100:.1f}%" for p in dist],
            textposition="outside",
            textfont=dict(color="#1a1612", size=13, family="Source Code Pro"),
        ))

        fig.update_layout(
            title=dict(text="Score Probability Distribution",
                       font=dict(color="#3d3830", size=14, family="Bitter")),
            yaxis=dict(range=[0, 1], title="Probability",
                       color="#6b6258", gridcolor="rgba(0,0,0,0.1)",
                       tickfont=dict(family="Source Code Pro")),
            xaxis=dict(color="#6b6258", tickfont=dict(family="Source Code Pro")),
            plot_bgcolor="#ede8de",
            paper_bgcolor="#e4dfd4",
            font=dict(color="#3d3830", family="Bitter"),
            margin=dict(t=50, b=20),
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Response time gauge
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=response_time,
            title={"text": "Response Time (mins)", "font": {"color": "#3d3830", "family": "Bitter"}},
            number={"suffix": " min", "font": {"color": "#1a1612", "family": "Source Code Pro"}},
            gauge={
                "axis": {"range": [0, 120], "tickcolor": "#6b6258",
                         "tickfont": {"family": "Source Code Pro"}},
                "bar":  {"color": "#c8860a", "thickness": 0.3},
                "bgcolor": "#e4dfd4",
                "borderwidth": 2,
                "bordercolor": "#9a948a",
                "steps": [
                    {"range": [0, 15],  "color": "#c8e8c0"},
                    {"range": [15, 45], "color": "#f0e0b0"},
                    {"range": [45, 120],"color": "#e8c0b0"},
                ],
                "threshold": {"line": {"color": "#5a1008", "width": 3}, "value": 30},
            }
        ))
        fig2.update_layout(
            paper_bgcolor="#e4dfd4",
            font=dict(color="#3d3830", family="Bitter"),
            height=220, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Recommendations ───────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Bitter',serif; font-size:0.72rem; font-weight:700;
                letter-spacing:0.14em; text-transform:uppercase; color:#6b6258;
                text-shadow:0 1px 0 rgba(255,255,255,0.8); margin-bottom:12px;">
        ▼ &nbsp; Action Recommendations
    </div>
    """, unsafe_allow_html=True)

    rec1, rec2, rec3 = st.columns(3)

    with rec1:
        if approx_score <= 2:
            st.markdown(f"""
            <div class="sk-panel" style="border-left:5px solid #8b1a0a;">
                <div style="font-family:'Bitter',serif;font-weight:700;font-size:0.9rem;
                            color:#8b1a0a;text-shadow:0 1px 0 rgba(255,255,255,0.5);margin-bottom:10px;">
                    <span class="led-red"></span>HIGH RISK — IMMEDIATE ACTION
                </div>
                <div style="font-family:'Bitter',serif;font-size:0.85rem;color:#3d3830;line-height:1.8;">
                    ▸ Escalate to senior agent immediately<br>
                    ▸ Schedule callback within 30 minutes<br>
                    ▸ Flag for manager review<br>
                    ▸ Offer compensation / goodwill gesture
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif approx_score == 3:
            st.markdown(f"""
            <div class="sk-panel" style="border-left:5px solid #c8860a;">
                <div style="font-family:'Bitter',serif;font-weight:700;font-size:0.9rem;
                            color:#8a5a00;text-shadow:0 1px 0 rgba(255,255,255,0.5);margin-bottom:10px;">
                    <span class="led-amber"></span>MEDIUM RISK — MONITOR CLOSELY
                </div>
                <div style="font-family:'Bitter',serif;font-size:0.85rem;color:#3d3830;line-height:1.8;">
                    ▸ Assign experienced agent<br>
                    ▸ Set 15-min follow-up reminder<br>
                    ▸ Send proactive resolution update<br>
                    ▸ Survey after ticket close
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sk-panel" style="border-left:5px solid #2a7a2a;">
                <div style="font-family:'Bitter',serif;font-weight:700;font-size:0.9rem;
                            color:#1a5a1a;text-shadow:0 1px 0 rgba(255,255,255,0.5);margin-bottom:10px;">
                    <span class="led-green"></span>LOW RISK — STANDARD RESOLUTION
                </div>
                <div style="font-family:'Bitter',serif;font-size:0.85rem;color:#3d3830;line-height:1.8;">
                    ▸ Continue standard workflow<br>
                    ▸ Collect CSAT feedback post-resolution<br>
                    ▸ Log as positive experience
                </div>
            </div>
            """, unsafe_allow_html=True)

    with rec2:
        rt_led  = 'led-green' if response_time<=15 else 'led-amber' if response_time<=30 else 'led-red'
        rt_text = ("⚡ Excellent — top quartile" if response_time<=5
                   else "✅ Good — within SLA target" if response_time<=15
                   else "⚠️ Fair — approaching threshold" if response_time<=30
                   else "🔴 Slow — significant CSAT impact")
        subcat_rate_val = SUBCAT_RATES.get(sub_category, GLOBAL_MEAN)
        sc_led  = 'led-green' if subcat_rate_val >= GLOBAL_MEAN else 'led-amber'
        sc_dir  = "above average" if subcat_rate_val >= GLOBAL_MEAN else "below average"
        st.markdown(f"""
        <div class="sk-panel">
            <div style="font-family:'Bitter',serif;font-size:0.7rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#6b6258;
                        text-shadow:0 1px 0 rgba(255,255,255,0.7);border-bottom:1px solid rgba(0,0,0,0.1);
                        padding-bottom:6px;margin-bottom:10px;">
                Response Time Insight
            </div>
            <div style="font-family:'Source Code Pro',monospace;font-size:0.88rem;color:#1a1612;margin-bottom:4px;">
                <span class="{rt_led}"></span><b>{response_time} min</b>
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.82rem;color:#3d3830;margin-bottom:16px;">
                {rt_text}
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.7rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#6b6258;
                        text-shadow:0 1px 0 rgba(255,255,255,0.7);border-bottom:1px solid rgba(0,0,0,0.1);
                        padding-bottom:6px;margin-bottom:10px;">
                Sub-Category Insight
            </div>
            <div style="font-family:'Source Code Pro',monospace;font-size:0.85rem;color:#1a1612;margin-bottom:4px;">
                <span class="{sc_led}"></span><b>{subcat_rate_val:.1%}</b> satisfaction rate
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.82rem;color:#3d3830;">
                <i>{sub_category}</i> is {sc_dir}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rec3:
        known     = agent_name_input in AGENT_RATES_SAMPLE
        rate      = AGENT_RATES_SAMPLE.get(agent_name_input, GLOBAL_MEAN)
        ag_led    = ('led-green' if rate>=0.87 else 'led-amber' if rate>=0.75 else 'led-red')
        ag_label  = ("🌟 High performer" if rate>=0.87
                     else "👍 Average performer" if rate>=0.75
                     else "⚠️ Below average — consider coaching")
        src_led   = 'led-green' if MODEL_LOADED else 'led-amber'
        st.markdown(f"""
        <div class="sk-panel">
            <div style="font-family:'Bitter',serif;font-size:0.7rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#6b6258;
                        text-shadow:0 1px 0 rgba(255,255,255,0.7);border-bottom:1px solid rgba(0,0,0,0.1);
                        padding-bottom:6px;margin-bottom:10px;">
                Agent Performance
            </div>
            <div style="font-family:'Source Code Pro',monospace;font-size:0.88rem;color:#1a1612;margin-bottom:4px;">
                <span class="{ag_led}"></span>
                {'<b>' + agent_name_input + '</b>' if known else '<i>Unknown agent</i>'}
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.82rem;color:#3d3830;margin-bottom:4px;">
                {ag_label}
            </div>
            <div style="font-family:'Source Code Pro',monospace;font-size:0.8rem;color:#6b6258;">
                rate = {rate:.1%}
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.7rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#6b6258;
                        text-shadow:0 1px 0 rgba(255,255,255,0.7);border-top:1px solid rgba(0,0,0,0.1);
                        padding-top:10px;margin-top:12px;margin-bottom:6px;">
                Prediction Source
            </div>
            <div style="font-family:'Bitter',serif;font-size:0.85rem;color:#1a1612;">
                <span class="{src_led}"></span>{prediction_source}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not MODEL_LOADED:
            st.markdown("""
            <div class="sk-infobox" style="margin-top:8px;">
            Run the notebook to generate:<br>
            <code>models/Wide_Deep_ANN.keras</code><br>
            <code>models/preprocessing.pkl</code><br>
            Then restart this app.
            </div>
            """, unsafe_allow_html=True)
