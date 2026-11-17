import os
import glob
import base64
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


try:
    from src.data_loader import load_dataset
    from src.preprocess import prepare_data
    from src.train_model import train_models, save_best_model
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance
    from src.detect_threat import generate_threat_alerts
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)


st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Paths & Constants
# -----------------------------------------------------------------------------
BG_IMAGE_PATH = os.path.join("assets", "C:\\Users\\shiva\\Downloads\\images (1).jpeg")  # Place your background image here
DEFAULT_DATA_PATH = os.path.join("data", "raw", "nsl_kdd.csv")
MODELS_DIR = "models"
GRAPHS_DIR = os.path.join("outputs", "graphs")
PREDICTIONS_DIR = os.path.join("outputs", "predictions")

ATTACK_CATEGORY_MAP = {
    "normal": "Normal",
    "neptune": "DoS", "smurf": "DoS", "back": "DoS", "teardrop": "DoS",
    "pod": "DoS", "land": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "portsweep": "Probe", "nmap": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "warezclient": "R2L", "guess_passwd": "R2L", "warezmaster": "R2L",
    "imap": "R2L", "ftp_write": "R2L", "multihop": "R2L", "phf": "R2L",
    "spy": "R2L", "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "rootkit": "U2R", "loadmodule": "U2R",
    "perl": "U2R", "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}



def inject_glassmorphism_theme(bg_path: str):
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(bg_path)[1].lstrip(".") or "png"
        bg_layer = f'linear-gradient(135deg, rgba(6, 10, 20, 0.80), rgba(11, 18, 34, 0.86)), url("data:image/{ext};base64,{encoded}")'
    else:
        # Fallback hyper-dark cyber grid radial background if no local image is present
        bg_layer = """
        radial-gradient(circle at 15% 20%, rgba(0, 240, 255, 0.14) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(255, 42, 109, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(16, 28, 52, 1) 0%, rgba(6, 9, 18, 1) 100%)
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

        /* Main Viewport Background */
        [data-testid="stAppViewContainer"] {{
            background: {bg_layer};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
            color: #e6edf7;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        /* Glassmorphic Sidebar */
        [data-testid="stSidebar"] {{
            background: rgba(11, 17, 30, 0.58) !important;
            backdrop-filter: blur(22px) saturate(170%);
            -webkit-backdrop-filter: blur(22px) saturate(170%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 8px 0 32px rgba(0, 0, 0, 0.45);
        }}

        [data-testid="stSidebar"] * {{
            color: #e2e8f0;
        }}

        /* Sidebar Navigation Radio Pills */
        [data-testid="stSidebar"] .stRadio > div {{
            gap: 8px;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 14px;
            transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1);
            cursor: pointer;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(0, 240, 255, 0.12);
            border-color: rgba(0, 240, 255, 0.45);
            transform: translateX(4px);
            box-shadow: 0 4px 18px rgba(0, 240, 255, 0.15);
        }}

        /* Glassmorphic Primary & Default Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, rgba(0, 210, 255, 0.22), rgba(58, 123, 213, 0.25)) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 240, 255, 0.45) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px;
            transition: all 0.28s ease !important;
            box-shadow: 0 4px 20px rgba(0, 210, 255, 0.18), inset 0 1px 0 rgba(255,255,255,0.25);
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            border-color: #00f0ff !important;
            box-shadow: 0 8px 28px rgba(0, 240, 255, 0.38), inset 0 1px 0 rgba(255,255,255,0.4);
        }}

        /* Hyper-Realistic Glass Metric Cards */
        .glass-kpi-card {{
            position: relative;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            backdrop-filter: blur(18px) saturate(160%);
            -webkit-backdrop-filter: blur(18px) saturate(160%);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-top: 1px solid rgba(255, 255, 255, 0.28);
            border-left: 1px solid rgba(255, 255, 255, 0.22);
            padding: 20px 22px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.38), inset 0 0 24px rgba(255, 255, 255, 0.02);
            transition: all 0.32s cubic-bezier(0.22, 1, 0.36, 1);
            overflow: hidden;
        }}

        .glass-kpi-card::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, var(--glow-color, rgba(0,240,255,0.16)) 0%, transparent 60%);
            opacity: 0;
            transition: opacity 0.35s ease;
            pointer-events: none;
        }}

        .glass-kpi-card:hover {{
            transform: translateY(-5px) scale(1.015);
            border-color: var(--accent-color, rgba(0, 240, 255, 0.6));
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.5), 0 0 24px var(--glow-color, rgba(0, 240, 255, 0.25));
        }}

        .glass-kpi-card:hover::before {{
            opacity: 1;
        }}

        .kpi-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }}

        .kpi-title {{
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
        }}

        .kpi-icon-orb {{
            width: 44px;
            height: 44px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 6px 16px rgba(0,0,0,0.35), inset 0 1px 1px rgba(255,255,255,0.4);
            transition: transform 0.35s ease, box-shadow 0.35s ease;
        }}

        .glass-kpi-card:hover .kpi-icon-orb {{
            transform: rotate(-6deg) scale(1.1);
            box-shadow: 0 0 20px var(--glow-color, rgba(0,240,255,0.45));
        }}

        .kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.95rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.15;
            text-shadow: 0 2px 12px rgba(0,0,0,0.4);
        }}

        .kpi-badge {{
            display: inline-block;
            margin-top: 10px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: var(--badge-bg, rgba(0, 240, 255, 0.14));
            color: var(--accent-color, #38bdf8);
            border: 1px solid var(--badge-border, rgba(0, 240, 255, 0.3));
        }}

        /* Glass Containers for Charts, Tables, and Expanders */
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"],
        div[data-testid="stForm"] {{
            background: rgba(15, 23, 42, 0.48) !important;
            backdrop-filter: blur(16px) saturate(150%);
            -webkit-backdrop-filter: blur(16px) saturate(150%);
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 16px !important;
            padding: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.32);
        }}

        /* Sidebar Brand Header Card */
        .sidebar-brand-card {{
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.14), rgba(124, 58, 237, 0.12));
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25);
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .status-pill {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 8px 0 14px 0;
            backdrop-filter: blur(10px);
        }}

        .status-online {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #6ee7b7;
        }}

        .status-offline {{
            background: rgba(244, 63, 94, 0.16);
            border: 1px solid rgba(244, 63, 94, 0.45);
            color: #fda4af;
        }}

        .pulse-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 10px currentColor;
            animation: pulseGlow 1.8s infinite ease-in-out;
        }}

        @keyframes pulseGlow {{
            0%, 100% {{ transform: scale(0.95); opacity: 0.75; }}
            50% {{ transform: scale(1.25); opacity: 1; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Hyper-Realistic 3D-Shaded SVG Icons & KPI Renderer
# -----------------------------------------------------------------------------
SVG_ICONS = {
    "radar": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="gCyan" x1="0" y1="0" x2="24" y2="24">
              <stop offset="0%" stop-color="#38bdf8"/>
              <stop offset="100%" stop-color="#0284c7"/>
            </linearGradient>
          </defs>
          <circle cx="12" cy="12" r="9.5" stroke="url(#gCyan)" stroke-width="1.8" stroke-dasharray="3 2"/>
          <circle cx="12" cy="12" r="5.5" stroke="#7dd3fc" stroke-width="1.5"/>
          <path d="M12 12L18.5 5.5" stroke="#e0f2fe" stroke-width="2" stroke-linecap="round"/>
          <circle cx="12" cy="12" r="2.2" fill="#38bdf8"/>
        </svg>
    """,
    "shield_ok": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="gEmerald" x1="0" y1="0" x2="24" y2="24">
              <stop offset="0%" stop-color="#34d399"/>
              <stop offset="100%" stop-color="#059669"/>
            </linearGradient>
          </defs>
          <path d="M12 2.5L4.5 5.8V11.5C4.5 16.55 7.7 21.2 12 22.5C16.3 21.2 19.5 16.55 19.5 11.5V5.8L12 2.5Z" fill="url(#gEmerald)" fill-opacity="0.25" stroke="url(#gEmerald)" stroke-width="1.8"/>
          <path d="M8.8 12.2L11 14.4L15.5 9.8" stroke="#a7f3d0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    """,
    "threat": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="gCrimson" x1="0" y1="0" x2="24" y2="24">
              <stop offset="0%" stop-color="#fb7185"/>
              <stop offset="100%" stop-color="#e11d48"/>
            </linearGradient>
          </defs>
          <path d="M12 3L2.5 19.5H21.5L12 3Z" fill="url(#gCrimson)" fill-opacity="0.25" stroke="url(#gCrimson)" stroke-width="1.8" stroke-linejoin="round"/>
          <path d="M12 9.5V14" stroke="#fecdd3" stroke-width="2.2" stroke-linecap="round"/>
          <circle cx="12" cy="17" r="1.2" fill="#fecdd3"/>
        </svg>
    """,
    "fingerprint": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="gViolet" x1="0" y1="0" x2="24" y2="24">
              <stop offset="0%" stop-color="#c084fc"/>
              <stop offset="100%" stop-color="#7c3aed"/>
            </linearGradient>
          </defs>
          <path d="M7 10.5C7 7.7 9.2 5.5 12 5.5C14.8 5.5 17 7.7 17 10.5V14.5" stroke="url(#gViolet)" stroke-width="1.8" stroke-linecap="round"/>
          <path d="M9.5 11.5C9.5 10.1 10.6 9 12 9C13.4 9 14.5 10.1 14.5 11.5V17.5" stroke="#e9d5ff" stroke-width="1.8" stroke-linecap="round"/>
          <path d="M12 12V19.5" stroke="#c084fc" stroke-width="2" stroke-linecap="round"/>
          <path d="M4.8 13.5C4.8 15.8 5.6 17.8 6.8 19.2" stroke="url(#gViolet)" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
    """,
    "cyber_shield_logo": """
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="logoGrad" x1="2" y1="2" x2="22" y2="22">
              <stop offset="0%" stop-color="#00f0ff"/>
              <stop offset="100%" stop-color="#8b5cf6"/>
            </linearGradient>
          </defs>
          <path d="M12 2L4 5.5V11.8C4 16.9 7.4 21.5 12 22.8C16.6 21.5 20 16.9 20 11.8V5.5L12 2Z" fill="url(#logoGrad)" fill-opacity="0.28" stroke="url(#logoGrad)" stroke-width="1.8"/>
          <circle cx="12" cy="11.5" r="3" stroke="#e0f2fe" stroke-width="1.7"/>
          <path d="M12 14.5V17.5" stroke="#00f0ff" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
    """,
}


def render_glass_metric(col, title: str, value: str, badge: str, icon_key: str, theme: str = "cyan"):
    themes = {
        "cyan": ("#38bdf8", "rgba(56, 189, 248, 0.25)", "rgba(56, 189, 248, 0.14)", "rgba(56, 189, 248, 0.35)"),
        "emerald": ("#34d399", "rgba(52, 211, 153, 0.25)", "rgba(52, 211, 153, 0.14)", "rgba(52, 211, 153, 0.35)"),
        "crimson": ("#fb7185", "rgba(251, 113, 133, 0.28)", "rgba(251, 113, 133, 0.16)", "rgba(251, 113, 133, 0.38)"),
        "violet": ("#c084fc", "rgba(192, 132, 252, 0.25)", "rgba(192, 132, 252, 0.14)", "rgba(192, 132, 252, 0.35)"),
    }
    accent, glow, badge_bg, badge_border = themes.get(theme, themes["cyan"])
    svg = SVG_ICONS.get(icon_key, SVG_ICONS["radar"])

    col.markdown(
        f"""
        <div class="glass-kpi-card" style="--accent-color:{accent}; --glow-color:{glow}; --badge-bg:{badge_bg}; --badge-border:{badge_border};">
            <div class="kpi-header">
                <span class="kpi-title">{title}</span>
                <div class="kpi-icon-orb">{svg}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-badge">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_glass_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        margin=dict(l=16, r=16, t=28, b=16),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.45)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# Activate Glassmorphism CSS
inject_glassmorphism_theme(BG_IMAGE_PATH)


# -----------------------------------------------------------------------------
# Cached Data & Artifact Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading NSL-KDD telemetry...")
def get_dataset(path: str) -> pd.DataFrame:
    if PIPELINE_AVAILABLE:
        return load_dataset(path)
    return pd.read_csv(path)


@st.cache_resource
def load_saved_artifacts(models_dir: str):
    if not os.path.exists(models_dir):
        return None, None, "Models directory not found."

    model_obj = None
    preprocessor_obj = None

    for f in os.listdir(models_dir):
        full_path = os.path.join(models_dir, f)
        if f.endswith((".pkl", ".joblib")):
            loaded = joblib.load(full_path)
            if isinstance(loaded, dict):
                model_obj = loaded.get("model", model_obj)
                preprocessor_obj = loaded.get("preprocessor", preprocessor_obj)
            elif "preprocess" in f.lower():
                preprocessor_obj = loaded
            else:
                model_obj = loaded

    if model_obj is None:
        return None, None, "No trained model artifact found in models/."
    return model_obj, preprocessor_obj, None


def run_full_pipeline(data_path: str):
    with st.status("Executing Threat Detection Pipeline...", expanded=True) as status:
        st.write("1. Loading NSL-KDD dataset...")
        df = load_dataset(data_path)

        st.write("2. Preprocessing network features...")
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

        st.write("3. Training ML classifiers...")
        models, results = train_models(X_train, X_test, y_train, y_test)
        best_model_name = max(results, key=lambda name: results[name]["f1_score"])
        best_model = models[best_model_name]

        st.write(f"4. Saving optimal model ({best_model_name})...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_best_model(best_model, preprocessor, best_model_name, MODELS_DIR)

        st.write("5. Generating confusion matrix & feature importance plots...")
        y_pred = best_model.predict(X_test)
        evaluate_model(y_test, y_pred, best_model_name)
        plot_confusion_matrix(y_test, y_pred, best_model_name, save_dir=GRAPHS_DIR)

        if hasattr(best_model, "feature_importances_"):
            plot_feature_importance(
                best_model=best_model,
                feature_names=feature_names,
                model_name=best_model_name,
                save_dir=GRAPHS_DIR,
                top_n=15,
            )

        st.write("6. Exporting threat alerts...")
        generate_threat_alerts(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            save_dir=PREDICTIONS_DIR,
        )

        st.session_state["training_results"] = results
        st.session_state["best_model_name"] = best_model_name
        status.update(label="Pipeline execution completed!", state="complete", expanded=False)

    st.cache_resource.clear()


# -----------------------------------------------------------------------------
# Glassmorphic Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand-card">
            <div>{SVG_ICONS["cyber_shield_logo"]}</div>
            <div>
                <div style="font-weight:700; font-size:1.05rem; letter-spacing:0.4px; color:#fff;">AEGIS IDS</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Neural Threat Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_path = st.text_input("Telemetry Dataset Path", value=DEFAULT_DATA_PATH)
    dataset_exists = os.path.exists(data_path)

    if dataset_exists:
        st.markdown(
            '<div class="status-pill status-online"><span class="pulse-dot"></span> NSL-KDD Telemetry Online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-offline"><span class="pulse-dot"></span> Dataset Path Offline</div>',
            unsafe_allow_html=True,
        )

    if PIPELINE_AVAILABLE:
        if st.button("⚡ Train & Evaluate Pipeline", use_container_width=True, disabled=not dataset_exists):
            run_full_pipeline(data_path)
    else:
        st.warning(f"Standalone mode (`src` import warning: {IMPORT_ERROR_MSG})")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    page = st.radio(
        "COMMAND MODULES",
        [
            "◈  Traffic & Dataset Explorer",
            "◈  Model Performance & Graphs",
            "◈  Threat Alerts Log",
            "◈  Live Packet Inspector",
        ],
    )


# -----------------------------------------------------------------------------
# View 1: Traffic & Dataset Explorer
# -----------------------------------------------------------------------------
if page == "◈  Traffic & Dataset Explorer":
    st.markdown("## 🛡️ Network Telemetry & NSL-KDD Overview")

    if not dataset_exists:
        st.error(f"Could not locate dataset at `{data_path}`. Update the path in the sidebar.")
        st.stop()

    df = get_dataset(data_path)
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP).fillna("Other")

    total_records = len(df)
    normal_count = int((df["label"] == "normal").sum())
    attack_count = total_records - normal_count
    attack_ratio = (attack_count / total_records) * 100
    unique_sigs = df["label"].nunique() - 1

    # Hyper-Realistic Glass KPI Row
    c1, c2, c3, c4 = st.columns(4)
    render_glass_metric(c1, "Total Network Flows", f"{total_records:,}", "43 Telemetry Features", "radar", "cyan")
    render_glass_metric(c2, "Verified Normal", f"{normal_count:,}", f"{100 - attack_ratio:.1f}% Baseline Traffic", "shield_ok", "emerald")
    render_glass_metric(c3, "Malicious Intrusions", f"{attack_count:,}", f"{attack_ratio:.1f}% Threat Density", "threat", "crimson")
    render_glass_metric(c4, "Attack Signatures", f"{unique_sigs}", "DoS • Probe • R2L • U2R", "fingerprint", "violet")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Attack Taxonomy Distribution")
        cat_counts = df["attack_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.52,
            color_discrete_sequence=["#34d399", "#fb7185", "#38bdf8", "#c084fc", "#fbbf24"],
        )
        st.plotly_chart(style_glass_plotly(fig_cat), use_container_width=True)

    with col_right:
        st.subheader("Top 10 Threat Signatures")
        top_labels = df["label"].value_counts().head(10).reset_index()
        top_labels.columns = ["Label", "Count"]
        fig_lbl = px.bar(
            top_labels,
            x="Count",
            y="Label",
            orientation="h",
            color="Count",
            color_continuous_scale=["#38bdf8", "#8b5cf6", "#fb7185"],
        )
        fig_lbl.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_glass_plotly(fig_lbl), use_container_width=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Protocol Vector Breakdown")
        proto_df = df.groupby(["protocol_type", "attack_category"]).size().reset_index(name="Count")
        fig_proto = px.bar(
            proto_df,
            x="protocol_type",
            y="Count",
            color="attack_category",
            barmode="group",
            color_discrete_sequence=["#fb7185", "#34d399", "#38bdf8", "#c084fc", "#fbbf24"],
        )
        st.plotly_chart(style_glass_plotly(fig_proto), use_container_width=True)

    with col_p2:
        st.subheader("Top 10 Targeted Services")
        top_srv = df[df["label"] != "normal"]["service"].value_counts().head(10).reset_index()
        top_srv.columns = ["Service", "Attack Count"]
        fig_srv = px.bar(
            top_srv,
            x="Service",
            y="Attack Count",
            color="Attack Count",
            color_continuous_scale=["#0ea5e9", "#f43f5e"],
        )
        st.plotly_chart(style_glass_plotly(fig_srv), use_container_width=True)

    with st.expander("Inspect Raw Packet Telemetry"):
        selected_cat = st.multiselect(
            "Filter by Attack Category",
            options=df["attack_category"].unique().tolist(),
            default=df["attack_category"].unique().tolist(),
        )
        st.dataframe(df[df["attack_category"].isin(selected_cat)].head(200), use_container_width=True)


# -----------------------------------------------------------------------------
# View 2: Model Performance & Graphs
# -----------------------------------------------------------------------------
elif page == "◈  Model Performance & Graphs":
    st.markdown("## 🧠 Classifier Benchmarks & Diagnostic Plots")

    if "training_results" in st.session_state:
        st.subheader("Model Evaluation Leaderboard")
        results_df = pd.DataFrame(st.session_state["training_results"]).T
        st.dataframe(results_df.style.highlight_max(axis=0, color="rgba(16, 185, 129, 0.35)"), use_container_width=True)
        st.success(f"Optimal Classifier Deployed: **{st.session_state.get('best_model_name', 'N/A')}**")

    st.subheader("Generated Confusion Matrix & Feature Importance")
    if os.path.exists(GRAPHS_DIR):
        graph_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, "*.png")))
        if graph_files:
            cols = st.columns(min(len(graph_files), 2))
            for idx, img_path in enumerate(graph_files):
                with cols[idx % 2]:
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.warning(f"No `.png` plots found in `{GRAPHS_DIR}`. Run **Train & Evaluate Pipeline** in the sidebar.")
    else:
        st.warning(f"Directory `{GRAPHS_DIR}` does not exist yet.")


# -----------------------------------------------------------------------------
# View 3: Threat Alerts Log
# -----------------------------------------------------------------------------
elif page == "◈  Threat Alerts Log":
    st.markdown("## 🚨 Intrusion Detection Alerts (`outputs/predictions`)")

    if os.path.exists(PREDICTIONS_DIR):
        alert_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.csv")))
        if alert_files:
            selected_file = st.selectbox("Select Alert Log File", alert_files)
            alerts_df = pd.read_csv(selected_file)

            c1, c2 = st.columns(2)
            render_glass_metric(c1, "Logged Flows", f"{len(alerts_df):,}", os.path.basename(selected_file), "radar", "cyan")
            if "prediction" in alerts_df.columns:
                flagged = int((alerts_df["prediction"] != 0).sum()) if pd.api.types.is_numeric_dtype(alerts_df["prediction"]) else int((alerts_df["prediction"] != "normal").sum())
                render_glass_metric(c2, "Flagged Threats", f"{flagged:,}", "Requires SOC Review", "threat", "crimson")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.dataframe(alerts_df, use_container_width=True)
            st.download_button(
                label="⬇️ Export Threat Alerts CSV",
                data=alerts_df.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(selected_file),
                mime="text/csv",
            )
        else:
            st.info(f"No CSV alert logs found in `{PREDICTIONS_DIR}` yet.")
    else:
        st.info(f"Directory `{PREDICTIONS_DIR}` not found. Run the pipeline first.")


# -----------------------------------------------------------------------------
# View 4: Live Packet Inspector
# -----------------------------------------------------------------------------
elif page == "◈  Live Packet Inspector":
    st.markdown("## 🔍 Real-Time Packet & Batch Inference")

    model, preprocessor, load_err = load_saved_artifacts(MODELS_DIR)
    if load_err:
        st.warning(f"{load_err} Run **Train & Evaluate Pipeline** in the sidebar first.")
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Flow Simulator", "Batch CSV Inference"])

    with tab_single:
        with st.form("single_flow_form"):
            c1, c2, c3 = st.columns(3)
            duration = c1.number_input("Duration (sec)", min_value=0, value=0)
            protocol_type = c2.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
            service = c3.selectbox("Service", ["http", "private", "domain_u", "smtp", "ftp_data", "eco_i", "other", "ecr_i", "telnet", "finger"])

            c4, c5, c6 = st.columns(3)
            flag = c4.selectbox("Connection Flag", ["SF", "S0", "REJ", "RSTR", "SH", "RSTO", "S1", "RSTOS0", "S3", "S2", "OTH"])
            src_bytes = c5.number_input("Source Bytes (`src_bytes`)", min_value=0, value=491)
            dst_bytes = c6.number_input("Destination Bytes (`dst_bytes`)", min_value=0, value=0)

            c7, c8, c9 = st.columns(3)
            count = c7.number_input("Connection Count (`count`)", min_value=0, value=2)
            srv_count = c8.number_input("Service Count (`srv_count`)", min_value=0, value=2)
            serror_rate = c9.slider("SYN Error Rate (`serror_rate`)", 0.0, 1.0, 0.0)

            submitted = st.form_submit_button("Analyze Packet Flow")

        if submitted:
            if not dataset_exists:
                st.error("Dataset required to populate baseline feature medians.")
                st.stop()

            base_df = get_dataset(data_path).drop(columns=["label", "difficulty_level"], errors="ignore")
            sample_row = {
                col: (float(base_df[col].median()) if pd.api.types.is_numeric_dtype(base_df[col]) else str(base_df[col].mode()[0]))
                for col in base_df.columns
            }
            sample_row.update({
                "duration": duration,
                "protocol_type": protocol_type,
                "service": service,
                "flag": flag,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "count": count,
                "srv_count": srv_count,
                "serror_rate": serror_rate,
            })

            input_df = pd.DataFrame([sample_row])
            try:
                X_input = preprocessor.transform(input_df) if preprocessor is not None else input_df
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0].max() if hasattr(model, "predict_proba") else None

                is_attack = (pred != 0) if isinstance(pred, (int, np.integer)) else (str(pred).lower() != "normal")
                if is_attack:
                    st.error(f"🚨 **INTRUSION DETECTED** — Class: `{pred}`" + (f" | Confidence: `{prob:.2%}`" if prob else ""))
                else:
                    st.success(f"✅ **NORMAL TRAFFIC** — Class: `{pred}`" + (f" | Confidence: `{prob:.2%}`" if prob else ""))
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    with tab_batch:
        uploaded_file = st.file_uploader("Upload NSL-KDD formatted CSV file", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            feature_df = batch_df.drop(columns=["label", "difficulty_level"], errors="ignore")
            try:
                X_batch = preprocessor.transform(feature_df) if preprocessor is not None else feature_df
                batch_df["predicted_threat"] = model.predict(X_batch)
                st.success(f"Scored {len(batch_df):,} network records.")
                st.dataframe(batch_df.head(100), use_container_width=True)
                st.download_button(
                    "⬇️ Download Batch Predictions",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_threat_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction error: {exc}")