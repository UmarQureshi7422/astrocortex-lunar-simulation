from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

st.set_page_config(
    page_title="ASTROCORTEX | Lunar Mission Control",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FEATURES = [
    "dust_concentration",
    "electrostatic_charge",
    "surface_temperature",
    "vibration",
]
RISK_ORDER = ["LOW", "MODERATE", "HIGH"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
}

BG = "#07111F"
PANEL = "#101E30"
PANEL_LIGHT = "#172941"
CYAN = "#53D8ED"
WHITE = "#EAF3FF"
MUTED = "#9CB0C8"
AMBER = "#F4B942"
RED = "#F06464"
GREEN = "#55D69E"
RISK_COLORS = {"LOW": GREEN, "MODERATE": AMBER, "HIGH": RED}

st.markdown("""
<style>
.stApp {background:#07111F;color:#EAF3FF}
header[data-testid="stHeader"] {background:#07111F}
section[data-testid="stSidebar"] {background:#101E30;border-right:1px solid #263C55}
section[data-testid="stSidebar"] * {color:#EAF3FF}
.block-container {padding-top:1.15rem;padding-bottom:2.4rem;max-width:1500px}
h1,h2,h3 {color:#EAF3FF !important}
h1 {letter-spacing:2px;font-weight:800}
p,label {color:#C2D0E2}
div[data-testid="stMetric"] {background:#101E30;border:1px solid #29445F;padding:13px;border-radius:12px;min-height:112px}
div[data-testid="stMetricValue"] {color:#53D8ED;font-size:clamp(1.15rem,2vw,2rem)}
div[data-testid="stMetricLabel"] {color:#AFC2D8}
div.stButton>button {background:#173D59;color:white;border:1px solid #53D8ED;border-radius:8px}
div.stButton>button:hover {background:#245C78;color:white}
div[data-testid="stAlert"] {border-radius:10px}
div[data-testid="stTabs"] button {color:#C5D7E9}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#53D8ED}
hr {border-color:#263C55}
[data-testid="stDataFrame"] {border:1px solid #29445F;border-radius:8px}
.mission-banner {background:linear-gradient(110deg,#122A42,#0C1A2B);border:1px solid #31536F;border-left:5px solid #53D8ED;border-radius:12px;padding:20px;margin-bottom:14px}
.mission-label {color:#53D8ED;font-size:11px;letter-spacing:3px;font-weight:bold}
.mission-title {color:#F0F6FF;font-size:29px;font-weight:800;letter-spacing:2px;margin-top:7px}
.mission-subtitle {color:#AFC2D8;font-size:13px;margin-top:5px}
div[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {padding-top:0.4rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename} is missing. Run: python train_model.py")
        loaded[name] = joblib.load(path)
    return loaded

@st.cache_data
def load_dataset():
    path = DATA_DIR / "astroCortex_simulated_data.csv"
    if not path.exists():
        raise FileNotFoundError("Dataset is missing. Run: python train_model.py")
    return pd.read_csv(path)

@st.cache_data
def load_comparison():
    path = DATA_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError("Model comparison is missing. Run: python train_model.py")
    return pd.read_csv(path)

try:
    models = load_models()
    df = load_dataset()
    comparison_df = load_comparison()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Project setup is incomplete: {exc}")
    st.info("Open a terminal in the AstroCortex folder and run `python train_model.py`, then restart Streamlit.")
    st.stop()

# Recreate the same held-out split used during training for reproducible validation.
X = df[FEATURES]
y = df["risk"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def dark_chart(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color("#38516B")
    ax.grid(color="#31445C", alpha=0.45, linestyle="--")
    return fig, ax

def draw_lunar_scene(risk, dust, charge):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 55); ax.axis("off")
    rng = np.random.default_rng(42)
    ax.scatter(rng.uniform(0,100,90), rng.uniform(17,54,90),
               s=rng.uniform(2,8,90), color="#A9C7E8", alpha=.55)
    terrain = Polygon(
        [(0,13),(10,16),(20,14),(30,17),(40,13),(52,15),(65,12),
         (78,16),(90,13),(100,15),(100,0),(0,0)],
        closed=True, facecolor="#5B6070", edgecolor="#A7AFC0", linewidth=1.2
    )
    ax.add_patch(terrain)
    for x, yy, radius in [(12,5,3),(33,8,4),(72,6,5),(91,8,3)]:
        ax.add_patch(Ellipse((x,yy), radius*2, radius, color="#353C4A", alpha=.85))
    ax.plot([48,44],[20,14],color="#C4D2E0",linewidth=3)
    ax.plot([58,62],[20,14],color="#C4D2E0",linewidth=3)
    ax.add_patch(Polygon([(47,21),(50,27),(56,27),(59,21)],closed=True,
                         facecolor="#D7E1EC",edgecolor=CYAN,linewidth=2))
    ax.add_patch(Rectangle((50,27),6,3,facecolor="#E8EDF3",edgecolor=CYAN))
    ax.plot([53,53],[30,34],color=CYAN,linewidth=1.5)
    ax.scatter([53],[34],color=CYAN,s=25)
    ax.add_patch(Rectangle((59,23),10,4,facecolor="#164E75",edgecolor=CYAN))
    for xx in [61,64,67]:
        ax.plot([xx,xx],[23,27],color="#67B9DD",linewidth=.6)
    particle_count = int(20 + dust*1.5) + {"LOW":0,"MODERATE":35,"HIGH":80}[risk]
    ax.scatter(rng.uniform(20,80,particle_count),rng.uniform(10,32,particle_count),
               s=rng.uniform(3,16,particle_count),color=RISK_COLORS[risk],alpha=.42)
    if charge >= 50:
        for radius in [7,10,13]:
            ax.add_patch(Circle((53,24),radius,fill=False,edgecolor=AMBER,alpha=.18))
    ax.text(3,50,"SIMULATED LUNAR SURFACE",color=CYAN,fontsize=11,fontweight="bold")
    ax.text(3,45,"CONCEPTUAL LANDER / REGOLITH / DUST ENVIRONMENT",color=WHITE,fontsize=8)
    ax.text(78,49,f"RISK: {risk}",color=RISK_COLORS[risk],fontsize=11,fontweight="bold")
    ax.text(78,44,f"DUST INDEX: {dust:.1f}",color=WHITE,fontsize=8)
    ax.text(78,40,f"CHARGE INDEX: {charge:.1f}",color=WHITE,fontsize=8)
    return fig

# SIDEBAR
st.sidebar.markdown("# 🌙 ASTROCORTEX")
st.sidebar.caption("LUNAR MISSION CONTROL · RESEARCH PROTOTYPE")
st.sidebar.markdown("---")
st.sidebar.markdown("### AI PREDICTION ENGINE")
selected_model_name = st.sidebar.selectbox("Select AI model", list(MODEL_FILES.keys()), index=2)
st.sidebar.caption(f"ACTIVE MODEL: {selected_model_name}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛰️ ENVIRONMENTAL TELEMETRY")
dust = st.sidebar.slider("Dust concentration index", 0.0, 100.0, 50.0, 1.0)
charge = st.sidebar.slider("Electrostatic charge index", 0.0, 100.0, 40.0, 1.0)
temperature = st.sidebar.slider("Surface temperature (°C)", -150.0, 80.0, -30.0, 1.0)
vibration = st.sidebar.slider("Surface vibration index", 0.0, 1.0, 0.30, 0.01)
st.sidebar.markdown("---")
st.sidebar.caption("SIMULATION STATUS: ACTIVE")
st.sidebar.caption("DATA SOURCE: SYNTHETIC")
st.sidebar.caption("NOT REAL LUNAR TELEMETRY")

# LIVE PREDICTION
selected_model = models[selected_model_name]
input_data = pd.DataFrame([{
    "dust_concentration": dust,
    "electrostatic_charge": charge,
    "surface_temperature": temperature,
    "vibration": vibration,
}], columns=FEATURES)
prediction = selected_model.predict(input_data)[0]
probabilities = selected_model.predict_proba(input_data)[0]
classes = list(selected_model.classes_)
probability_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
confidence = float(np.max(probabilities)) * 100
risk_color = RISK_COLORS[prediction]

# HEADER
st.markdown("""
<div class="mission-banner">
  <div class="mission-label">LUNAR RESEARCH OPERATIONS / ASTROCORTEX</div>
  <div class="mission-title">🌙 ASTROCORTEX</div>
  <div class="mission-subtitle">AI-BASED LUNAR DUST HAZARD PREDICTION & ENVIRONMENTAL MONITORING</div>
</div>
""", unsafe_allow_html=True)
st.warning("SIMULATION MODE: Sensor readings and environmental scenarios are synthetic. No real lunar telemetry is being received.")

st.markdown("### MISSION STATUS")
c1,c2,c3,c4 = st.columns([1.25,1,1,1])
c1.metric("ACTIVE AI MODEL", selected_model_name)
c2.metric("PREDICTED DUST RISK", prediction)
c3.metric("CLASSIFIER CONFIDENCE", f"{confidence:.1f}%")
best = comparison_df.sort_values("F1",ascending=False).iloc[0]
c4.metric("TOP MODEL (F1)", str(best["Model"]), f"{best['F1']:.1%} weighted F1")

tab1,tab2,tab3,tab4 = st.tabs([
    "🌕 Mission Overview","🤖 AI Analytics","📊 Model Validation","📡 Sensor Trends"
])

with tab1:
    st.markdown("### LUNAR ENVIRONMENT VISUALIZATION")
    st.caption("Conceptual 2D visualization of a lander in a simulated lunar dust environment.")
    scene = draw_lunar_scene(prediction,dust,charge)
    st.pyplot(scene,use_container_width=True)
    plt.close(scene)

    st.markdown("### ENVIRONMENTAL SENSOR TELEMETRY")
    cols = st.columns(4)
    sensor_values = [
        ("DUST INDEX",dust,100, dust/100),
        ("CHARGE INDEX",charge,100, charge/100),
        ("TEMPERATURE",temperature,230,(temperature+150)/230),
        ("VIBRATION",vibration,1,vibration),
    ]
    for col,(label,value,maximum,fraction) in zip(cols,sensor_values):
        with col:
            st.caption(label)
            st.progress(float(np.clip(fraction,0,1)))
            suffix = " °C" if label=="TEMPERATURE" else ""
            col.metric(label,f"{value:.2f}{suffix}")

    st.markdown("### AI HAZARD ASSESSMENT")
    left,right = st.columns([1,2])
    with left:
        st.markdown(
            f"""<div style="background:{PANEL};border:1px solid {risk_color};
            border-radius:12px;padding:22px;text-align:center;">
            <div style="color:{MUTED};font-size:12px;">PREDICTED HAZARD LEVEL</div>
            <div style="color:{risk_color};font-size:28px;font-weight:bold;margin-top:10px;">{prediction}</div>
            </div>""", unsafe_allow_html=True
        )
    recommendations = {
        "LOW":"Continue routine monitoring in this illustrative scenario.",
        "MODERATE":"Increase monitoring frequency and consider precautionary measures.",
        "HIGH":"Issue a high-risk alert and consider protective operating measures.",
    }
    with right:
        st.markdown("#### MISSION RECOMMENDATION")
        if prediction == "HIGH":
            st.error(recommendations[prediction], icon="🚨")
        elif prediction == "MODERATE":
            st.warning(recommendations[prediction], icon="⚠️")
        else:
            st.success(recommendations[prediction], icon="✅")
        st.caption("These are illustrative research recommendations, not validated lunar operating procedures.")

with tab2:
    st.markdown("### AI PREDICTION CENTRE")
    left,right = st.columns(2)
    with left:
        st.markdown("#### CLASS PROBABILITIES")
        display_probs = probability_df.copy()
        display_probs["Probability (%)"] = (display_probs.pop("Probability")*100).round(2)
        display_probs.columns = ["Risk category","Probability (%)"]
        st.dataframe(display_probs,hide_index=True,use_container_width=True)
        fig,ax = dark_chart((7,3.6))
        ax.bar(classes,probabilities,color=[RISK_COLORS[x] for x in classes],width=.55)
        ax.set_ylim(0,1); ax.set_ylabel("Classifier probability"); ax.set_title("AI risk classification")
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    with right:
        st.markdown("#### ACTIVE SENSOR INPUTS")
        st.dataframe(input_data,hide_index=True,use_container_width=True)
        st.metric("Highest class probability",f"{confidence:.2f}%")
        st.caption("Classifier confidence is not a calibrated probability of a real lunar hazard.")
    st.markdown("### COMPARATIVE AI PERFORMANCE")
    st.dataframe(comparison_df.style.format({
        "Accuracy":"{:.1%}","Precision":"{:.1%}","Recall":"{:.1%}","F1":"{:.1%}"
    }),hide_index=True,use_container_width=True)
    fig,ax = dark_chart((10,3.8))
    bars = ax.bar(comparison_df["Model"],comparison_df["F1"],color=[CYAN,AMBER,GREEN],width=.55)
    ax.set_ylim(0,1); ax.set_ylabel("Weighted F1-score"); ax.set_title("Three-model comparison · held-out test set")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f"{bar.get_height():.1%}",
                ha="center",color=WHITE,fontsize=9)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tab3:
    st.markdown("### MODEL VALIDATION CENTRE")
    st.caption("Metrics are computed on a held-out 20% split of the synthetic dataset.")
    validation_name = st.selectbox("Choose model for validation",list(MODEL_FILES.keys()),key="validation_model")
    validation_model = models[validation_name]
    predictions = validation_model.predict(X_test)
    cm = confusion_matrix(y_test,predictions,labels=RISK_ORDER)
    fig,ax = dark_chart((7.5,4.5))
    im = ax.imshow(cm,cmap="Blues")
    ax.set_title(f"CONFUSION MATRIX · {validation_name}")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(RISK_ORDER,color=WHITE); ax.set_yticklabels(RISK_ORDER,color=WHITE)
    threshold = cm.max()/2 if cm.max() else 0
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>threshold else "#07111F",fontweight="bold")
    fig.colorbar(im,ax=ax)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    report = classification_report(y_test,predictions,labels=RISK_ORDER,output_dict=True,zero_division=0)
    st.markdown("#### CLASSIFICATION REPORT")
    st.dataframe(pd.DataFrame(report).transpose().round(3),use_container_width=True)
    st.caption("Rows in the confusion matrix are actual labels; columns are model predictions.")

with tab4:
    st.markdown("### SIMULATED ENVIRONMENTAL TELEMETRY")
    st.caption("A reproducible 30-step synthetic sequence. It is not a live sensor feed.")
    rng = np.random.default_rng(42)
    n = 30
    trend = pd.DataFrame({
        "Time step":np.arange(1,n+1),
        "Dust index":np.clip(dust+np.cumsum(rng.normal(0,2,n)),0,100),
        "Charge index":np.clip(charge+np.cumsum(rng.normal(0,2,n)),0,100),
        "Temperature (°C)":np.clip(temperature+np.cumsum(rng.normal(0,3,n)),-150,80),
        "Vibration index":np.clip(vibration+np.cumsum(rng.normal(0,.02,n)),0,1),
    })
    trend_features = trend[["Dust index","Charge index","Temperature (°C)","Vibration index"]].copy()
    trend_features.columns = FEATURES
    trend["Predicted risk"] = selected_model.predict(trend_features)
    trend["Risk score"] = trend["Predicted risk"].map({"LOW":0,"MODERATE":1,"HIGH":2})

    chart1,chart2 = st.columns(2)
    with chart1:
        fig,ax = dark_chart((7,3.5))
        ax.plot(trend["Time step"],trend["Dust index"],color=CYAN,linewidth=2,label="Dust")
        ax.set_title("DUST CONCENTRATION INDEX"); ax.set_xlabel("Simulation step"); ax.set_ylabel("Index (0–100)")
        ax.legend(facecolor=PANEL,labelcolor=WHITE)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    with chart2:
        fig,ax = dark_chart((7,3.5))
        ax.plot(trend["Time step"],trend["Charge index"],color=AMBER,linewidth=2,label="Charge")
        ax.set_title("ELECTROSTATIC CHARGE INDEX"); ax.set_xlabel("Simulation step"); ax.set_ylabel("Index (0–100)")
        ax.legend(facecolor=PANEL,labelcolor=WHITE)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    chart3,chart4 = st.columns(2)
    with chart3:
        fig,ax = dark_chart((7,3.5))
        ax.plot(trend["Time step"],trend["Temperature (°C)"],color=RED,linewidth=2)
        ax.set_title("SURFACE TEMPERATURE"); ax.set_xlabel("Simulation step"); ax.set_ylabel("Temperature (°C)")
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    with chart4:
        fig,ax = dark_chart((7,3.5))
        ax.plot(trend["Time step"],trend["Vibration index"],color=GREEN,linewidth=2)
        ax.set_title("SURFACE VIBRATION"); ax.set_xlabel("Simulation step"); ax.set_ylabel("Index (0–1)")
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    fig,ax = dark_chart((11,3.5))
    ax.step(trend["Time step"],trend["Risk score"],where="mid",color=CYAN,linewidth=2)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["LOW","MODERATE","HIGH"])
    ax.set_ylim(-.2,2.2); ax.set_xlabel("Simulation step"); ax.set_ylabel("Predicted category")
    ax.set_title("AI-PREDICTED DUST HAZARD TREND")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.dataframe(trend,hide_index=True,use_container_width=True)
    csv = trend.to_csv(index=False).encode("utf-8")
    st.download_button("Download current synthetic telemetry (CSV)",csv,
                       file_name="astrocortex_synthetic_telemetry.csv",mime="text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:22px;background:#101E30;border:1px solid #29445F;border-radius:12px;">
  <div style="color:#53D8ED;font-size:20px;font-weight:bold;letter-spacing:3px;">🌙 ASTROCORTEX</div>
  <div style="color:#AFC2D8;font-size:12px;margin-top:8px;">AI-BASED LUNAR DUST HAZARD PREDICTION SYSTEM</div>
  <div style="color:#7189A4;font-size:11px;margin-top:9px;">ISSSP 2026 · SYNTHETIC RESEARCH SIMULATION · NOT FLIGHT-QUALIFIED</div>
</div>
""",unsafe_allow_html=True)
