import streamlit as st
import pandas as pd
import joblib
import time
import random
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads the model and metadata saved in Phase 3."""
    # The dictionary contains 'model', 'label_mapping', and 'cols_to_drop'
    data = joblib.load('ciciot_lightgbm_model.pkl')
    return data['model'], data['label_mapping'], data['cols_to_drop']

@st.cache_data
def load_simulation_data(file_choice):
    """Loads dataset. train.csv is recommended to access Benign samples."""
    return pd.read_csv(file_choice)

# Initialize assets
model, mapping, dropped_cols = load_assets()
inv_mapping = {v: k for k, v in mapping.items()} #

# --- 3. HELPER FUNCTIONS ---
def get_safe_metric(df, possible_names):
    """Prevents KeyErrors by checking for various CICIOT23 column naming styles."""
    for name in possible_names:
        if name in df.columns:
            return df[name].values[0]
    return 0.0

# --- 4. SIDEBAR & SETTINGS ---
st.sidebar.title("🛡️ Simulation Settings")
# train.csv is default because test.csv was found to have 0 benign samples
data_source = st.sidebar.selectbox("Select Data Source", ["train.csv", "test.csv"], index=0)
is_running = st.sidebar.toggle("Start Live Monitor", value=False)
simulation_speed = st.sidebar.slider("Update Interval (Sec)", 0.5, 5.0, 2.0)
# Default to 95% to ensure benign traffic is the standard "normal" state
benign_bias = st.sidebar.slider("Target Benign Probability", 0.0, 1.0, 0.95)

# Prepare biased pools
df_sim = load_simulation_data(data_source)

# Use the specific label 'BenignTraffic' found in the dataset for filtering
df_benign = df_sim[df_sim['label'] == 'BenignTraffic']
df_attacks = df_sim[df_sim['label'] != 'BenignTraffic']

st.sidebar.divider()
st.sidebar.write(f"📊 Benign rows found: {len(df_benign)}")
st.sidebar.write(f"📊 Attack rows found: {len(df_attacks)}")
st.sidebar.info(f"Model Baseline Accuracy: 99.75%")

# --- 5. MAIN UI ---
st.title("🛡️ Real-Time IoT Network Intrusion Monitor")
st.markdown("Monitoring live traffic patterns using the LightGBM production model.")

if is_running:
    # Empty container for content replacement to ensure smooth transitions
    monitor_placeholder = st.empty()

    while True:
        # Step A: Biased Sampling Logic
        if random.random() < benign_bias and not df_benign.empty:
            sample = df_benign.sample(1).copy()
        else:
            sample = df_attacks.sample(1).copy()
        
        idx = sample.index[0]
        actual_label = sample['label'].values[0]

        # Step B: Preprocessing
        # Drop the 19 features identified as low impact during Phase 1 & 3
        X_input = sample.drop(columns=dropped_cols + ['label'], errors='ignore')

        # Step C: Inference & Probability Analysis
        # Get full probability distribution
        probs = model.predict_proba(X_input)[0]
        
        # Map probabilities to class names and sort
        class_probs = {inv_mapping[i]: p for i, p in enumerate(probs)}
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top two possibilities for "This or That" analysis
        top_class, top_val = sorted_probs[0]
        second_class, second_val = sorted_probs[1]

        # Step D: Update the UI
        with monitor_placeholder.container():
            st.divider()
            st.subheader(f"🔍 Analyzing Stream: Packet ID {idx}")
            
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                if top_class == "Benign":
                    st.success(f"### Current State: {top_class} ({top_val*100:.1f}%)")
                else:
                    st.error(f"### ALERT: {top_class} Detected! ({top_val*100:.1f}%)")
                
                # Multi-class possibility snippet
                st.write(f"**Analysis Detail:** The model is {top_val*100:.1f}% sure this is **{top_class}**, "
                         f"but there is a {second_val*100:.1f}% chance it could be **{second_class}**.")
            
            with res_col2:
                # Real-time confidence vs. Phase 4 baseline
                st.metric(
                    label="Prediction Confidence", 
                    value=f"{top_val*100:.2f}%", 
                    delta="Baseline: 99.75%"
                )

            # Live Metrics Display (Robust column handling to prevent KeyErrors)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Flow Duration", f"{get_safe_metric(sample, ['flow_duration', 'Flow_Duration']):.4f}")
            m_col2.metric("Max Packet Len", f"{get_safe_metric(sample, ['Max_Pkt_Len', 'Pkt_Len_Max', 'Pkt Len Max'])}")
            m_col3.metric("Fwd Packets/s", f"{get_safe_metric(sample, ['fwd_pkts_s', 'Fwd_Pkts_s']):.2f}")
            m_col4.metric("IAT Mean", f"{get_safe_metric(sample, ['Flow_IAT_Mean', 'iat_mean']):.4f}")

            # Probability Breakdown
            st.write("**Top 3 Classification Possibilities:**")
            p_cols = st.columns(3)
            for i in range(3):
                label, val = sorted_probs[i]
                p_cols[i].progress(val, text=f"{label}: {val*100:.1f}%")

            with st.expander("Diagnostic Details"):
                st.write(f"CSV Ground Truth Label: **{actual_label}**")
                st.write("**Model Input Vector (14 pruned features):**")
                st.dataframe(X_input)

        # Step E: Transition Delay
        time.sleep(simulation_speed)

else:
    # Offline / Welcome State
    st.divider()
    st.info("Monitor is OFFLINE. Select a data source and toggle 'Start Live Monitor'.")
    st.markdown("### Test Dataset Overview (Phase 4 Results)")
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.write("**Traffic Distribution in Test Set:**")
        # Visualizing the imbalance that makes Benign traffic rare in test.csv
        st.bar_chart(df_test['label'].value_counts())
    with stat_col2:
        st.write("**Final Model Performance:**")
        st.markdown(
            "- **Accuracy:** 99.75%\n"
            "- **DDoS F1-Score:** 1.00\n"
            "- **Mirai F1-Score:** 1.00\n"
            "- **Web Attack F1-Score:** 0.34 (rare class sample issue)"
        ) # Emphasizing the rare class challenge