import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Intelligent Resource Allocation",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# -------------------- STYLING --------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #dfe9f3, #ffffff);
        color: #222222;
    }
    h1, h2, h3, h4 {
        text-align: center;
        color: #2e3a59;
    }
    .prediction-box {
        background-color: #d5f5e3;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #145a32;
    }
    footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL & SCALER --------------------
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(base_path, "scaler.pkl"), "rb"))

# -------------------- APP TITLE --------------------
st.title("🧠 Intelligent Resource Allocation")
st.subheader("🚀 Predict optimal cloud resource usage in real-time")
st.markdown("---")

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    cpu = st.number_input("🖥️ CPU Usage (%)", min_value=0, max_value=100, step=1)
    memory = st.number_input("💾 Memory Usage (GB)", min_value=0.0, step=0.1)

with col2:
    bandwidth = st.number_input("🌐 Bandwidth (Mbps)", min_value=0, step=1)
    latency = st.number_input("⏱️ Latency (ms)", min_value=0.0, step=0.1)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Allocation"):
    input_data = np.array([[cpu, memory, bandwidth, latency]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    st.markdown(f"""
        <div class='prediction-box'>
            🎯 Predicted Resource Allocation: {prediction[0]:.2f}
        </div>
    """, unsafe_allow_html=True)

    # Optional Graph
    fig, ax = plt.subplots()
    ax.bar(["CPU", "Memory", "Bandwidth", "Latency"], [cpu, memory, bandwidth, latency])
    ax.set_title("📊 Input Metrics Overview")
    ax.set_ylabel("Values")
    st.pyplot(fig)

# -------------------- SIDEBAR INFO --------------------
st.sidebar.title("📘 About Project")
st.sidebar.markdown("""
This project uses **Machine Learning (Linear Regression)**  
to predict resource allocation based on:
- CPU Usage  
- Memory Usage  
- Bandwidth  
- Latency  

**Goal:** Optimize cloud system performance.  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developed by:** Swaleh Khan (B.Tech CSE 2025)")

# -------------------- FOOTER --------------------
st.markdown("""
<footer>
    © 2025 Intelligent Resource Allocation | Department of Computer Science
</footer>
""", unsafe_allow_html=True)



