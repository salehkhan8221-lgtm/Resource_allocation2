import streamlit as st
import pickle
import numpy as np
import os

# Load model and scaler
base_path = 
os.path.dirname(__file__)
model = 
pickle.load(open(os.path.join(base_path, "model.pkl") , "rb"))
scaler =
pickle.load(open(os.path.join(base_path, "scaler.pkl") , "rb"))

st.title("🧠 Intelligent Resource Allocation")
st.markdown("Predict optimal resource allocation based on system metrics")

cpu = st.number_input("CPU Usage (%)", min_value=0, max_value=100, step=1)
memory = st.number_input("Memory Usage (GB)", min_value=0.0, step=0.1)
bandwidth = st.number_input("Bandwidth (Mbps)", min_value=0, step=1)
latency = st.number_input("Latency (ms)", min_value=0.0, step=0.1)

if st.button("Predict Allocation"):
    input_data = np.array([[cpu, memory, bandwidth, latency]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)
    st.success(f"Predicted Resource Allocation: {prediction[0]:.2f}")
