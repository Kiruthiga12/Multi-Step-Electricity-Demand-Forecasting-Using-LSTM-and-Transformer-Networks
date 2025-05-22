import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, create_sequences
from lstm_model import build_lstm_model
from transformer_model import build_transformer_model
import tensorflow as tf
from utils import evaluate_model

# Page config
st.set_page_config(page_title="Electricity Forecasting", layout="wide")

st.title("⚡ Electricity Consumption Forecasting")
st.markdown("Compare **LSTM** vs **Transformer** model predictions")

# Load data and models
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Reshape if necessary
if len(y_test.shape) > 1:
    y_test = y_test.squeeze()

# Sidebar model selection
model_choice = st.sidebar.selectbox("Choose Model", ["LSTM", "Transformer"])

# Load model
@st.cache_resource
def load_model(name):
    if name == "LSTM":
        return tf.keras.models.load_model("lstm_model.keras")
    else:
        return tf.keras.models.load_model("transformer_model.keras")

model = load_model(model_choice)

# Predict
with st.spinner(f"Predicting using {model_choice}..."):
    preds = model.predict(X_test)
    preds = preds.squeeze()

# Evaluate
r2, mae, rmse = evaluate_model(y_test, preds)

st.subheader("📊 Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")

# Plot
st.subheader("📈 Actual vs Predicted")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, label="Actual", linewidth=2)
ax.plot(preds, label=f"{model_choice} Prediction", linestyle='--')
ax.set_xlabel("Time Step")
ax.set_ylabel("Electricity Consumption")
ax.set_title(f"{model_choice} Forecast vs Actual")
ax.legend()
st.pyplot(fig)
