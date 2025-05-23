# Multi-Step Electricity Demand Forecasting Using LSTM and Transformer Networks

⚡ Multi-Step Electricity Demand Forecasting Using LSTM and Transformer Networks

<div align="center"> 
  <img src="https://img.shields.io/badge/Python-3.12.7-blue?style=flat-square&logo=python" /> 
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=flat-square&logo=tensorflow" /> 
  <img src="https://img.shields.io/badge/Project%20Type-TimeSeries-blueviolet?style=flat-square" /> 
  <img src="https://img.shields.io/badge/Model-LSTM%20%7C%20Transformer-green?style=flat-square" /> 
</div>

### 📝 Project Summary

  This project focuses on multi-step forecasting of electricity consumption using deep learning models such as LSTM (Long Short-Term Memory) and Transformer 

networks. It leverages hourly power grid data, including features like solar, wind, nuclear, coal, and biomass generation to predict the next 6 hours of 

electricity demand, enabling better energy management and grid stability.

### 🚀 Objectives

📊 Forecast electricity consumption 6 hours ahead using past 24-hour historical data.

🧠 Build and compare two deep learning models: LSTM and Transformer.

🔁 Evaluate model performance using R², MAE, and RMSE for robust comparison and selection.

### 🔧 Tools & Technologies

🐍 Python 3.12.7

🧠 TensorFlow / Keras

🧪 Scikit-learn

📊 NumPy / Pandas

📈 Matplotlib / Seaborn

📏 MinMaxScaler for feature normalization

📓 Jupyter Notebook for development

🌐 Streamlit for interactive web app deployment

### 📉 Feature Engineering Highlights
  
  ⏱️ Time-based features: hour, dayofweek, is_weekend, month
  
  🔁 Lag features: lag_1, lag_2, lag_3
  
  📊 Rolling statistics: 6, 12, and 24-hour rolling mean & std
  
  🔄 Sequence generation for supervised learning using past 24 hours to predict next 6 hours

### 🧠 Model Architecture
  
 ### ✅ LSTM Model
  
  Stacked LSTM layers with Dropout
     
  Input Shape: (24, number of features)
     
  Output: 6 future values of electricity consumption

### ✅ Transformer Model
    
  Custom-built encoder-only Transformer
    
  Multi-Head Self Attention
    
  Feed Forward Dense layers
    
  Flattened output layer for 6-hour prediction

### 📈 Model Evaluation
  
  Metric	    LSTM Model	     Transformer Model
   
  R²	         0.8539	            0.8658
   
  MAE	         0.0405           	0.0380
    
  RMSE	       0.0576             0.0552

### 📊 Visualizations

Multi-step Forecast Plot

<p align="center"> <img src="assets/multistep_forecast.png" alt="Multi-step Forecast" width="700"> </p>

Training and Validation Loss

<p align="center"> <img src="assets/loss_curves.png" alt="Loss Curves" width="700"> </p>

### 📌 Key Takeaways
   
   📈 Multi-step forecasting is essential for grid stability and energy demand management.
   
   🤖 Transformers outperformed LSTM for short-horizon sequence prediction.
   
   🧠 Deep learning is highly effective in modeling complex temporal dependencies in power systems.

### 🌐 Streamlit Web App

An interactive Streamlit dashboard was developed for real-time visualization and multi-step electricity demand forecasting using LSTM and Transformer models.

### 🔍 Features

📈 Upload and visualize new electricity consumption data.

🧮 Run forecasts directly in the browser with LSTM and Transformer models.

📊 Display of forecast results alongside actual values with interactive line plots.

📉 Metrics panel showing MAE, RMSE, and R² Score for model evaluation.

### 🛠 How It Works

Developed using Streamlit integrated with Jupyter Notebook-trained models.

Backend prediction uses saved .keras model weights and MinMaxScaler for consistent input scaling.

📷 Streamlit App Screenshot

<div align="center"> <img src="your_screenshot_filename.png" alt="Streamlit App Screenshot" width="600"/> </div>

⚠️ The live deployment is not hosted online. However, the Streamlit code is fully functional and can be run locally.

### 🔧 Run Locally

```bash
cd streamlit_app (location where the app.py is created)

pip install -r requirements.txt

streamlit run app.py
