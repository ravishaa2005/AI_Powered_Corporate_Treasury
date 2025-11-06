# 🤖 AI-Powered Corporate Treasury Intelligence and Risk Prediction System  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-102230?style=for-the-badge&logo=tensorflow&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Bi--LSTM-8E44AD?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📘 Overview  

**AI-Powered Corporate Treasury** is an AI-driven financial analytics system built using **Python** and **Machine Learning** to evaluate corporate performance and predict financial risk.  
It employs a **Bi-LSTM model** to compute various key performance metrics like **Risk Score, Profitability Score, Growth Score, Sector Score**, and **Overall Score**, helping companies and investors make data-backed financial decisions.  

This system is trained on a dataset of **231 Indian companies**, integrating both **financial** and **macroeconomic indicators** to generate accurate forecasts and risk assessments.  

---

## ⚙️ Core Features  

- **Hybrid ML Framework:** Uses Bi-LSTM for score prediction and Agglomerative Clustering for risk segmentation.  
- **Comprehensive Scoring System:** Calculates Risk, Profitability, Growth, Sector, and Overall Scores for companies.  
- **Financial Forecasting:** Predicts the next quarter’s **risk level** for a new company based on user input.  
- **Model Evaluation:** Compared multiple models - LSTM, Bi-LSTM, XGBoost, LightGBM, and Random Forest - with Bi-LSTM performing best.  
- **Dynamic Clustering:** Applies **Agglomerative Clustering (Average Linkage)** with **3 clusters** to group companies by risk.  
- **Interactive Visualization:** Displays cluster groups and financial trends for easy interpretation.  

---

## 🧩 Dataset Details  

The dataset includes **231 Indian companies** with the following features:  

| Feature | Description |
|----------|-------------|
| Company | Company name |
| Sector | Industry sector |
| Revenue | Total revenue generated |
| Expenses | Operating and fixed costs |
| EBITDA | Earnings Before Interest, Tax, Depreciation, and Amortization |
| Operating Margin % | Profitability indicator |
| Depreciation | Value depreciation on assets |
| Interest | Interest expense |
| PBT | Profit Before Tax |
| Tax | Tax amount paid |
| Net Profit | Final profit after tax |
| EPS | Earnings per share |
| GST | Goods & Services Tax rate |
| CorpTax% | Corporate tax rate |
| Inflation% | Inflation percentage |
| RepoRate% | RBI repo rate |
| USDINR_Close | USD to INR closing exchange rate |

---

## 🧠 Model Architecture  

- **Bi-LSTM Model:** Captures sequential dependencies in financial data, improving prediction accuracy.  
- **Agglomerative Clustering:** Groups companies into 3 clusters (Low, Medium, High Risk) for the next quarter.  
- **Evaluation Metrics:** Mean Squared Error (MSE), R² Score, and Accuracy for classification tasks.  

---

## 🖥️ Frontend Structure  

The **frontend** of the project is built using **Streamlit** and includes the following main files:  

| File Name | Description |
|------------|-------------|
| `app.py` | Main Streamlit app file for user interaction and dashboard visualization |
| `bilstm_clustering.py` | Contains the Bi-LSTM model implementation and clustering logic |
| `score_calculation.py` | Handles score computation (Risk, Growth, Profitability, Sector, and Overall) |

---

## ⚙️ Setup & Configuration  

### 🧾 Prerequisites  
Ensure the following tools are installed on your system:  
- Python 3.9 or above  
- pip (Python package manager)  
- Virtual environment setup  

---

### 🛠 Installation Steps  

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI_Powered_Corporate_Treasury.git
cd AI_Powered_Corporate_Treasury

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # (For Windows)
# source venv/bin/activate   # (For Mac/Linux)

# Install required dependencies
pip install tensorflow streamlit pandas scikit-learn
pip install plotly
pip install openpyxl
```
### Run the Application
Once all dependencies are installed, launch the application with:
```bash
streamlit run app.py
```
This will open a Streamlit dashboard in your browser, where you can:

- Upload new company data (Excel/CSV)
- View predicted Risk, Profitability, Growth, and Overall Scores
- Visualize risk clusters and trends with interactive Plotly charts
- Forecast the next-quarter risk level for any company

### 📊 Results
- Best Model: Bi-LSTM (outperformed LSTM, XGBoost, LightGBM, and Random Forest)
- Clustering Method: Agglomerative Clustering (Average Linkage, 3 clusters)
- Output: Predicts next-quarter risk level and visualizes clusters interactively.

### 🚀 Future Enhancements
- Integration with real-time financial APIs for live data analysis.
- Addition of automated alert systems for high-risk companies.
- Advanced visualization dashboards with time-series comparison.
- Deployment on cloud platforms for enterprise-scale access.

### 👩‍💻 Contributors
- Ravisha Arora
- Drishta Grover
- Nikhil Khandelwal
