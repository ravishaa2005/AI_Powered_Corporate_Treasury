import streamlit as st
import pandas as pd
import os
from io import BytesIO
from score_calculation import calculate_scores
from bilstm_clustering import predict_future_scores_and_cluster

st.set_page_config(page_title="AI-Powered Corporate Risk Prediction", layout="wide")
st.title("AI-Powered Corporate Risk Prediction")

# --- Predefined required field names ---
REQUIRED_FIELDS = [
    'FiscalDate', 'Company', 'Sector', 'Revenue', 'Expenses', 'EBITDA',
    'Operating Margin %', 'Depreciation', 'Interest', 'PBT', 'Tax', 'Net Profit',
    'EPS', 'GST', 'CorpTax%', 'Inflation%', 'RepoRate%', 'USDINR_Close'
]

st.markdown("""
### 📁 File Upload
Upload your company's quarterly financial statement (Excel or CSV with at least 16 quarters). 
If you do not have a 'CorpTax%' column, map GST or equivalent, or just fill with 0s.
""")
uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith("csv"):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)
        st.dataframe(df_in.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()
    
    st.markdown("""
    ### 🗂️ Column Mapping
    Map your columns to required dashboard fields:
    """)

    mapping = {}
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Required Fields")
        for req in REQUIRED_FIELDS:
            st.markdown(f"- `{req}`")
    with col2:
        st.markdown("#### Your Data Columns (Select for Each)")
        with st.form("mapping_form"):
            for req in REQUIRED_FIELDS:
                # Find the best guess from user's data for dropdown default
                guess = None
                for colname in df_in.columns:
                    if req.lower().replace(" ","") in colname.lower().replace(" ",""):
                        guess = colname;
                        break
                mapping[req] = st.selectbox(
                    f"Map `{req}` to", [None] + list(df_in.columns), index=(df_in.columns.get_loc(guess) + 1) if guess else 0, key=req
                )
            confirmed = st.form_submit_button("✅ Confirm Mapping")

    if confirmed:
        try:
            # Create mapped DataFrame
            newcols = {v:req for req,v in mapping.items() if v}
            if len(newcols) != len(REQUIRED_FIELDS):
                missing = list(set(REQUIRED_FIELDS) - set([req for req,v in mapping.items() if v]))
                st.warning(f"Please map all required fields: {', '.join(missing)}")
                st.stop()
            df_map = df_in.rename(columns=newcols)
            st.success("Column mapping applied! Preview below:")
            st.dataframe(df_map.head(16), use_container_width=True)
        except Exception as e:
            st.error(f"Mapping Error: {e}")
            st.stop()

        # --- Score calculation ---
        st.markdown("#### ⚙️ Score Calculation")
        with st.spinner("Calculating scores..."):
            try:
                df_scored = calculate_scores(df_map)
                st.dataframe(df_scored.tail(8), use_container_width=True)
            except Exception as e:
                st.error(f"Score Calculation Failed: {e}")
                st.stop()

        # --- Prediction & Clustering ---
        st.markdown("#### 🤖 Prediction & Clustering")
        with st.spinner("Predicting next quarter scores and clustering risk..."):
            try:
                df_pred, risk_level = predict_future_scores_and_cluster(df_scored)
            except FileNotFoundError:
                st.error("❌ BiLSTM/scaler model files not found in 'financial_model/financial_model/'. Please add them and retry.")
                st.stop()
            except Exception as e:
                st.error(f"Clustering/Prediction Error: {e}")
                st.stop()
        st.dataframe(df_pred.tail(8), use_container_width=True)

        # --- Show cluster for each company (use color box) ---
        unique_levels = df_pred['Risk_Level'].unique()
        for comp in df_pred['Company'].unique():
            level = df_pred[df_pred['Company']==comp]['Risk_Level'].iloc[-1]
            if level == "Low":
                st.success(f"🟢 {comp}: Low Risk")
            elif level == "Medium":
                st.warning(f"🟡 {comp}: Medium Risk")
            elif level == "High":
                st.error(f"🔴 {comp}: High Risk")
            else:
                st.info(f"{comp}: {level}")

        # --- Optional: Historical vs predicted trend chart (plotly) ---
        import plotly.graph_objs as go
        st.markdown('#### 📈 Historical vs Predicted Overall Score Trend')
        for comp in df_pred['Company'].unique():
            comp_df = df_pred[df_pred['Company'] == comp].sort_values('FiscalDate')
            x_hist = comp_df['FiscalDate']
            y_hist = comp_df['Overall_Score']
            y_pred = comp_df['NextQ_Overall_Score'].dropna().iloc[-1] if 'NextQ_Overall_Score' in comp_df.columns else None
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='lines+markers', name='Historical'))
            if y_pred is not None:
                predict_x = x_hist.iloc[-1] + pd.DateOffset(months=3)
                fig.add_trace(go.Scatter(x=[predict_x], y=[y_pred], mode='markers', marker=dict(color='red', size=12), name='Predicted NextQ'))
            fig.update_layout(title=f"{comp} - Overall Score", xaxis_title='FiscalDate', yaxis_title='Overall Score (0-100)')
            st.plotly_chart(fig, use_container_width=True)

        # --- Download results ---
        st.markdown("#### ⬇️ Download Final Predictions")
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, file_name="predicted_risk_scores.csv", mime="text/csv")

