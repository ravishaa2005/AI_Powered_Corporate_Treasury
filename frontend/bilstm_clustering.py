from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

"""Model artifact path resolution.

We prefer a path relative to the repository so the app works on any machine.
Search candidates in order and use the first that exists; otherwise default to
`<repo>/financial_model/financial_model`.
"""

# Candidate directories to probe
_here = Path(__file__).parent
_candidates = [
    _here / "financial_model" / "financial_model",
    _here.parent / "financial_model" / "financial_model",
    Path.cwd() / "financial_model" / "financial_model",
]

MODEL_DIR = None
for _p in _candidates:
    if _p.exists():
        MODEL_DIR = _p
        break
if MODEL_DIR is None:
    # Fallback to the expected relative location; existence will be validated later
    MODEL_DIR = _here / "financial_model" / "financial_model"

# Artifact paths
BILSTM_MODEL_FILE = MODEL_DIR / "bilstm_model.keras"
SCALER_X_FILE = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_FILE = MODEL_DIR / "scaler_y.pkl"
TRAINING_DF = MODEL_DIR / "scaled_training_df.pkl"
CLUSTER_MODEL_FILE = MODEL_DIR / "cluster_model.pkl"

# ============================================================
# Load Model + Predict Future Scores + Cluster Risk Levels
# ============================================================
def predict_future_scores_and_cluster(df):
    # --- Check if required artifacts exist ---
    if not (BILSTM_MODEL_FILE.exists() and SCALER_X_FILE.exists() and SCALER_Y_FILE.exists()):
        missing = []
        if not BILSTM_MODEL_FILE.exists():
            missing.append(BILSTM_MODEL_FILE.name)
        if not SCALER_X_FILE.exists():
            missing.append(SCALER_X_FILE.name)
        if not SCALER_Y_FILE.exists():
            missing.append(SCALER_Y_FILE.name)
        raise FileNotFoundError(
            f"Missing model/scaler files in '{MODEL_DIR}'. Add: {', '.join(missing)}"
        )

    # --- Load model and scalers ---
    model = tf.keras.models.load_model(BILSTM_MODEL_FILE)
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    df_train = pd.read_pickle(TRAINING_DF) if TRAINING_DF.exists() else None

    SEQ_LEN = 16
    TARGETS = [
        "Risk_Score", "Growth_Score", "Profitability_Score",
        "Stability_Score", "Sector_Score", "Overall_Score"
    ]
    MACRO_FEATURES = ["GST", "CorpTax%", "Inflation%", "RepoRate%", "USDINR_Close"]
    SEQ_FEATURES = TARGETS + MACRO_FEATURES

    prediction_rows = []

    # --- Generate predictions for each company ---
    for company, comp_df in df.groupby("Company"):
        comp_df_sorted = comp_df.sort_values("FiscalDate")

        if len(comp_df_sorted) < SEQ_LEN:
            continue

        seq_df = comp_df_sorted[SEQ_FEATURES].copy()

        # Fill missing columns (if any)
        for col in SEQ_FEATURES:
            if col not in seq_df.columns:
                if df_train is not None and col in df_train.columns:
                    seq_df[col] = df_train[col].mean()
                else:
                    seq_df[col] = 0

        last_seq = seq_df[SEQ_FEATURES].values[-SEQ_LEN:]
        last_seq_scaled = scaler_X.transform(last_seq)
        last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)

        # Dummy company input if model expects two inputs
        company_input = np.array([[0]], dtype=np.int32)

        # --- Predict next-quarter scores ---
        try:
            pred_scaled = model.predict([last_seq_scaled, company_input], verbose=0)
        except Exception as e:
            raise ValueError(f"Model prediction failed: {e}")

        pred = scaler_y.inverse_transform(pred_scaled)[0]
        pred_dict = dict(zip([f"NextQ_{t}" for t in TARGETS], pred))
        pred_dict["Company"] = company
        prediction_rows.append(pred_dict)

    # --- Combine predictions with original data ---
    pred_df = pd.DataFrame(prediction_rows)
    final_df = pd.merge(df, pred_df, on="Company", how="left")

    nextq_col = "NextQ_Risk_Score"
    if nextq_col not in final_df.columns:
        raise ValueError("Prediction failed: 'NextQ_Risk_Score' not found in output.")

    # --- Apply clustering ---
    if CLUSTER_MODEL_FILE.exists():
        cluster_model = joblib.load(CLUSTER_MODEL_FILE)
        clusters = cluster_model.fit_predict(final_df[[nextq_col]].fillna(0))
    else:
        # Create new clustering if no pre-trained model
        clusters = AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(
            final_df[[nextq_col]].fillna(0)
        )

    final_df["Risk_Cluster"] = clusters

    # --- Map clusters to human-readable levels ---
    cluster_scores = final_df.groupby("Risk_Cluster")[nextq_col].mean().sort_values()
    risk_map = {
        cluster: level for cluster, level in zip(cluster_scores.index, ["Low", "Medium", "High"])
    }
    final_df["Risk_Level"] = final_df["Risk_Cluster"].map(risk_map)

    # --- Final risk label output ---
    if final_df["Company"].nunique() == 1:
        risk_label = final_df["Risk_Level"].iloc[-1]
    else:
        risk_label = final_df.groupby("Company")["Risk_Level"].last().to_dict()

    return final_df, risk_label
