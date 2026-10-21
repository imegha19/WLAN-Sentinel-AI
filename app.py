from pathlib import Path

import joblib
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "wlan_demo_model.joblib"
MULTICLASS_MODEL_PATH = PROJECT_DIR / "wlan_multiclass_synthetic_model.joblib"
MULTICLASS_ENCODER_PATH = PROJECT_DIR / "wlan_multiclass_label_encoder.joblib"
REVIEW_DB_PATH = PROJECT_DIR / "wlan_reviews.db"

DEFAULT_CHUNK_SIZE = 10_000
MIN_CHUNK_SIZE = 1_000
MAX_CHUNK_SIZE = 50_000
PREVIEW_ROWS = 20
MAX_NUMERIC_SAMPLE_ROWS = 5_000

st.set_page_config(
    page_title="WLAN Sentinel AI | Security Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= FRONTEND STYLING =========================
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #071521; color: #e6eef5; font-size: 0.96rem; }
    [data-testid="stHeader"] { background: rgba(7,21,33,.96); border-bottom: 1px solid rgba(37,70,87,.35); }
    [data-testid="stSidebar"] { background: #0b1c2a; border-right: 1px solid #244252; }
    [data-testid="stSidebar"] section { padding-top: 1.25rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.45; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #79e2cf; }
    h1, h2, h3 { color: #eaf4f7; letter-spacing: -.03em; }
    h1 { font-size: clamp(1.75rem, 2.7vw, 2.25rem) !important; font-weight: 720 !important; margin-bottom: .25rem !important; }
    h2 { font-size: clamp(1.25rem, 1.8vw, 1.55rem) !important; font-weight: 650 !important; margin-top: .45rem !important; }
    h3 { font-size: 1.08rem !important; font-weight: 620 !important; }
    .stApp p, .stApp label, .stApp li { color: #c5d4df; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #102735, #0d202d); border: 1px solid #244655; border-radius: 10px; padding: 12px 14px; min-height: 94px; box-shadow: 0 3px 12px rgba(0,0,0,.08); }
    [data-testid="stMetricLabel"] { color: #a8bfcc; font-size: .88rem; }
    [data-testid="stMetricValue"] { color: #e9f7f7; font-size: clamp(1.25rem, 1.9vw, 1.85rem); font-weight: 650; }
    [data-testid="stMetricDelta"] { font-size: .8rem; }
    .stButton > button, .stDownloadButton > button { background: #087f73; color: #fff; border: 1px solid #168f81; border-radius: 7px; font-weight: 600; min-height: 2.4rem; transition: background .15s ease, border-color .15s ease; box-shadow: none; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #0b9a89; border-color: #63e6d2; color: #fff; }
    [data-testid="stFileUploader"] { background: #0e2432; border: 1px dashed #356879; border-radius: 10px; padding: 4px; }
    [data-testid="stDataFrame"] { border: 1px solid #254657; border-radius: 8px; overflow: hidden; }
    [data-testid="stPlotlyChart"] { border: 1px solid rgba(36,70,85,.55); border-radius: 10px; padding: .35rem .35rem .1rem; background: rgba(13,32,45,.32); }
    [data-testid="stAlert"] { border-radius: 9px; border: 1px solid rgba(89,166,255,.16); }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius: 8px; }
    [data-testid="stExpander"] { border: 1px solid #203b4b; border-radius: 8px; background: rgba(13,32,45,.22); }
    [data-testid="stFileUploader"] section { padding: .7rem; }
    [data-testid="stAlert"] p { line-height: 1.5; }
    hr { border-color: #203b4b; margin: 1.2rem 0; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1.8rem; max-width: 1320px; }
    [data-testid="stVerticalBlock"] { gap: .72rem; }
    small, .stCaption { color: #8fa6b5 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================= PERSISTENT HUMAN REVIEW STORAGE =========================
def init_review_db():
    """Create the local SQLite review store if it does not already exist."""
    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS review_decisions (
                source_file TEXT NOT NULL,
                record_index INTEGER NOT NULL,
                predicted_class TEXT NOT NULL,
                model_score TEXT,
                review_status TEXT NOT NULL DEFAULT 'Pending review',
                analyst_decision TEXT NOT NULL DEFAULT 'Not decided',
                analyst_notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT,
                PRIMARY KEY (source_file, record_index, predicted_class)
            )"""
        )

def load_review(source_file, record_index, predicted_class):
    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        row = conn.execute(
            """SELECT review_status, analyst_decision, analyst_notes, updated_at
               FROM review_decisions
               WHERE source_file=? AND record_index=? AND predicted_class=?""",
            (str(source_file), int(record_index), str(predicted_class)),
        ).fetchone()
    if row:
        return {"Review Status": row[0], "Analyst Decision": row[1], "Analyst Notes": row[2], "Updated At": row[3] or ""}
    return {"Review Status": "Pending review", "Analyst Decision": "Not decided", "Analyst Notes": "", "Updated At": ""}

def load_review_history(limit=500):
    """Load saved decisions directly from SQLite, across app sessions and uploads."""
    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        rows = conn.execute(
            """SELECT source_file, record_index, predicted_class, model_score,
                      review_status, analyst_decision, analyst_notes, updated_at
               FROM review_decisions
               ORDER BY CASE WHEN updated_at IS NULL OR updated_at = '' THEN 1 ELSE 0 END,
                        updated_at DESC
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
    columns = [
        "Source file", "Record ID", "Predicted class", "Model score (%)",
        "Review status", "Analyst decision", "Analyst notes", "Updated at",
    ]
    return pd.DataFrame(rows, columns=columns)


def save_review(source_file, record_index, predicted_class, model_score, decision, notes):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(REVIEW_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO review_decisions
               (source_file, record_index, predicted_class, model_score, review_status,
                analyst_decision, analyst_notes, updated_at)
               VALUES (?, ?, ?, ?, 'Reviewed', ?, ?, ?)
               ON CONFLICT(source_file, record_index, predicted_class) DO UPDATE SET
                model_score=excluded.model_score, review_status='Reviewed',
                analyst_decision=excluded.analyst_decision,
                analyst_notes=excluded.analyst_notes, updated_at=excluded.updated_at""",
            (str(source_file), int(record_index), str(predicted_class), str(model_score), str(decision), str(notes), timestamp),
        )
    return timestamp

init_review_db()

# ========================= SESSION STATE =========================
if "latest_predictions" not in st.session_state:
    st.session_state["latest_predictions"] = None
if "prediction_filename" not in st.session_state:
    st.session_state["prediction_filename"] = None
if "multiclass_predictions" not in st.session_state:
    st.session_state["multiclass_predictions"] = None
if "multiclass_prediction_filename" not in st.session_state:
    st.session_state["multiclass_prediction_filename"] = None

# ========================= SIDEBAR NAVIGATION =========================
st.sidebar.title("WLAN Sentinel AI")
st.sidebar.caption("WIRELESS SECURITY CONSOLE")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Traffic Analysis", "Dataset Inspector", "Threat Alerts", "About"],
)
st.sidebar.divider()
st.sidebar.caption("ACADEMIC PROTOTYPE")
st.sidebar.caption("Synthetic data · Demo mode")

# ========================= DASHBOARD =========================
if page == "Dashboard":
    st.title("Security overview")
    st.caption("WLAN Sentinel AI  /  Analyst console")
    st.write("Analyze uploaded traffic datasets and review model-generated alerts.")
    st.warning(
        "Academic prototype: the multiclass model uses synthetic data. "
        "Predictions are demonstrations, not confirmed attacks or validated "
        "real-world WLAN detection results."
    )

    dashboard_results = st.session_state.get("multiclass_predictions")
    if dashboard_results is not None and "Multiclass Prediction" in dashboard_results.columns:
        predictions = dashboard_results["Multiclass Prediction"].astype(str)
        evaluated_mask = predictions != "Not evaluated"
        evaluated_count = int(evaluated_mask.sum())
        normal_count = int((predictions == "Normal").sum())
        other_count = int(((predictions != "Normal") & evaluated_mask).sum())
        total_count = len(dashboard_results)
        normal_pct = (normal_count / evaluated_count * 100) if evaluated_count else 0
        other_pct = (other_count / evaluated_count * 100) if evaluated_count else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Uploaded", f"{total_count:,}")
        m2.metric("Evaluated", f"{evaluated_count:,}")
        m3.metric("Normal", f"{normal_count:,}", f"{normal_pct:.1f}%")
        m4.metric("Flagged for review", f"{other_count:,}", f"{other_pct:.1f}%")

        st.subheader("Prediction distribution")
        st.caption(
            "Counts from the most recent multiclass run. Other simulated classes "
            "are not confirmed security incidents."
        )
        chart_counts = (
            predictions[evaluated_mask]
            .value_counts()
            .rename_axis("Predicted Class")
            .reset_index(name="Records")
            .sort_values("Records", ascending=True)
        )
        if not chart_counts.empty:
            dashboard_chart = px.bar(
                chart_counts,
                x="Records",
                y="Predicted Class",
                orientation="h",
                text="Records",
                color="Predicted Class",
                color_discrete_sequence=[
                    "#70d6a7", "#59a6ff", "#ffb15c", "#ff7078",
                    "#a6eedb", "#b49cff", "#e6c96f", "#83c5be"
                ],
            )
            dashboard_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf7",
                showlegend=False,
                margin=dict(l=10, r=25, t=12, b=8),
                height=350,
                xaxis_title="Number of records",
                yaxis_title="Predicted class",
            )
            dashboard_chart.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(dashboard_chart, use_container_width=True)
        st.caption(
            f"Latest source file: {st.session_state.get('multiclass_prediction_filename') or 'uploaded dataset'}"
        )
    else:
        st.info(
            "No multiclass results are available in this session yet. Open Traffic Analysis, "
            "upload synthetic_awid3_style_demo.csv, and click Run Multiclass Predictions. "
            "The Dashboard will then show the actual results from that run."
        )

    st.divider()
    left, right = st.columns([1, 1])
    with right:
        st.subheader("Prototype Scope")
        st.info(
            "This project demonstrates CSV inspection and ML classification on synthetic data. "
            "Live Wi-Fi packet capture, automatic blocking, and real-time monitoring are not implemented."
        )
        st.caption(
            "Official AWID3 data has not yet been used for validation. Do not interpret model outputs "
            "as confirmed real-world attacks."
        )

# ========================= TRAFFIC ANALYSIS =========================
elif page == "Traffic Analysis":
    st.title("Traffic Analysis")
    st.write(
        "Upload a manageable CSV file to inspect its structure and, when "
        "compatible, run the demonstration ML model."
    )
    st.info(
        "For larger files, use Dataset Inspector. This page loads the entire "
        "CSV into memory and is intended for smaller compatible files."
    )
    uploaded_file = st.file_uploader(
        "Upload a network traffic dataset",
        type=["csv"],
        key="traffic_analysis_uploader",
        help="The demo model requires duration, packet_count, and bytes_sent columns.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV file to begin inspecting network traffic.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV contains no records.")
            elif len(df.columns) == 0:
                st.error("The uploaded CSV contains no columns.")
            else:
                st.success("CSV uploaded and read successfully.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Records", f"{len(df):,}")
                col2.metric("Columns", f"{len(df.columns):,}")
                col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)
                with st.expander("Column Data Types"):
                    types_df = pd.DataFrame({
                        "Column": df.columns.astype(str),
                        "Data Type": [str(dtype) for dtype in df.dtypes],
                    })
                    st.dataframe(types_df, use_container_width=True)
                with st.expander("Missing Values by Column"):
                    missing_df = pd.DataFrame({
                        "Column": df.columns.astype(str),
                        "Missing Values": df.isna().sum().values,
                    })
                    st.dataframe(missing_df, use_container_width=True)
                with st.expander("Numerical Summary"):
                    numeric_df = df.select_dtypes(include="number")
                    if numeric_df.empty:
                        st.info("No numerical columns were found.")
                    else:
                        st.dataframe(numeric_df.describe().round(2), use_container_width=True)
                st.divider()
                st.subheader("ML Model Demonstration")
                st.warning(
                    "This model was trained on synthetic data, not a validated "
                    "WLAN dataset. Predictions are demonstrations, not confirmed attacks."
                )
                required_features = ["duration", "packet_count", "bytes_sent"]
                if not MODEL_PATH.exists():
                    st.error("Saved model not found. Run `python train_model.py`.")
                elif not all(column in df.columns for column in required_features):
                    st.info(
                        "This CSV is not compatible with the current demo model. "
                        "It needs duration, packet_count, and bytes_sent."
                    )
                else:
                    st.caption("A label column is not used as a model input.")
                    if st.button("Run ML Demo Predictions", key="run_ml_predictions"):
                        try:
                            saved_model = joblib.load(MODEL_PATH)
                            model = saved_model["model"]
                            inputs = df[required_features].copy()
                            for column in required_features:
                                inputs[column] = pd.to_numeric(inputs[column], errors="coerce")
                            valid_rows = inputs.notna().all(axis=1)
                            results = df.copy()
                            results["Demo Prediction"] = "Not evaluated"
                            if valid_rows.any():
                                results.loc[valid_rows, "Demo Prediction"] = model.predict(
                                    inputs.loc[valid_rows]
                                )
                            st.session_state["latest_predictions"] = results
                            st.session_state["prediction_filename"] = uploaded_file.name
                            st.success("Demonstration predictions completed.")
                        except Exception as error:
                            st.error(f"Could not run the model: {error}")
                latest = st.session_state["latest_predictions"]
                if latest is not None and st.session_state["prediction_filename"] == uploaded_file.name:
                    st.subheader("Prediction Results")
                    counts = latest["Demo Prediction"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", f"{len(latest):,}")
                    c2.metric("Demo Normal", int(counts.get("Normal", 0)))
                    c3.metric("Demo Suspicious", int(counts.get("Suspicious", 0)))
                    st.caption(
                        "These prototype predictions do not prove that real WLAN attacks were detected."
                    )
                    st.dataframe(latest, use_container_width=True)
                    st.download_button(
                        "Download Prediction Results CSV",
                        data=latest.to_csv(index=False).encode("utf-8"),
                        file_name="wlan_demo_predictions.csv",
                        mime="text/csv",
                    )

                # Separate multiclass workflow: uses its own model and encoder.
                st.divider()
                st.subheader("Multiclass WLAN Traffic Classification")
                st.write(
                    "This separate model predicts one of eight simulated traffic "
                    "classes using the numeric features it was trained on."
                )
                st.warning(
                    "Synthetic-data demonstration only. The predicted class and "
                    "model score are not verified real-world attack detections."
                )

                if not MULTICLASS_MODEL_PATH.exists() or not MULTICLASS_ENCODER_PATH.exists():
                    st.info(
                        "The multiclass model files were not found. Run "
                        "`python train_multiclass_model.py` in the project folder first."
                    )
                else:
                    try:
                        multiclass_bundle = joblib.load(MULTICLASS_MODEL_PATH)
                        multiclass_model = multiclass_bundle["model"]
                        feature_columns = multiclass_bundle["feature_columns"]
                        multiclass_encoder = joblib.load(MULTICLASS_ENCODER_PATH)

                        missing_features = [
                            column for column in feature_columns if column not in df.columns
                        ]

                        if missing_features:
                            st.info(
                                "This CSV does not contain the feature columns required "
                                "by the multiclass model. Upload "
                                "`synthetic_awid3_style_demo.csv` to try it. "
                                f"Missing columns: {', '.join(missing_features[:8])}"
                                + (" ..." if len(missing_features) > 8 else "")
                            )
                        else:
                            st.caption(
                                f"Compatible dataset detected: {len(feature_columns)} "
                                "numeric model features are available."
                            )
                            if st.button(
                                "Run Multiclass Predictions",
                                key="run_multiclass_predictions",
                            ):
                                inputs = df[feature_columns].copy()
                                for column in feature_columns:
                                    inputs[column] = pd.to_numeric(
                                        inputs[column], errors="coerce"
                                    )

                                valid_rows = inputs.notna().all(axis=1)
                                results = df.copy()
                                results["Multiclass Prediction"] = "Not evaluated"
                                results["Model Score (%)"] = pd.NA

                                if valid_rows.any():
                                    predicted_ids = multiclass_model.predict(
                                        inputs.loc[valid_rows]
                                    )
                                    predicted_labels = multiclass_encoder.inverse_transform(
                                        predicted_ids.astype(int)
                                    )
                                    results.loc[
                                        valid_rows, "Multiclass Prediction"
                                    ] = predicted_labels

                                    if hasattr(multiclass_model, "predict_proba"):
                                        probabilities = multiclass_model.predict_proba(
                                            inputs.loc[valid_rows]
                                        )
                                        results.loc[
                                            valid_rows, "Model Score (%)"
                                        ] = (
                                            probabilities.max(axis=1) * 100
                                        ).round(2)

                                st.session_state["multiclass_predictions"] = results
                                st.session_state["multiclass_prediction_filename"] = uploaded_file.name
                                st.success(
                                    "Multiclass demonstration predictions completed."
                                )

                            multiclass_latest = st.session_state["multiclass_predictions"]
                            if (
                                multiclass_latest is not None
                                and st.session_state["multiclass_prediction_filename"]
                                == uploaded_file.name
                            ):
                                st.subheader("Multiclass Prediction Results")
                                prediction_counts = (
                                    multiclass_latest["Multiclass Prediction"]
                                    .value_counts()
                                )
                                metric_cols = st.columns(4)
                                metric_cols[0].metric(
                                    "Rows", f"{len(multiclass_latest):,}"
                                )
                                metric_cols[1].metric(
                                    "Evaluated",
                                    int(
                                        (multiclass_latest["Multiclass Prediction"]
                                         != "Not evaluated").sum()
                                    ),
                                )
                                metric_cols[2].metric(
                                    "Normal",
                                    int(prediction_counts.get("Normal", 0)),
                                )
                                metric_cols[3].metric(
                                    "Flagged",
                                    int(
                                        sum(
                                            count
                                            for label, count in prediction_counts.items()
                                            if label not in ("Normal", "Not evaluated")
                                        )
                                    ),
                                )
                                st.caption(
                                    "Model Score is the model's highest class score, "
                                    "not a calibrated probability of a real attack."
                                )

                                # Prediction distribution chart.
                                st.subheader("Predicted Class Distribution")
                                class_counts = (
                                    multiclass_latest["Multiclass Prediction"]
                                    .value_counts(dropna=False)
                                    .rename_axis("Predicted Class")
                                    .reset_index(name="Records")
                                )
                                class_counts = class_counts[
                                    class_counts["Predicted Class"] != "Not evaluated"
                                ]
                                if not class_counts.empty:
                                    chart = px.bar(
                                        class_counts.sort_values("Records", ascending=True),
                                        x="Records",
                                        y="Predicted Class",
                                        orientation="h",
                                        text="Records",
                                        title="Number of records assigned to each simulated class",
                                        color="Predicted Class",
                                        color_discrete_sequence=[
                                            "#70d6a7", "#59a6ff", "#ffb15c",
                                            "#ff7078", "#a6eedb", "#b49cff",
                                            "#e6c96f", "#83c5be"
                                        ],
                                    )
                                    chart.update_layout(
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#e5edf7",
                                        showlegend=False,
                                        margin=dict(l=10, r=20, t=35, b=8),
                    height=320,
                                        xaxis_title="Number of records",
                                        yaxis_title="Predicted class",
                                    )
                                    chart.update_traces(textposition="outside", cliponaxis=False)
                                    st.plotly_chart(chart, use_container_width=True)

                                # Explicit normal vs non-normal summary.
                                evaluated_mask = multiclass_latest[
                                    "Multiclass Prediction"
                                ] != "Not evaluated"
                                normal_count = int(
                                    (
                                        multiclass_latest["Multiclass Prediction"]
                                        == "Normal"
                                    ).sum()
                                )
                                non_normal_mask = (
                                    multiclass_latest["Multiclass Prediction"].notna()
                                    & (multiclass_latest["Multiclass Prediction"] != "Normal")
                                    & (
                                        multiclass_latest["Multiclass Prediction"]
                                        != "Not evaluated"
                                    )
                                )
                                non_normal_count = int(non_normal_mask.sum())
                                evaluated_count = int(evaluated_mask.sum())
                                st.subheader("Prediction summary")
                                summary_cols = st.columns(3)
                                summary_cols[0].metric("Evaluated rows", f"{evaluated_count:,}")
                                summary_cols[1].metric(
                                    "Normal predictions",
                                    f"{normal_count:,}",
                                    f"{(normal_count / evaluated_count * 100):.1f}%"
                                    if evaluated_count else None,
                                )
                                summary_cols[2].metric(
                                    "Other simulated classes",
                                    f"{non_normal_count:,}",
                                    f"{(non_normal_count / evaluated_count * 100):.1f}%"
                                    if evaluated_count else None,
                                )
                                st.info(
                                    "“Other simulated classes” means predictions other than "
                                    "Normal. It does not mean these rows are confirmed attacks."
                                )

                                # Filterable table for non-normal simulated classes.
                                st.subheader("Flagged records")
                                st.caption(
                                    "These are rows assigned a non-Normal simulated class; "
                                    "review them as examples, not confirmed incidents."
                                )
                                if non_normal_count:
                                    st.dataframe(
                                        multiclass_latest.loc[non_normal_mask].head(500),
                                        use_container_width=True,
                                    )
                                    if non_normal_count > 500:
                                        st.caption(
                                            "Showing the first 500 non-Normal rows. "
                                            "The full results download includes every row."
                                        )
                                else:
                                    st.success(
                                        "No rows were assigned a non-Normal simulated class."
                                    )

                                with st.expander("View all prediction rows"):
                                    st.dataframe(
                                        multiclass_latest.head(500),
                                        use_container_width=True,
                                    )
                                    if len(multiclass_latest) > 500:
                                        st.caption(
                                            "Showing the first 500 rows in this table; "
                                            "the download contains all rows."
                                        )

                                st.download_button(
                                    "Download Multiclass Results CSV",
                                    data=multiclass_latest.to_csv(index=False).encode("utf-8"),
                                    file_name="wlan_multiclass_synthetic_predictions.csv",
                                    mime="text/csv",
                                    key="download_multiclass_results",
                                )
                    except Exception as error:
                        st.error(f"Could not run multiclass predictions: {error}")

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"The uploaded file could not be read as a valid CSV: {error}")
        except Exception as error:
            st.error(f"An unexpected error occurred: {error}")

# ========================= DATASET INSPECTOR =========================
elif page == "Dataset Inspector":
    st.title("Large Dataset Inspector")
    st.write(
        "Inspect CSV files chunk by chunk. Only a small preview and a bounded "
        "numeric sample are retained, rather than every chunk."
    )
    st.warning(
        "Streamlit buffers uploaded files in memory. Chunked Pandas processing "
        "reduces additional DataFrame memory, but does not remove upload buffering."
    )
    chunk_size = st.slider(
        "Rows per processing chunk",
        min_value=MIN_CHUNK_SIZE,
        max_value=MAX_CHUNK_SIZE,
        value=DEFAULT_CHUNK_SIZE,
        step=1_000,
        help="Use smaller chunks if memory is limited.",
    )
    st.caption(
        f"Current chunk size: {chunk_size:,} rows. Inspection does not train "
        "the synthetic model or assume AWID3 column names."
    )
    large_file = st.file_uploader(
        "Choose a CSV dataset for chunked inspection",
        type=["csv"],
        key="large_dataset_uploader",
        help="Start with sample_traffic.csv to verify the inspector.",
    )
    if large_file is not None:
        file_size_mb = large_file.size / (1024 * 1024)
        st.write(f"**File:** {large_file.name}")
        st.write(f"**Uploaded size:** {file_size_mb:.2f} MB")
        if file_size_mb > 500:
            st.warning(
                "This is a large upload. Close unnecessary applications and "
                "ensure enough free memory before inspecting it."
            )
        if st.button("Inspect Dataset in Chunks", key="inspect_large_dataset"):
            try:
                large_file.seek(0)
                chunk_iterator = pd.read_csv(
                    large_file, chunksize=chunk_size, low_memory=True
                )
                total_rows = 0
                chunk_count = 0
                missing_counts = None
                column_names = None
                dtype_names = None
                preview_df = None
                numeric_sample_parts = []
                numeric_sample_row_count = 0
                progress = st.progress(0, text="Reading CSV chunks...")
                status = st.empty()

                for chunk in chunk_iterator:
                    chunk_count += 1
                    total_rows += len(chunk)
                    if column_names is None:
                        column_names = list(chunk.columns)
                        dtype_names = {
                            str(column): str(dtype)
                            for column, dtype in chunk.dtypes.items()
                        }
                        missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")
                    missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)

                    if preview_df is None:
                        preview_df = chunk.head(PREVIEW_ROWS).copy()
                    elif len(preview_df) < PREVIEW_ROWS:
                        remaining = PREVIEW_ROWS - len(preview_df)
                        preview_df = pd.concat(
                            [preview_df, chunk.head(remaining)], ignore_index=True
                        )

                    if numeric_sample_row_count < MAX_NUMERIC_SAMPLE_ROWS:
                        numeric_chunk = chunk.select_dtypes(include="number")
                        if not numeric_chunk.empty:
                            remaining_sample = MAX_NUMERIC_SAMPLE_ROWS - numeric_sample_row_count
                            sample_part = numeric_chunk.head(remaining_sample).copy()
                            numeric_sample_parts.append(sample_part)
                            numeric_sample_row_count += len(sample_part)

                    status.write(
                        f"Processed {chunk_count:,} chunk(s) — {total_rows:,} rows so far."
                    )
                    # Total chunks are unknown until EOF, so avoid a misleading
                    # percentage. This bar is activity feedback, finalized at EOF.
                    progress.progress(
                        min(0.95, 0.05 + (chunk_count % 18) / 20),
                        text=f"Processed {total_rows:,} rows...",
                    )
                    del chunk

                progress.progress(1.0, text="Inspection complete.")
                status.success("Chunked inspection completed.")

                if total_rows == 0 or column_names is None:
                    st.error("The CSV contains no data rows.")
                else:
                    st.success("Dataset inspection completed successfully.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Records", f"{total_rows:,}")
                    c2.metric("Columns", f"{len(column_names):,}")
                    c3.metric("Chunks Read", f"{chunk_count:,}")
                    c4.metric("Total Missing Cells", f"{int(missing_counts.sum()):,}")
                    st.subheader("Dataset Preview")
                    st.dataframe(preview_df, use_container_width=True)
                    st.subheader("Column Summary")
                    summary_df = pd.DataFrame({
                        "Column": [str(column) for column in column_names],
                        "First-Chunk Data Type": [
                            dtype_names.get(str(column), "unknown")
                            for column in column_names
                        ],
                        "Missing Values": [
                            int(missing_counts.get(column, 0))
                            for column in column_names
                        ],
                    })
                    summary_df["Missing %"] = (
                        summary_df["Missing Values"] / total_rows * 100
                    ).round(2)
                    st.dataframe(summary_df, use_container_width=True)
                    with st.expander("Numeric Summary (bounded sample)"):
                        if numeric_sample_parts:
                            sample_df = pd.concat(numeric_sample_parts, ignore_index=True)
                            st.caption(
                                f"This summary uses at most {MAX_NUMERIC_SAMPLE_ROWS:,} "
                                "numeric rows, not the full dataset."
                            )
                            st.dataframe(sample_df.describe().round(3), use_container_width=True)
                        else:
                            st.info("No numeric columns were found in the sampled chunks.")
                    st.info(
                        "Inspection only: no model was trained and no intrusion "
                        "predictions were made for this dataset."
                    )
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
                st.error(
                    "Could not inspect this CSV. Check its encoding, delimiter, "
                    f"and format. Details: {error}"
                )
            except Exception as error:
                st.error(f"Dataset inspection failed: {error}")

# ========================= THREAT ALERTS =========================
elif page == "Threat Alerts":
    st.title("Threat Alerts")
    st.caption("Review model-flagged records and record a human decision.")

    latest = st.session_state.get("latest_predictions")
    multiclass_alerts = st.session_state.get("multiclass_predictions")

    # Only show the empty-state message when neither prediction workflow has results.
    if latest is None and multiclass_alerts is None:
        st.info("No predictions yet. Go to **Traffic Analysis**, upload `synthetic_awid3_style_demo.csv`, and click **Run Multiclass Predictions**.")

    # Keep the older three-feature demo results available, but don't let its empty-state
    # message appear when multiclass predictions already exist.
    if latest is not None and "Demo Prediction" in latest.columns:
        with st.expander("Older basic demo-model results"):
            st.caption("Optional legacy demo. The multiclass review queue below is the main workflow.")
            st.dataframe(latest, use_container_width=True, hide_index=True)

    if multiclass_alerts is not None:
        st.divider()
        st.subheader("Flagged records")
        st.caption("Synthetic demo predictions · Flags are not confirmed attacks.")
        if "Multiclass Prediction" in multiclass_alerts.columns:
            multiclass_suspicious = multiclass_alerts[
                (multiclass_alerts["Multiclass Prediction"] != "Normal")
                & (multiclass_alerts["Multiclass Prediction"] != "Not evaluated")
            ].copy()

            # Explainable, rule-based triage labels for the demo. These are NOT
            # calibrated risk scores or claims about real-world attack severity.
            high_priority_classes = {"Deauth", "Evil_Twin", "KRACK_like", "KR00K_like"}
            medium_priority_classes = {"Malware_like", "SQL_Injection", "SSDP"}

            def _flag_reason(predicted_class):
                return (
                    f"Flag rule triggered: the model predicted '{predicted_class}', "
                    "which is not the 'Normal' class. This requests human review; "
                    "it does not confirm an attack."
                )

            def _prototype_priority(predicted_class):
                if predicted_class in high_priority_classes:
                    return "High (prototype triage)"
                if predicted_class in medium_priority_classes:
                    return "Medium (prototype triage)"
                return "Review (prototype triage)"

            if not multiclass_suspicious.empty:
                multiclass_suspicious["Flag Status"] = "🚩 Flagged for review"
                multiclass_suspicious["Flag Reason"] = multiclass_suspicious[
                    "Multiclass Prediction"
                ].map(_flag_reason)
                multiclass_suspicious["Prototype Priority"] = multiclass_suspicious[
                    "Multiclass Prediction"
                ].map(_prototype_priority)

            a1, a2, a3 = st.columns(3)
            a1.metric("Multiclass Records", f"{len(multiclass_alerts):,}")
            a2.metric("Flagged", f"{len(multiclass_suspicious):,}")
            a3.metric(
                "Not Evaluated",
                int(
                    (multiclass_alerts["Multiclass Prediction"] == "Not evaluated").sum()
                ),
            )
            if not multiclass_suspicious.empty:
                alert_counts = (
                    multiclass_alerts["Multiclass Prediction"]
                    .value_counts()
                    .drop(labels=["Not evaluated"], errors="ignore")
                    .rename_axis("Predicted Class")
                    .reset_index(name="Records")
                )
                alert_chart = px.bar(
                    alert_counts.sort_values("Records", ascending=True),
                    x="Records",
                    y="Predicted Class",
                    orientation="h",
                    text="Records",
                    title="All simulated prediction classes",
                    color="Predicted Class",
                    color_discrete_sequence=[
                        "#70d6a7", "#59a6ff", "#ffb15c", "#ff7078",
                        "#a6eedb", "#b49cff", "#e6c96f", "#83c5be"
                    ],
                )
                alert_chart.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5edf7",
                    showlegend=False,
                    margin=dict(l=10, r=20, t=35, b=8),
                    height=320,
                )
                alert_chart.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(alert_chart, use_container_width=True)
                st.subheader("Alert queue")
                st.caption("Non-Normal predictions are queued for review. Priority is for demo sorting only.")
                filter_col1, filter_col2 = st.columns([1, 1])
                with filter_col1:
                    priority_filter = st.selectbox(
                        "Filter by prototype priority",
                        ["All priorities", "High (prototype triage)", "Medium (prototype triage)", "Review (prototype triage)"],
                        key="flag_priority_filter",
                    )
                with filter_col2:
                    class_options = ["All predicted classes"] + sorted(
                        multiclass_suspicious["Multiclass Prediction"].astype(str).unique().tolist()
                    )
                    class_filter = st.selectbox(
                        "Filter by predicted class", class_options, key="flag_class_filter"
                    )
                queue_view = multiclass_suspicious
                if priority_filter != "All priorities":
                    queue_view = queue_view[queue_view["Prototype Priority"] == priority_filter]
                if class_filter != "All predicted classes":
                    queue_view = queue_view[queue_view["Multiclass Prediction"].astype(str) == class_filter]
                st.caption(f"Showing {len(queue_view):,} of {len(multiclass_suspicious):,} flagged records.")
                preferred_columns = [
                    "Flag Status", "Prototype Priority", "Flag Reason",
                    "Multiclass Prediction", "Model Score (%)"
                ]
                visible_columns = [c for c in preferred_columns if c in queue_view.columns]
                other_columns = [c for c in queue_view.columns if c not in visible_columns]
                compact_columns = [c for c in ["Multiclass Prediction", "Model Score (%)", "Prototype Priority", "Flag Status"] if c in queue_view.columns]
                st.dataframe(
                    queue_view[compact_columns].head(100),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Showing up to 100 records here. Use the selector below to review one record at a time.")
                st.download_button(
                    "Download Multiclass Alert Rows",
                    data=multiclass_suspicious.to_csv(index=False).encode("utf-8"),
                    file_name="wlan_multiclass_synthetic_alerts.csv",
                    mime="text/csv",
                    key="download_multiclass_alert_rows",
                )

                # Human-in-the-loop review: flags are predictions, not confirmed attacks.
                st.divider()
                st.subheader("Review a record")
                st.caption("Choose a record, make a decision, and save it to review history.")

                source_file = st.session_state.get("multiclass_prediction_filename") or "uploaded_dataset.csv"
                review_records = {}
                flagged_indices = list(multiclass_suspicious.index)
                for record_idx in flagged_indices:
                    predicted_for_row = multiclass_suspicious.loc[record_idx, "Multiclass Prediction"]
                    review_records[record_idx] = load_review(source_file, record_idx, predicted_for_row)
                st.session_state["human_review_records"] = review_records

                pending_count = sum(
                    review_records[idx]["Review Status"] == "Pending review"
                    for idx in flagged_indices
                )
                reviewed_count = sum(
                    review_records[idx]["Review Status"] == "Reviewed"
                    for idx in flagged_indices
                )
                r1, r2, r3 = st.columns(3)
                r1.metric("Flagged", f"{len(flagged_indices):,}")
                r2.metric("Pending", f"{pending_count:,}")
                r3.metric("Reviewed", f"{reviewed_count:,}")

                st.caption(
                    "Flag rule: predicted class is not Normal. A flag is only a prompt "
                    "for investigation and does not confirm malicious activity."
                )
                if flagged_indices:
                    selected_idx = st.selectbox(
                        "Choose a flagged record to review",
                        options=flagged_indices,
                        format_func=lambda idx: (
                            f"Record {idx} — "
                            f"{multiclass_suspicious.loc[idx, 'Multiclass Prediction']}"
                        ),
                        key="human_review_selected_record",
                    )
                    selected_record = multiclass_suspicious.loc[selected_idx]
                    st.markdown(f"### Case review: Record {selected_idx}")
                    st.caption("Inspect the summary and optional traffic details, then record your decision.")
                    summary_cols = st.columns(3)
                    summary_cols[0].metric(
                        "Predicted class",
                        str(selected_record.get("Multiclass Prediction", "Unknown")),
                    )
                    score_value = selected_record.get("Model Score (%)", pd.NA)
                    if pd.notna(score_value):
                        summary_cols[1].metric("Model score", f"{float(score_value):.2f}%")
                    else:
                        summary_cols[1].metric("Model score", "Not available")
                    summary_cols[2].metric("Flag status", "Flagged for review")

                    st.info(
                        f"Reason for review: the model predicted **{selected_record.get('Multiclass Prediction', 'Unknown')}** instead of **Normal**. "
                        "This is a model-generated flag, not confirmation of an attack."
                    )

                    # Keep technical details available without crowding the main review form.
                    try:
                        model_bundle_for_review = joblib.load(MULTICLASS_MODEL_PATH)
                        review_feature_columns = model_bundle_for_review.get("feature_columns", [])
                    except Exception:
                        review_feature_columns = []

                    available_review_features = [
                        column for column in review_feature_columns
                        if column in multiclass_suspicious.columns
                    ]
                    with st.expander("View traffic features"):
                        if available_review_features:
                            feature_values = multiclass_suspicious.loc[
                                [selected_idx], available_review_features
                            ].T.reset_index()
                            feature_values.columns = ["Feature", "Value"]
                            st.dataframe(feature_values, use_container_width=True, hide_index=True)
                        else:
                            st.caption("Feature values are unavailable for this record.")
                    with st.expander("View complete record"):
                        st.dataframe(
                            multiclass_suspicious.loc[[selected_idx]],
                            use_container_width=True,
                            hide_index=True,
                        )

                    existing_review = review_records[selected_idx]
                    decision_options = [
                        "Needs further investigation",
                        "Likely benign",
                        "Escalate for investigation",
                    ]
                    default_decision = existing_review.get("Analyst Decision", "Not decided")
                    default_index = (
                        decision_options.index(default_decision)
                        if default_decision in decision_options else 0
                    )
                    st.markdown("**Your decision**")
                    analyst_decision = st.selectbox(
                        "Decision",
                        options=decision_options,
                        index=default_index,
                        format_func=lambda value: {
                            "Needs further investigation": "Investigate",
                            "Likely benign": "Likely benign",
                            "Escalate for investigation": "Escalate",
                        }.get(value, value),
                        key=f"human_review_decision_{selected_idx}",
                        label_visibility="collapsed",
                    )
                    analyst_notes = st.text_area(
                        "Notes (optional)",
                        value=existing_review.get("Analyst Notes", ""),
                        key=f"human_review_notes_{selected_idx}",
                        placeholder="Add a short note, or leave blank.",
                        height=80,
                    )
                    if st.button("Save decision", type="primary", key=f"save_human_review_{selected_idx}"):
                        selected_score = selected_record.get("Model Score (%)", "")
                        updated_at = save_review(
                            source_file, selected_idx,
                            selected_record.get("Multiclass Prediction", "Unknown"),
                            selected_score, analyst_decision, analyst_notes,
                        )
                        review_records[selected_idx] = {
                            "Review Status": "Reviewed",
                            "Analyst Decision": analyst_decision,
                            "Analyst Notes": analyst_notes,
                            "Updated At": updated_at,
                        }
                        st.session_state["human_review_records"] = review_records
                        st.success("Decision saved to SQLite review history.")
                        st.rerun()

                    review_export = multiclass_suspicious.copy()
                    review_export["Review Status"] = [
                        review_records[idx]["Review Status"] for idx in review_export.index
                    ]
                    review_export["Analyst Decision"] = [
                        review_records[idx]["Analyst Decision"] for idx in review_export.index
                    ]
                    review_export["Analyst Notes"] = [
                        review_records[idx]["Analyst Notes"] for idx in review_export.index
                    ]
                    st.subheader("Review history")
                    history_df = load_review_history(limit=5000)
                    if history_df.empty:
                        st.caption("No decisions have been saved yet.")
                    else:
                        reviewed_total = int((history_df["Review status"] == "Reviewed").sum())
                        pending_total = int((history_df["Review status"] != "Reviewed").sum())
                        h1, h2, h3 = st.columns(3)
                        h1.metric("Saved records", f"{len(history_df):,}")
                        h2.metric("Reviewed", f"{reviewed_total:,}")
                        h3.metric("Pending", f"{pending_total:,}")

                        history_filter = st.selectbox(
                            "Show history",
                            ["All saved records", "Reviewed only", "Pending only"],
                            key="review_history_filter",
                        )
                        visible_history = history_df.copy()
                        if history_filter == "Reviewed only":
                            visible_history = visible_history[
                                visible_history["Review status"] == "Reviewed"
                            ]
                        elif history_filter == "Pending only":
                            visible_history = visible_history[
                                visible_history["Review status"] != "Reviewed"
                            ]

                        # Keep the main table narrow; show notes in a separate detail view.
                        compact_history = visible_history[[
                            "Record ID", "Predicted class", "Review status",
                            "Analyst decision", "Updated at"
                        ]].copy()
                        st.dataframe(
                            compact_history,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Record ID": st.column_config.NumberColumn("Record", width="small"),
                                "Predicted class": st.column_config.TextColumn("Class", width="medium"),
                                "Review status": st.column_config.TextColumn("Status", width="small"),
                                "Analyst decision": st.column_config.TextColumn("Decision", width="medium"),
                                "Updated at": st.column_config.TextColumn("Updated", width="medium"),
                            },
                        )
                        with st.expander("View saved notes and source files"):
                            st.dataframe(
                                visible_history[[
                                    "Source file", "Record ID", "Analyst notes",
                                    "Model score (%)", "Updated at"
                                ]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        st.download_button(
                            "Download complete review history (CSV)",
                            data=history_df.to_csv(index=False).encode("utf-8"),
                            file_name="wlan_human_review_history.csv",
                            mime="text/csv",
                            key="download_human_review_log",
                        )

# ========================= ABOUT =========================
elif page == "About":
    st.title("About WLAN Sentinel AI")
    st.write(
        """
        **WLAN Sentinel AI** is a prototype for demonstrating an AI-assisted
        wireless network traffic analysis workflow.
        """
    )
    st.subheader("Current Features")
    st.markdown(
        """
        - CSV traffic dataset upload and inspection
        - Memory-aware chunked large-CSV inspection
        - Missing-value and numerical-summary analysis
        - Random Forest demonstration predictions for compatible small CSVs
        - Separate eight-class synthetic multiclass prediction workflow
        - Prediction results tables and CSV export
        - Human-in-the-loop review with SQLite persistence
        - Review history and CSV export
        - Dark navy and teal cybersecurity interface
        """
    )
    st.subheader("Current Limitations")
    st.markdown(
        """
        - The model is trained on synthetic demonstration data.
        - Real WLAN dataset validation has not been completed.
        - Detection accuracy on real traffic is unknown.
        - Chunked inspection reduces extra DataFrame memory, but Streamlit
          still buffers the uploaded file.
        - Large uploads may exceed available RAM or processing time.
        - Live Wi-Fi packet capture is not implemented.
        - Automated threat response or network blocking is not implemented.
        - SQLite review storage is local to the app filesystem; hosted ephemeral filesystems may reset.
        """
    )
    st.subheader("Technology Stack")
    st.markdown(
        """
        - Python and Streamlit
        - Pandas and NumPy
        - Scikit-learn Random Forest
        - Plotly
        """
    )
    st.info(
        "Future work: inspect and preprocess an appropriate real WLAN dataset, "
        "train and evaluate a dataset-specific model, measure false positives "
        "and false negatives, and document limitations."
    )
