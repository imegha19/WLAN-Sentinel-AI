from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "wlan_demo_model.joblib"
MULTICLASS_MODEL_PATH = PROJECT_DIR / "wlan_multiclass_synthetic_model.joblib"
MULTICLASS_ENCODER_PATH = PROJECT_DIR / "wlan_multiclass_label_encoder.joblib"

DEFAULT_CHUNK_SIZE = 10_000
MIN_CHUNK_SIZE = 1_000
MAX_CHUNK_SIZE = 50_000
PREVIEW_ROWS = 20
MAX_NUMERIC_SAMPLE_ROWS = 5_000

st.set_page_config(
    page_title="WLAN Sentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= FRONTEND STYLING =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #091521 0%, #10252d 100%);
        color: #e8f1f2;
    }
    [data-testid="stSidebar"] {
        background: #0d1d28;
        border-right: 1px solid #28414a;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #9ce8cf; }
    h1, h2, h3 { color: #a6ead5; letter-spacing: -0.4px; }
    .stApp p, .stApp label, .stApp li { color: #d9e5e9; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #162d3a, #132530);
        border: 1px solid #31515a;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
    }
    [data-testid="stMetricLabel"] { color: #b9d0d5; }
    [data-testid="stMetricValue"] { color: #f0faf7; }
    .stButton > button, .stDownloadButton > button {
        background: #147d70;
        color: #ffffff;
        border: 1px solid #35a895;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #1b9b87;
        border-color: #8ae3cb;
        color: #ffffff;
    }
    [data-testid="stFileUploader"] {
        background: rgba(22, 45, 58, 0.65);
        border: 1px dashed #4b827e;
        border-radius: 12px;
        padding: 10px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #29434e;
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 10px; }
    hr { border-color: #29434e; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
st.sidebar.title("🛡️ WLAN Sentinel AI")
st.sidebar.caption("AI-Based Wireless LAN Security")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Traffic Analysis", "Dataset Inspector", "Threat Alerts", "About"],
)
st.sidebar.divider()
st.sidebar.caption("Academic ML prototype")
st.sidebar.caption("Synthetic-data model — not real-world validated")

# ========================= DASHBOARD =========================
if page == "Dashboard":
    st.title("🛡️ WLAN Sentinel AI")
    st.subheader("AI-Based Intrusion Detection for Wireless LAN Security")
    st.write(
        "Explore network traffic, inspect uploaded datasets, and demonstrate "
        "a machine-learning prediction workflow."
    )
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
        m1.metric("Records uploaded", f"{total_count:,}")
        m2.metric("Rows evaluated", f"{evaluated_count:,}")
        m3.metric("Normal predictions", f"{normal_count:,}", f"{normal_pct:.1f}%")
        m4.metric("Other simulated classes", f"{other_count:,}", f"{other_pct:.1f}%")

        st.subheader("Latest Prediction Distribution")
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
                margin=dict(l=10, r=25, t=20, b=10),
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
    with left:
        st.subheader("System Workflow")
        st.markdown(
            """
            **01 · Upload traffic data**

            Import a CSV file containing network traffic features.

            **02 · Inspect and prepare**

            Review columns, data types, missing values, and numerical summaries.

            **03 · Run the ML demonstration**

            Apply the saved Random Forest model to compatible feature columns.

            **04 · Review results**

            Explore predicted classes, inspect flagged rows, and download a results CSV.
            """
        )
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
                                    "Other Simulated Classes",
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
                                        margin=dict(l=10, r=20, t=55, b=10),
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
                                st.subheader("Normal vs. Other Simulated Predictions")
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
                                st.subheader("Rows Predicted as Non-Normal")
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
    st.write("Review demonstration predictions generated from an uploaded CSV.")
    st.warning(
        "These are model-generated demonstration labels, not verified security "
        "incidents or confirmed WLAN intrusions."
    )
    latest = st.session_state["latest_predictions"]
    if latest is None:
        st.info(
            "No prediction results are available yet. Open Traffic Analysis, "
            "upload a compatible CSV, and click Run ML Demo Predictions."
        )
    else:
        st.caption(f"Source file: {st.session_state['prediction_filename']}")
        if "Demo Prediction" in latest.columns:
            suspicious_rows = latest[latest["Demo Prediction"] == "Suspicious"]
            not_evaluated = latest[latest["Demo Prediction"] == "Not evaluated"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Records", f"{len(latest):,}")
            col2.metric("Demo Suspicious Predictions", f"{len(suspicious_rows):,}")
            col3.metric("Not Evaluated", f"{len(not_evaluated):,}")
            if suspicious_rows.empty:
                st.info(
                    "The model did not label any evaluated rows as Suspicious "
                    "in this demonstration run."
                )
            else:
                st.subheader("Rows Labelled Suspicious by the Demo Model")
                st.dataframe(suspicious_rows, use_container_width=True)
            with st.expander("View All Prediction Results"):
                st.dataframe(latest, use_container_width=True)
        else:
            st.info("Results do not contain prediction labels. Run the model again.")

    multiclass_alerts = st.session_state.get("multiclass_predictions")
    if multiclass_alerts is not None:
        st.divider()
        st.subheader("Multiclass Demonstration Results")
        st.caption(
            "These labels come from the synthetic-data multiclass model and are "
            "not confirmed security incidents."
        )
        if "Multiclass Prediction" in multiclass_alerts.columns:
            multiclass_suspicious = multiclass_alerts[
                (multiclass_alerts["Multiclass Prediction"] != "Normal")
                & (multiclass_alerts["Multiclass Prediction"] != "Not evaluated")
            ]
            a1, a2, a3 = st.columns(3)
            a1.metric("Multiclass Records", f"{len(multiclass_alerts):,}")
            a2.metric("Non-Normal Simulated Labels", f"{len(multiclass_suspicious):,}")
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
                    margin=dict(l=10, r=20, t=55, b=10),
                )
                alert_chart.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(alert_chart, use_container_width=True)
                st.dataframe(multiclass_suspicious.head(500), use_container_width=True)
                st.download_button(
                    "Download Multiclass Alert Rows",
                    data=multiclass_suspicious.to_csv(index=False).encode("utf-8"),
                    file_name="wlan_multiclass_synthetic_alerts.csv",
                    mime="text/csv",
                    key="download_multiclass_alert_rows",
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
        - Review of demonstration labels on the Threat Alerts page
        - Dark teal cybersecurity-themed interface
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
