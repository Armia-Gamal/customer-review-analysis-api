import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
from threading import Thread
import io

API_URL = "https://armia-gamal-customer-review-analysis-gpu.hf.space/analyze"

st.set_page_config(
    page_title="Customer Review Analysis",
    layout="wide"
)

# =========================
# Session State
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# =========================
# Sidebar
# =========================
st.sidebar.title("Review Analysis")

uploaded_file = st.sidebar.file_uploader(
    "Upload Reviews File (CSV)",
    type=["csv"]
)

st.sidebar.markdown("---")

selected_view = st.sidebar.radio(
    "Choose Analysis View",
    (
        "Sentiment Distribution",
        "Severity Distribution",
        "Top Issues"
    )
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### About This App
This AI-powered system analyzes customer reviews using LLM's.

It performs:
- Sentiment classification
- Automated issue extraction
- Issue frequency analysis
- Severity classification

Developed by **Armia Gamal** and **Sara Essam**
""")

# =========================
# Main Header
# =========================
st.title("Customer Review Analysis Dashboard")

st.write(
    "Upload a dataset containing a 'review_text' column."
)

# =========================
# Auto Run When File Uploaded
# =========================
if uploaded_file is not None:

    if st.session_state.last_uploaded_name != uploaded_file.name:

        st.session_state.analysis_done = False
        st.session_state.analysis_data = None
        st.session_state.last_uploaded_name = uploaded_file.name

        try:
            df_preview = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception:
                st.error("Unable to read CSV file.")
                st.stop()

        if "review_text" not in df_preview.columns:
            st.error("CSV must contain a 'review_text' column.")
            st.stop()

        # =========================
        # Clean Data
        # =========================
        df_preview = df_preview.dropna(subset=["review_text"])
        df_preview["review_text"] = df_preview["review_text"].astype(str)
        df_preview = df_preview[df_preview["review_text"].str.strip() != ""]

        if len(df_preview) == 0:
            st.error("No valid reviews found after cleaning.")
            st.stop()

        num_reviews = len(df_preview)

        estimated_speed = 25
        estimated_time = max(5, int(num_reviews / estimated_speed))

        st.subheader("Processing Status")

        progress_bar = st.progress(0)
        status_text = st.empty()
        eta_text = st.empty()

        api_response = {}

        def call_api():
            try:
                # Convert cleaned dataframe to CSV in memory
                csv_buffer = io.StringIO()
                df_preview.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                files = {
                    "file": (
                        "clean_reviews.csv",
                        csv_bytes,
                        "text/csv"
                    )
                }

                r = requests.post(
                    API_URL,
                    files=files,
                    timeout=300
                )

                api_response["response"] = r

            except Exception as e:
                api_response["error"] = str(e)

        thread = Thread(target=call_api)
        thread.start()

        start_time = time.time()

        while thread.is_alive():
            elapsed = time.time() - start_time
            progress = min(elapsed / estimated_time, 0.95)
            remaining = max(0, estimated_time - elapsed)

            progress_bar.progress(progress)
            status_text.text(f"Processing reviews ({int(progress * 100)}%)")
            eta_text.text(f"Estimated time remaining: {int(remaining)} seconds")

            time.sleep(1)

        thread.join()

        progress_bar.progress(1.0)
        status_text.text("Analysis completed")
        eta_text.text("Estimated time remaining: 0 seconds")

        # =========================
        # Handle Errors
        # =========================
        if "error" in api_response:
            st.error("API connection failed.")
            st.code(api_response["error"])
            st.stop()

        response = api_response.get("response")

        if response is None:
            st.error("API did not return a response.")
            st.stop()

        if response.status_code != 200:
            st.error("Analysis API failed.")
            st.code(response.text)
            st.stop()

        try:
            st.session_state.analysis_data = response.json()
        except Exception:
            st.error("Invalid JSON returned from API.")
            st.code(response.text)
            st.stop()

        st.session_state.analysis_done = True
        st.success("Analysis completed successfully")

# =========================
# Show Visualizations
# =========================
if st.session_state.analysis_done:

    data = st.session_state.analysis_data

    sentiment_df = pd.DataFrame(
        data.get("summary", {}).get("sentiment_breakdown", [])
    )

    issues_df = pd.DataFrame(
        data.get("issue_analysis", {}).get("issues", [])
    )

    total_reviews = data.get("summary", {}).get("total_reviews", 0)
    total_negative_reviews = (
        issues_df["count"].sum() if not issues_df.empty else 0
    )
    total_issue_categories = len(issues_df)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Negative Reviews", int(total_negative_reviews))
    col3.metric("Issue Categories", int(total_issue_categories))

    st.markdown("---")

    if selected_view == "Sentiment Distribution":

        st.header("Sentiment Distribution")

        if not sentiment_df.empty:
            fig = plt.figure()
            plt.pie(
                sentiment_df["percentage"],
                labels=sentiment_df["sentiment"],
                autopct="%1.1f%%"
            )
            plt.axis("equal")
            st.pyplot(fig)
        else:
            st.warning("No sentiment data available.")

    elif selected_view == "Severity Distribution":

        st.header("Issue Severity Distribution")

        if not issues_df.empty:
            severity_weighted = issues_df.groupby("severity")["count"].sum()

            fig = plt.figure()
            plt.pie(
                severity_weighted.values,
                labels=severity_weighted.index,
                autopct="%1.1f%%"
            )
            plt.axis("equal")
            st.pyplot(fig)
        else:
            st.warning("No severity data available.")

    elif selected_view == "Top Issues":

        st.header("Top Customer Issues")

        if not issues_df.empty:
            top_issues = issues_df.sort_values(
                "count",
                ascending=False
            ).head(10)

            fig = plt.figure(figsize=(9, 5))
            plt.barh(top_issues["issue"], top_issues["count"])
            plt.gca().invert_yaxis()
            st.pyplot(fig)
        else:
            st.warning("No issues detected.")
