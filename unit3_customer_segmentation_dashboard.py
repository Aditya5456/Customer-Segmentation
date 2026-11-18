"""Unit 3 - Customer Segmentation Interactive Dashboard

Run:
    pip install streamlit pandas plotly scikit-learn
    streamlit run unit3_customer_segmentation_dashboard.py

The app loads the included Test.csv by default. You can also upload a CSV
from the sidebar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration and project reference data
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation | Unit 3",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# These are the cluster summaries and evaluation values reported by the
# referenced project. They are displayed as the repository's model results;
# the demographic filters below do not retrain K-Means.
CLUSTER_PROFILES = [
    {
        "cluster": "Cluster 0",
        "customers": 1142,
        "age": 54.8,
        "work_experience": 2.1,
        "family_size": 2.6,
        "profession": "Artist",
        "spending": "Average",
        "married": "97%",
        "graduated": "76%",
        "label": "Middle-aged Average-spenders",
    },
    {
        "cluster": "Cluster 1",
        "customers": 834,
        "age": 31.9,
        "work_experience": 3.3,
        "family_size": 2.8,
        "profession": "Healthcare",
        "spending": "Low",
        "married": "8%",
        "graduated": "54%",
        "label": "Young Low-spenders",
    },
    {
        "cluster": "Cluster 2",
        "customers": 384,
        "age": 39.0,
        "work_experience": 2.2,
        "family_size": 3.6,
        "profession": "Engineer",
        "spending": "Low",
        "married": "54%",
        "graduated": "39%",
        "label": "Middle-aged Low-spenders",
    },
    {
        "cluster": "Cluster 3",
        "customers": 267,
        "age": 39.6,
        "work_experience": 2.6,
        "family_size": 2.9,
        "profession": "Artist",
        "spending": "Low",
        "married": "51%",
        "graduated": "53%",
        "label": "Middle-aged Low-spenders",
    },
]

MODEL_METRICS = [
    ("Silhouette Score", "0.145", "Higher is generally better; this indicates overlapping clusters."),
    ("Davies-Bouldin Index", "2.311", "Lower is generally better."),
    ("Calinski-Harabasz Index", "233.9", "Higher is generally better."),
    ("Mean ARI across seeds", "0.441", "Measures similarity between clusterings across random seeds."),
    ("Mean ARI across subsamples", "0.315", "Measures how consistently customers are grouped across samples."),
]

AGE_BANDS = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]

FEATURE_DESCRIPTIONS = {
    "ID": "Unique customer identifier; normally excluded from model training.",
    "Gender": "Customer gender category.",
    "Ever_Married": "Whether the customer has ever been married.",
    "Age": "Customer age in years.",
    "Graduated": "Whether the customer is a graduate.",
    "Profession": "Customer's profession category.",
    "Work_Experience": "Number of years of work experience.",
    "Spending_Score": "Categorical spending level: Low, Average, or High.",
    "Family_Size": "Number of people in the customer's family.",
    "Var_1": "Additional categorical variable supplied with the dataset.",
}

EXPECTED_MISSING = {
    "Ever_Married": 50,
    "Graduated": 24,
    "Profession": 38,
    "Work_Experience": 269,
    "Family_Size": 113,
    "Var_1": 32,
}


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(uploaded_file) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file), None
        except Exception as exc:  # noqa: BLE001 - show a useful UI error
            return None, f"Could not read the uploaded CSV: {exc}"
    try:
        dataset_path = Path(__file__).with_name("Test.csv")
        return pd.read_csv(dataset_path), None
    except Exception as exc:  # noqa: BLE001
        return None, (
            "Could not read the included Test.csv. Upload a CSV in the "
            f"sidebar instead. Details: {exc}"
        )


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Age" not in data.columns:
        data["Age Band"] = "Unknown"
        return data

    age = pd.to_numeric(data["Age"], errors="coerce")
    bins = [0, 24, 34, 44, 54, 64, float("inf")]
    labels = AGE_BANDS
    data["Age Band"] = pd.cut(
        age, bins=bins, labels=labels, right=True, include_lowest=True
    ).astype("object")
    data["Age Band"] = data["Age Band"].fillna("Unknown")
    return data


def display_value(value: object) -> str:
    if pd.isna(value):
        return "Not specified"
    return str(value)


# ---------------------------------------------------------------------------
# Sidebar: dataset source and filters
# ---------------------------------------------------------------------------

st.title("👥 Customer Segmentation Intelligence Dashboard")
st.caption(
    "Unit 3 • K-Means clustering • Customer demographic and spending analysis"
)

with st.sidebar:
    st.header("Dashboard controls")
    uploaded = st.file_uploader("Upload Test.csv (optional)", type=["csv"])
    st.caption("Without an upload, the app tries to load the CSV from the project GitHub repository.")

    raw_df, load_error = load_data(uploaded)
    if load_error:
        st.error(load_error)
        st.stop()
    if raw_df is None or raw_df.empty:
        st.error("The dataset is empty or could not be loaded.")
        st.stop()

    df = add_age_band(raw_df)

    st.divider()
    st.subheader("Filters")

    gender_values = sorted(df["Gender"].dropna().astype(str).unique()) if "Gender" in df else []
    spend_order = ["Low", "Average", "High"]
    spend_values = (
        [v for v in spend_order if v in df["Spending_Score"].dropna().astype(str).unique()]
        if "Spending_Score" in df
        else []
    )
    age_values = [v for v in AGE_BANDS if v in df["Age Band"].astype(str).unique()]

    # Explicit widget keys make the Reset button reliable across reruns.
    if "gender_filter" not in st.session_state:
        st.session_state["gender_filter"] = gender_values
    if "spending_filter" not in st.session_state:
        st.session_state["spending_filter"] = spend_values
    if "age_filter" not in st.session_state:
        st.session_state["age_filter"] = age_values

    def reset_filters() -> None:
        st.session_state["gender_filter"] = gender_values
        st.session_state["spending_filter"] = spend_values
        st.session_state["age_filter"] = age_values

    selected_gender = st.multiselect("Gender", gender_values, key="gender_filter")
    selected_spending = st.multiselect("Spending score", spend_values, key="spending_filter")
    selected_age = st.multiselect("Age group", age_values, key="age_filter")

    st.button("Reset filters", use_container_width=True, on_click=reset_filters)

# Apply only dataset-level filters. This does not change precomputed K-Means labels.
filtered = df.copy()
if "Gender" in filtered:
    filtered = filtered[filtered["Gender"].astype(str).isin(selected_gender)]
if "Spending_Score" in filtered:
    filtered = filtered[filtered["Spending_Score"].astype(str).isin(selected_spending)]
filtered = filtered[filtered["Age Band"].astype(str).isin(selected_age)]


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

st.subheader("Dataset snapshot")
base_total = len(df)
filtered_total = len(filtered)
low_pct = 0.0
if filtered_total and "Spending_Score" in filtered:
    low_pct = (filtered["Spending_Score"].astype(str).eq("Low").mean()) * 100

avg_age = pd.to_numeric(filtered.get("Age", pd.Series(dtype=float)), errors="coerce").mean()
metric_cols = st.columns(4)
metric_cols[0].metric("Customers in view", f"{filtered_total:,}", f"of {base_total:,} total")
metric_cols[1].metric("Low spenders", f"{low_pct:.1f}%")
metric_cols[2].metric("Average age", f"{avg_age:.1f}" if pd.notna(avg_age) else "—")
metric_cols[3].metric("K-Means clusters", "4", "K = 4 in the referenced project")

if filtered.empty:
    st.warning("No customers match the selected filters. Change or reset the filters to see the charts.")


# ---------------------------------------------------------------------------
# Interactive charts
# ---------------------------------------------------------------------------

st.subheader("Customer analytics")
left, right = st.columns(2)

with left:
    st.markdown("**Spending score by age group**")
    if not filtered.empty and {"Age Band", "Spending_Score"}.issubset(filtered.columns):
        chart_data = (
            filtered.groupby(["Age Band", "Spending_Score"], observed=False)
            .size()
            .reset_index(name="Customers")
        )
        chart_data["Age Band"] = pd.Categorical(chart_data["Age Band"], AGE_BANDS, ordered=True)
        chart_data = chart_data.sort_values("Age Band")
        fig = px.bar(
            chart_data,
            x="Age Band",
            y="Customers",
            color="Spending_Score",
            barmode="group",
            category_orders={"Age Band": AGE_BANDS, "Spending_Score": spend_order},
            labels={"Age Band": "Age group", "Spending_Score": "Spending score"},
        )
        fig.update_layout(legend_title_text="Spending score", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Age and spending columns are required for this chart.")

with right:
    st.markdown("**Customers by gender**")
    if not filtered.empty and "Gender" in filtered:
        gender_counts = filtered["Gender"].fillna("Missing").astype(str).value_counts().rename_axis("Gender").reset_index(name="Customers")
        fig = px.bar(gender_counts, x="Gender", y="Customers", color="Gender", text="Customers")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gender column is not available or no customers match the filters.")

left2, right2 = st.columns(2)
with left2:
    st.markdown("**Profession distribution**")
    if not filtered.empty and "Profession" in filtered:
        profession_counts = filtered["Profession"].fillna("Missing").astype(str).value_counts().reset_index()
        profession_counts.columns = ["Profession", "Customers"]
        fig = px.bar(
            profession_counts.sort_values("Customers"),
            x="Customers",
            y="Profession",
            orientation="h",
            text="Customers",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Profession column is not available or no customers match the filters.")

with right2:
    st.markdown("**Spending score distribution**")
    if not filtered.empty and "Spending_Score" in filtered:
        spend_counts = filtered["Spending_Score"].fillna("Missing").astype(str).value_counts().reindex(spend_order, fill_value=0).reset_index()
        spend_counts.columns = ["Spending score", "Customers"]
        fig = px.pie(spend_counts, names="Spending score", values="Customers", hole=0.45)
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Spending_Score column is not available or no customers match the filters.")


# ---------------------------------------------------------------------------
# Cluster profiles and model evaluation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("K-Means cluster profiles")
st.caption(
    "These summaries are the reported profiles from the referenced project. "
    "The sidebar filters update the dataset charts above, but do not retrain "
    "K-Means or recalculate these four cluster profiles."
)

cluster_cols = st.columns(4)
for col, profile in zip(cluster_cols, CLUSTER_PROFILES):
    with col:
        st.markdown(f"**{profile['cluster']}**")
        st.metric("Customers", f"{profile['customers']:,}")
        st.write(f"**Profile:** {profile['label']}")
        st.write(f"Age: {profile['age']:.1f} years")
        st.write(f"Typical profession: {profile['profession']}")
        st.write(f"Spending: {profile['spending']}")
        st.write(f"Married: {profile['married']}")
        st.write(f"Graduated: {profile['graduated']}")

st.subheader("Model evaluation")
metric_cols = st.columns(len(MODEL_METRICS))
for col, (name, value, help_text) in zip(metric_cols, MODEL_METRICS):
    with col:
        st.metric(name, value, help=help_text)

st.warning(
    "Interpretation: the Silhouette Score (0.145) suggests that the groups "
    "overlap and are not sharply separated. Treat the clusters as broad customer "
    "profiles, not perfectly distinct categories."
)


# ---------------------------------------------------------------------------
# Dataset inspection, missing values, and methodology
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Dataset explorer")
show_cols = st.multiselect(
    "Choose columns to display",
    options=list(raw_df.columns),
    default=list(raw_df.columns),
)
st.dataframe(filtered[show_cols] if show_cols else filtered, use_container_width=True, height=300)

with st.expander("Missing values in the original dataset"):
    missing = raw_df.isna().sum().rename_axis("Feature").reset_index(name="Missing values")
    missing["Missing %"] = (missing["Missing values"] / max(len(raw_df), 1) * 100).round(2)
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.caption("The figures above are calculated from the CSV currently loaded, rather than hardcoded.")

with st.expander("Feature dictionary"):
    dictionary = pd.DataFrame(
        [{"Feature": col, "Meaning": FEATURE_DESCRIPTIONS.get(col, "Feature supplied in the dataset.")}
         for col in raw_df.columns]
    )
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

with st.expander("Preprocessing and algorithm"):
    st.markdown(
        """
        **Preprocessing pipeline reported for the project**
        1. Remove the customer `ID` because it is an identifier, not a useful behavioural feature.
        2. Fill missing numerical values using the median.
        3. Fill missing categorical values using the mode.
        4. Encode `Spending_Score` in order: Low = 0, Average = 1, High = 2.
        5. Apply One-Hot Encoding to the remaining categorical features.
        6. Apply StandardScaler so numerical features are on comparable scales.
        7. Apply K-Means with **K = 4**, `random_state = 42`, and `n_init = 10`.

        **Visual analysis:** Elbow method, PCA, and t-SNE were used to inspect
        cluster structure. Silhouette, Davies-Bouldin, Calinski-Harabasz, and
        Adjusted Rand Index (ARI) were used for evaluation/stability checks.
        """
    )

with st.expander("What the graphs tell us"):
    st.markdown(
        """
        - **Spending by age:** compares Low, Average, and High spending levels across age groups.
        - **Gender chart:** shows how many customers of each gender are in the current filtered view.
        - **Profession chart:** highlights the most common professions in the current filtered view.
        - **Spending distribution:** shows the proportion of each spending category in the current filtered view.
        - **Cluster profiles:** summarise the four customer groups reported by the original project.
        """
    )

st.markdown("---")
st.caption(
    "Academic demonstration only. K-Means is an unsupervised method; the cluster "
    "profiles describe patterns in this dataset and should not be treated as fixed truths about individuals."
)
