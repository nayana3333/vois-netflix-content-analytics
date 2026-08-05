"""
Netflix Content Strategy Analysis -- interactive dashboard.

Same data, same feature engineering, same hypothesis tests as
notebooks/Netflix_Analysis.ipynb, wrapped in an explorable Streamlit app.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_percentage_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

RANDOM_STATE = 42
DATA_PATH = str(Path(__file__).parent / "data" / "Netflix_Dataset.csv")

AUDIENCE_MAP = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "G": "Kids", "TV-G": "Kids",
    "PG": "Family/Teens", "TV-PG": "Family/Teens",
    "PG-13": "Teens", "TV-14": "Teens",
    "R": "Adults", "TV-MA": "Adults", "NC-17": "Adults",
    "NR": "Unrated", "UR": "Unrated",
}

st.set_page_config(page_title="Netflix Content Strategy Analysis", layout="wide", page_icon="🎬")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["release_date_parsed"] = pd.to_datetime(df["Release_Date"], errors="coerce")
    df["year_added"] = df["release_date_parsed"].dt.year

    df["duration_num"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = np.where(
        df["Duration"].astype(str).str.contains("Season"), "seasons",
        np.where(df["Duration"].astype(str).str.contains("min"), "minutes", "unknown"),
    )

    df["genres_list"] = df["Type"].astype(str).str.split(",").apply(lambda x: [g.strip() for g in x])
    df["num_genres"] = df["genres_list"].apply(len)

    df["country_list"] = df["Country"].fillna("Unknown").astype(str).str.split(",").apply(
        lambda x: [c.strip() for c in x]
    )
    df["num_countries"] = df["country_list"].apply(len)
    df["primary_country"] = df["country_list"].apply(lambda x: x[0] if x else "Unknown")

    df["audience_segment"] = df["Rating"].map(AUDIENCE_MAP)
    df["is_movie"] = (df["Category"] == "Movie").astype(int)
    return df


@st.cache_resource
def train_classifier(_df: pd.DataFrame):
    df = _df
    model_df = df.dropna(subset=["duration_num", "year_added", "num_genres", "num_countries", "audience_segment"]).copy()
    model_df = model_df[model_df["audience_segment"] != "Unrated"]

    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(model_df["genres_list"]), columns=mlb.classes_, index=model_df.index
    )
    top_genres = model_df["genres_list"].explode().value_counts().head(15).index.tolist()
    genre_dummies = genre_dummies[top_genres]
    cat_dummy = pd.get_dummies(model_df["Category"], prefix="cat")

    X = pd.concat(
        [model_df[["duration_num", "year_added", "num_genres", "num_countries"]], cat_dummy, genre_dummies], axis=1
    )
    y = model_df["audience_segment"]

    baseline_acc = y.value_counts(normalize=True).max()
    baseline_label = y.value_counts().idxmax()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "clf": clf, "columns": X.columns, "top_genres": top_genres,
        "categories": sorted(model_df["Category"].unique().tolist()),
        "baseline_acc": baseline_acc, "baseline_label": baseline_label,
        "acc": acc, "f1": f1, "importances": importances,
    }


df = load_data(DATA_PATH)

st.title("🎬 Netflix Content Strategy Analysis")
st.caption(
    "Decision-science analysis of 7,789 Netflix titles (2008–2021) — four business hypotheses, "
    "each tested statistically rather than eyeballed. Data Analysis Internship project · "
    "Vodafone Idea Foundation (VOIS) × Edunet Foundation."
)

# ---- KPI row -----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Titles analyzed", f"{len(df):,}")
c2.metric("Movies / TV Shows", f"{(df['Category']=='Movie').sum():,} / {(df['Category']=='TV Show').sum():,}")
country_counts_full = df["primary_country"].value_counts()
hhi_full = ((country_counts_full / country_counts_full.sum()) ** 2).sum() * 10000
c3.metric("Country sourcing HHI", f"{hhi_full:.0f}", help="Herfindahl-Hirschman Index of primary-country share. >1500 = moderately concentrated.")
c4.metric("Years covered", f"{int(df['year_added'].min())}–{int(df['year_added'].max())}")

tabs = st.tabs([
    "Hypothesis Summary", "Content Mix", "Genre Growth", "Country Sourcing",
    "Audience Segments", "Clustering", "Predict a Title", "Growth Forecast",
])

# ---- Tab: Hypothesis Summary --------------------------------------------
with tabs[0]:
    st.subheader("Four business hypotheses, tested")
    yearly_cat = df.dropna(subset=["year_added"]).groupby(["year_added", "Category"]).size().unstack(fill_value=0)
    yearly_cat["tv_share"] = yearly_cat["TV Show"] / (yearly_cat["Movie"] + yearly_cat["TV Show"])
    yearly_cat = yearly_cat[(yearly_cat.index >= 2015) & (yearly_cat.index <= 2020)]
    slope_mix, _, r_mix, p_mix, _ = stats.linregress(yearly_cat.index, yearly_cat["tv_share"])

    exploded = df.dropna(subset=["year_added"]).explode("genres_list")
    top8 = exploded["genres_list"].value_counts().head(8).index.tolist()
    n_sig, n_genres = 0, len(top8)
    for g in top8:
        yc = exploded[exploded["genres_list"] == g].groupby("year_added").size()
        yc = yc[(yc.index >= 2015) & (yc.index <= 2020)]
        if len(yc) >= 4:
            _, _, _, p_g, _ = stats.linregress(yc.index, yc.values)
            if p_g < 0.05:
                n_sig += 1

    model = train_classifier(df)

    summary = pd.DataFrame({
        "Hypothesis": [
            "H1: TV-share trend is real, not noise",
            "H2: Top genres show significant growth trends",
            "H3: Country sourcing is concentrated (HHI > 1500)",
            "H4: Metadata predicts audience segment above baseline",
        ],
        "Test": [
            f"Linear regression, p = {p_mix:.3f}",
            f"{n_sig}/{n_genres} genres significant (p < 0.05)",
            f"HHI = {hhi_full:.0f}",
            f"RF accuracy {model['acc']:.0%} vs baseline {model['baseline_acc']:.0%}",
        ],
        "Result": [
            "✅ Not supported" if p_mix >= 0.05 else "✅ Supported",
            "✅ Supported" if n_sig >= 3 else "⚠️ Partial",
            "✅ Supported" if hhi_full > 1500 else "❌ Not supported",
            "✅ Supported" if model["acc"] > model["baseline_acc"] + 0.10 else "⚠️ Weak",
        ],
    })
    st.dataframe(summary, width='stretch', hide_index=True)
    st.info(
        "H1 is deliberately reported as **not supported** (p = 0.924) — the year-to-year TV-share "
        "swings are statistically indistinguishable from noise. Presenting a null result honestly, "
        "instead of only showcasing wins, is the point of testing hypotheses rather than eyeballing charts."
    )

# ---- Tab: Content Mix ----------------------------------------------------
with tabs[1]:
    st.subheader("Movies vs. TV Shows, 2015–2020")
    fig = go.Figure()
    fig.add_bar(x=yearly_cat.index, y=yearly_cat["Movie"], name="Movies")
    fig.add_bar(x=yearly_cat.index, y=yearly_cat["TV Show"], name="TV Shows")
    fig.add_trace(go.Scatter(
        x=yearly_cat.index, y=yearly_cat["tv_share"] * 100, name="TV Show share (%)",
        yaxis="y2", mode="lines+markers", line=dict(color="black", width=3),
    ))
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="Titles added"),
        yaxis2=dict(title="TV Show share (%)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')
    st.metric("TV-share trend", f"{slope_mix*100:+.2f} pts/year", f"p = {p_mix:.3f}")
    st.caption(
        "Not supported: the trend line is not distinguishable from random year-to-year noise "
        "at the 5% significance level — don't over-commit production budget to a TV-Show pivot on this alone."
    )

# ---- Tab: Genre Growth ---------------------------------------------------
with tabs[2]:
    st.subheader("Genre growth trends, 2015–2020")
    all_genres = exploded["genres_list"].value_counts().index.tolist()
    default_genres = top8
    chosen = st.multiselect("Genres to compare (defaults to the top 8 by volume)", all_genres, default=default_genres)
    rows = []
    for g in chosen:
        yc = exploded[exploded["genres_list"] == g].groupby("year_added").size()
        yc = yc[(yc.index >= 2015) & (yc.index <= 2020)]
        if len(yc) >= 4:
            slope_g, _, r_g, p_g, _ = stats.linregress(yc.index, yc.values)
            rows.append({"genre": g, "titles_2015_20": int(yc.sum()), "yearly_slope": round(slope_g, 1),
                         "p_value": round(p_g, 4), "significant (p<0.05)": p_g < 0.05})
    if rows:
        genre_trend = pd.DataFrame(rows).sort_values("yearly_slope", ascending=False)
        fig = px.bar(
            genre_trend, x="yearly_slope", y="genre", orientation="h",
            color="significant (p<0.05)", color_discrete_map={True: "#2ca02c", False: "#c7c7c7"},
            labels={"yearly_slope": "Titles added per year (trend slope)"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(genre_trend, width='stretch', hide_index=True)
    else:
        st.warning("Pick at least one genre with 4+ years of data.")

# ---- Tab: Country Sourcing ------------------------------------------------
with tabs[3]:
    st.subheader("Where Netflix sources content from")
    top_n = st.slider("Show top N countries", 5, 25, 12)
    top_n_df = country_counts_full.head(top_n)
    fig = px.bar(
        x=top_n_df.values, y=top_n_df.index, orientation="h",
        labels={"x": "Titles", "y": "Primary country"}, color=top_n_df.values, color_continuous_scale="magma",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    shares = country_counts_full / country_counts_full.sum()
    top5_share = shares.head(5).sum() * 100
    us_share = shares.get("United States", 0) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("HHI", f"{hhi_full:.0f}")
    m2.metric("Top-5 country share", f"{top5_share:.1f}%")
    m3.metric("US share alone", f"{us_share:.1f}%")
    st.caption("HHI > 1500 is generally read as moderately-to-highly concentrated — a single-market dependency risk worth diversifying.")

# ---- Tab: Audience Segments -----------------------------------------------
with tabs[4]:
    st.subheader("Audience segment mix by content category")
    seg_df = df.dropna(subset=["audience_segment"])
    seg_df = seg_df[seg_df["audience_segment"] != "Unrated"]
    ct = pd.crosstab(seg_df["Category"], seg_df["audience_segment"])
    chi2_stat, p_chi2, _, _ = stats.chi2_contingency(ct)
    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).reset_index().melt(id_vars="Category", var_name="Segment", value_name="pct")
    fig = px.bar(ct_pct, x="Category", y="pct", color="Segment", barmode="stack", labels={"pct": "% of category"})
    st.plotly_chart(fig, width='stretch')
    st.metric("Chi-square test of independence", f"χ² = {chi2_stat:.1f}", f"p {'< 0.0001' if p_chi2 < 0.0001 else f'= {p_chi2:.4f}'}")
    st.caption("Category and audience segment are statistically associated — Movies skew more toward Adults/Teens, TV Shows carry proportionally more Kids content.")

# ---- Tab: Clustering --------------------------------------------------
with tabs[5]:
    st.subheader("Content clustering (K-Means)")
    feat_cols = ["duration_num", "year_added", "num_genres", "num_countries", "is_movie"]
    cluster_df = df.dropna(subset=feat_cols).copy()
    X_scaled = StandardScaler().fit_transform(cluster_df[feat_cols])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        k = st.slider("Number of clusters (k)", 2, 6, 3)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        cluster_df["cluster"] = km.labels_.astype(str)
        sil = silhouette_score(X_scaled, km.labels_)
        st.metric("Silhouette score", f"{sil:.3f}", help="Best k found in the notebook via silhouette search is k=3 (score 0.433).")
        profile = cluster_df.groupby("cluster")[feat_cols].mean().round(2)
        profile["count"] = cluster_df.groupby("cluster").size()
        st.dataframe(profile, width='stretch')
    with col_b:
        x_axis = st.selectbox("X axis", feat_cols, index=0)
        y_axis = st.selectbox("Y axis", feat_cols, index=3)
        fig = px.scatter(
            cluster_df.sample(min(2000, len(cluster_df)), random_state=RANDOM_STATE),
            x=x_axis, y=y_axis, color="cluster", opacity=0.6,
        )
        st.plotly_chart(fig, width='stretch')
    st.caption(
        "At k=3 the notebook names these cohorts: **Standard Single-Market Movies** (high duration_num in minutes, "
        "is_movie≈1), **International Co-Productions** (high num_countries), and **TV Series** (is_movie≈0)."
    )

# ---- Tab: Predict a Title --------------------------------------------------
with tabs[6]:
    st.subheader("Try the audience-segment classifier live")
    st.caption(
        f"Random Forest, {model['acc']:.0%} accuracy vs. a {model['baseline_acc']:.0%} majority-class baseline "
        f"(always predicting '{model['baseline_label']}'). Fill in a hypothetical title's metadata and see what it predicts."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        in_category = st.selectbox("Category", model["categories"])
        in_year = st.slider("Year added", 2008, 2021, 2019)
    with col2:
        in_duration = st.slider("Duration (minutes if Movie, seasons if TV Show)", 1, 200, 90)
        in_num_countries = st.slider("Number of production countries", 1, 10, 1)
    with col3:
        in_genres = st.multiselect("Genres", model["top_genres"], default=[model["top_genres"][0]])

    if st.button("Predict audience segment", type="primary"):
        row = pd.DataFrame([{c: 0 for c in model["columns"]}])
        row["duration_num"] = in_duration
        row["year_added"] = in_year
        row["num_genres"] = max(len(in_genres), 1)
        row["num_countries"] = in_num_countries
        cat_col = f"cat_{in_category}"
        if cat_col in row.columns:
            row[cat_col] = 1
        for g in in_genres:
            if g in row.columns:
                row[g] = 1
        row = row.reindex(columns=model["columns"], fill_value=0)

        pred = model["clf"].predict(row)[0]
        proba = pd.Series(model["clf"].predict_proba(row)[0], index=model["clf"].classes_).sort_values(ascending=False)
        st.success(f"Predicted audience segment: **{pred}**")
        fig = px.bar(x=proba.values, y=proba.index, orientation="h", labels={"x": "Probability", "y": "Segment"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("What the model relies on most")
    imp = model["importances"].head(10).sort_values()
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": "Feature"})
    st.plotly_chart(fig, width='stretch')

# ---- Tab: Growth Forecast ---------------------------------------------------
with tabs[7]:
    st.subheader("Catalog growth forecast")
    yearly_total = df.dropna(subset=["year_added"]).groupby("year_added").size().reset_index(name="count")
    yearly_total = yearly_total[(yearly_total["year_added"] >= 2015) & (yearly_total["year_added"] <= 2020)].sort_values("year_added")

    train_y = yearly_total.iloc[:-2]
    test_y = yearly_total.iloc[-2:]
    holdout_model = LinearRegression().fit(train_y[["year_added"]], train_y["count"])
    holdout_preds = holdout_model.predict(test_y[["year_added"]])
    mape = mean_absolute_percentage_error(test_y["count"], holdout_preds)

    horizon = st.slider("Forecast horizon (years beyond 2020)", 1, 5, 2)
    final_model = LinearRegression().fit(yearly_total[["year_added"]], yearly_total["count"])
    fitted = final_model.predict(yearly_total[["year_added"]])
    future_years = pd.DataFrame({"year_added": list(range(2021, 2021 + horizon))})
    future_preds = final_model.predict(future_years)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly_total["year_added"], y=yearly_total["count"], mode="lines+markers", name="Historical"))
    fig.add_trace(go.Scatter(x=yearly_total["year_added"], y=fitted, mode="lines", name="Linear fit", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=future_years["year_added"], y=future_preds, mode="lines+markers", name="Forecast", line=dict(dash="dot")))
    st.plotly_chart(fig, width='stretch')
    st.metric("Holdout MAPE (2019–2020, unseen during fit)", f"{mape*100:.1f}%")
    st.caption(
        "2020 actually fell short of the naive linear extrapolation — plausibly a COVID-19 production slowdown. "
        "Treat the forecast as a directional range, not a precise number."
    )

st.divider()
st.caption(
    "Built by Nayana S — Data Analysis Intern, Vodafone Idea Foundation (VOIS) × Edunet Foundation. "
    "[View the full notebook & README on GitHub](https://github.com/nayana3333/vois-netflix-content-analytics)."
)
