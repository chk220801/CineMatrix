"""CineMatrix: linear-algebra movie recommender for Streamlit Community Cloud."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data import USERS, MOVIES, RATINGS
from recommender import recommend

st.set_page_config(page_title="CineMatrix", page_icon="🎬", layout="wide")

st.sidebar.title("🎬 CineMatrix")
st.sidebar.caption("Movie recommendations powered by linear algebra")

uploaded = st.sidebar.file_uploader(
    "Upload ratings CSV (users as rows, movies as columns; leave missing ratings blank)",
    type=["csv"],
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, index_col=0)
        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            st.sidebar.error("The CSV must contain at least 2 users and 2 movies.")
            st.stop()
        df = df.apply(pd.to_numeric, errors="coerce")
        R = df.to_numpy(dtype=float)
        users = df.index.astype(str).tolist()
        movies = df.columns.astype(str).tolist()
        if np.isinf(R).any():
            st.sidebar.error("The CSV contains infinite values. Please clean the file and upload again.")
            st.stop()
        if np.isnan(R).all(axis=1).any():
            st.sidebar.error("Every user must have at least one rating.")
            st.stop()
        if np.nanmin(R) < 1 or np.nanmax(R) > 5:
            st.sidebar.error("Ratings must be between 1 and 5. Leave missing ratings blank.")
            st.stop()
    except Exception as exc:
        st.sidebar.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    users, movies, R = USERS, MOVIES, RATINGS
    st.sidebar.caption("Using the built-in sample dataset (6 users × 5 movies).")

max_k = max(1, min(R.shape) - 1)
default_k = min(2, max_k)
k = st.sidebar.slider("PCA components (k)", 1, max_k, default_k)
target = st.sidebar.selectbox("Recommend for user", users)

try:
    res = recommend(R, k)
except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
    st.error(f"Unable to analyze this ratings matrix: {exc}")
    st.stop()

p = res["pca"]
V = p["components"]
ex = p["explained"] * 100
fmt = lambda a, r, c: pd.DataFrame(a, index=r, columns=c)

st.title("🎬 CineMatrix")
st.caption(
    "Ratings → preprocessing → matrix rank → vector similarity → eigen-decomposition/PCA "
    "→ projection → personalized recommendations"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users × Movies", f"{R.shape[0]} × {R.shape[1]}")
c2.metric("Missing ratings", int(np.isnan(R).sum()))
c3.metric("Rank of centered matrix", res["rank"]["rank_centered"])
c4.metric(f"Variance in {k} PCs", f"{ex[:k].sum():.1f}%")

tabs = st.tabs(["📋 Data", "🧮 Rank", "🔗 Similarity", "📈 PCA", "🎯 Projection", "⭐ Recommendations"])

with tabs[0]:
    left, right = st.columns(2)
    left.subheader("Ratings matrix (R)")
    left.dataframe(fmt(R, users, movies).style.format("{:.0f}", na_rep="–"), use_container_width=True)
    right.subheader("Centered matrix (C)")
    right.dataframe(
        fmt(res["C"], users, movies).style.format("{:.2f}").background_gradient("RdBu", vmin=-3, vmax=3),
        use_container_width=True,
    )
    st.latex(r"C_{ij}=R_{ij}-\mu_i \text{ if observed, else } 0")
    st.caption("Each user's observed ratings are centered around that user's mean.")

with tabs[1]:
    st.write(
        f"Rank of zero-filled R = **{res['rank']['rank_raw']}**  ·  "
        f"Rank of centered C = **{res['rank']['rank_centered']}**"
    )
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sv = res["rank"]["singular_values"]
    ax.bar(range(1, len(sv) + 1), sv)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.set_title("Singular values of centered matrix C")
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    st.latex(r"\mathrm{sim}(u,v)=\frac{\langle u,v\rangle}{\|u\|\|v\|}")
    which = st.radio("Similarity between", ["Users", "Movies"], horizontal=True)
    S, names = (res["user_sim"], users) if which == "Users" else (res["movie_sim"], movies)
    st.dataframe(
        fmt(S, names, names).style.format("{:.2f}").background_gradient("coolwarm", vmin=-1, vmax=1),
        use_container_width=True,
    )
    if which == "Users":
        i = users.index(target)
        similarities = pd.Series(S[i], index=users).drop(target).sort_values(ascending=False)
        if not similarities.empty:
            st.write(
                f"Most similar to **{target}**: **{similarities.index[0]}** "
                f"({similarities.iloc[0]:.2f})"
            )
            st.write(
                f"Least similar to **{target}**: **{similarities.index[-1]}** "
                f"({similarities.iloc[-1]:.2f})"
            )

with tabs[3]:
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(1, len(ex) + 1)
    ax.bar(x, ex, label="Individual variance")
    ax.plot(x, np.cumsum(ex), "o-", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot")
    ax.legend()
    left.pyplot(fig)
    plt.close(fig)
    count = min(3, V.shape[1])
    right.dataframe(
        fmt(V[:, :count].T, [f"PC{i+1} ({ex[i]:.1f}%)" for i in range(count)], movies)
        .style.format("{:.2f}").background_gradient("RdBu", vmin=-1, vmax=1),
        use_container_width=True,
    )
    right.caption("Each row shows a principal component: a dominant direction of movie preference.")

with tabs[4]:
    Z = res["Z"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    y_values = Z[:, 1] if k > 1 else np.zeros(len(users))
    ax.scatter(Z[:, 0], y_values, s=75, zorder=3)
    for i, user in enumerate(users):
        ax.annotate(user, (Z[i, 0], y_values[i]), xytext=(6, 4), textcoords="offset points")
    for j, movie in enumerate(movies):
        vx = V[j, 0] * 3.5
        vy = V[j, 1] * 3.5 if V.shape[1] > 1 and k > 1 else 0
        ax.arrow(0, 0, vx, vy, head_width=0.07, length_includes_head=True)
        ax.text(vx * 1.08, vy * 1.08, movie, fontsize=8)
    ax.set_xlabel(f"PC1 ({ex[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ex[1]:.1f}%)" if k > 1 else "No second component selected")
    ax.set_title("User preference projection and movie directions")
    st.pyplot(fig)
    plt.close(fig)
    st.latex(r"Z = C V_k,\quad \hat R = Z V_k^\top + \mu")

with tabs[5]:
    user_index = users.index(target)
    rows = [
        (movies[item["movie"]], round(item["predicted"], 2), round(item["user_mean"], 2),
         "✅ Recommend" if item["recommend"] else "⛔ Skip")
        for item in res["recs"] if item["user"] == user_index
    ]
    st.subheader(f"Recommendations for {target}")
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=["Movie", "Predicted rating", "Your average", "Verdict"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("This user has rated every movie in the dataset, so there are no missing movies to recommend.")
    st.subheader("Predictions for all users")
    all_rows = [
        (users[item["user"]], movies[item["movie"]], round(item["predicted"], 2),
         "✅ Recommend" if item["recommend"] else "⛔ Skip")
        for item in res["recs"]
    ]
    st.dataframe(
        pd.DataFrame(all_rows, columns=["User", "Movie", "Predicted rating", "Recommendation"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Decision rule: recommend an unrated movie when its predicted rating is at least the user's own average.")
