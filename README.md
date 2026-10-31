# CineMatrix — Streamlit Movie Recommender

An interactive movie recommendation demo that explains its linear-algebra pipeline:
ratings preprocessing, matrix rank, cosine similarity, eigen-decomposition/PCA,
low-dimensional projection, and rating predictions.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Extract this ZIP file.
2. Create a GitHub repository and upload the **contents** of this folder (not the ZIP itself).
3. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit Community Cloud will install packages from `requirements.txt`.

The app uses a built-in sample dataset by default. You can optionally upload a CSV with
users as rows, movie titles as columns, and ratings from 1 to 5. Leave unrated cells blank.
The first column should contain user names/IDs and the remaining columns should contain movie ratings.

## Notes

- `.streamlit/config.toml` contains the dark theme configuration.
- Do not commit passwords, API keys, or `.streamlit/secrets.toml` to GitHub.
- Streamlit Community Cloud needs a GitHub repository; this ZIP is the prepared source bundle,
  not a live hosted URL.
