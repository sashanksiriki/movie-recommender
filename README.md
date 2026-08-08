# 🎬 Movie Recommender

A simple content-based movie recommender built with TF-IDF + cosine similarity,
served as a Streamlit app.

## Files

- `app.py` — the Streamlit app (loads data, builds the model, and renders the UI)
- `movies.csv` — dataset (**you must add this yourself**, see below)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. **Create a GitHub repo** and push these files to it, including `movies.csv`.
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **https://share.streamlit.io** and sign in with GitHub.

3. Click **"New app"**, then select:
   - Repository: `<your-username>/<your-repo>`
   - Branch: `main`
   - Main file path: `app.py`

4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch
   your app automatically. You'll get a permanent URL like:
   `https://<your-app-name>.streamlit.app`

5. Any time you `git push` new changes to `main`, the deployed app updates
   automatically.

## Notes

- The model (TF-IDF + cosine similarity) is rebuilt at app startup and cached
  with `st.cache_data` / `st.cache_resource`, so there's no need to upload
  `.pkl` files — this keeps the repo small and avoids version-mismatch issues
  between local pickles and the cloud environment.
- Make sure `movies.csv` has at least these columns: `genres`, `keywords`,
  `overview`, `title`.
