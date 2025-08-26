# Streamlit Cloud Deployment

## Files in this folder
- requirements.txt  → Python dependencies
- runtime.txt       → Python version (3.11)
- .streamlit/config.toml → Optional UI/server settings

## Quick start
1. Push your repo (containing `streamlit_job_comp_mc_app.py`) to GitHub.
2. Place these files at the repo root.
3. In Streamlit Cloud, set app entry point to: `streamlit_job_comp_mc_app.py`
4. Deploy. Cloud uses `requirements.txt` and `runtime.txt` automatically.

### Notes
- If you need system packages later, add `packages.txt` with apt package names.
- Store any secrets in **App → Settings → Secrets** (Streamlit Cloud UI), not in your repo.
