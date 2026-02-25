# Streamlit Cloud Deployment

## Files used
- `requirements.txt` (dependencies)
- `runtime.txt` (Python version)
- `config.toml` (optional Streamlit config in this repo)
- `app/app.py` (app entry point)

## Quick start
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the app entry point to `app/app.py`.
4. Deploy.

## Notes
- Add `packages.txt` later only if you need OS-level packages.
- Put secrets in Streamlit Cloud app settings, not in the repo.
