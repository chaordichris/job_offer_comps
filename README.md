# job_offer_comps

Lightweight Streamlit app for comparing job offers using a real-options lens.

## What it does

The app compares two offers using:
- cash compensation (base + target bonus + sign-on)
- equity as an option-like payoff (Black-Scholes-style approximation)
- probability of liquidity/exit
- layoff downside and severance
- strategic "real option" value (learning, network, flexibility, remote value)

This is not a tax or legal model. It is a decision-support tool for thinking about upside, downside, and optionality.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Notes on the model

- `Equity Option PV` is a simplified expected present value estimate.
- `Strategic Option Value` is user-entered and should reflect long-term career optionality.
- Sensitivity analysis helps test whether the preferred offer changes under different volatility/rate assumptions.
