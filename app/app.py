from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass
class Offer:
    name: str = ""
    base_salary: float = 0.0
    target_bonus_pct: float = 0.0
    sign_on: float = 0.0
    annual_equity_grant_value: float = 0.0
    equity_volatility: float = 0.0
    time_to_liquidity_years: float = 0.0
    vesting_years: float = 0.0
    risk_free_rate: float = 0.0
    company_value_now: float = 0.0
    expected_company_value_at_liquidity: float = 0.0
    option_strike_value: float = 0.0
    exit_probability: float = 0.0
    layoff_probability: float = 0.0
    severance_months: float = 0.0
    decision_flex_value: float = 0.0
    remote_value: float = 0.0
    learning_value: float = 0.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, years: float, rate: float, vol: float) -> float:
    if years <= 0:
        return max(spot - strike, 0.0)
    if vol <= 0:
        return max(spot - strike * math.exp(-rate * years), 0.0)
    if spot <= 0 or strike <= 0:
        return 0.0

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    return spot * norm_cdf(d1) - strike * math.exp(-rate * years) * norm_cdf(d2)


def discount(value: float, rate: float, years: float) -> float:
    return value / ((1 + rate) ** years)


def evaluate_offer(offer: Offer) -> dict[str, float]:
    annual_bonus = offer.base_salary * offer.target_bonus_pct
    annual_cash = offer.base_salary + annual_bonus

    # Treat equity as a call option on company value (real-world approximation).
    spot = offer.expected_company_value_at_liquidity
    strike = offer.option_strike_value
    call_ratio = 0.0
    if offer.company_value_now > 0:
        call_ratio = black_scholes_call(
            spot=spot,
            strike=strike,
            years=offer.time_to_liquidity_years,
            rate=offer.risk_free_rate,
            vol=offer.equity_volatility,
        ) / offer.company_value_now

    vested_fraction = min(1.0, max(0.0, offer.time_to_liquidity_years / max(offer.vesting_years, 0.25)))
    gross_equity_option_value = offer.annual_equity_grant_value * call_ratio * vested_fraction
    probability_adjusted_equity = gross_equity_option_value * offer.exit_probability
    equity_pv = discount(probability_adjusted_equity, offer.risk_free_rate, offer.time_to_liquidity_years)

    severance_value = (offer.base_salary / 12.0) * offer.severance_months * offer.layoff_probability
    layoff_penalty = annual_cash * 0.5 * offer.layoff_probability

    strategic_option_value = offer.decision_flex_value + offer.remote_value + offer.learning_value

    one_year_pv = annual_cash + offer.sign_on + equity_pv + severance_value + strategic_option_value - layoff_penalty
    upside_score = probability_adjusted_equity + strategic_option_value
    downside_risk = layoff_penalty - severance_value

    return {
        "Annual Cash": annual_cash,
        "Sign-On": offer.sign_on,
        "Equity Option PV": equity_pv,
        "Expected Severance Value": severance_value,
        "Strategic Option Value": strategic_option_value,
        "Layoff Penalty": layoff_penalty,
        "1Y Total Expected Value": one_year_pv,
        "Upside Score": upside_score,
        "Downside Risk": downside_risk,
    }


def offer_inputs(label: str, defaults: dict[str, float | str]) -> Offer:
    st.subheader(label)
    c1, c2 = st.columns(2)

    with c1:
        name = st.text_input(f"{label} name", value=str(defaults["name"]), key=f"{label}_name")
        base_salary = st.number_input(f"{label} base salary", min_value=0.0, value=float(defaults["base_salary"]), step=5000.0)
        target_bonus_pct = st.slider(f"{label} target bonus (%)", 0.0, 100.0, float(defaults["target_bonus_pct"]) * 100, 1.0) / 100.0
        sign_on = st.number_input(f"{label} sign-on", min_value=0.0, value=float(defaults["sign_on"]), step=5000.0)
        annual_equity_grant_value = st.number_input(
            f"{label} annual equity grant (current fair value)", min_value=0.0, value=float(defaults["annual_equity_grant_value"]), step=5000.0
        )
        option_strike_value = st.number_input(
            f"{label} option strike (company value basis)", min_value=1.0, value=float(defaults["option_strike_value"]), step=1000000.0
        )
        company_value_now = st.number_input(
            f"{label} company value now", min_value=1.0, value=float(defaults["company_value_now"]), step=1000000.0
        )
        expected_company_value_at_liquidity = st.number_input(
            f"{label} expected company value at liquidity", min_value=1.0, value=float(defaults["expected_company_value_at_liquidity"]), step=1000000.0
        )

    with c2:
        equity_volatility = st.slider(f"{label} equity volatility", 0.05, 1.50, float(defaults["equity_volatility"]), 0.05)
        time_to_liquidity_years = st.slider(f"{label} years to liquidity", 0.5, 10.0, float(defaults["time_to_liquidity_years"]), 0.5)
        vesting_years = st.slider(f"{label} vesting period (years)", 1.0, 6.0, float(defaults["vesting_years"]), 0.5)
        exit_probability = st.slider(f"{label} exit/liquidity probability", 0.0, 1.0, float(defaults["exit_probability"]), 0.05)
        layoff_probability = st.slider(f"{label} layoff probability (1Y)", 0.0, 1.0, float(defaults["layoff_probability"]), 0.05)
        severance_months = st.slider(f"{label} severance months", 0.0, 12.0, float(defaults["severance_months"]), 0.5)
        decision_flex_value = st.number_input(
            f"{label} future decision flexibility value", min_value=0.0, value=float(defaults["decision_flex_value"]), step=1000.0
        )
        remote_value = st.number_input(f"{label} remote/commute value", value=float(defaults["remote_value"]), step=1000.0)
        learning_value = st.number_input(f"{label} learning/network value", value=float(defaults["learning_value"]), step=1000.0)

    return Offer(
        name=name,
        base_salary=base_salary,
        target_bonus_pct=target_bonus_pct,
        sign_on=sign_on,
        annual_equity_grant_value=annual_equity_grant_value,
        equity_volatility=equity_volatility,
        time_to_liquidity_years=time_to_liquidity_years,
        vesting_years=vesting_years,
        risk_free_rate=0.04,
        company_value_now=company_value_now,
        expected_company_value_at_liquidity=expected_company_value_at_liquidity,
        option_strike_value=option_strike_value,
        exit_probability=exit_probability,
        layoff_probability=layoff_probability,
        severance_months=severance_months,
        decision_flex_value=decision_flex_value,
        remote_value=remote_value,
        learning_value=learning_value,
    )


def sensitivity_table(offer_a: Offer, offer_b: Offer, rates: list[float], vols: list[float]) -> pd.DataFrame:
    rows = []
    for r in rates:
        for v in vols:
            a = Offer(**{**offer_a.__dict__, "risk_free_rate": r, "equity_volatility": v})
            b = Offer(**{**offer_b.__dict__, "risk_free_rate": r, "equity_volatility": v})
            av = evaluate_offer(a)["1Y Total Expected Value"]
            bv = evaluate_offer(b)["1Y Total Expected Value"]
            rows.append({
                "Rate": r,
                "Vol": v,
                "Winner": a.name if av >= bv else b.name,
                "Delta": round(av - bv, 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Job Offer Real Options Comparator", layout="wide")
    st.title("Job Offer Comparator (Real Options Lens)")
    st.caption("Lightweight model: salary + bonus + option-like equity value + flexibility/learning option value - downside risk.")

    st.sidebar.header("Global Assumptions")
    risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.04, 0.005)
    comparison_horizon = st.sidebar.slider("Comparison horizon (years)", 1, 5, 1)
    st.sidebar.markdown(
        "This app uses a practical approximation: equity is treated as a call option and strategic factors are modeled as user-entered real-option value."
    )

    default_a = {
        "name": "Offer A",
        "base_salary": 180000.0,
        "target_bonus_pct": 0.15,
        "sign_on": 20000.0,
        "annual_equity_grant_value": 50000.0,
        "equity_volatility": 0.55,
        "time_to_liquidity_years": 4.0,
        "vesting_years": 4.0,
        "company_value_now": 1000000000.0,
        "expected_company_value_at_liquidity": 1800000000.0,
        "option_strike_value": 1200000000.0,
        "exit_probability": 0.35,
        "layoff_probability": 0.10,
        "severance_months": 2.0,
        "decision_flex_value": 10000.0,
        "remote_value": 8000.0,
        "learning_value": 12000.0,
    }
    default_b = {
        "name": "Offer B",
        "base_salary": 210000.0,
        "target_bonus_pct": 0.10,
        "sign_on": 10000.0,
        "annual_equity_grant_value": 30000.0,
        "equity_volatility": 0.25,
        "time_to_liquidity_years": 2.0,
        "vesting_years": 4.0,
        "company_value_now": 2000000000000.0,
        "expected_company_value_at_liquidity": 2400000000000.0,
        "option_strike_value": 2100000000000.0,
        "exit_probability": 0.80,
        "layoff_probability": 0.07,
        "severance_months": 3.0,
        "decision_flex_value": 6000.0,
        "remote_value": 2000.0,
        "learning_value": 7000.0,
    }

    left, right = st.columns(2)
    with left:
        offer_a = offer_inputs("Offer A", default_a)
    with right:
        offer_b = offer_inputs("Offer B", default_b)

    offer_a.risk_free_rate = risk_free_rate
    offer_b.risk_free_rate = risk_free_rate
    offer_a = Offer(**{**offer_a.__dict__, "time_to_liquidity_years": max(offer_a.time_to_liquidity_years, float(comparison_horizon))})
    offer_b = Offer(**{**offer_b.__dict__, "time_to_liquidity_years": max(offer_b.time_to_liquidity_years, float(comparison_horizon))})

    a_metrics = evaluate_offer(offer_a)
    b_metrics = evaluate_offer(offer_b)
    comparison = pd.DataFrame([a_metrics, b_metrics], index=[offer_a.name, offer_b.name]).T

    st.divider()
    st.subheader("Comparison")
    st.dataframe(comparison.style.format("${:,.0f}"), use_container_width=True)

    a_total = a_metrics["1Y Total Expected Value"]
    b_total = b_metrics["1Y Total Expected Value"]
    winner = offer_a.name if a_total >= b_total else offer_b.name
    delta = abs(a_total - b_total)

    c1, c2, c3 = st.columns(3)
    c1.metric("Preferred Offer", winner)
    c2.metric("Expected Value Delta", f"${delta:,.0f}")
    c3.metric(
        "Equity Upside Difference",
        f"${abs(a_metrics['Equity Option PV'] - b_metrics['Equity Option PV']):,.0f}",
    )

    st.subheader("Sensitivity (Rate x Volatility)")
    sens = sensitivity_table(
        offer_a,
        offer_b,
        rates=[max(0.0, risk_free_rate - 0.02), risk_free_rate, min(0.10, risk_free_rate + 0.02)],
        vols=[0.2, 0.4, 0.6, 0.8],
    )
    st.dataframe(sens, use_container_width=True)

    st.subheader("How to Use the Real Options Lens")
    st.markdown(
        """
- `Equity Option PV`: expected present value of equity using a Black-Scholes-style call approximation and exit probability.
- `Strategic Option Value`: your estimate of career flexibility, network access, and learning compounding.
- `Layoff Penalty`: expected downside to near-term cash flow (partially offset by severance).
- Use sensitivity to test whether your decision flips under different volatility/rate assumptions.
        """
    )


if __name__ == "__main__":
    main()
