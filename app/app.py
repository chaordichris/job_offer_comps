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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f5f1e8;
            --ink: #181512;
            --muted: #6a6258;
            --line: #d7cfbf;
            --accent: #1f3b2d;
            --accent-soft: #e4eadf;
        }
        .stApp {
            background: linear-gradient(180deg, #f3efe6 0%, #f8f5ee 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 0.8rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
            color: var(--ink);
            font-weight: 500;
        }
        h1 {
            font-size: 2.2rem;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }
        h2 {
            font-size: 1.05rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            margin-top: 0.3rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .deck {
            color: var(--muted);
            max-width: 68ch;
            line-height: 1.55;
            margin-bottom: 1.25rem;
            font-size: 1rem;
        }
        .rule {
            border-top: 1px solid var(--line);
            margin: 0.8rem 0 1.15rem 0;
        }
        .note {
            border-left: 2px solid var(--line);
            padding: 0.15rem 0 0.15rem 0.8rem;
            color: var(--muted);
            font-size: 0.9rem;
            background: transparent;
        }
        [data-testid="stMetric"] {
            background: transparent;
            border: 0;
            border-top: 1px solid var(--line);
            padding: 0.55rem 0.15rem 0.25rem 0.15rem;
            border-radius: 0;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.7rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            letter-spacing: -0.02em;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.36);
            border-radius: 0;
            box-shadow: none;
        }
        .offer-header {
            border-top: 2px solid var(--ink);
            padding-top: 0.55rem;
            margin-bottom: 0.6rem;
        }
        .offer-header h3 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .verdict {
            border-top: 2px solid var(--ink);
            border-bottom: 1px solid var(--line);
            padding: 1rem 0 0.9rem 0;
            margin: 0.65rem 0 1.1rem 0;
            line-height: 1.35;
        }
        .verdict strong {
            font-size: 1.18rem;
            font-weight: 600;
        }
        .marginal {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.4;
            border-top: 1px solid var(--line);
            padding-top: 0.55rem;
        }
        .smallcaps {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: var(--muted);
        }
        .inline-def {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.25rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.28);
            padding: 0.15rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
            background: rgba(249,246,239,0.9);
        }
        [data-testid="stTabs"] button {
            border-radius: 0 !important;
        }
        .caption-row {
            display: grid;
            grid-template-columns: 2.2fr 1fr;
            gap: 1.5rem;
            align-items: start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def format_comparison_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        df.style.format("${:,.0f}")
        .set_properties(**{"text-align": "right"})
        .set_table_styles(
            [
                {"selector": "th.col_heading", "props": "text-align:right; font-weight:600;"},
                {"selector": "th.row_heading", "props": "text-align:left; color:#6a6258; font-weight:500;"},
                {"selector": "td", "props": "border-bottom:1px solid #e6dece;"},
                {"selector": "th", "props": "border-bottom:1px solid #d7cfbf;"},
                {"selector": "table", "props": "font-size:0.92rem;"},
            ]
        )
    )


def offer_inputs(label: str, defaults: dict[str, float | str]) -> Offer:
    st.markdown(f'<div class="offer-header"><h3>{label}</h3></div>', unsafe_allow_html=True)
    name = st.text_input(f"{label} name", value=str(defaults["name"]), key=f"{label}_name", label_visibility="collapsed")

    with st.expander("Compensation", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            base_salary = st.number_input(f"{label} base salary", min_value=0.0, value=float(defaults["base_salary"]), step=5000.0)
            target_bonus_pct = st.slider(f"{label} target bonus (%)", 0.0, 100.0, float(defaults["target_bonus_pct"]) * 100, 1.0) / 100.0
            sign_on = st.number_input(f"{label} sign-on", min_value=0.0, value=float(defaults["sign_on"]), step=5000.0)
        with c2:
            annual_equity_grant_value = st.number_input(
                f"{label} annual equity grant (current fair value)", min_value=0.0, value=float(defaults["annual_equity_grant_value"]), step=5000.0
            )
            vesting_years = st.slider(f"{label} vesting period (years)", 1.0, 6.0, float(defaults["vesting_years"]), 0.5)
            time_to_liquidity_years = st.slider(f"{label} years to liquidity", 0.5, 10.0, float(defaults["time_to_liquidity_years"]), 0.5)

    with st.expander("Equity / Exit assumptions", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            company_value_now = st.number_input(
                f"{label} company value now", min_value=1.0, value=float(defaults["company_value_now"]), step=1000000.0
            )
            option_strike_value = st.number_input(
                f"{label} option strike (company value basis)", min_value=1.0, value=float(defaults["option_strike_value"]), step=1000000.0
            )
            expected_company_value_at_liquidity = st.number_input(
                f"{label} expected company value at liquidity", min_value=1.0, value=float(defaults["expected_company_value_at_liquidity"]), step=1000000.0
            )
        with c2:
            equity_volatility = st.slider(f"{label} equity volatility", 0.05, 1.50, float(defaults["equity_volatility"]), 0.05)
            exit_probability = st.slider(f"{label} exit/liquidity probability", 0.0, 1.0, float(defaults["exit_probability"]), 0.05)
            layoff_probability = st.slider(f"{label} layoff probability (1Y)", 0.0, 1.0, float(defaults["layoff_probability"]), 0.05)
            severance_months = st.slider(f"{label} severance months", 0.0, 12.0, float(defaults["severance_months"]), 0.5)

    with st.expander("Career optionality", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            decision_flex_value = st.number_input(
                f"{label} future decision flexibility value", min_value=0.0, value=float(defaults["decision_flex_value"]), step=1000.0
            )
        with c2:
            remote_value = st.number_input(f"{label} remote/commute value", value=float(defaults["remote_value"]), step=1000.0)
        with c3:
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
    inject_styles()

    st.markdown('<div class="eyebrow">Offer Analysis</div>', unsafe_allow_html=True)
    st.title("Job Offers, Read as Optionality")
    st.markdown(
        '<div class="deck">A restrained decision worksheet inspired by editorial sports design: compare immediate cash, probabilistic equity upside, and the quieter compounding value of flexibility and learning.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="smallcaps">Global assumptions</div>', unsafe_allow_html=True)
    risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.04, 0.005)
    comparison_horizon = st.sidebar.slider("Comparison horizon (years)", 1, 5, 1)
    st.sidebar.caption("Equity is treated as a call-option proxy; strategic factors are user-entered real-option value.")

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

    left, right = st.columns([1, 1], gap="large")
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

    a_total = a_metrics["1Y Total Expected Value"]
    b_total = b_metrics["1Y Total Expected Value"]
    winner = offer_a.name if a_total >= b_total else offer_b.name
    delta = abs(a_total - b_total)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="caption-row">
          <div class="inline-def">
            <span class="smallcaps">Method</span><br>
            Expected value = immediate cash + probability-adjusted equity option present value + strategic option value
            + expected severance value − layoff penalty. Use this as a comparative lens, not a precise forecast.
          </div>
          <div class="marginal">
            The inputs that most often change the winner are <em>exit probability</em>, <em>expected exit value</em>, and your estimate of <em>career optionality</em>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="verdict"><strong>Verdict:</strong> {winner} leads by {fmt_money(delta)} in 1-year expected value under current assumptions.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.1, 1, 1])
    c1.metric("Preferred Offer", winner)
    c2.metric("Expected Value Delta", f"${delta:,.0f}")
    c3.metric(
        "Equity Upside Difference",
        f"${abs(a_metrics['Equity Option PV'] - b_metrics['Equity Option PV']):,.0f}",
    )

    tabs = st.tabs(["Breakdown", "Sensitivity", "Notes"])

    with tabs[0]:
        st.subheader("Value Breakdown")
        st.markdown('<div class="inline-def">Read vertically. The top line is the decision summary; the lower lines explain why.</div>', unsafe_allow_html=True)
        st.dataframe(format_comparison_table(comparison), use_container_width=True)

        summary_rows = [
            {
                "Component": "Cash (base + bonus + sign-on)",
                offer_a.name: a_metrics["Annual Cash"] + a_metrics["Sign-On"],
                offer_b.name: b_metrics["Annual Cash"] + b_metrics["Sign-On"],
            },
            {
                "Component": "Equity option present value",
                offer_a.name: a_metrics["Equity Option PV"],
                offer_b.name: b_metrics["Equity Option PV"],
            },
            {
                "Component": "Strategic option value",
                offer_a.name: a_metrics["Strategic Option Value"],
                offer_b.name: b_metrics["Strategic Option Value"],
            },
            {
                "Component": "Downside adjustment (penalty - severance)",
                offer_a.name: a_metrics["Downside Risk"],
                offer_b.name: b_metrics["Downside Risk"],
            },
        ]
        summary_df = pd.DataFrame(summary_rows).set_index("Component")
        st.dataframe(format_comparison_table(summary_df), use_container_width=True)

    with tabs[1]:
        st.subheader("Sensitivity (Rate x Volatility)")
        st.markdown('<div class="inline-def">This grid shows where the preferred offer flips as financing conditions and equity uncertainty change.</div>', unsafe_allow_html=True)
        sens = sensitivity_table(
            offer_a,
            offer_b,
            rates=[max(0.0, risk_free_rate - 0.02), risk_free_rate, min(0.10, risk_free_rate + 0.02)],
            vols=[0.2, 0.4, 0.6, 0.8],
        )
        pivot = sens.pivot(index="Vol", columns="Rate", values="Winner")
        st.dataframe(pivot, use_container_width=True)
        with st.expander("Detailed sensitivity rows", expanded=False):
            st.dataframe(sens, use_container_width=True)

    with tabs[2]:
        st.markdown(
            """
            <div class="note">
            <strong>Model notes.</strong> This is a decision aid, not a valuation report.
            Equity is treated as an option-like payoff with user-specified probability of liquidity.
            The most important inputs are usually exit probability, expected company value at liquidity, and your career optionality estimates.
            </div>
            """,
            unsafe_allow_html=True,
        )
        left_note, right_note = st.columns([1.2, 1], gap="large")
        with left_note:
            st.markdown(
                """
                - `Equity Option PV`: Black-Scholes-style approximation scaled to your grant and discounted.
                - `Strategic Option Value`: your subjective estimate of flexibility, learning, and network compounding.
                - `Layoff Penalty`: expected near-term cash disruption, partially offset by severance.
                """
            )
        with right_note:
            st.markdown(
                """
                <div class="marginal">
                Tufte principle here: keep explanatory text adjacent to the numbers it explains.
                Adjust one assumption at a time and watch whether the verdict changes.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
