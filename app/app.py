# Streamlit app: Job Compensation Monte Carlo (Indiana + County taxes, HSA/401k, vesting, NPV)
# -------------------------------------------------------------------------------------------
# How to run locally:
#   1) pip install streamlit numpy pandas altair openpyxl xlsxwriter
#   2) streamlit run streamlit_job_comp_mc_app.py
#
# Notes:
# - Defaults reflect the assumptions we've been using (Indiana state tax, optional county tax, 6-year cliff vest for Job 1,
#   triangular distributions for Job 1 bonus and Job 2 raises, max HSA + 401k deferrals, 5% growth and discount).
# - You can toggle parameters in the sidebar, choose horizons (5/10/20), and export CSV/XLSX summaries.
# - This app keeps IRS limits constant for simplicity; you can override amounts manually.

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------
# Tax & contribution assumptions (editable)
# --------------------------------------------
# 2025-ish brackets (single filer) used for modeling; adjust as needed.
FED_STD_DED = 15000.0
FED_BRACKETS = [
    (0.00,    11600.0, 0.10),
    (11600.0, 47150.0, 0.12),
    (47150.0, 100525.0, 0.22),
    (100525.0, 191950.0, 0.24),
    (191950.0, 243725.0, 0.32),
    (243725.0, 609350.0, 0.35),
    (609350.0, float('inf'), 0.37),
]

IN_STATE_RATE = 0.03  # Indiana state income tax (flat). County added separately via UI.

# FICA
SS_RATE = 0.062
SS_WAGE_BASE = 176_100.0
MED_RATE = 0.0145
ADD_MED_RATE = 0.009
ADD_MED_THRESH = 200_000.0

# IRS limits (can be overridden in UI)
DEFAULT_EMP_401K_LIMIT = 23_500.0
DEFAULT_EMP_HSA_LIMIT_SINGLE = 4_300.0
DEFAULT_EMP_HSA_LIMIT_FAMILY = 8_550.0
K_OVERALL_LIMIT = 70_000.0  # employee + employer additions

# --------------------------------------------
# Helpers
# --------------------------------------------

def federal_tax_from_taxable(taxable: float, brackets: List[Tuple[float, float, float]]) -> float:
    if taxable <= 0:
        return 0.0
    tax = 0.0
    for lower, upper, rate in brackets:
        if taxable > lower:
            taxed_amt = min(taxable, upper) - lower
            if taxed_amt > 0:
                tax += taxed_amt * rate
        else:
            break
    return tax

@dataclass
class JobParams:
    name: str
    base_start: float
    base_raise_fixed: float | None  # If None, uses triangular raises each year.
    raise_tri: Tuple[float, float, float] | None  # (low, mode, high) for annual raise if stochastic
    bonus_rate_fixed: float | None  # If None, uses triangular bonus each year.
    bonus_tri: Tuple[float, float, float] | None  # (low, mode, high) for bonus if stochastic (fraction of base)
    employer_401k_rate_on_total: float  # % of (base + bonus) paid by employer (0 if none)
    vesting_type: str  # 'cliff' or 'graded' or 'immediate'
    cliff_years: int = 6  # applies when vesting_type == 'cliff'
    graded_schedule: List[Tuple[int, float]] | None = None  # list of (year, vested_percent) e.g., [(1,0.25),...]

@dataclass
class GlobalParams:
    horizons: List[int]
    n_trials: int
    invest_return: float
    discount_rate: float
    emp_401k_limit: float
    emp_hsa_limit: float
    hsa_pre_fica: bool
    include_county_tax: bool
    county_rate: float  # additional flat rate on taxable income (approximation)
    std_deduction: float
    fed_brackets: List[Tuple[float, float, float]]

# Tax computation per year

def compute_taxes_and_takehome(gross_wages: float, emp_401k: float, emp_hsa: float, gp: GlobalParams) -> Dict[str, float]:
    # FICA wages exclude HSA but include 401k deferrals
    fica_wages = max(0.0, gross_wages - (emp_hsa if gp.hsa_pre_fica else 0.0))
    ss_tax = SS_RATE * min(fica_wages, SS_WAGE_BASE)
    med_tax = MED_RATE * fica_wages
    add_med_tax = ADD_MED_RATE * max(0.0, fica_wages - ADD_MED_THRESH)
    fica = ss_tax + med_tax + add_med_tax

    # Income taxes — simplify by using same taxable base for federal/state/county
    taxable_income = max(0.0, gross_wages - emp_401k - emp_hsa - gp.std_deduction)
    federal_tax = federal_tax_from_taxable(taxable_income, gp.fed_brackets)
    state_tax = IN_STATE_RATE * taxable_income
    county_tax = (gp.county_rate * taxable_income) if gp.include_county_tax else 0.0

    take_home = gross_wages - emp_401k - emp_hsa - federal_tax - state_tax - county_tax - fica
    return {
        "federal_tax": federal_tax,
        "state_tax": state_tax,
        "county_tax": county_tax,
        "fica": fica,
        "take_home": take_home,
    }

# Vesting helpers

def vested_fraction(year_index_1based: int, job: JobParams) -> float:
    if job.vesting_type == 'immediate':
        return 1.0
    if job.vesting_type == 'cliff':
        return 1.0 if year_index_1based >= job.cliff_years else 0.0
    if job.vesting_type == 'graded' and job.graded_schedule:
        # schedule is cumulative fraction vested at each year (1-based)
        y = year_index_1based
        frac = 0.0
        for yr, f in job.graded_schedule:
            if y >= yr:
                frac = f
        return min(1.0, max(0.0, frac))
    return 0.0

# Monte Carlo engine (both jobs together)

def npv_of_flows(flows: List[float], rate: float) -> float:
    return sum(cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(flows))


def simulate_jobs(years: int, n_trials: int, job1: JobParams, job2: JobParams, gp: GlobalParams) -> pd.DataFrame:
    records: List[Dict] = []

    # Pre-draw stochastic series
    # Job 1 bonus% triangular each year (can include 0 for "not guaranteed")
    if job1.bonus_rate_fixed is None and job1.bonus_tri is not None:
        j1_bonus_draws = np.random.triangular(job1.bonus_tri[0], job1.bonus_tri[1], job1.bonus_tri[2], size=(n_trials, years))
    else:
        j1_bonus_draws = np.full((n_trials, years), job1.bonus_rate_fixed or 0.0)

    # Job 2 raises triangular each year
    if job2.base_raise_fixed is None and job2.raise_tri is not None:
        j2_raise_draws = np.random.triangular(job2.raise_tri[0], job2.raise_tri[1], job2.raise_tri[2], size=(n_trials, years))
    else:
        j2_raise_draws = np.full((n_trials, years), job2.base_raise_fixed or 0.0)

    # Job 1 fixed raises per year (if provided)
    j1_raise_draws = np.full((n_trials, years), job1.base_raise_fixed or 0.0)

    # Job 1 bonus draws already set; if fixed, j1_bonus_draws is constant

    for trial in range(n_trials):
        # --------------------------
        # Job 1 path
        # --------------------------
        base1 = job1.base_start
        emp_401k_bal_1 = 0.0
        emp_hsa_bal_1 = 0.0
        er_401k_bal_1 = 0.0  # employer contributions accumulate with growth
        er_bal_progress = []  # to compute vesting increments precisely
        takehomes_1: List[float] = []

        for yr in range(1, years + 1):
            bonus_pct = j1_bonus_draws[trial, yr - 1]
            bonus1 = bonus_pct * base1
            gross1 = base1 + bonus1

            emp_401k = min(gp.emp_401k_limit, gross1)
            emp_hsa = min(gp.emp_hsa_limit, max(0.0, gross1 - emp_401k))

            # Employer 401k contribution on (base + bonus), limited by overall additions
            tentative_er = job1.employer_401k_rate_on_total * gross1
            er_401k = min(tentative_er, max(0.0, K_OVERALL_LIMIT - emp_401k))

            taxes1 = compute_taxes_and_takehome(gross1, emp_401k, emp_hsa, gp)
            takehomes_1.append(taxes1["take_home"])

            # Grow prior balances then add this year's contributions
            emp_401k_bal_1 = emp_401k_bal_1 * (1 + gp.invest_return) + emp_401k
            emp_hsa_bal_1 = emp_hsa_bal_1 * (1 + gp.invest_return) + emp_hsa
            er_401k_bal_1 = er_401k_bal_1 * (1 + gp.invest_return) + er_401k
            er_bal_progress.append(er_401k_bal_1)

            # Next year's base (fixed raise for J1)
            base1 *= (1 + j1_raise_draws[trial, yr - 1])

        # Vested ER flows for NPV under vesting schedule
        vested_flows_1 = [0.0] * years
        prev_vested_amt = 0.0
        for yr in range(1, years + 1):
            vest_frac = vested_fraction(yr, job1)
            total_er_at_yr = er_bal_progress[yr - 1]
            vested_amt = vest_frac * total_er_at_yr
            increment = max(0.0, vested_amt - prev_vested_amt)
            vested_flows_1[yr - 1] = increment
            prev_vested_amt = vested_amt

        vested_er_total_1 = prev_vested_amt
        ending_fv_1 = emp_401k_bal_1 + emp_hsa_bal_1 + vested_er_total_1
        npv_1 = npv_of_flows(takehomes_1, gp.discount_rate) + npv_of_flows(vested_flows_1, gp.discount_rate)

        records.append({
            "job": job1.name,
            "years": years,
            "sum_takehome": sum(takehomes_1),
            "ending_fv_total_invested": ending_fv_1,
            "npv_cash_plus_vested_er": npv_1,
        })

        # --------------------------
        # Job 2 path
        # --------------------------
        base2 = job2.base_start
        emp_401k_bal_2 = 0.0
        emp_hsa_bal_2 = 0.0
        takehomes_2: List[float] = []

        for yr in range(1, years + 1):
            raise_pct = j2_raise_draws[trial, yr - 1]
            bonus2 = (job2.bonus_rate_fixed or 0.0) * base2
            gross2 = base2 + bonus2

            emp_401k = min(gp.emp_401k_limit, gross2)
            emp_hsa = min(gp.emp_hsa_limit, max(0.0, gross2 - emp_401k))

            taxes2 = compute_taxes_and_takehome(gross2, emp_401k, emp_hsa, gp)
            takehomes_2.append(taxes2["take_home"])

            emp_401k_bal_2 = emp_401k_bal_2 * (1 + gp.invest_return) + emp_401k
            emp_hsa_bal_2 = emp_hsa_bal_2 * (1 + gp.invest_return) + emp_hsa

            # Next year's base after stochastic raise
            base2 *= (1 + raise_pct)

        ending_fv_2 = emp_401k_bal_2 + emp_hsa_bal_2
        npv_2 = npv_of_flows(takehomes_2, gp.discount_rate)

        records.append({
            "job": job2.name,
            "years": years,
            "sum_takehome": sum(takehomes_2),
            "ending_fv_total_invested": ending_fv_2,
            "npv_cash_plus_vested_er": npv_2,
        })

    return pd.DataFrame.from_records(records)

# --------------------------------------------
# UI
# --------------------------------------------
st.set_page_config(page_title="Job Compensation Monte Carlo (IN)", layout="wide")
st.title("Job Compensation Monte Carlo – Indiana + County taxes")

with st.sidebar:
    st.header("Global settings")
    horizons = st.multiselect("Horizon years", [5, 10, 20], default=[5, 10, 20])
    n_trials = st.slider("Monte Carlo trials", 500, 20000, 5000, step=500)
    invest_return = st.slider("Investment growth (HSA & 401k)", 0.0, 0.12, 0.05, 0.005)
    discount_rate = st.slider("NPV discount rate", 0.0, 0.12, 0.05, 0.005)

    st.divider()
    st.subheader("Taxes & limits")
    include_county = st.checkbox("Include county income tax", value=True)
    county_rate = st.slider("County tax rate", 0.0, 0.035, 0.02, 0.0025, help="Approx. Marion County ~2% (set yours)")

    filing_single = st.checkbox("Single filer (affects std deduction)", value=True)
    std_ded = st.number_input("Standard deduction", 0.0, 30000.0, FED_STD_DED if filing_single else 30000.0, 100.0)

    st.caption("Federal brackets baked in; update code for future years if needed.")

    st.divider()
    st.subheader("HSA & 401(k) deferrals (employee)")
    hsa_family = st.checkbox("Use family HSA limit", value=False)
    emp_hsa_limit = st.number_input("Annual HSA deferral", 0.0, 20000.0, DEFAULT_EMP_HSA_LIMIT_FAMILY if hsa_family else DEFAULT_EMP_HSA_LIMIT_SINGLE, 50.0)
    emp_401k_limit = st.number_input("Annual 401(k) deferral", 0.0, 100000.0, DEFAULT_EMP_401K_LIMIT, 500.0)
    hsa_pre_fica = st.checkbox("HSA is pre-FICA (cafeteria plan)", value=True)

    st.divider()
    st.subheader("Export")
    auto_run = st.checkbox("Run on parameter change", value=True)

st.subheader("Job setup")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Job 1")
    j1_base = st.number_input("Starting base (Job 1)", 0.0, 2_000_000.0, 125_000.0, 1000.0)
    j1_raise = st.number_input("Annual raise (Job 1, fixed %)", 0.0, 0.20, 0.03, 0.005)
    st.markdown("Bonus (% of base) – triangular each year")
    j1_bonus_low = st.number_input("Bonus low", 0.0, 1.0, 0.00, 0.01)
    j1_bonus_mode = st.number_input("Bonus mode", 0.0, 1.0, 0.20, 0.01)
    j1_bonus_high = st.number_input("Bonus high", 0.0, 1.0, 0.30, 0.01)
    j1_er_rate = st.number_input("Employer 401(k) on (base+bonus)", 0.0, 1.0, 0.15, 0.01)

    vest_type = st.selectbox("Vesting type (employer 401k)", ["cliff", "graded", "immediate"], index=0)
    cliff_years = st.number_input("Cliff years (if cliff)", 1, 10, 6)
    graded_str = st.text_input("Graded schedule (if graded): year:percent,comma...", "1:0.2,2:0.4,3:0.6,4:0.8,5:1.0")

with col2:
    st.markdown("### Job 2")
    j2_base = st.number_input("Starting base (Job 2)", 0.0, 2_000_000.0, 96_000.0, 1000.0)
    st.markdown("Annual raises (Job 2) – triangular each year")
    j2_r_low = st.number_input("Raise low", 0.0, 1.0, 0.07, 0.005)
    j2_r_mode = st.number_input("Raise mode", 0.0, 1.0, 0.135, 0.005)
    j2_r_high = st.number_input("Raise high", 0.0, 1.0, 0.20, 0.005)
    j2_bonus_fixed = st.number_input("Bonus rate (fixed % of base)", 0.0, 1.0, 0.10, 0.01)

# Parse graded schedule input
graded_schedule = None
if vest_type == "graded":
    try:
        parts = [p.strip() for p in graded_str.split(",") if p.strip()]
        graded_schedule = []
        for p in parts:
            y, frac = p.split(":")
            graded_schedule.append((int(y), float(frac)))
        graded_schedule.sort(key=lambda x: x[0])
    except Exception:
        st.warning("Could not parse graded schedule; defaulting to 4-year graded 25/50/75/100%.")
        graded_schedule = [(1, 0.25), (2, 0.50), (3, 0.75), (4, 1.00)]

job1 = JobParams(
    name="Job 1",
    base_start=j1_base,
    base_raise_fixed=j1_raise,
    raise_tri=None,
    bonus_rate_fixed=None,
    bonus_tri=(j1_bonus_low, j1_bonus_mode, j1_bonus_high),
    employer_401k_rate_on_total=j1_er_rate,
    vesting_type=vest_type,
    cliff_years=cliff_years,
    graded_schedule=graded_schedule,
)

job2 = JobParams(
    name="Job 2",
    base_start=j2_base,
    base_raise_fixed=None,
    raise_tri=(j2_r_low, j2_r_mode, j2_r_high),
    bonus_rate_fixed=j2_bonus_fixed,
    bonus_tri=None,
    employer_401k_rate_on_total=0.0,
    vesting_type="immediate",
)

gp = GlobalParams(
    horizons=horizons,
    n_trials=n_trials,
    invest_return=invest_return,
    discount_rate=discount_rate,
    emp_401k_limit=emp_401k_limit,
    emp_hsa_limit=emp_hsa_limit,
    hsa_pre_fica=hsa_pre_fica,
    include_county_tax=include_county,
    county_rate=county_rate,
    std_deduction=std_ded,
    fed_brackets=FED_BRACKETS,
)

# Run button / auto-run
run = auto_run or st.button("Run simulation")

if run and horizons:
    all_trials: List[pd.DataFrame] = []
    np.random.seed(42)  # reproducible runs per parameter set
    for y in horizons:
        dfy = simulate_jobs(y, n_trials, job1, job2, gp)
        all_trials.append(dfy)
    trials_df = pd.concat(all_trials, ignore_index=True)

    # Summaries
    def pct(a, p):
        return float(np.percentile(a, p))

    summary_rows = []
    for y in horizons:
        for job in ["Job 1", "Job 2"]:
            sub = trials_df[(trials_df["years"] == y) & (trials_df["job"] == job)]
            for metric in ["sum_takehome", "ending_fv_total_invested", "npv_cash_plus_vested_er"]:
                arr = sub[metric].to_numpy()
                summary_rows.append({
                    "years": y,
                    "job": job,
                    "metric": metric,
                    "p10": pct(arr, 10),
                    "p25": pct(arr, 25),
                    "p50": pct(arr, 50),
                    "p75": pct(arr, 75),
                    "p90": pct(arr, 90),
                    "mean": float(arr.mean()),
                })
    summary_df = pd.DataFrame(summary_rows)

    # Wide pivot for easy lookups & charts
    wide = summary_df.pivot_table(index=["years", "job"], columns="metric", values=["p10", "p25", "p50", "p75", "p90", "mean"]).reset_index()
    wide.columns = ["|".join([str(c) for c in col if c != ""]) if isinstance(col, tuple) else str(col) for col in wide.columns]
    # create key column
    if "years" in wide.columns and "job" in wide.columns:
        wide.insert(0, "key", wide["years"].astype(str) + "|" + wide["job"])

    st.success("Simulation complete")
    st.dataframe(summary_df)

    # Dashboard-like controls for charts
    st.subheader("Charts")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        chart_year = st.selectbox("Chart year", horizons, index=0)
    with colB:
        metric_friendly = {
            "Sum Take-home ($)": "sum_takehome",
            "Ending FV: HSA+Emp401k+Vested ER ($)": "ending_fv_total_invested",
            "NPV: Cash + Vested ER ($)": "npv_cash_plus_vested_er",
        }
        metric_name = st.selectbox("Metric", list(metric_friendly.keys()), index=0)
        metric_code = metric_friendly[metric_name]
    with colC:
        st.write(" ")
        st.write(" ")
        st.caption("Use metric to toggle Take-home vs NPV vs Ending FV")

    # Selected year band chart (p10/p50/p90) for both jobs
    yr = chart_year
    def get_val(job, stat):
        coln = f"{stat}|{metric_code}"
        row = wide[(wide["years"] == yr) & (wide["job"] == job)]
        return float(row[coln].iloc[0]) if not row.empty else np.nan

    band_data = pd.DataFrame({
        "job": ["Job 1", "Job 1", "Job 1", "Job 2", "Job 2", "Job 2"],
        "stat": ["p10", "p50", "p90", "p10", "p50", "p90"],
        "value": [get_val("Job 1", s) for s in ["p10", "p50", "p90"]] + [get_val("Job 2", s) for s in ["p10", "p50", "p90"]],
    })

    band_chart = alt.Chart(band_data).mark_bar().encode(
        x=alt.X("job:N", title="Job"),
        y=alt.Y("value:Q", title=metric_name),
        color="stat:N",
        column=None,
        tooltip=["job", "stat", alt.Tooltip("value:Q", format=",")],
    ).properties(width=200, height=300)
    st.altair_chart(band_chart, use_container_width=True)

    # Trend lines across 5/10/20 years for p10/p50/p90 for each job
    long_rows = []
    for y in horizons:
        for job in ["Job 1", "Job 2"]:
            for stat in ["p10", "p50", "p90"]:
                long_rows.append({
                    "years": y,
                    "job": job,
                    "stat": stat,
                    "value": get_val(job, stat) if y == yr else float(wide[(wide["years"] == y) & (wide["job"] == job)][f"{stat}|{metric_code}"].iloc[0])
                })
    trend_df = pd.DataFrame(long_rows)

    trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X("years:O", title="Horizon (years)"),
        y=alt.Y("value:Q", title=metric_name),
        color="job:N",
        strokeDash="stat:N",
        tooltip=["years", "job", "stat", alt.Tooltip("value:Q", format=",")],
    ).properties(height=350)
    st.altair_chart(trend_chart, use_container_width=True)

    # Difference table (J2 - J1) for quick comparison
    rows = []
    for y in horizons:
        r = {"years": y}
        for stat in ["p10", "p50", "p90", "mean"]:
            r[f"{stat}_diff"] = (
                float(wide[(wide["years"] == y) & (wide["job"] == "Job 2")][f"{stat}|{metric_code}"].iloc[0])
                - float(wide[(wide["years"] == y) & (wide["job"] == "Job 1")][f"{stat}|{metric_code}"].iloc[0])
            )
        rows.append(r)
    diff_df = pd.DataFrame(rows)
    st.markdown("### Difference (Job 2 − Job 1)")
    st.dataframe(diff_df)

    # Downloads (CSV/XLSX)
    st.subheader("Download results")
    @st.cache_data
    def build_csvs(trials_df: pd.DataFrame, summary_df: pd.DataFrame) -> Dict[str, bytes]:
        csv_trials = trials_df.to_csv(index=False).encode("utf-8")
        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
        return {"trials.csv": csv_trials, "summary.csv": csv_summary}

    blobs = build_csvs(trials_df, summary_df)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download summary (CSV)", data=blobs["summary.csv"], file_name="summary.csv", mime="text/csv")
    with c2:
        st.download_button("Download trials (CSV)", data=blobs["trials.csv"], file_name="trials.csv", mime="text/csv")

    # XLSX bundle
    @st.cache_data
    def build_xlsx(summary_df: pd.DataFrame, trials_df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Summary_Pctiles", index=False)
            trials_10 = trials_df[trials_df["years"] == 10]
            trials_10.to_excel(writer, sheet_name="Raw_Trials_10y", index=False)
        return output.getvalue()

    xlsx_bytes = build_xlsx(summary_df, trials_df)
    st.download_button("Download Excel bundle", data=xlsx_bytes, file_name="job_comp_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Adjust parameters in the sidebar, then click **Run simulation** (or enable auto-run).")
