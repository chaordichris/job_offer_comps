from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------
# Tax & contribution assumptions (editable)
# --------------------------------------------
FED_STD_DED = 15000.0  # single filer (approx.)
FED_BRACKETS = [
    (0.00,    11600.0, 0.10),
    (11600.0, 47150.0, 0.12),
    (47150.0, 100525.0, 0.22),
    (100525.0, 191950.0, 0.24),
    (191950.0, 243725.0, 0.32),
    (243725.0, 609350.0, 0.35),
    (609350.0, float('inf'), 0.37),
]
IN_STATE_RATE = 0.03  # Indiana flat income tax (county handled separately)

# FICA
SS_RATE = 0.062
SS_WAGE_BASE = 176_100.0
MED_RATE = 0.0145
ADD_MED_RATE = 0.009
ADD_MED_THRESH = 200_000.0

# IRS limits (override in UI if needed)
DEFAULT_EMP_401K_LIMIT = 23_500.0
DEFAULT_EMP_HSA_LIMIT_SINGLE = 4_300.0
DEFAULT_EMP_HSA_LIMIT_FAMILY = 8_550.0
K_OVERALL_LIMIT = 70_000.0  # employee + employer

# --------------------------------------------
# Data classes
# --------------------------------------------
@dataclass
class JobParams:
    name: str
    base_start: float
    # Raises: fixed % or triangular (low, mode, high)
    raise_type: str  # 'fixed' | 'tri'
    base_raise_fixed: Optional[float] = None
    raise_tri: Optional[Tuple[float, float, float]] = None
    # Bonuses: fixed % of base or triangular % of base
    bonus_type: str  # 'fixed' | 'tri'
    bonus_rate_fixed: Optional[float] = None
    bonus_tri: Optional[Tuple[float, float, float]] = None
    # Employer 401k & vesting
    employer_401k_rate_on_total: float = 0.0  # % of (base + bonus)
    vesting_type: str = 'immediate'          # 'immediate' | 'cliff' | 'graded'
    cliff_years: int = 0
    graded_schedule: Optional[List[Tuple[int, float]]] = None

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
    county_rate: float
    std_deduction: float
    fed_brackets: List[Tuple[float, float, float]]

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


def compute_taxes_and_takehome(gross_wages: float, emp_401k: float, emp_hsa: float, gp: GlobalParams) -> Dict[str, float]:
    # FICA wages exclude HSA but include 401k deferrals
    fica_wages = max(0.0, gross_wages - (emp_hsa if gp.hsa_pre_fica else 0.0))
    ss_tax = SS_RATE * min(fica_wages, SS_WAGE_BASE)
    med_tax = MED_RATE * fica_wages
    add_med_tax = ADD_MED_RATE * max(0.0, fica_wages - ADD_MED_THRESH)
    fica = ss_tax + med_tax + add_med_tax

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


def vested_fraction(year_index_1based: int, job: JobParams) -> float:
    if job.vesting_type == 'immediate':
        return 1.0
    if job.vesting_type == 'cliff':
        return 1.0 if year_index_1based >= max(1, job.cliff_years) else 0.0
    if job.vesting_type == 'graded' and job.graded_schedule:
        y = year_index_1based
        frac = 0.0
        for yr, f in job.graded_schedule:
            if y >= yr:
                frac = f
        return float(min(1.0, max(0.0, frac)))
    return 0.0


def npv_of_flows(flows: List[float], rate: float) -> float:
    return sum(cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(flows))

# --------------------------------------------
# Monte Carlo engine
# --------------------------------------------

def _draw_series(n_trials: int, years: int, typ: str, fixed: Optional[float], tri: Optional[Tuple[float, float, float]]):
    if typ == 'fixed':
        return np.full((n_trials, years), fixed or 0.0)
    lo, mo, hi = tri or (0.0, 0.0, 0.0)
    return np.random.triangular(lo, mo, hi, size=(n_trials, years))


def simulate_jobs(years: int, n_trials: int, jobA: JobParams, jobB: JobParams, gp: GlobalParams) -> pd.DataFrame:
    records: List[Dict] = []

    # Pre-draw stochastic series
    a_raise = _draw_series(n_trials, years, jobA.raise_type, jobA.base_raise_fixed, jobA.raise_tri)
    a_bonus = _draw_series(n_trials, years, jobA.bonus_type, jobA.bonus_rate_fixed, jobA.bonus_tri)
    b_raise = _draw_series(n_trials, years, jobB.raise_type, jobB.base_raise_fixed, jobB.raise_tri)
    b_bonus = _draw_series(n_trials, years, jobB.bonus_type, jobB.bonus_rate_fixed, jobB.bonus_tri)

    # ---- Job A
    for trial in range(n_trials):
        # Job A path
        base = jobA.base_start
        emp_401k_bal = 0.0
        emp_hsa_bal = 0.0
        er_401k_bal = 0.0
        er_prog: List[float] = []
        takehomes: List[float] = []

        for yr in range(1, years + 1):
            bonus_pct = float(a_bonus[trial, yr - 1])
            raise_pct = float(a_raise[trial, yr - 1])
            bonus = bonus_pct * base
            gross = base + bonus

            emp_401k = min(gp.emp_401k_limit, gross)
            emp_hsa = min(gp.emp_hsa_limit, max(0.0, gross - emp_401k))
            tentative_er = jobA.employer_401k_rate_on_total * gross
            er_401k = min(tentative_er, max(0.0, K_OVERALL_LIMIT - emp_401k))

            taxes = compute_taxes_and_takehome(gross, emp_401k, emp_hsa, gp)
            takehomes.append(taxes["take_home"])

            emp_401k_bal = emp_401k_bal * (1 + gp.invest_return) + emp_401k
            emp_hsa_bal = emp_hsa_bal * (1 + gp.invest_return) + emp_hsa
            er_401k_bal = er_401k_bal * (1 + gp.invest_return) + er_401k
            er_prog.append(er_401k_bal)

            base *= (1 + raise_pct)

        # Vested flows for NPV
        vested_flows = [0.0] * years
        prev_vested = 0.0
        for yr in range(1, years + 1):
            frac = vested_fraction(yr, jobA)
            total_er = er_prog[yr - 1]
            vested_amt = frac * total_er
            inc = max(0.0, vested_amt - prev_vested)
            vested_flows[yr - 1] = inc
            prev_vested = vested_amt

        ending_vested_er = prev_vested
        ending_fv = emp_401k_bal + emp_hsa_bal + ending_vested_er
        npv_val = npv_of_flows(takehomes, gp.discount_rate) + npv_of_flows(vested_flows, gp.discount_rate)

        records.append({
            "job": jobA.name,
            "years": years,
            "sum_takehome": sum(takehomes),
            "ending_fv_total_invested": ending_fv,
            "npv_cash_plus_vested_er": npv_val,
        })

        # Job B path
        base = jobB.base_start
        emp_401k_bal = 0.0
        emp_hsa_bal = 0.0
        er_401k_bal = 0.0
        er_prog = []
        takehomes = []

        for yr in range(1, years + 1):
            bonus_pct = float(b_bonus[trial, yr - 1])
            raise_pct = float(b_raise[trial, yr - 1])
            bonus = bonus_pct * base
            gross = base + bonus

            emp_401k = min(gp.emp_401k_limit, gross)
            emp_hsa = min(gp.emp_hsa_limit, max(0.0, gross - emp_401k))
            tentative_er = jobB.employer_401k_rate_on_total * gross
            er_401k = min(tentative_er, max(0.0, K_OVERALL_LIMIT - emp_401k))

            taxes = compute_taxes_and_takehome(gross, emp_401k, emp_hsa, gp)
            takehomes.append(taxes["take_home"])

            emp_401k_bal = emp_401k_bal * (1 + gp.invest_return) + emp_401k
            emp_hsa_bal = emp_hsa_bal * (1 + gp.invest_return) + emp_hsa
            er_401k_bal = er_401k_bal * (1 + gp.invest_return) + er_401k
            er_prog.append(er_401k_bal)

            base *= (1 + raise_pct)

        vested_flows = [0.0] * years
        prev_vested = 0.0
        for yr in range(1, years + 1):
            frac = vested_fraction(yr, jobB)
            total_er = er_prog[yr - 1]
            vested_amt = frac * total_er
            inc = max(0.0, vested_amt - prev_vested)
            vested_flows[yr - 1] = inc
            prev_vested = vested_amt

        ending_vested_er = prev_vested
        ending_fv = emp_401k_bal + emp_hsa_bal + ending_vested_er
        npv_val = npv_of_flows(takehomes, gp.discount_rate) + npv_of_flows(vested_flows, gp.discount_rate)

        records.append({
            "job": jobB.name,
            "years": years,
            "sum_takehome": sum(takehomes),
            "ending_fv_total_invested": ending_fv,
            "npv_cash_plus_vested_er": npv_val,
        })

    return pd.DataFrame.from_records(records)

# --------------------------------------------
# UI
# --------------------------------------------
st.set_page_config(page_title="Job Compensation Monte Carlo (IN)", layout="wide")
st.title("Job Compensation Monte Carlo – Indiana + County taxes")

# Presets so inputs are selectable (not hard-coded)
PRESETS = {
    "Original Job 1": dict(
        base_start=125000.0,
        raise_type='fixed', base_raise_fixed=0.03,
        bonus_type='tri', bonus_tri=(0.0, 0.20, 0.30),
        employer_401k_rate_on_total=0.15,
        vesting_type='cliff', cliff_years=6,
    ),
    "Original Job 2": dict(
        base_start=96000.0,
        raise_type='tri', raise_tri=(0.07, 0.135, 0.20),
        bonus_type='fixed', bonus_rate_fixed=0.10,
        employer_401k_rate_on_total=0.0,
        vesting_type='immediate',
    ),
    "Blank Custom": dict(
        base_start=100000.0,
        raise_type='fixed', base_raise_fixed=0.03,
        bonus_type='fixed', bonus_rate_fixed=0.10,
        employer_401k_rate_on_total=0.00,
        vesting_type='immediate',
    ),
}

with st.sidebar:
    st.header("Global settings")
    horizons = st.multiselect("Horizon years", [5, 10, 20], default=[5, 10, 20])
    n_trials = st.slider("Monte Carlo trials", 500, 20000, 5000, step=500)
    invest_return = st.slider("Investment growth (HSA & 401k)", 0.0, 0.12, 0.05, 0.005)
    discount_rate = st.slider("NPV discount rate", 0.0, 0.12, 0.05, 0.005)

    st.divider()
    st.subheader("Taxes & limits")
    include_county = st.checkbox("Include county income tax", value=True)
    county_rate = st.slider("County tax rate", 0.0, 0.035, 0.02, 0.0025, help="Approx. Marion County ~2%")

    filing_single = st.checkbox("Single filer (affects std deduction)", value=True)
    std_ded = st.number_input("Standard deduction", 0.0, 50000.0, FED_STD_DED if filing_single else 30000.0, 100.0)

    st.caption("Federal brackets baked in; update code for future years if needed.")

    st.divider()
    st.subheader("HSA & 401(k) deferrals (employee)")
    hsa_family = st.checkbox("Use family HSA limit", value=False)
    emp_hsa_limit = st.number_input("Annual HSA deferral", 0.0, 20000.0, DEFAULT_EMP_HSA_LIMIT_FAMILY if hsa_family else DEFAULT_EMP_HSA_LIMIT_SINGLE, 50.0)
    emp_401k_limit = st.number_input("Annual 401(k) deferral", 0.0, 100000.0, DEFAULT_EMP_401K_LIMIT, 500.0)
    hsa_pre_fica = st.checkbox("HSA is pre-FICA (cafeteria plan)", value=True)

    st.divider()
    auto_run = st.checkbox("Run on parameter change", value=True)

st.subheader("Job setup")

# Symmetric job control builder

def job_controls(job_label: str, defaults: Dict, key_prefix: str) -> JobParams:
    name = st.text_input(f"Name ({job_label})", value=job_label, key=f"{key_prefix}_name")
    base = st.number_input(f"Starting base ({job_label})", 0.0, 2_000_000.0, defaults.get('base_start', 100000.0), 1000.0, key=f"{key_prefix}_base")

    # Raises
    rt_default = 'fixed' if defaults.get('raise_type','fixed')=='fixed' else 'tri'
    raise_choice = st.radio(f"Annual raises ({job_label})", ["Fixed %", "Triangular"], index=0 if rt_default=='fixed' else 1, key=f"{key_prefix}_raise_type")
    if raise_choice == "Fixed %":
        r_fixed = st.number_input(f"Raise (fixed %, {job_label})", 0.0, 1.0, defaults.get('base_raise_fixed', 0.03), 0.005, key=f"{key_prefix}_rfix")
        r_tri = None
        r_type = 'fixed'
    else:
        lo, mo, hi = defaults.get('raise_tri', (0.02, 0.03, 0.05))
        lo = st.number_input(f"Raise low ({job_label})", 0.0, 1.0, lo, 0.005, key=f"{key_prefix}_rlo")
        mo = st.number_input(f"Raise mode ({job_label})", 0.0, 1.0, mo, 0.005, key=f"{key_prefix}_rmo")
        hi = st.number_input(f"Raise high ({job_label})", 0.0, 1.0, hi, 0.005, key=f"{key_prefix}_rhi")
        r_fixed = None
        r_tri = (lo, mo, hi)
        r_type = 'tri'

    # Bonuses
    bt_default = 'fixed' if defaults.get('bonus_type','tri')=='fixed' else 'tri'
    bonus_choice = st.radio(f"Bonus (% of base, {job_label})", ["Fixed %", "Triangular"], index=0 if bt_default=='fixed' else 1, key=f"{key_prefix}_btype")
    if bonus_choice == "Fixed %":
        b_fixed = st.number_input(f"Bonus rate (fixed %, {job_label})", 0.0, 1.0, defaults.get('bonus_rate_fixed', 0.10), 0.01, key=f"{key_prefix}_bfix")
        b_tri = None
        b_type = 'fixed'
    else:
        blo, bmo, bhi = defaults.get('bonus_tri', (0.0, 0.20, 0.30))
        blo = st.number_input(f"Bonus low ({job_label})", 0.0, 1.0, blo, 0.01, key=f"{key_prefix}_blo")
        bmo = st.number_input(f"Bonus mode ({job_label})", 0.0, 1.0, bmo, 0.01, key=f"{key_prefix}_bmo")
        bhi = st.number_input(f"Bonus high ({job_label})", 0.0, 1.0, bhi, 0.01, key=f"{key_prefix}_bhi")
        b_fixed = None
        b_tri = (blo, bmo, bhi)
        b_type = 'tri'

    er_rate = st.number_input(f"Employer 401(k) on (base+bonus), {job_label}", 0.0, 1.0, defaults.get('employer_401k_rate_on_total', 0.0), 0.01, key=f"{key_prefix}_er")

    vest_type = st.selectbox(f"Vesting type ({job_label})", ["immediate", "cliff", "graded"], index=["immediate","cliff","graded"].index(defaults.get('vesting_type','immediate')), key=f"{key_prefix}_vest")
    cliff_years = 0
    graded_sched = None
    if vest_type == "cliff":
        cliff_years = st.number_input(f"Cliff years ({job_label})", 1, 10, defaults.get('cliff_years', 6), key=f"{key_prefix}_cliff")
    elif vest_type == "graded":
        gs_default = defaults.get('graded_schedule_str', "1:0.25,2:0.50,3:0.75,4:1.0")
        gs_str = st.text_input(f"Graded schedule year:frac, {job_label}", gs_default, key=f"{key_prefix}_gs")
        try:
            graded_sched = []
            for p in [x.strip() for x in gs_str.split(',') if x.strip()]:
                y, f = p.split(':')
                graded_sched.append((int(y), float(f)))
            graded_sched.sort(key=lambda x: x[0])
        except Exception:
            st.warning(f"Could not parse graded schedule for {job_label}; defaulting to 4-year 25/50/75/100%.")
            graded_sched = [(1,0.25),(2,0.50),(3,0.75),(4,1.00)]

    return JobParams(
        name=name,
        base_start=base,
        raise_type=r_type,
        base_raise_fixed=r_fixed,
        raise_tri=r_tri,
        bonus_type=b_type,
        bonus_rate_fixed=b_fixed,
        bonus_tri=b_tri,
        employer_401k_rate_on_total=er_rate,
        vesting_type=vest_type,
        cliff_years=cliff_years,
        graded_schedule=graded_sched,
    )

# --- Presets (choose what each job starts from) ---
PRESETS = {
    "Original Job 1": dict(
        base_start=125000.0,
        raise_type="fixed", base_raise_fixed=0.03,
        bonus_type="tri", bonus_tri=(0.00, 0.20, 0.30),
        employer_401k_rate_on_total=0.15,
        vesting_type="cliff", cliff_years=6,
    ),
    "Original Job 2": dict(
        base_start=96000.0,
        raise_type="tri", raise_tri=(0.07, 0.135, 0.20),
        bonus_type="fixed", bonus_rate_fixed=0.10,
        employer_401k_rate_on_total=0.00,
        vesting_type="immediate",
    ),
    "Blank Custom": dict(
        base_start=100000.0,
        raise_type="fixed", base_raise_fixed=0.03,
        bonus_type="fixed", bonus_rate_fixed=0.10,
        employer_401k_rate_on_total=0.00,
        vesting_type="immediate",
    ),
}

# Place these right under st.subheader("Job setup")
st.markdown("### Presets")
col_p1, col_p2 = st.columns(2)
with col_p1:
    preset1 = st.selectbox("Preset for Job 1", list(PRESETS.keys()), index=0, key="preset1")
with col_p2:
    preset2 = st.selectbox("Preset for Job 2", list(PRESETS.keys()), index=1, key="preset2")

# Editors (use the chosen preset as defaults)
col1, col2 = st.columns(2)
with col1:
    jobA = job_controls("Job 1", defaults=PRESETS[preset1], key_prefix="j1")
with col2:
    jobB = job_controls("Job 2", defaults=PRESETS[preset2], key_prefix="j2")


# Global params
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

# Run
run = auto_run or st.button("Run simulation")

if run and horizons:
    all_trials: List[pd.DataFrame] = []
    np.random.seed(42)  # reproducible per parameter set
    for y in horizons:
        dfy = simulate_jobs(y, n_trials, jobA, jobB, gp)
        all_trials.append(dfy)
    trials_df = pd.concat(all_trials, ignore_index=True)

    # Summaries
    def pct(a, p):
        return float(np.percentile(a, p))

    summary_rows = []
    for y in horizons:
        for job_name in [jobA.name, jobB.name]:
            sub = trials_df[(trials_df["years"] == y) & (trials_df["job"] == job_name)]
            for metric in ["sum_takehome", "ending_fv_total_invested", "npv_cash_plus_vested_er"]:
                arr = sub[metric].to_numpy()
                summary_rows.append({
                    "years": y,
                    "job": job_name,
                    "metric": metric,
                    "p10": pct(arr, 10),
                    "p25": pct(arr, 25),
                    "p50": pct(arr, 50),
                    "p75": pct(arr, 75),
                    "p90": pct(arr, 90),
                    "mean": float(arr.mean()),
                })
    summary_df = pd.DataFrame(summary_rows)

    # Wide pivot for charts
    wide = summary_df.pivot_table(index=["years", "job"], columns="metric", values=["p10", "p25", "p50", "p75", "p90", "mean"]).reset_index()
    wide.columns = ["|".join([str(c) for c in col if c != ""]) if isinstance(col, tuple) else str(col) for col in wide.columns]
    if "years" in wide.columns and "job" in wide.columns:
        wide.insert(0, "key", wide["years"].astype(str) + "|" + wide["job"])

    st.success("Simulation complete")
    st.dataframe(summary_df)

    # Charts
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
        st.caption("Use Metric to toggle Take-home vs NPV vs Ending FV")

    yr = chart_year
    def get_val(job_name: str, stat: str):
        coln = f"{stat}|{metric_code}"
        row = wide[(wide["years"] == yr) & (wide["job"] == job_name)]
        return float(row[coln].iloc[0]) if not row.empty else np.nan

    band_data = pd.DataFrame({
        "job": [jobA.name, jobA.name, jobA.name, jobB.name, jobB.name, jobB.name],
        "stat": ["p10", "p50", "p90", "p10", "p50", "p90"],
        "value": [get_val(jobA.name, s) for s in ["p10", "p50", "p90"]] + [get_val(jobB.name, s) for s in ["p10", "p50", "p90"]],
    })

    band_chart = alt.Chart(band_data).mark_bar().encode(
        x=alt.X("job:N", title="Job"),
        y=alt.Y("value:Q", title=metric_name),
        color="stat:N",
        tooltip=["job", "stat", alt.Tooltip("value:Q", format=",")],
    ).properties(width=200, height=300)
    st.altair_chart(band_chart, use_container_width=True)

    long_rows = []
    for y in horizons:
        for job_name in [jobA.name, jobB.name]:
            for stat in ["p10", "p50", "p90"]:
                long_rows.append({
                    "years": y,
                    "job": job_name,
                    "stat": stat,
                    "value": float(wide[(wide["years"] == y) & (wide["job"] == job_name)][f"{stat}|{metric_code}"].iloc[0])
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

    # Difference table (JobB − JobA)
    rows = []
    for y in horizons:
        r = {"years": y}
        for stat in ["p10", "p50", "p90", "mean"]:
            r[f"{stat}_diff"] = (
                float(wide[(wide["years"] == y) & (wide["job"] == jobB.name)][f"{stat}|{metric_code}"].iloc[0])
                - float(wide[(wide["years"] == y) & (wide["job"] == jobA.name)][f"{stat}|{metric_code}"].iloc[0])
            )
        rows.append(r)
    diff_df = pd.DataFrame(rows)
    st.markdown(f"### Difference ({jobB.name} − {jobA.name})")
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
    st.info("Pick presets, enter values, then click **Run simulation** (or enable auto-run).")

