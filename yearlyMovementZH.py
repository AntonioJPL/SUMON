import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import os
from mongo_utils import MongoDB
from datetime import datetime, timedelta
import sys
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import calendar
import json

print("Generating the year plots...")

if len(sys.argv) > 1:
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
else:
    date = datetime.now()

today = date
one_year_ago = today - relativedelta(years=1)
current_year = today.year
data_year = one_year_ago.year

fixed_start = datetime(2018, 1, 1)

output_dir = "D:/github/SUMON-Repo/html/contents"

stress_points = np.array([292, 136, 63, 50, 37, 32, 20])
cycles_points = np.array([1e4, 1e5, 1e6, 2e6, 5e6, 1e7, 1e8])


def sn_curve(N, a, b):
    N = np.clip(N, 1e3, 1e12)
    return a * N ** (-b)


params, _ = curve_fit(sn_curve, cycles_points, stress_points)
a_sn, b_sn = params


def estimate_cycles(stress_input):
    return (stress_input / a_sn) ** (-1 / b_sn)


def estimate_stress(cycles_input):
    return a_sn * cycles_input ** (-b_sn)


def fig_to_responsive_json(fig):
    fig.update_layout(autosize=True)
    spec = fig.to_plotly_json()
    spec.get("layout", {}).pop("width", None)
    spec.get("layout", {}).pop("height", None)
    return spec


def build_global_accumulated_damage(docs, year_now):
    docs_sorted = sorted(docs, key=lambda d: d["T"])
    if not docs_sorted:
        return [], [], [], [], 0.0, 0.0, None, None

    historical_dates = []
    historical_values = []
    year_dates = []
    year_values = []

    accumulated_total = 0.0
    accumulated_yearly = 0.0

    for d in docs_sorted:
        dmg = d.get("DMG", 0.0) or 0.0
        if dmg == 0:
            continue

        accumulated_total += dmg
        year_of_doc = d["T"].year

        if year_of_doc < year_now:
            historical_dates.append(d["T"])
            historical_values.append(accumulated_total)
        elif year_of_doc == year_now:
            year_dates.append(d["T"])
            year_values.append(accumulated_total)
            accumulated_yearly += dmg

    first_date = docs_sorted[0]["T"]
    last_date = docs_sorted[-1]["T"]

    return (
        historical_dates,
        historical_values,
        year_dates,
        year_values,
        accumulated_total,
        accumulated_yearly,
        first_date,
        last_date,
    )


def build_monthly_series(docs, value_key):
    monthly_totals = defaultdict(float)
    total_value = 0.0

    for entry in docs:
        val = entry.get(value_key, 0.0) or 0.0
        month = entry["T"].month
        monthly_totals[month] += val
        total_value += val

    monthly_values = []
    accumulated_monthly = []
    running = 0.0

    for m in range(1, 13):
        v = monthly_totals.get(m, 0.0)
        if v == 0.0:
            monthly_values.append(None)
            if not accumulated_monthly:
                accumulated_monthly.append(None)
            else:
                accumulated_monthly.append(accumulated_monthly[-1])
        else:
            monthly_values.append(v)
            running += v
            accumulated_monthly.append(running)

    return monthly_values, accumulated_monthly, total_value


def max_non_none(series):
    return max((v for v in series if v is not None), default=0.0)


all_damage_docs = MongoDB.getDamageValues(MongoDB, fixed_start, today)

(
    hist_dates,
    hist_values,
    year_dates,
    year_values,
    accumulated_total_damage,
    accumulated_yearly_damage,
    first_date,
    last_date,
) = build_global_accumulated_damage(all_damage_docs, data_year)

if not first_date or not last_date:
    print("No damage data found. Aborting.")
    sys.exit(0)

total_calendar_days = (last_date - first_date).days + 1
n_years_observed = total_calendar_days / 365.25 if total_calendar_days > 0 else 0.0
typical_yearly_damage = accumulated_total_damage / n_years_observed if n_years_observed > 0 else 0.0

projected_dates = []
projected_values = []
estimated_date = None

if accumulated_yearly_damage > 0 and year_dates:
    remaining_damage = max(0.0, 1.0 - accumulated_total_damage)
    estimated_years = int(remaining_damage / accumulated_yearly_damage)
    if 0 < estimated_years < 25500:
        estimated_date = year_dates[-1] + relativedelta(years=estimated_years)
        projected_dates = [
            year_dates[-1] + relativedelta(years=i) for i in range(0, estimated_years + 1)
        ]
        projected_values = [
            accumulated_total_damage + accumulated_yearly_damage * i
            for i in range(0, estimated_years + 1)
        ]

baseline_dates = []
baseline_values = []
baseline_break_date = None
if typical_yearly_damage > 0:
    years_to_break_baseline = 1.0 / typical_yearly_damage
    n_year_steps = int(np.ceil(years_to_break_baseline))
    for i in range(0, n_year_steps + 1):
        date_i = datetime(first_date.year + i, 1, 1)
        value_i = typical_yearly_damage * i
        if value_i > 1.0:
            value_i = 1.0
        baseline_dates.append(date_i)
        baseline_values.append(value_i)
    baseline_break_date = datetime(first_date.year, 1, 1) + relativedelta(years=n_year_steps)

fig_year_projection = go.Figure()

if baseline_dates:
    fig_year_projection.add_trace(go.Scatter(
        x=baseline_dates,
        y=baseline_values,
        mode="lines",
        name="Historical baseline (average yearly damage)",
        line=dict(color="green", width=2),
        showlegend=True,
    ))
    if baseline_break_date is not None:
        fig_year_projection.add_trace(go.Scatter(
            x=[baseline_break_date],
            y=[1],
            mode="markers+text",
            name="Estimated breaking point (historical trend)",
            marker=dict(color="green", size=12),
            text=[baseline_break_date.strftime("%Y-%m-%d")],
            textposition="top center",
            textfont=dict(color="green", size=18),
            showlegend=False,
        ))

if hist_dates:
    fig_year_projection.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        mode="lines+markers",
        name="Actual accumulated damage (previous years)",
        line=dict(color="blue"),
    ))

if year_dates:
    fig_year_projection.add_trace(go.Scatter(
        x=year_dates,
        y=year_values,
        mode="lines+markers",
        name="Actual accumulated damage (current year)",
        line=dict(color="purple"),
    ))

fig_year_projection.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=1,
    y1=1,
    line=dict(color="red", width=2),
)

if projected_dates and projected_values and estimated_date:
    fig_year_projection.add_trace(go.Scatter(
        x=projected_dates,
        y=projected_values,
        mode="lines",
        name="Projection to breaking point (current-year trend)",
        line=dict(color="red", dash="dash"),
    ))
    fig_year_projection.add_trace(go.Scatter(
        x=[estimated_date],
        y=[1],
        mode="markers+text",
        name="Estimated breaking point (current-year trend)",
        marker=dict(color="red", size=12),
        text=[estimated_date.strftime("%Y-%m-%d")],
        textposition="top center",
        textfont=dict(color="red", size=18),
    ))

percent_total = round(accumulated_total_damage * 100.0, 2)
label_percent = f"{percent_total}%<br>{data_year}"

if year_dates:
    fig_year_projection.add_trace(go.Scatter(
        x=[year_dates[-1]],
        y=[accumulated_total_damage],
        mode="markers+text",
        name=f"Actual state {data_year}",
        marker=dict(color="purple", size=12),
        text=label_percent,
        textposition="middle right",
        textfont=dict(color="purple", size=18),
        showlegend=False,
    ))

fig_year_projection.update_layout(
    font=dict(size=20),
    title=dict(
        text="Accumulated Damage and Projection to Break Point (Year)",
        y=0.97,
        x=0,
        xanchor="left",
        yanchor="top",
    ),
    xaxis_title="Date",
    yaxis_title="Accumulated Damage",
    xaxis=dict(tickfont=dict(size=18)),
    yaxis=dict(type="log", range=[-4, 1.2], dtick=1, tickfont=dict(size=16)),
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=-0.05,
    ),
    margin=dict(t=100),
)

fig_year_projection.update_xaxes(showline=True, linewidth=2, linecolor="black")
fig_year_projection.update_yaxes(showline=True, linewidth=2, linecolor="black")

os.makedirs(os.path.join(output_dir, "projection_plots"), exist_ok=True)
json_path_proj = os.path.join(output_dir, "projection_plots", f"Projection_{data_year}.json")
spec_proj = fig_to_responsive_json(fig_year_projection)
with open(json_path_proj, "w") as f:
    json.dump(spec_proj, f, default=str)

current_year_docs = MongoDB.getDamageValues(MongoDB, one_year_ago, today)
prev_year_docs = MongoDB.getDamageValues(
    MongoDB, one_year_ago - relativedelta(years=1), today - relativedelta(years=1)
)
prevprev_year_docs = MongoDB.getDamageValues(
    MongoDB, one_year_ago - relativedelta(years=2), today - relativedelta(years=2)
)

monthly_current_damage, acc_current_damage, total_current_damage = build_monthly_series(
    current_year_docs, "DMG"
)
monthly_prev_damage, acc_prev_damage, total_prev_damage = build_monthly_series(
    prev_year_docs, "DMG"
)
monthly_prevprev_damage, acc_prevprev_damage, total_prevprev_damage = build_monthly_series(
    prevprev_year_docs, "DMG"
)

estimated_30_lifetime_all = accumulated_total_damage * 30.0
estimated_60_lifetime_all = accumulated_total_damage * 60.0

estimated_30_current = total_current_damage * 30.0
estimated_60_current = total_current_damage * 60.0
estimated_30_prev = total_prev_damage * 30.0
estimated_60_prev = total_prev_damage * 60.0
estimated_30_prevprev = total_prevprev_damage * 30.0
estimated_60_prevprev = total_prevprev_damage * 60.0

month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

fig_month_damage = go.Figure()

fig_month_damage.add_trace(go.Bar(
    x=month_labels,
    y=monthly_current_damage,
    name=f"Monthly Damage {one_year_ago.year}",
    marker=dict(color="steelblue"),
    opacity=0.6,
))
fig_month_damage.add_trace(go.Scatter(
    x=month_labels,
    y=acc_current_damage,
    mode="lines+markers",
    line=dict(color="dodgerblue", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {one_year_ago.year}",
))

fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_60_lifetime_all,
    y1=estimated_60_lifetime_all,
    line=dict(color="dodgerblue", width=2, dash="solid"),
)
fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_30_lifetime_all,
    y1=estimated_30_lifetime_all,
    line=dict(color="steelblue", width=2, dash="dot"),
)

fig_month_damage.add_trace(go.Bar(
    x=month_labels,
    y=monthly_prev_damage,
    name=f"Monthly Damage {one_year_ago.year - 1}",
    marker=dict(color="darkorange"),
    opacity=0.3,
))
fig_month_damage.add_trace(go.Scatter(
    x=month_labels,
    y=acc_prev_damage,
    mode="lines+markers",
    line=dict(color="orangered", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {one_year_ago.year - 1}",
    opacity=0.3,
))

fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_60_prev,
    y1=estimated_60_prev,
    line=dict(color="orangered", width=2, dash="solid"),
    opacity=0.75,
)
fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_30_prev,
    y1=estimated_30_prev,
    line=dict(color="darkorange", width=2, dash="dot"),
    opacity=0.75,
)

fig_month_damage.add_trace(go.Bar(
    x=month_labels,
    y=monthly_prevprev_damage,
    name=f"Monthly Damage {one_year_ago.year - 2}",
    marker=dict(color="darkgreen"),
    opacity=0.3,
))
fig_month_damage.add_trace(go.Scatter(
    x=month_labels,
    y=acc_prevprev_damage,
    mode="lines+markers",
    line=dict(color="springgreen", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {one_year_ago.year - 2}",
    opacity=0.3,
))

fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_60_prevprev,
    y1=estimated_60_prevprev,
    line=dict(color="springgreen", width=2, dash="solid"),
    opacity=0.75,
)
fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=estimated_30_prevprev,
    y1=estimated_30_prevprev,
    line=dict(color="darkgreen", width=2, dash="dot"),
    opacity=0.75,
)

fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=1,
    y1=1,
    line=dict(color="black", width=2, dash="solid"),
    opacity=0.75,
)
fig_month_damage.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0,
    x1=1,
    y0=0.8,
    y1=0.8,
    line=dict(color="black", width=2, dash="dot"),
    opacity=0.75,
)

fig_month_damage.update_layout(
    font=dict(size=20),
    title=dict(
        text="Monthly and accumulated damage evolution with limits",
        y=0.97,
        x=0,
        xanchor="left",
        yanchor="top",
    ),
    xaxis_title="Month",
    yaxis_title="Damage",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=-0.05,
    ),
    margin=dict(t=100),
)

fig_month_damage.update_yaxes(
    type="log",
    tickformat=".0e",
    tickvals=[1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    range=[-6, 1],
    tickfont=dict(size=18),
    showline=True,
    linewidth=2,
    linecolor="black",
)
fig_month_damage.update_xaxes(
    tickfont=dict(size=18),
    minor=dict(ticklen=8, tickcolor="black"),
    showline=True,
    linewidth=2,
    linecolor="black",
)

os.makedirs(os.path.join(output_dir, "accumulation_plots"), exist_ok=True)
json_path_month_damage = os.path.join(output_dir, "accumulation_plots", f"Accumulated_{data_year}.json")
spec_month_damage = fig_to_responsive_json(fig_month_damage)
with open(json_path_month_damage, "w") as f:
    json.dump(spec_month_damage, f, default=str)

monthly_cycles_current, acc_cycles_current, _ = build_monthly_series(current_year_docs, "CYCLES")
monthly_cycles_prev, acc_cycles_prev, _ = build_monthly_series(prev_year_docs, "CYCLES")
monthly_cycles_prevprev, acc_cycles_prevprev, _ = build_monthly_series(prevprev_year_docs, "CYCLES")

hover_template_cycles_year = (
    "%{x}<br>"
    "%{fullData.name}: %{y:.0f}<extra></extra>"
)

fig_cycles_year = go.Figure()

fig_cycles_year.add_trace(go.Bar(
    x=month_labels,
    y=monthly_cycles_current,
    name=f"Monthly cycles {one_year_ago.year}",
    marker=dict(color="steelblue"),
    opacity=0.6,
    hovertemplate=hover_template_cycles_year,
))
fig_cycles_year.add_trace(go.Scatter(
    x=month_labels,
    y=acc_cycles_current,
    mode="lines+markers",
    line=dict(color="dodgerblue", width=3),
    marker=dict(size=6),
    name=f"Accumulated cycles {one_year_ago.year}",
    hovertemplate=hover_template_cycles_year,
))

fig_cycles_year.add_trace(go.Bar(
    x=month_labels,
    y=monthly_cycles_prev,
    name=f"Monthly cycles {one_year_ago.year - 1}",
    marker=dict(color="darkorange"),
    opacity=0.3,
    hovertemplate=hover_template_cycles_year,
))
fig_cycles_year.add_trace(go.Scatter(
    x=month_labels,
    y=acc_cycles_prev,
    mode="lines+markers",
    line=dict(color="orangered", width=3),
    marker=dict(size=6),
    name=f"Accumulated cycles {one_year_ago.year - 1}",
    opacity=0.3,
    hovertemplate=hover_template_cycles_year,
))

fig_cycles_year.add_trace(go.Bar(
    x=month_labels,
    y=monthly_cycles_prevprev,
    name=f"Monthly cycles {one_year_ago.year - 2}",
    marker=dict(color="darkgreen"),
    opacity=0.3,
    hovertemplate=hover_template_cycles_year,
))
fig_cycles_year.add_trace(go.Scatter(
    x=month_labels,
    y=acc_cycles_prevprev,
    mode="lines+markers",
    line=dict(color="springgreen", width=3),
    marker=dict(size=6),
    name=f"Accumulated cycles {one_year_ago.year - 2}",
    opacity=0.3,
    hovertemplate=hover_template_cycles_year,
))

fig_cycles_year.update_layout(
    font=dict(size=20),
    title=dict(
        text="Monthly and accumulated cycles (year comparison)",
        y=0.97,
        x=0,
        xanchor="left",
        yanchor="top",
    ),
    xaxis_title="Month",
    yaxis_title="Cycles",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=-0.05,
    ),
    margin=dict(t=100),
)

max_cycles_val = max(
    max_non_none(acc_cycles_current),
    max_non_none(acc_cycles_prev),
    max_non_none(acc_cycles_prevprev),
)

if max_cycles_val > 1:
    max_exp = int(np.ceil(np.log10(max_cycles_val)))
    min_exp = 0
    tickvals_cycles = [10 ** e for e in range(min_exp, max_exp + 1)]
    fig_cycles_year.update_yaxes(
        type="log",
        tickformat=".0e",
        tickvals=tickvals_cycles,
        range=[min_exp, max_exp],
        tickfont=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor="black",
    )
else:
    fig_cycles_year.update_yaxes(
        tickfont=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor="black",
    )

fig_cycles_year.update_xaxes(
    tickfont=dict(size=18),
    minor=dict(ticklen=8, tickcolor="black"),
    showline=True,
    linewidth=2,
    linecolor="black",
)

os.makedirs(os.path.join(output_dir, "cycles_plots"), exist_ok=True)
json_path_cycles_year = os.path.join(output_dir, "cycles_plots", f"Cycles_{data_year}.json")
spec_cycles_year = fig_to_responsive_json(fig_cycles_year)
with open(json_path_cycles_year, "w") as f:
    json.dump(spec_cycles_year, f, default=str)

print("Yearly plots (damage and cycles) have been generated.")
