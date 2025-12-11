# Import necessary libraries
import pandas as pd  # For data manipulation and analysis, especially with CSV files
import numpy as np   # For numerical operations, especially array manipulations
import plotly.express as px  # For creating interactive plots easily
import plotly.graph_objects as go  # For more control over plot creation
from scipy.optimize import curve_fit  # For curve fitting, used here for the S-N curve
import os  # For interacting with the operating system, like creating directories and paths
from mongo_utils import MongoDB  # Custom utility for MongoDB interactions (assumed to be in a local file)
from datetime import datetime, timedelta  # For handling dates and times
import sys  # For accessing command-line arguments
from pytz import UTC  # For timezone handling (Coordinated Universal Time)
from dateutil.relativedelta import relativedelta
import calendar
import json

print("Generating the month plots...")  # Initial script status message

# ---------------------------------------------------------------------------
#  Generic helpers
# ---------------------------------------------------------------------------

def fig_to_responsive_json(fig):
    """
    Convert a Plotly figure into a responsive JSON spec (no fixed width/height).
    This keeps the plot fluid in the frontend.
    """
    fig.update_layout(autosize=True)
    spec = fig.to_plotly_json()
    spec.get("layout", {}).pop("width", None)
    spec.get("layout", {}).pop("height", None)
    return spec


def max_non_none(series):
    """
    Return the maximum non-None value in a sequence, or 0 if all values are None.
    """
    return max((v for v in series if v is not None), default=0)


def build_accumulated_cycles(docs):
    """
    Build an accumulated cycles series from a list of Mongo documents.
    Keeps None in positions where there are no cycles for that day.
    The structure and behavior match the original per-year loops.
    """
    accumulated = []
    total = 0
    for element in docs:
        if not element:
            accumulated.append(None)
            continue
        cycles_val = element.get("CYCLES", 0)
        if cycles_val and cycles_val != 0:
            total += cycles_val
            accumulated.append(total)
        else:
            accumulated.append(None)
    return accumulated


def build_day_tick_values(accumulated_dates):
    """
    Build a reduced list of tick values for the X axis (dates within a month),
    preserving the original selection logic:
    - Start at the first day
    - Add roughly every ~5th day, alternating pattern for even/odd day counts
    - Always include the last day
    """
    start = datetime(accumulated_dates[0].year, accumulated_dates[0].month, 1)
    end = start + pd.offsets.MonthEnd(0)
    date_range = pd.date_range(start=start, end=end, freq="D")

    tick_vals = [start]
    last_inserted = 0

    if end.day % 2 == 0:
        # Even number of days in the month
        for index, date in enumerate(date_range):
            if index % 2 == 0 and index > last_inserted + 4:
                if date not in tick_vals:
                    tick_vals.append(date)
                    last_inserted = index
    else:
        # Odd number of days in the month
        for index, date in enumerate(date_range):
            if index % 2 != 0 and index > last_inserted + 5:
                if date not in tick_vals:
                    tick_vals.append(date)
                    last_inserted = index

    if end not in tick_vals:
        tick_vals.append(end)

    return tick_vals


# ---------------------------------------------------------------------------
#  S-N curve configuration (Stress-Number of Cycles)
# ---------------------------------------------------------------------------

# Define known stress points and corresponding number of cycles to failure
stress = np.array([292, 136, 63, 50, 37, 32, 20])  # Stress values in MPa
cycles = np.array([1e4, 1e5, 1e6, 2e6, 5e6, 1e7, 1e8])  # Number of cycles


def sn_curve(N, a, b):
    """
    Calculates stress (S) given number of cycles (N) based on S = a * N^(-b).
    Clips N to avoid issues with very low or very high cycle counts.
    """
    N = np.clip(N, 1e3, 1e12)  # Clip N to a practical range
    return a * N ** (-b)


# Fit the S-N curve function to the experimental data to find parameters 'a' and 'b'
params, _ = curve_fit(sn_curve, cycles, stress)
a, b = params  # Unpack the fitted parameters


def estimate_cycles(stress_input):
    """Estimate the number of cycles to failure for a given stress input."""
    return (stress_input / a) ** (-1 / b)


def estimate_stress(cycles_input):
    """Estimate the stress for a given number of cycles."""
    return a * cycles_input ** (-b)


def detect_abrupt_movements(values):
    """
    Apply the same state-machine logic used originally to detect abrupt movements
    (potential fatigue cycles) in the daily Zenith Angle time series.

    Returns the list of detected movements, each one as:
    {
        'start_value': ZA_start,
        'start_time' : T_start,
        'end_value'  : ZA_end,
        'end_time'   : T_end
    }
    """
    abrupt_movements = []
    prev_value = None

    # State variables for cycle detection
    start_down_value = None
    start_up_value = None
    deepest_point = None
    deepest_time = None
    highest_point = None
    highest_time = None
    start_down_candidate = None
    start_up_candidate = None

    for current_value in values:
        current_za = current_value["ZA"]
        current_time = current_value["T"]

        # Track global min/max in the day
        if not deepest_point or current_za < deepest_point:
            deepest_point = current_za
            deepest_time = current_time
            # Reset highest if we found a new deepest point
            highest_point = None
            highest_time = None
        if not highest_point or current_za > highest_point:
            highest_point = current_za
            highest_time = current_time

        # Original state machine logic (unchanged)
        if not start_up_value and not start_down_value:
            # Initial state or after finishing a cycle
            start_up_value = current_value
        elif (
            not start_down_value
            and start_up_value
            and start_down_candidate
            and current_value["ZA"] - start_down_candidate["ZA"] > 0.25
        ):
            # Confirmed downward trend start after an upward phase
            start_down_value = start_down_candidate
            start_up_value = None
            start_down_candidate = None
        elif (
            not start_down_value
            and not start_down_candidate
            and start_up_value
            and prev_value
            and current_value["ZA"] - prev_value["ZA"] > 0
        ):
            # Potential start of a downward trend
            start_down_candidate = prev_value
        elif (
            not start_down_value
            and start_down_candidate
            and start_up_value
            and current_value["ZA"] - start_down_candidate["ZA"] < -0.1
        ):
            # Downward candidate invalidated
            start_down_candidate = None
        elif (
            start_down_value
            and start_up_candidate
            and not start_up_value
            and current_value["ZA"] > start_down_value["ZA"]
            and current_value["ZA"] - start_up_candidate["ZA"] < -0.25
        ):
            # Confirmed upward trend start after a downward phase → complete cycle
            start_up_value = start_up_candidate
            abrupt_movements.append(
                {
                    "start_value": start_down_value["ZA"],
                    "start_time": start_down_value["T"],
                    "end_value": start_up_value["ZA"],
                    "end_time": start_up_value["T"],
                }
            )
            start_up_candidate = None
            start_down_value = None
        elif (
            start_down_value
            and start_up_candidate
            and not start_up_value
            and current_value["ZA"] < start_down_value["ZA"]
            and current_value["ZA"] - start_up_candidate["ZA"] < -0.25
        ):
            # Direction change: new downward phase
            start_down_value = None
            start_up_value = start_up_candidate
            start_up_candidate = None
        elif (
            start_down_value
            and not start_up_value
            and not start_up_candidate
            and prev_value
            and current_value["ZA"] - prev_value["ZA"] < 0
        ):
            # Potential start of an upward trend
            start_up_candidate = prev_value
        elif (
            start_down_value
            and start_up_candidate
            and not start_up_value
            and current_value["ZA"] - start_up_candidate["ZA"] > 0.1
        ):
            # Upward candidate invalidated
            start_up_candidate = None

        prev_value = current_value

    # Final incomplete cycle handling (same semantics as original)
    if start_down_value and start_up_candidate:
        abrupt_movements.append(
            {
                "start_value": start_down_value["ZA"],
                "start_time": start_down_value["T"],
                "end_value": start_up_candidate["ZA"],
                "end_time": start_up_candidate["T"],
            }
        )
    elif start_down_value:
        abrupt_movements.append(
            {
                "start_value": start_down_value["ZA"],
                "start_time": start_down_value["T"],
                "end_value": prev_value["ZA"],
                "end_time": prev_value["T"],
            }
        )

    # Ensure the largest daily fluctuation (deepest↔highest) is included
    if highest_point and deepest_point:
        largest = {
            "start_value": deepest_point,
            "start_time": deepest_time,
            "end_value": highest_point,
            "end_time": highest_time,
        }
        if largest not in abrupt_movements:
            abrupt_movements.append(largest)

    return abrupt_movements


def compute_daily_damage(values, conversion_df):
    """
    Compute Miner's rule damage for a single day:
    - Detect abrupt movements (cycles) in ZA
    - Convert ZA amplitudes to stress amplitudes via conversion_df (deg→MPa)
    - Group by stress amplitude and apply S-N curve for Miner's rule
    Returns:
        accumulated_damage_today (float)
    """
    abrupt_movements = detect_abrupt_movements(values)
    grouped_values = {}  # stress amplitude (as string) → number of cycles

    # Convert each movement into stress amplitude and count occurrences
    for element in abrupt_movements:
        start_value = element["start_value"]
        end_value = element["end_value"]

        # Only consider plausible ZA ranges
        if 0 <= start_value < 100 and 0 <= end_value < 100:
            matching_row_start = conversion_df.query(f"Degree == {round(start_value)}")
            matching_row_end = conversion_df.query(f"Degree == {round(end_value)}")

            start_stress = (
                round(matching_row_start.iloc[0]["MPa"])
                if not matching_row_start.empty
                else None
            )
            end_stress = (
                round(matching_row_end.iloc[0]["MPa"])
                if not matching_row_end.empty
                else None
            )

            if start_stress is not None and end_stress is not None:
                stress_value = abs(start_stress - end_stress)
                if stress_value > 0:
                    key = str(stress_value)
                    grouped_values[key] = grouped_values.get(key, 0) + 1

    # Apply Miner's rule using the S-N curve
    accumulated_damage_today = 0.0
    for stress_str, num_cycles_at_stress in grouped_values.items():
        rounded_stress = round(float(stress_str), 2)
        max_cycles_to_failure = estimate_cycles(rounded_stress)
        if max_cycles_to_failure > 0:
            accumulated_damage_today += num_cycles_at_stress / max_cycles_to_failure

    return accumulated_damage_today


# ---------------------------------------------------------------------------
#  Date configuration
# ---------------------------------------------------------------------------

# Check if a date is provided as a command-line argument
if len(sys.argv) > 1:
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
else:
    date = datetime.now()

today = date
# Start of previous month
yesterday = today - relativedelta(months=1) + relativedelta(day=1)

# Base output directory for all plots (same path as before)
output_dir = r"D:\github\SUMON-Repo\html\contents"

# ---------------------------------------------------------------------------
#  Initialization for the main loop
# ---------------------------------------------------------------------------

currentDate = yesterday
accumulated_total_damage = 0.0
daily_values = []
accumulated_values = []
accumulated_dates = []

limit_value = 1.0   # Hard limit based on 60-year lifetime
soft_value = 0.8    # Soft limit (80% of hard limit)

currentMonth = currentDate.month
currentYear = currentDate.year

# These lists are kept for compatibility, even if not used directly
accumulated_limit = []
accumulated_soft = []

# ---------------------------------------------------------------------------
#  Main daily loop
# ---------------------------------------------------------------------------

while currentDate.date() <= today.date():
    # -----------------------------------------------------------------------
    # Daily damage computation section
    # -----------------------------------------------------------------------
    if currentDate.date() < today.date():
        # Time interval: from 17:00 D to 07:59:59 D+1 (UTC, ms timestamps)
        T_min = int(
            currentDate.replace(
                hour=17, minute=0, second=0, microsecond=0, tzinfo=UTC
            ).timestamp()
            * 1000
        )
        next_day = currentDate + timedelta(days=1)
        T_max = int(
            next_day.replace(
                hour=7, minute=59, second=59, microsecond=0, tzinfo=UTC
            ).timestamp()
            * 1000
        )

        # Fetch daily ZA data
        values = MongoDB.getDailyZenith(MongoDB, T_min, T_max)

        # Load deg→MPa conversion (kept inside loop to preserve original behavior)
        conversion = pd.read_csv("./deg_to_stress.csv")

        # Compute damage for this day using the same logic as before
        accumulated_damage_today = compute_daily_damage(values, conversion)
        accumulated_total_damage += accumulated_damage_today

        # Store data for monthly plots
        if yesterday.date() <= currentDate.date() <= today.date():
            daily_values.append(
                accumulated_damage_today if accumulated_damage_today != 0 else None
            )
            accumulated_dates.append(currentDate.date())

            if accumulated_values:
                if accumulated_total_damage != 0:
                    accumulated_values.append(accumulated_total_damage)
                else:
                    accumulated_values.append(accumulated_values[-1])
            else:
                accumulated_values.append(
                    accumulated_total_damage if accumulated_total_damage != 0 else None
                )

    # -----------------------------------------------------------------------
    # Monthly boundary or last day → projection plot
    # -----------------------------------------------------------------------
    if currentDate.month != currentMonth or currentDate.date() == today.date():
        currentMonth = currentDate.month
        projected_dates = None

        current_value = accumulated_values[-1]
        current_date_for_proj = accumulated_dates[-1]

        # Estimate number of months to reach damage = 1
        estimated_months = int(1 / accumulated_total_damage) if accumulated_total_damage > 0 else 0
        if 0 < estimated_months < 305000:
            estimated_date = current_date_for_proj + relativedelta(months=estimated_months)
            projected_dates = [
                current_date_for_proj + relativedelta(months=i)
                for i in range(1, estimated_months + 1)
            ]
            projected_values = [current_value * i for i in range(1, estimated_months + 1)]
        else:
            estimated_date = None
            projected_values = None

        # === Plot: Accumulated Damage and Projection ===
        fig4 = go.Figure()

        # Actual accumulated damage
        fig4.add_trace(
            go.Scatter(
                x=accumulated_dates,
                y=accumulated_values,
                mode="lines+markers",
                name="Accumulated Damage (actual)",
                line=dict(color="blue"),
            )
        )

        # Hard limit at damage = 1
        fig4.add_shape(
            type="line",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=1,
            y1=1,
            line=dict(color="red", width=2),
        )

        # Projection to breaking point
        if projected_dates and projected_values and estimated_date:
            fig4.add_trace(
                go.Scatter(
                    x=projected_dates,
                    y=projected_values,
                    mode="lines",
                    name="Projection to Breaking Point <br> based on month data",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig4.add_trace(
                go.Scatter(
                    x=[estimated_date],
                    y=[1],
                    mode="markers+text",
                    name="Estimated reach of <br> Breaking Point",
                    marker=dict(color="red", size=12),
                    text=[f"{estimated_date}"],
                    textposition="bottom center",
                    textfont=dict(color="red", size=18),
                )
            )

        # Marker for current month
        if currentDate.date() == today.date():
            percent = round((accumulated_total_damage / limit_value) * 100, 2)
            fixed_date = currentDate - relativedelta(months=1)
            label = f"{percent}%<br>{calendar.month_abbr[fixed_date.month]} - {fixed_date.year}"
            fig4.add_trace(
                go.Scatter(
                    x=[currentDate],
                    y=[accumulated_total_damage],
                    mode="markers+text",
                    name=f"{currentDate.month-1} - {currentDate.year}",
                    marker=dict(color="purple", size=12),
                    text=label,
                    textposition="middle right",
                    textfont=dict(color="purple", size=18),
                    showlegend=False,
                )
            )

        fig4.update_layout(
            font=dict(size=20),
            title=dict(
                text="Accumulated Damage and Projection to Break Point",
                y=0.97,
            x=0,
                xanchor="left",
                yanchor="top",
            ),
            xaxis_title="Date",
            yaxis_title="Accumulated Damage",
            xaxis=dict(tickfont=dict(size=18)),
            yaxis=dict(type="log", range=[-4, 1], dtick=1, tickfont=dict(size=16)),
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
        fig4.update_xaxes(showline=True, linewidth=2, linecolor="black")
        fig4.update_yaxes(showline=True, linewidth=2, linecolor="black")

        os.makedirs(os.path.join(output_dir, "projection_plots"), exist_ok=True)
        fixed_date = currentDate - relativedelta(months=1)
        json_path4 = os.path.join(
            output_dir,
            "projection_plots",
            f"Projection_{fixed_date.year}-{fixed_date.month}.json",
        )
        spec4 = fig_to_responsive_json(fig4)
        with open(json_path4, "w") as f:
            json.dump(spec4, f, default=str)

    # -----------------------------------------------------------------------
    # Last day of range → monthly damage + cycles comparison plots
    # -----------------------------------------------------------------------
    if currentDate.date() == today.date():
        # ------------ Damage values for 3 years ------------
        currentYear_docs = MongoDB.getDamageValues(MongoDB, yesterday, today)

        totalYearDamage = sum(el["DMG"] for el in currentYear_docs if el is not None)
        estimated_30_lifetime_1 = totalYearDamage * 12 * 30
        estimated_60_lifetime_1 = totalYearDamage * 12 * 60

        previousYear_docs = MongoDB.getDamageValues(
            MongoDB, yesterday - relativedelta(years=1), today - relativedelta(years=1)
        )
        previousPreviousYear_docs = MongoDB.getDamageValues(
            MongoDB, yesterday - relativedelta(years=2), today - relativedelta(years=2)
        )

        totalPrevYearDamage = 0
        accumulatedPrevDamage = []
        for el in previousYear_docs:
            totalPrevYearDamage += el["DMG"]
            accumulatedPrevDamage.append(totalPrevYearDamage if totalPrevYearDamage else None)
        estimated_30_lifetime_lastYear = totalPrevYearDamage * 12 * 30
        estimated_60_lifetime_lastYear = totalPrevYearDamage * 12 * 60

        totalPrevPrevYearDamage = 0
        accumulatedPrevPrevDamage = []
        for el in previousPreviousYear_docs:
            totalPrevPrevYearDamage += el["DMG"]
            accumulatedPrevPrevDamage.append(
                totalPrevPrevYearDamage if totalPrevPrevYearDamage else None
            )
        estimated_30_lifetime_last_lastYear = totalPrevPrevYearDamage * 12 * 30
        estimated_60_lifetime_last_lastYear = totalPrevPrevYearDamage * 12 * 60

        # ------------ Accumulated cycles for 3 years ------------
        accumulatedCyclesCurrent = build_accumulated_cycles(currentYear_docs)
        accumulatedPrevCycles = build_accumulated_cycles(previousYear_docs)
        accumulatedPrevPrevCycles = build_accumulated_cycles(previousPreviousYear_docs)

        # -------------------------------------------------------------------
        # Plot 2: Daily + accumulated damage evolution
        # -------------------------------------------------------------------
        fig2 = go.Figure()

        if daily_values and accumulated_values:
            # Current year bars + line
            fig2.add_trace(
                go.Bar(
                    x=accumulated_dates,
                    y=daily_values,
                    name=f"Daily Damage {yesterday.year}",
                    marker=dict(color="steelblue"),
                    opacity=0.6,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=accumulated_dates,
                    y=accumulated_values,
                    mode="lines+markers",
                    line=dict(color="dodgerblue", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated Damage {yesterday.year}",
                )
            )

            # Current year hard/soft lifetime limits
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_60_lifetime_1,
                y1=estimated_60_lifetime_1,
                line=dict(color="dodgerblue", width=2, dash="solid"),
            )
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_30_lifetime_1,
                y1=estimated_30_lifetime_1,
                line=dict(color="steelblue", width=2, dash="dot"),
            )

        # Previous year damage (aligned in current year)
        if previousYear_docs and accumulatedPrevDamage:
            aligned_prev_dates = [
                el["T"].date() + relativedelta(year=yesterday.year) for el in previousYear_docs
            ]
            fig2.add_trace(
                go.Bar(
                    x=aligned_prev_dates,
                    y=[el["DMG"] if el["DMG"] != 0 else None for el in previousYear_docs],
                    name=f"Daily Damage {yesterday.year-1}",
                    marker=dict(color="darkorange"),
                    opacity=0.3,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=aligned_prev_dates,
                    y=accumulatedPrevDamage,
                    mode="lines+markers",
                    line=dict(color="orangered", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated Damage {yesterday.year-1}",
                    opacity=0.3,
                )
            )
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_60_lifetime_lastYear,
                y1=estimated_60_lifetime_lastYear,
                line=dict(color="orangered", width=2, dash="solid"),
                opacity=0.75,
            )
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_30_lifetime_lastYear,
                y1=estimated_30_lifetime_lastYear,
                line=dict(color="darkorange", width=2, dash="dot"),
                opacity=0.75,
            )

        # Previous-previous year damage (aligned)
        if previousPreviousYear_docs and accumulatedPrevPrevDamage:
            aligned_prev_prev_dates = [
                el["T"].date() + relativedelta(year=yesterday.year)
                for el in previousPreviousYear_docs
            ]
            fig2.add_trace(
                go.Bar(
                    x=aligned_prev_prev_dates,
                    y=[
                        el["DMG"] if el["DMG"] != 0 else None
                        for el in previousPreviousYear_docs
                    ],
                    name=f"Daily Damage {yesterday.year-2}",
                    marker=dict(color="darkgreen"),
                    opacity=0.3,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=aligned_prev_prev_dates,
                    y=accumulatedPrevPrevDamage,
                    mode="lines+markers",
                    line=dict(color="springgreen", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated Damage {yesterday.year-2}",
                    opacity=0.3,
                )
            )
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_60_lifetime_last_lastYear,
                y1=estimated_60_lifetime_last_lastYear,
                line=dict(color="springgreen", width=2, dash="solid"),
                opacity=0.75,
            )
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=estimated_30_lifetime_last_lastYear,
                y1=estimated_30_lifetime_last_lastYear,
                line=dict(color="darkgreen", width=2, dash="dot"),
                opacity=0.75,
            )

            # Absolute reference limits
            fig2.add_shape(
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
            fig2.add_shape(
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

        fig2.update_layout(
            font=dict(size=20),
            title=dict(
                text="Daily and accumulated evolution with limits",
                y=0.97,
                x=0,
                xanchor="left",
                yanchor="top",
            ),
            xaxis_title="Date",
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
        fig2.update_yaxes(
            type="log",
            tickformat=".0e",
            tickvals=[1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            range=[-6, 1],
            tickfont=dict(size=18),
            showline=True,
            linewidth=2,
            linecolor="black",
        )

        # Shared x-axis ticks for both damage and cycles plots
        tickVals = build_day_tick_values(accumulated_dates)

        # -------------------------------------------------------------------
        # NEW PLOT: Daily and accumulated cycles (monthly comparison)
        # -------------------------------------------------------------------
        fig_cycles = go.Figure()

        # Hover template: show cycles as normal integers
        hover_template_cycles = "%{x|%d %b}<br>%{fullData.name}: %{y:.0f}<extra></extra>"

        # Current year cycles
        if currentYear_docs and accumulatedCyclesCurrent:
            dates_current = [el["T"].date() for el in currentYear_docs]
            fig_cycles.add_trace(
                go.Bar(
                    x=dates_current,
                    y=[
                        el.get("CYCLES", 0) if el.get("CYCLES", 0) != 0 else None
                        for el in currentYear_docs
                    ],
                    name=f"Daily cycles {yesterday.year}",
                    marker=dict(color="steelblue"),
                    opacity=0.6,
                    hovertemplate=hover_template_cycles,
                )
            )
            fig_cycles.add_trace(
                go.Scatter(
                    x=dates_current,
                    y=accumulatedCyclesCurrent,
                    mode="lines+markers",
                    line=dict(color="dodgerblue", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated cycles {yesterday.year}",
                    hovertemplate=hover_template_cycles,
                )
            )

        # Previous year cycles (aligned)
        if previousYear_docs and accumulatedPrevCycles:
            dates_prev = [
                el["T"].date() + relativedelta(year=yesterday.year) for el in previousYear_docs
            ]
            fig_cycles.add_trace(
                go.Bar(
                    x=dates_prev,
                    y=[
                        el.get("CYCLES", 0) if el.get("CYCLES", 0) != 0 else None
                        for el in previousYear_docs
                    ],
                    name=f"Daily cycles {yesterday.year-1}",
                    marker=dict(color="darkorange"),
                    opacity=0.3,
                    hovertemplate=hover_template_cycles,
                )
            )
            fig_cycles.add_trace(
                go.Scatter(
                    x=dates_prev,
                    y=accumulatedPrevCycles,
                    mode="lines+markers",
                    line=dict(color="orangered", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated cycles {yesterday.year-1}",
                    opacity=0.3,
                    hovertemplate=hover_template_cycles,
                )
            )

        # Previous-previous year cycles (aligned)
        if previousPreviousYear_docs and accumulatedPrevPrevCycles:
            dates_prev_prev = [
                el["T"].date() + relativedelta(year=yesterday.year)
                for el in previousPreviousYear_docs
            ]
            fig_cycles.add_trace(
                go.Bar(
                    x=dates_prev_prev,
                    y=[
                        el.get("CYCLES", 0) if el.get("CYCLES", 0) != 0 else None
                        for el in previousPreviousYear_docs
                    ],
                    name=f"Daily cycles {yesterday.year-2}",
                    marker=dict(color="darkgreen"),
                    opacity=0.3,
                    hovertemplate=hover_template_cycles,
                )
            )
            fig_cycles.add_trace(
                go.Scatter(
                    x=dates_prev_prev,
                    y=accumulatedPrevPrevCycles,
                    mode="lines+markers",
                    line=dict(color="springgreen", width=3),
                    marker=dict(size=6),
                    name=f"Accumulated cycles {yesterday.year-2}",
                    opacity=0.3,
                    hovertemplate=hover_template_cycles,
                )
            )

        # Layout for cycles plot
        fig_cycles.update_layout(
            font=dict(size=20),
            title=dict(
                text="Daily and accumulated cycles (monthly comparison)",
                y=0.97,
                x=0,
                xanchor="left",
                yanchor="top",
            ),
            xaxis_title="Date",
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
        fig_cycles.update_xaxes(
            tickformat="%d<br>%b",
            tickvals=tickVals,
            tickfont=dict(size=18),
            minor=dict(ticklen=8, tickcolor="black"),
            showline=True,
            linewidth=2,
            linecolor="black",
        )

        # Y-axis for cycles: log scale if there is enough dynamic range
        max_cycles_value = max(
            max_non_none(accumulatedCyclesCurrent),
            max_non_none(accumulatedPrevCycles),
            max_non_none(accumulatedPrevPrevCycles),
        )

        if max_cycles_value > 1:
            max_exp = int(np.ceil(np.log10(max_cycles_value)))
            min_exp = 0  # 10^0 = 1
            tickvals_cycles = [10 ** e for e in range(min_exp, max_exp + 1)]
            fig_cycles.update_yaxes(
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
            fig_cycles.update_yaxes(
                tickfont=dict(size=18),
                showline=True,
                linewidth=2,
                linecolor="black",
            )

        # Reuse tickVals for damage plot X axis
        fig2.update_xaxes(
            tickformat="%d<br>%b",
            tickvals=tickVals,
            tickfont=dict(size=18),
            minor=dict(ticklen=8, tickcolor="black"),
            showline=True,
            linewidth=2,
            linecolor="black",
        )

        # ---------- Save plots as JSON ----------
        os.makedirs(os.path.join(output_dir, "accumulation_plots"), exist_ok=True)
        fixed_date = currentDate - relativedelta(months=1)
        json_path2 = os.path.join(
            output_dir,
            "accumulation_plots",
            f"Accumulated_{fixed_date.year}-{fixed_date.month}.json",
        )
        spec2 = fig_to_responsive_json(fig2)
        with open(json_path2, "w") as f:
            json.dump(spec2, f, default=str)

        os.makedirs(os.path.join(output_dir, "cycles_plots"), exist_ok=True)
        json_path_cycles = os.path.join(
            output_dir,
            "cycles_plots",
            f"Cycles_{fixed_date.year}-{fixed_date.month}.json",
        )
        spec_cycles = fig_to_responsive_json(fig_cycles)
        with open(json_path_cycles, "w") as f:
            json.dump(spec_cycles, f, default=str)

        print("General monthly plots have been generated")

    currentDate += timedelta(days=1)

# ---------------------------------------------------------------------------
# Optional S-N curve plot generation
# ---------------------------------------------------------------------------
if False:  # Optional to generate the S-N curve
    fig3 = go.Figure()

    N_plot = np.logspace(4, 9, 500)
    stress_plot = sn_curve(N_plot, a, b)

    fig3.add_trace(
        go.Scatter(
            x=N_plot,
            y=stress_plot,
            mode="lines",
            name="S-N curve",
            line=dict(color="steelblue", width=3),
        )
    )

    fig3.add_trace(
        go.Scatter(
            x=cycles,
            y=stress,
            mode="markers",
            name="Spots",
            marker=dict(size=8, color="darkorange"),
        )
    )

    fig3.update_xaxes(
        type="log",
        title="Cycles (N)",
        tickformat=".1e",
        range=[4, 9],
        tickfont=dict(size=32),
    )
    fig3.update_yaxes(title="Stress (MPa)", tickfont=dict(size=32))

    fig3.update_layout(font=dict(size=28), title="S-N Curve", template="plotly_white")

    png_path3 = os.path.join(
        output_dir, f"S-N_Curve_{yesterday.year}-{yesterday.month}.png"
    )
    fig3.write_image(png_path3, width=1920, height=1080, scale=1)
