# Import necessary libraries
import pandas as pd  # For data manipulation and analysis, especially with CSV files
import numpy as np   # For numerical operations, especially array manipulations
import plotly.express as px # For creating interactive plots easily
import plotly.graph_objects as go # For more control over plot creation
from scipy.optimize import curve_fit # For curve fitting, used here for the S-N curve
import os # For interacting with the operating system, like creating directories and paths
from mongo_utils import MongoDB # Custom utility for MongoDB interactions (assumed to be in a local file)
from datetime import datetime, timedelta # For handling dates and times
import sys # For accessing command-line arguments
from pytz import UTC # For timezone handling (Coordinated Universal Time)
from dateutil.relativedelta import relativedelta
import calendar
from collections import defaultdict
import json

print("Generating the year plots...")
# --- Date Configuration ---
# Check if a date is provided as a command-line argument
if len(sys.argv) > 1:
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d") # Use the provided date string
else:
    date = datetime.now() # No date provided
 
# Define the end date for the analysis period
#today = datetime.now()
today = date
#Yesterday value is start of past month
yesterday = today - relativedelta(years=1)

fixed_start = datetime(2018, 1, 1)
# --- S-N Curve Configuration (Stress-Number of Cycles) ---
# Define known stress points and corresponding number of cycles to failure
stress = np.array([292, 136, 63, 50, 37, 32, 20]) # Stress values in MPa
cycles = np.array([1e4, 1e5, 1e6, 2e6, 5e6, 1e7, 1e8]) # Number of cycles

# Define the S-N curve function (Basquin's equation form)
def sn_curve(N, a, b):
    """
    Calculates stress (S) given number of cycles (N) based on S = a * N^(-b).
    Clips N to avoid issues with very low or very high cycle counts.
    """
    N = np.clip(N, 1e3, 1e12) # Clip N to a practical range
    return a * N**(-b)

# Fit the S-N curve function to the experimental data to find parameters 'a' and 'b'
params, _ = curve_fit(sn_curve, cycles, stress)
a, b = params # Unpack the fitted parameters

# Define helper functions based on the fitted S-N curve
def estimate_cycles(stress_input):
    """Estimates the number of cycles to failure for a given stress input."""
    return (stress_input / a) ** (-1 / b)

def estimate_stress(cycles_input):
    """Estimates the stress for a given number of cycles."""
    return a * cycles_input**(-b)


def fig_to_responsive_json(fig):
    fig.update_layout(autosize=True)
    #fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    spec = fig.to_plotly_json()
    spec.get('layout', {}).pop('width', None)
    spec.get('layout', {}).pop('height', None)
    return spec

# --- Initialization for Loop and Accumulated Values ---
currentDate = yesterday # Get the starting day of the month
accumulated_total_damage = 0 # Initialize total accumulated damage
daily_values = [] # List to store daily damage values
accumulated_values = [] # List to store cumulative damage values over time
accumulated_dates = [] # List to store dates corresponding to accumulated values
limit_value = 1 # Predefined daily damage limit (hard limit) Based on a 60 year lifetime
soft_value = 0.8 # Predefined daily damage limit (soft limit) 80% of limit_day_value
accumulated_limit = [] # List to store the cumulative hard limit over time
accumulated_soft = [] # List to store the cumulative soft limit over time
currentMonth = currentDate.month
currentYear = currentDate.year
accumulated_yearly_damage = 0
accumulated_year_dates = []
accumulated_year_values = []

values = MongoDB.getDamageValues(MongoDB, fixed_start, today)

for value in values:
    if value['DMG'] != 0:
        accumulated_total_damage += value['DMG']
        if value['T'].year != currentYear:
            accumulated_dates.append(value['T'])
            if accumulated_values:
                accumulated_values.append(accumulated_values[len(accumulated_values)-1] + value['DMG'])
            else: 
                accumulated_values.append(value['DMG'])
        else:
            accumulated_year_dates.append(value['T'])
            if accumulated_year_values:
                accumulated_year_values.append(accumulated_year_values[len(accumulated_year_values)-1] + value['DMG'])
            else: 
                accumulated_year_values.append(accumulated_values[-1]+value['DMG'])
    if value['T'].year == currentYear:
        accumulated_yearly_damage += value['DMG']


# --- Save Daily Plot and Data ---
output_dir = './html/contents' # Define output directory for plots
# Define image path (though img_path is defined, it's not used; png_path1 is used)
# img_path = os.path.join(output_dir, f'dayCycles_{currentDate.date()}.png')

# Create a DataFrame from the raw daily values and save to CSV
#df = pd.DataFrame(values, columns=["T", "ZA"])
#df['T'] = pd.to_datetime(df['T'], unit='ms', errors='coerce') # Convert timestamp to datetime
#df = df.dropna(subset=['T', 'ZA']) # Drop rows with invalid date/ZA
#df.to_csv(f'dataFiles/data_{currentDate.date()}.csv', index=False) # Save daily data

#sys.exit(0) Day stopper     

# Current value and date
current_value = accumulated_year_values[-1]
current_date = accumulated_year_dates[-1]

# Estimate how many days until the value reaches 1

estimated_years = int((1-accumulated_total_damage) / accumulated_yearly_damage)
if estimated_years < 25500:
    estimated_date = current_date + relativedelta(years=estimated_years)
    projected_dates = [current_date + relativedelta(years=i) for i in range(0, estimated_years + 1)]
    projected_values = [accumulated_total_damage+(accumulated_yearly_damage * i) for i in range(0, estimated_years)]


#sys.exit(0)

# === Plot ===
fig4 = go.Figure()

# Blue line - actual values
fig4.add_trace(go.Scatter(
    x=accumulated_dates,
    y=accumulated_values,
    mode='lines+markers',
    name='Accumulated Damage (past)',
    line=dict(color='blue')
))
# Purple line - current year values
fig4.add_trace(go.Scatter(
    x=accumulated_year_dates,
    y=accumulated_year_values,
    mode='lines+markers',
    name='Accumulated Damage (current year)',
    line=dict(color='purple')
))

#Red limit line
fig4.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=1, y1=1,
    line=dict(color="red", width=2)
)

# Red dashed line - projection
if projected_dates:
    fig4.add_trace(go.Scatter(
        x=projected_dates,
        y=projected_values,
        mode='lines',
        name='Projection to Breaking Point bassed on all the data',
        line=dict(color='red', dash='dash')
    ))

    # Mark the point where damage reaches 1
    fig4.add_trace(go.Scatter(
        x=[estimated_date],
        y=[1],
        mode='markers+text',
        name='Estimated reach of Breaking Point',
        marker=dict(color='red', size=12),
        text=[f"{estimated_date}"],
        textposition="bottom center",
        textfont=dict(color="red", size=18),
    ))
percent = round(accumulated_total_damage * 100, 2)
label = f"{percent}%<br>{currentDate.year}"

fig4.add_trace(go.Scatter(
    x=[accumulated_year_dates[-1]],
    y=[accumulated_total_damage],
    mode='markers+text',
    name=f"{currentDate.year}",
    marker=dict(color='purple', size=12),
    text=label,
    textposition="middle right",
    textfont=dict(color="purple", size=18),
    showlegend=False
))

fig4.update_layout(
    font=dict(size=20),
    title=dict(
        text="Accumulated Damage and Projection to Break Point (Year)",
        y=0.97,
        x=0,
        xanchor="left",
        yanchor="top"
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
        x=-0.05
    ),
    margin=dict(t=100)
)

fig4.update_xaxes(showline=True, linewidth=2, linecolor='black') #Plot border on X axis
fig4.update_yaxes(showline=True, linewidth=2, linecolor='black') #Plot border on Y axis
os.makedirs(output_dir+"/projection_plots", exist_ok=True) # Ensure the output directory exists
fixed_date = currentDate
json_path4 = os.path.join(output_dir+"/projection_plots", f'Projection_{fixed_date.year}.json')
spec4 = fig_to_responsive_json(fig4)
with open(json_path4, "w") as f:
    json.dump(spec4, f, default=str)

currentYear = currentDate.year
# --- Post-Loop Analysis and Plotting ---
estimated_30_lifetime = accumulated_total_damage* 30
estimated_60_lifetime = accumulated_total_damage* 60
monthly_totals = defaultdict(float)
currentYear = MongoDB.getDamageValues(MongoDB, yesterday, today)
accumulatedDamage = []
for entry in currentYear:
    month = entry['T'].month
    monthly_totals[month] += entry['DMG']

for element in range(1,13):
    if element in monthly_totals:
        if not accumulatedDamage:
            accumulatedDamage.append(monthly_totals[element])
        else:
            if accumulatedDamage[len(accumulatedDamage)-1] != None:
                accumulatedDamage.append(monthly_totals[element] + accumulatedDamage[len(accumulatedDamage)-1])
            else:
                accumulatedDamage.append(monthly_totals[element])
    else:
        if not accumulatedDamage:
            accumulatedDamage.append(None)
        else:
            accumulatedDamage.append(0 + accumulatedDamage[len(accumulatedDamage)-1])


monthly_value = [monthly_totals[m] if m in monthly_totals else None for m in range(1, 13)]

totalYearDamage = 0
for element in currentYear:
    totalYearDamage += element['DMG']
estimated_30_lifetime_1 = totalYearDamage* 30
estimated_60_lifetime_1 = totalYearDamage* 60

previousYear = MongoDB.getDamageValues(MongoDB, yesterday-relativedelta(years=1), today-relativedelta(years=1))
totalPrevYearDamage = 0
prev_monthly_totals = defaultdict(float)
accumulatedPrevDamage = []
for entry in previousYear:
    month = entry['T'].month
    prev_monthly_totals[month] += entry['DMG']
    totalPrevYearDamage += entry['DMG']

for element in range(1,13):
    if element in prev_monthly_totals:
        if not accumulatedPrevDamage:
            accumulatedPrevDamage.append(prev_monthly_totals[element])
        else:
            if accumulatedPrevDamage[len(accumulatedPrevDamage)-1] != None:
                accumulatedPrevDamage.append(prev_monthly_totals[element] + accumulatedPrevDamage[len(accumulatedPrevDamage)-1])
            else:
                accumulatedPrevDamage.append(prev_monthly_totals[element])
    else:
        if not accumulatedPrevDamage:
            accumulatedPrevDamage.append(None)
        else:
            accumulatedPrevDamage.append(0 + accumulatedPrevDamage[len(accumulatedPrevDamage)-1])


prev_monthly_value = [prev_monthly_totals[m] if m in prev_monthly_totals else None for m in range(1, 13)]
estimated_30_lifetime_lastYear = totalPrevYearDamage* 30
estimated_60_lifetime_lastYear = totalPrevYearDamage* 60

previousPreviousYear = MongoDB.getDamageValues(MongoDB, yesterday-relativedelta(years=2), today-relativedelta(years=2))
prev_prev_monthly_totals = defaultdict(float)
totalPrevPrevYearDamage = 0
accumulatedPrevPrevDamage = []
for entry in previousPreviousYear:
    month = entry['T'].month
    prev_prev_monthly_totals[month] += entry['DMG']
    totalPrevPrevYearDamage += entry['DMG']

for element in range(1,13):
    if element in prev_prev_monthly_totals:
        if not accumulatedPrevPrevDamage:
            if prev_prev_monthly_totals[element] != 0:
                accumulatedPrevPrevDamage.append(prev_prev_monthly_totals[element])
        else:
            if accumulatedPrevPrevDamage[len(accumulatedPrevPrevDamage)-1] != None:
                accumulatedPrevPrevDamage.append(prev_prev_monthly_totals[element] + accumulatedPrevPrevDamage[len(accumulatedPrevPrevDamage)-1])
            else:
                if prev_prev_monthly_totals[element] != 0:
                    accumulatedPrevPrevDamage.append(prev_prev_monthly_totals[element])
    else:
        if not accumulatedPrevPrevDamage:
            accumulatedPrevPrevDamage.append(None)
        else:
            accumulatedPrevPrevDamage.append(0 + accumulatedPrevPrevDamage[len(accumulatedPrevPrevDamage)-1])

prev_prev_monthly_value = [prev_prev_monthly_totals[m] if m in prev_prev_monthly_totals else None for m in range(1, 13)]
estimated_30_lifetime_last_lastYear = totalPrevPrevYearDamage* 30
estimated_60_lifetime_last_lastYear = totalPrevPrevYearDamage* 60

# --- Plot 2: Daily and Accumulated Damage Evolution ---
fig2 = go.Figure() # Initialize a new Plotly figure

# Add a bar trace for daily damage values
fig2.add_trace(go.Bar(
    x=[calendar.month_abbr[element] for element in range(1,13)],
    y=monthly_value,
    name=f"Daily Damage {yesterday.year}",
    marker=dict(color="steelblue"),
    opacity=0.6
))

# Add a line trace for accumulated damage
fig2.add_trace(go.Scatter(
    x=[calendar.month_abbr[element] for element in range(1, 13)],
    y=accumulatedDamage,
    mode="lines+markers",
    line=dict(color="dodgerblue", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {yesterday.year}"
))

# Add a line trace for the accumulated hard limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_60_lifetime, y1=estimated_60_lifetime,
    line=dict(color="dodgerblue", width=2, dash="solid"),
)

# Add a line trace for the accumulated soft limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_30_lifetime, y1=estimated_30_lifetime,
    line=dict(color="steelblue", width=2, dash="dot"),
)

# Add a bar trace for daily damage values
fig2.add_trace(go.Bar(
    x=[calendar.month_abbr[element] for element in range(1,13)],
    y=prev_monthly_value,
    name=f"Daily Damage {yesterday.year-1}",
    marker=dict(color="darkorange"),
    opacity=0.3
))

# Add a line trace for accumulated damage
fig2.add_trace(go.Scatter(
    x=[calendar.month_abbr[element] for element in range(1, 13)],
    y=accumulatedPrevDamage,
    mode="lines+markers",
    line=dict(color="orangered", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {yesterday.year-1}",
    opacity=0.3
))

# Add a line trace for the accumulated hard limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_60_lifetime_lastYear, y1=estimated_60_lifetime_lastYear,
    line=dict(color="orangered", width=2, dash="solid"),
    opacity=0.75
)

# Add a line trace for the accumulated soft limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_30_lifetime_lastYear, y1=estimated_30_lifetime_lastYear,
    line=dict(color="darkorange", width=2, dash="dot"),
    opacity=0.75
)

# Add a bar trace for daily damage values
fig2.add_trace(go.Bar(
    x=[calendar.month_abbr[element] for element in range(1,13)],
    y=prev_prev_monthly_value,
    name=f"Daily Damage {yesterday.year-2}",
    marker=dict(color="darkgreen"),
    opacity=0.3
))

# Add a line trace for accumulated damage
fig2.add_trace(go.Scatter(
    x=[calendar.month_abbr[element] for element in range(1, 13)],
    y=accumulatedPrevPrevDamage,
    mode="lines+markers",
    line=dict(color="springgreen", width=3),
    marker=dict(size=6),
    name=f"Accumulated Damage {yesterday.year-2}",
    opacity=0.3
))

# Add a line trace for the accumulated hard limit
#Red limit line
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_60_lifetime_last_lastYear, y1=estimated_60_lifetime_last_lastYear,
    line=dict(color="springgreen", width=2, dash="solid"),
    opacity=0.75
)

# Add a line trace for the accumulated soft limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=estimated_30_lifetime_last_lastYear, y1=estimated_30_lifetime_last_lastYear,
    line=dict(color="darkgreen", width=2, dash="dot"),
    opacity=0.75
)

fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=1, y1=1,
    line=dict(color="black", width=2, dash="solid"),
    opacity=0.75
)

# Add a line trace for the accumulated soft limit
fig2.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=0, x1=1,
    y0=0.8, y1=0.8,
    line=dict(color="black", width=2, dash="dot"),
    opacity=0.75
)

# Update layout for the second plot
fig2.update_layout(
    font=dict(size=20),
    title=dict(
        text="Monthly and accumulated evolution with limits",
        y=0.97,
        x=0,
        xanchor="left",
        yanchor="top"
    ),
    xaxis_title="Date",
    yaxis_title="Damage",
    template="plotly_white", # Use a clean plot template
    #legend=dict(orientation="v", x=1.02, y=0.95) # Position the legend
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=-0.05
    ),
    margin=dict(t=100)
)

# Set y-axis to logarithmic scale for better visualization of damage values
fig2.update_yaxes(
    type="log",
    tickformat=".0e", # Format y-axis ticks in scientific notation
    tickvals=[1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    range=[-6, 1],
    tickfont=dict(size=18),
    showline=True,
    linewidth=2,
    linecolor='black'
)

dates = pd.date_range(start=datetime(currentYear[0]['T'].year, 1, 1), end=datetime(currentYear[0]['T'].year, 12, 1), freq='MS')
tickvals = [calendar.month_abbr[d.to_pydatetime().month] for d in dates]
# Format x-axis for better readability of dates
fig2.update_xaxes(
    tickformat="%b", # Format date display
    tickvals=tickvals,
    tickfont=dict(size=18),
    minor=dict(ticklen=8, tickcolor="black"),
    showline=True,
    linewidth=2,
    linecolor='black'
)

# Define path and save the accumulated damage plot
os.makedirs(output_dir+"/accumulation_plots", exist_ok=True) # Ensure the output directory exists
fixed_date = currentDate
json_path2 = os.path.join(output_dir+"/accumulation_plots", f'Accumulated_{fixed_date.year}.json')
spec2 = fig_to_responsive_json(fig2)
with open(json_path2, "w") as f:
    json.dump(spec2, f,  default=str)

print("General plot have been generated") # Final confirmation message

if False: #Optional to generate the sn curve
    # --- Plot 3: S-N Curve ---
    fig3 = go.Figure() # Initialize a new Plotly figure

    # Generate points for plotting the S-N curve smoothly
    N_plot = np.logspace(4, 9, 500)  # Generate 500 points from 10^4 to 10^9 on a log scale
    stress_plot = sn_curve(N_plot, a, b) # Calculate stress values using the fitted S-N curve

    # Add a line trace for the fitted S-N curve
    fig3.add_trace(go.Scatter(
        x=N_plot,
        y=stress_plot,
        mode='lines',
        name='S-N curve',
        line=dict(color='steelblue', width=3)
    ))

    # Add a scatter trace for the original experimental data points
    fig3.add_trace(go.Scatter(
        x=cycles, # Original cycle data
        y=stress, # Original stress data
        mode='markers',
        name='Spots', # Legend name for data points
        marker=dict(size=8, color='darkorange')
    ))

    # Update x-axis for the S-N curve plot
    fig3.update_xaxes(
        type='log', # Set x-axis to logarithmic scale
        title='Cycles (N)',
        tickformat='.1e', # Scientific notation for ticks
        range=[4, 9],  # Set x-axis range (log10(1e4) to log10(1e9))
        tickfont = dict(size=32)
    )

    # Update y-axis for the S-N curve plot
    fig3.update_yaxes(
        title='Stress (MPa)',
        tickfont = dict(size=32)
    )

    # Update layout for the S-N curve plot
    fig3.update_layout(
        font=dict(size=28),
        title='S-N Curve',
        template='plotly_white'
    )

    # Define path and save the S-N curve plot
    png_path3 = os.path.join(output_dir, f'S-N_Curve_{yesterday.year}-{yesterday.month}.png')
    fig3.write_image(png_path3, width=1920, height=1080, scale=1)