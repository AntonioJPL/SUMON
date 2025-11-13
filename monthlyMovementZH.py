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
import json

print('Generating the month plots...') # Initial script status message

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
yesterday = today - relativedelta(months=1) + relativedelta(day=1)
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

# --- Main Processing Loop: Iterates through each day in the specified range ---
while currentDate.date() <= today.date():
    if currentDate.date() < today.date():

        # Define time range for data fetching (from 12:00 PM current day to 11:59:59 AM next day in UTC)
        # Timestamps are converted to milliseconds for MongoDB query
        T_min = int(currentDate.replace(hour=17, minute=0, second=0, microsecond=0, tzinfo=UTC).timestamp() * 1000)
        next_day = currentDate + timedelta(days=1)
        T_max = int(next_day.replace(hour=7, minute=59, second=59, microsecond=0, tzinfo=UTC).timestamp() * 1000)

        # Fetch daily Zenith Angle (ZA) data from MongoDB for the defined time range
        # The MongoDB.getDailyZenith method is assumed to be defined in 'mongo_utils.py'
        values = MongoDB.getDailyZenith(MongoDB, T_min, T_max) # `MongoDB` is passed as the first argument, which might be a class instance or a module reference

        # Prepare data for plotting and analysis
        datetime_objects = [datetime.fromtimestamp(item['T'] / 1000, tz=UTC) for item in values] # Convert timestamps to datetime objects
        za_values = [item['ZA'] for item in values] # Extract Zenith Angle values

        # Format time and ZA values for display or tables (not directly used in plots later)
        formatted_time_table = [t.strftime('%Y-%m-%d %H:%M') for t in datetime_objects]
        formatted_za_values = ["{:02.2f}".format(val) for val in za_values]

        # --- Plot 1: Daily Zenith Angle Variation ---
        #fig1 = go.Figure() # Initialize a Plotly graph object
        # Add a scatter trace for Zenith Angle vs. Time
        #fig1.add_trace(go.Scatter(x=datetime_objects, y=za_values, mode='lines+markers', name='Zenith Angle'))

        # Update layout for the first plot
        #fig1.update_layout(
        #    font=dict(size=28),
        #    xaxis=dict(tickfont=dict(size=32)),
        #    yaxis=dict(tickfont=dict(size=32)),
        #    xaxis_tickformat='%Y-%m-%d %H:%M', # Format x-axis ticks as date and time
        #    yaxis_tickformat='02.2f', # Format y-axis ticks to two decimal places
        #    yaxis_title='Zenith Angle', # Set y-axis title
        #    xaxis_title='Time (WEST)' # Set x-axis title (Note: Data is in UTC, title says WEST)
        #)

        # Load conversion data from CSV (degrees to stress in MPa)
        conversion = pd.read_csv('./deg_to_stress.csv')

        # --- Abrupt Movement Identification Logic ---
        # Initialize variables for detecting significant changes (cycles) in Zenith Angle
        abrupt_movements = [] # List to store identified abrupt movements
        prev_value = None # Stores the previous data point {ZA, T}
        # Variables for identifying start and end points of a potential cycle
        start_down_value = None # Marks the start of a downward ZA trend (potential cycle start)
        start_up_value = None   # Marks the start of an upward ZA trend (potential cycle end)
        deepest_point = None    # Tracks the minimum ZA value in the current segment
        deepest_time = None     # Timestamp of the deepest_point
        highest_point = None    # Tracks the maximum ZA value in the current segment
        highest_time = None     # Timestamp of the highest_point
        # Candidate points for cycle boundaries, refined as data is processed
        start_down_candidate = None
        start_up_candidate = None

        # Iterate through each data point (ZA and timestamp) for the current day
        for i, current_value in enumerate(values):
            current_za = current_value['ZA']
            current_time = current_value['T']

            # Identify the overall highest and deepest ZA points for the day
            if not deepest_point or current_za < deepest_point:
                deepest_point = current_za
                deepest_time = current_time
                # Reset highest point if a new deepest point is found (implies a new trend segment)
                highest_point = None
                highest_time = None
            if not highest_point or current_za > highest_point:
                highest_point = current_za
                highest_time = current_time
            
            # Logic to identify cycles based on ZA changes (this is complex state-based logic)
            # This section attempts to define a "cycle" by looking for patterns of ZA decreasing then increasing.
            # Thresholds (e.g., 0.25, -0.1, -0.25, 0.1) define significant changes.

            if not start_up_value and not start_down_value: # Initial state or after a cycle is completed
                start_up_value = current_value # Assume an upward trend might start
            elif not start_down_value and start_up_value and start_down_candidate and current_value['ZA'] - start_down_candidate['ZA'] > 0.25:
                # Confirmed downward trend start after an upward phase
                start_down_value = start_down_candidate
                start_up_value = None # Reset start_up_value, looking for end of downward trend
                start_down_candidate = None
            elif not start_down_value and not start_down_candidate and start_up_value and current_value['ZA'] - prev_value['ZA'] > 0:
                # Potential start of a downward trend (ZA increased then previous was lower)
                if prev_value: # Ensure prev_value exists
                    start_down_candidate = prev_value
            elif not start_down_value and start_down_candidate and start_up_value and current_value['ZA'] - start_down_candidate['ZA'] < -0.1:
                # Downward candidate invalidated, ZA not decreasing enough from candidate
                start_down_candidate = None
            elif start_down_value and start_up_candidate and not start_up_value and current_value['ZA'] > start_down_value['ZA'] and current_value['ZA'] - start_up_candidate['ZA'] < -0.25:
                # Confirmed upward trend start after a downward phase (cycle completed)
                start_up_value = start_up_candidate
                abrupt_movements.append({
                                        'start_value': start_down_value['ZA'],
                                        'start_time': start_down_value['T'],
                                        'end_value': start_up_value['ZA'],
                                        'end_time': start_up_value['T']
                                    })
                # Reset for next cycle detection
                start_up_candidate = None
                start_down_value = None
            elif start_down_value and start_up_candidate and not start_up_value and current_value['ZA'] < start_down_value['ZA'] and current_value['ZA'] - start_up_candidate['ZA'] < -0.25:
                start_down_value = None
                start_up_value = start_up_candidate
                start_up_candidate = None
            elif start_down_value and not start_up_value and not start_up_candidate  and current_value['ZA'] - prev_value['ZA'] < 0:
                # Potential start of an upward trend (ZA decreased then previous was higher)
                if prev_value: # Ensure prev_value exists
                    start_up_candidate = prev_value
            elif start_down_value and start_up_candidate and not start_up_value and current_value['ZA'] - start_up_candidate['ZA'] > 0.1:
                # Upward candidate invalidated, ZA not increasing enough from candidate
                start_up_candidate = None
            
            prev_value = current_value # Update previous value for the next iteration
        
        # Handle a potential incomplete cycle at the end of the data
        if start_down_value and start_up_candidate:
            abrupt_movements.append({
                                        'start_value': start_down_value['ZA'],
                                        'start_time': start_down_value['T'],
                                        'end_value': start_up_candidate['ZA'], # Use candidate as end
                                        'end_time': start_up_candidate['T']
                                    })
        elif start_down_value:
            abrupt_movements.append({
                                        'start_value': start_down_value['ZA'],
                                        'start_time': start_down_value['T'],
                                        'end_value': prev_value['ZA'], # Use prev value as end
                                        'end_time': prev_value['T']
                                    })

        # Ensure the overall largest daily fluctuation (deepest to highest) is included as a movement
        if highest_point and deepest_point:
            element = {
                'start_value': deepest_point,
                'start_time': deepest_time,
                'end_value': highest_point,
                'end_time': highest_time
            }
            if element not in abrupt_movements: # Avoid duplicating if already identified
                abrupt_movements.append(element)

        # --- Process and Plot Abrupt Movements ---
        # Iterate through a copy of abrupt_movements for safe removal if needed (though not used here)
        for element in abrupt_movements[:]:
            #delta = element['end_value'] - element['start_value'] # Calculate ZA change for the movement
            #if delta < 2:
            #    abrupt_movements.remove(element)
            #else:
            # Convert timestamps to datetime objects for plotting
            t0 = datetime.fromtimestamp(element['start_time']/1000, tz=UTC)
            t1 = datetime.fromtimestamp(element['end_time']/1000, tz=UTC)
            y0, y1 = element['start_value'], element['end_value'] # Get ZA start and end values
            # Add a line trace to fig1 for each identified abrupt movement
            #fig1.add_trace(go.Scatter(
            #    x=[t0, t1],
            #    y=[y0, y1],
            #    mode='lines',
            #    line=dict(color='crimson', width=4), # Style the line
            #    name='Section' # Label for the legend
            #))

        # --- Damage Calculation ---
        def process_movement(element, grouped_values):
            """
            Processes a single abrupt movement to calculate its contribution to damage.
            Converts ZA change to stress and counts occurrences of each stress level.
            """
            start_value = element['start_value'] # ZA at the start of the movement
            end_value = element['end_value'] # ZA at the end of the movement
            start_stress = None
            end_stress = None
            stress_value = None

            # Check if ZA values are within a plausible range (0-100 degrees)
            if start_value < 100 and end_value < 100 and start_value >= 0 and end_value >= 0:
                # Query the conversion table to find MPa stress corresponding to rounded ZA degrees
                matching_row_start = conversion.query(f"Degree == {round(start_value)}")
                matching_row_end = conversion.query(f"Degree == {round(end_value)}")

                # Safely get the MPa value if a match was found
                if not matching_row_start.empty:
                    start_stress = round(matching_row_start.iloc[0]["MPa"])
                if not matching_row_end.empty:
                    end_stress = round(matching_row_end.iloc[0]["MPa"])

                # If both start and end stresses are found, calculate the stress amplitude
                if start_stress is not None and end_stress is not None:
                    stress_value = abs(start_stress - end_stress) # Stress amplitude of the cycle
                    if stress_value > 0:
                        # Group by stress value and count occurrences (cycles at that stress level)
                        if str(stress_value) in grouped_values:
                            grouped_values[str(stress_value)] += 1
                        else:
                            grouped_values[str(stress_value)] = 1
            
        grouped_values = {} # Dictionary to store stress amplitudes and their counts
        accumulated_damage_today = 0 # Initialize damage for the current day

        # Process each identified abrupt movement to populate grouped_values
        for element in abrupt_movements:
            process_movement(element, grouped_values)

        # Calculate Miner's rule damage for the current day
        for stress_amplitude_str, num_cycles_at_stress in grouped_values.items():
            rounded_stress = round(float(stress_amplitude_str), 2) # Convert stress key to float
            # Estimate max cycles to failure at this stress level using the S-N curve
            max_cycles_to_failure = estimate_cycles(rounded_stress)
            # Add the damage fraction (actual cycles / cycles to failure)
            if max_cycles_to_failure > 0: # Avoid division by zero
                accumulated_damage_today += num_cycles_at_stress / max_cycles_to_failure
            
        accumulated_total_damage += accumulated_damage_today # Add today's damage to the overall total

        # Store daily and accumulated values for later plotting
        if currentDate.date() >= yesterday.date() and currentDate.date() <= today.date():
            daily_values.append(accumulated_damage_today if accumulated_damage_today != 0 else None)
            accumulated_dates.append(currentDate.date()) # Use the actual date object for plotting
            if accumulated_values:
                accumulated_values.append(accumulated_total_damage if accumulated_total_damage != 0 else accumulated_values[len(accumulated_values)-1])
            else:
                if accumulated_total_damage != 0:
                    accumulated_values.append(accumulated_total_damage)
                else:
                    accumulated_values.append(None)

        # --- Save Daily Plot and Data ---
        output_dir = './html/contents' # Define output directory for plots
        # Define image path (though img_path is defined, it's not used; png_path1 is used)
        # img_path = os.path.join(output_dir, f'dayCycles_{currentDate.date()}.png')

        # Create a DataFrame from the raw daily values and save to CSV
        #df = pd.DataFrame(values, columns=["T", "ZA"])
        #df['T'] = pd.to_datetime(df['T'], unit='ms', errors='coerce') # Convert timestamp to datetime
        #df = df.dropna(subset=['T', 'ZA']) # Drop rows with invalid date/ZA
        #df.to_csv(f'dataFiles/data_{currentDate.date()}.csv', index=False) # Save daily data

        #os.makedirs(output_dir+"/zenith_plots", exist_ok=True) # Ensure the output directory exists
        # Define path for saving the daily Zenith plot
        #png_path1 = os.path.join(output_dir+"/zenith_plots", f'Zenith_{currentDate.date()}.png')
        #fig1.update_xaxes(showline=True, linewidth=2, linecolor='black') #Plot border on X axis
        #fig1.update_yaxes(showline=True, linewidth=2, linecolor='black') #Plot border on Y axis
        #fig1.write_image(png_path1, width=1920, height=1080, scale=1) # Save the plot as a PNG

    if currentDate.month != currentMonth or currentDate.date() == today.date():
        currentMonth = currentDate.month
        projected_dates = None

        # Current value and date
        current_value = accumulated_values[-1]
        current_date = accumulated_dates[-1]

        # Estimate how many days until the value reaches 1

        estimated_months = int(1 / accumulated_total_damage)
        if estimated_months < 305000:
            estimated_date = current_date + relativedelta(months=estimated_months)
            projected_dates = [current_date + relativedelta(months=i) for i in range(1, estimated_months + 1)]
            projected_values = [current_value * i for i in range(1, estimated_months + 1)]
       
        # === Plot ===
        fig4 = go.Figure()

        # Blue line - actual values
        fig4.add_trace(go.Scatter(
            x=accumulated_dates,
            y=accumulated_values,
            mode='lines+markers',
            name='Accumulated Damage (actual)',
            line=dict(color='blue')
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
                name='Projection to Breaking Point <br> based on month data',
                line=dict(color='red', dash='dash')
            ))

            # Mark the point where damage reaches 1
            fig4.add_trace(go.Scatter(
                x=[estimated_date],
                y=[1],
                mode='markers+text',
                name='Estimated reach of <br> Breaking Point',
                marker=dict(color='red', size=12),
                text=[f"{estimated_date}"],
                textposition="bottom center",
                textfont=dict(color="red", size=18),
            ))

        if currentDate.date() == today.date():
            percent = round((accumulated_total_damage / limit_value) * 100, 2)
            fixed_date = currentDate - relativedelta(months=1)
            label = f"{percent}%<br>{calendar.month_abbr[fixed_date.month]} - {fixed_date.year}"
            fig4.add_trace(go.Scatter(
                x=[currentDate],
                y=[accumulated_total_damage],
                mode='markers+text',
                name=f"{currentDate.month-1} - {currentDate.year}",
                marker=dict(color='purple', size=12),
                text=label,
                textposition="middle right",
                textfont=dict(color="purple", size=18),
                showlegend=False
            ))

        fig4.update_layout(
            font=dict(size=20),
            title=dict(
                text="Accumulated Damage and Projection to Break Point",
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
        fixed_date = currentDate-relativedelta(months=1)
        json_path4 = os.path.join(output_dir+"/projection_plots", f'Projection_{fixed_date.year}-{fixed_date.month}.json')
        spec4 = fig_to_responsive_json(fig4)
        with open(json_path4, "w") as f:
            json.dump(spec4, f, default=str)

    if currentDate.date() == today.date():
        currentYear = currentDate.year
        # --- Post-Loop Analysis and Plotting ---
        currentYear = MongoDB.getDamageValues(MongoDB, yesterday, today)
        totalYearDamage = 0
        for element in currentYear:
            if element != None:
                totalYearDamage += element['DMG']
        estimated_30_lifetime_1 = totalYearDamage*12 * 30
        estimated_60_lifetime_1 = totalYearDamage*12 * 60

        previousYear = MongoDB.getDamageValues(MongoDB, yesterday-relativedelta(years=1), today-relativedelta(years=1))
        previousPreviousYear = MongoDB.getDamageValues(MongoDB, yesterday-relativedelta(years=2), today-relativedelta(years=2))

        totalPrevYearDamage = 0
        accumulatedPrevDamage= []
        for element in previousYear:
            totalPrevYearDamage += element['DMG']
            accumulatedPrevDamage.append(totalPrevYearDamage if totalPrevYearDamage else None)
        estimated_30_lifetime_lastYear = totalPrevYearDamage*12 * 30
        estimated_60_lifetime_lastYear = totalPrevYearDamage*12 * 60

        totalPrevPrevYearDamage = 0
        accumulatedPrevPrevDamage= []
        for element in previousPreviousYear:
            totalPrevPrevYearDamage += element['DMG']
            accumulatedPrevPrevDamage.append(totalPrevPrevYearDamage if totalPrevPrevYearDamage else None)
        estimated_30_lifetime_last_lastYear = totalPrevPrevYearDamage*12 * 30
        estimated_60_lifetime_last_lastYear = totalPrevPrevYearDamage*12 * 60
        
        # --- Plot 2: Daily and Accumulated Damage Evolution ---
        fig2 = go.Figure() # Initialize a new Plotly figure

        if daily_values and accumulated_values:
            # Add a bar trace for daily damage values
            fig2.add_trace(go.Bar(
                x=accumulated_dates,
                y=daily_values,
                name=f"Daily Damage {yesterday.year}",
                marker=dict(color="steelblue"),
                opacity=0.6
            ))
            # Add a line trace for accumulated damage
            fig2.add_trace(go.Scatter(
                x=accumulated_dates,
                y=accumulated_values,
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
                y0=estimated_60_lifetime_1, y1=estimated_60_lifetime_1,
                line=dict(color="dodgerblue", width=2, dash="solid"),
            )

            # Add a line trace for the accumulated soft limit
            fig2.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0, x1=1,
                y0=estimated_30_lifetime_1, y1=estimated_30_lifetime_1,
                line=dict(color="steelblue", width=2, dash="dot"),
            )

        if previousYear and accumulatedPrevDamage:
            # Add a bar trace for daily damage values
            fig2.add_trace(go.Bar(
                x=[element['T'].date()+relativedelta(year=yesterday.year) for element in previousYear],
                y=[element['DMG'] if element['DMG'] != 0 else None for element in previousYear],
                name=f"Daily Damage {yesterday.year-1}",
                marker=dict(color="darkorange"),
                opacity=0.3
            ))
            # Add a line trace for accumulated damage
            fig2.add_trace(go.Scatter(
                x=[element['T'].date()+relativedelta(year=yesterday.year) for element in previousYear],
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
        
        if previousPreviousYear and accumulatedPrevPrevDamage:
            # Add a bar trace for daily damage values
            fig2.add_trace(go.Bar(
                x=[element['T'].date()+relativedelta(year=yesterday.year) for element in previousPreviousYear],
                y=[element['DMG'] if element['DMG'] != 0 else None for element in previousPreviousYear],
                name=f"Daily Damage {yesterday.year-2}",
                marker=dict(color="darkgreen"),
                opacity=0.3
            ))
            # Add a line trace for accumulated damage
            fig2.add_trace(go.Scatter(
                x=[element['T'].date()+relativedelta(year=yesterday.year) for element in previousPreviousYear],
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
                text="Daily and accumulated evolution with limits",
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
        
        #Select the dates for the xaxes
        start = datetime(accumulated_dates[0].year, accumulated_dates[0].month, 1)
        end = start + pd.offsets.MonthEnd(0)
        date_range = pd.date_range(start=start, end=end, freq='D')
        tickVals = []
        tickVals.append(start)
        lastInserted = 0
        if end.day % 2 == 0:
            for index, date in enumerate(date_range):
                if index % 2 == 0 and index > lastInserted + 4:
                    if date not in tickVals:
                        tickVals.append(date)
                        lastInserted = index
        else:
            for index, date in enumerate(date_range):
                if index % 2 != 0 and index > lastInserted + 5:
                    if date not in tickVals:
                        tickVals.append(date)
                        lastInserted = index
        
        if end not in tickVals:
            tickVals.append(end)

        # Format x-axis for better readability of dates
        fig2.update_xaxes(
            tickformat="%d<br>%b", # Format date display
            tickvals=tickVals,
            tickfont=dict(size=18),
            minor=dict(ticklen=8, tickcolor="black"),
            showline=True,
            linewidth=2,
            linecolor='black'
        )

        # Define path and save the accumulated damage plot
        os.makedirs(output_dir+"/accumulation_plots", exist_ok=True) # Ensure the output directory exists
        fixed_date = currentDate-relativedelta(months=1)
        json_path2 = os.path.join(output_dir+"/accumulation_plots", f'Accumulated_{fixed_date.year}-{fixed_date.month}.json')
        spec2 = fig_to_responsive_json(fig2)
        with open(json_path2, "w") as f:
            json.dump(spec2, f,  default=str)

        print("General monthly plots have been generated") # Final confirmation message
    currentDate +=  timedelta(days=1)

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