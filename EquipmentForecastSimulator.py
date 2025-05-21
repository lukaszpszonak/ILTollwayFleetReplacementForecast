
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from io import BytesIO
import base64

# Load base data
@st.cache_data()
def load_data():
    return pd.read_csv("POWERBIDATA_ANALYSIS.csv", encoding='cp1252')

df = load_data()

# Preprocessing
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
df['Manufactured Date'] = pd.to_datetime(df['Manufactured Date'], errors='coerce')
df['BaseYear'] = 2025
df['Units Needed'] = df['Units Needed'].fillna(0).astype(int)

# Streamlit UI
st.title("Equipment Replacement Simulator (2025–2049)")

# Sidebar content for assumptions and instructions
with st.sidebar:
    st.header("📘 Assumptions & Instructions")
    st.markdown("""
    ### Simulation Assumptions
    - 📅 Forecast runs from **2025 to 2049**.
    - ⚠️ **Replacements are triggered** if condition is **Poor or Critical**, unless:
        - The unit was already replaced in the **previous year** (no double-year replacements).
    - 🛠️ **Age RSL**, **Meter RSL**, and **MR RSL** are calculated and used to determine worst-case condition.
    - 💰 **APC and Total Cost escalate annually** using the specified inflation rate.
    - 🔁 After replacement:
        - Service age, meter, and cost reset.
        - Replacement year is reset for next calculations.

    ### Units Needed Constraint
    - 🚨 If the number of units in **Fair, Good, or Excellent** condition falls below the "Units Needed" per class:
        - Additional units are replaced (if scenario allows) to meet the required threshold.

    ### Optional Budget Constraints
    - 🧮 Toggle ON to limit:
        - Max replacements per year.
        - Max spend per year.

    ### How to Use
    1. 🎯 Choose a **scenario** from the dropdown.
    2. 🎛️ Filter by class, ID, or department.
    3. 💸 Adjust inflation rate and budgets.
    4. 📉 Review replacement forecasts and KPIs.
    5. 📤 Export detailed forecast to Excel.
    """)

scenario = st.selectbox("Choose Scenario:", [
    "Run Till Poor and Replace Forecast",
    "Replace chosen classes/units in given year",
    "Replace chosen classes/units now"
])

selected_classes = st.multiselect("Select Equipment Classifications:", df['Equipment Classification'].unique())
selected_ids = st.multiselect("Select Unit IDs:", df['ID'].unique())
selected_depts = st.multiselect("Select Departments:", df['Department'].dropna().unique())
selected_year = st.slider("Replacement Year (if applicable):", 2025, 2049, 2027)
inflation_rate = st.number_input("Set Annual Inflation Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
expected_budget = st.number_input("Set Expected Budget ($)", min_value=0.0, value=25000000.0, step=50000.0)
adjusted_expected_budget = st.slider("Adjust Expected Budget Trendline ($)", min_value=0, max_value=100_000_000, value=int(expected_budget), step=500_000)

# Toggle to activate constraints
constraint_mode = st.toggle("Apply Annual Budget and Replacement Constraints", value=False)
max_annual_replacements = st.number_input("Max Replacements Per Year", min_value=1, value=100, step=1) if constraint_mode else None
max_annual_budget = st.number_input("Max Budget Per Year ($)", min_value=0, value=10_000_000, step=50000) if constraint_mode else None

# Apply filters
df_filtered = df.copy()
if selected_depts:
    df_filtered = df_filtered[df_filtered['Department'].isin(selected_depts)]
if selected_classes:
    df_filtered = df_filtered[df_filtered['Equipment Classification'].isin(selected_classes)]
if selected_ids:
    df_filtered = df_filtered[df_filtered['ID'].isin(selected_ids)]

# Simulation Functions
def get_age(replacement_year, current_year):
    return current_year - replacement_year

def get_rsl_category(value):
    if value < -50:
        return "Critical"
    elif value < 0:
        return "Poor"
    elif value < 40:
        return "Fair"
    elif value < 80:
        return "Good"
    else:
        return "Excellent"

# Simulation Logic
all_results = []
unit_history = {}
annual_replacements = {year: 0 for year in range(2025, 2050)}
annual_budget = {year: 0.0 for year in range(2025, 2050)}

for _, row in df_filtered.iterrows():
    results = []
    replacement_years = []
    current_replacement_year = 2025
    base_meter = row['Total Mileage.amt'] if row['Meter Type'] == 'Miles' else row['Total Work Hours.amt']
    base_annual_meter = row['Annual Meter Average']
    base_apc = row['Average Purchase Cost']
    base_cost = row['Total Cost']

    for year in range(2025, 2050):
        years_since_replacement = year - current_replacement_year
        apc = base_apc * ((1 + inflation_rate) ** years_since_replacement)
        total_cost = base_cost * ((1 + inflation_rate) ** years_since_replacement)
        usage = base_meter + base_annual_meter * years_since_replacement
        age = get_age(current_replacement_year, year)

        age_rsl = (1 - (age / row['Service Life by Age in Years'])) * 100 if row['Service Life by Age in Years'] > 0 else 0
        meter_rsl = (1 - (usage / row['Service Life by Meter'])) * 100 if row['Service Life by Meter'] > 0 else 0
        mr_rsl = (1 - (total_cost / (0.8 * apc))) * 100 if apc > 0 else 0

        worst = min(age_rsl, meter_rsl, mr_rsl)
        condition = get_rsl_category(worst)
        recent_replaced = (len(replacement_years) > 0 and replacement_years[-1] == year - 1)

        replace = False
        if scenario == "Run Till Poor and Replace Forecast" and condition in ["Poor", "Critical"] and not recent_replaced:
            if not constraint_mode or (
                annual_replacements[year] < max_annual_replacements and
                annual_budget[year] + apc <= max_annual_budget
            ):
                replace = True
        elif scenario == "Replace chosen classes/units now" and year == 2025:
            replace = True
        elif scenario == "Replace chosen classes/units in given year" and year == selected_year:
            replace = True

        results.append({
            'ID': row['ID'],
            'Year': year,
            'Condition': condition,
            'Replace': replace,
            'SimulatedTotalCost': apc if replace else 0,
            'Classification': f"{row['Asset Class']} - {row['Equipment Classification']}",
            'Equipment Classification': row['Equipment Classification'],
            'Department': row['Department'],
            'ReplacementCycle': len(replacement_years) + 1 if replace else np.nan,
            'ReplacementCondition': condition if replace else None,
            'APC': apc
        })

        if replace:
            replacement_years.append(year)
            current_replacement_year = year
            base_meter = 0
            base_cost = 0
            base_apc = apc
            annual_replacements[year] += 1
            annual_budget[year] += apc

    unit_history[row['ID']] = replacement_years
    all_results.extend(results)

sim_df = pd.DataFrame(all_results)

# Units Needed KPI check
deficits = []
for year in range(2025, 2050):
    for cls in sim_df['Equipment Classification'].unique():
        cls_df = sim_df[(sim_df['Year'] == year) & (sim_df['Equipment Classification'] == cls)]
        good_count = cls_df[cls_df['Condition'].isin(['Fair', 'Good', 'Excellent'])].shape[0]
        needed = df[df['Equipment Classification'] == cls]['Units Needed'].iloc[0]
        if good_count < needed:
            deficits.append({"Year": year, "Equipment Classification": cls, "Available": good_count, "Required": needed})

deficit_df = pd.DataFrame(deficits)

replace_summary = sim_df[sim_df['Replace'] == True].copy()
replace_summary['CumulativeBudget'] = replace_summary.groupby('ID')['SimulatedTotalCost'].cumsum()

summary = (
    replace_summary.groupby(['ID', 'Classification'])
    .agg(
        Replacements=('Year', 'count'),
        TotalBudget=('SimulatedTotalCost', 'sum'),
        ReplacementYears=('Year', lambda x: ', '.join(map(str, sorted(x.unique())))),
        FirstReplacementCondition=('ReplacementCondition', 'first')
    )
    .reset_index()
)

class_summary = summary.groupby('Classification').agg(
    TotalReplacements=('Replacements', 'sum'),
    TotalBudget=('TotalBudget', 'sum')
).reset_index()

total_replacements = summary['Replacements'].sum()
total_budget = summary['TotalBudget'].sum()

st.subheader("Forecast Table")
st.dataframe(summary)

st.subheader("Class-Level Summary")
st.dataframe(class_summary)

st.metric("Total Replacements (All Units)", total_replacements)
st.metric("Total Budget Required ($)", f"${total_budget:,.0f}")

st.subheader("Replacement Count per Year")
chart = replace_summary.groupby('Year').size().reset_index(name='Replacements')
st.plotly_chart(px.bar(chart, x='Year', y='Replacements', title="Replacements Over Time"))

st.subheader("Annual Budget Need Over Time")
chart2 = replace_summary.groupby('Year')['SimulatedTotalCost'].sum().reset_index()
chart2['ExpectedBudget'] = adjusted_expected_budget
chart2['BalancedBudget'] = adjusted_expected_budget / 25
st.plotly_chart(
    px.line(chart2, x='Year', y=['SimulatedTotalCost', 'BalancedBudget', 'ExpectedBudget'],
            labels={'value': 'Dollars', 'variable': 'Line'},
            title="Annual Budget Forecast vs. Targets")
)

st.subheader("Detailed Replacement Timeline with Filters")
selected_class_filter = st.multiselect("Filter by Classification", replace_summary['Classification'].unique())
selected_dept_filter = st.multiselect("Filter by Department", replace_summary['Department'].dropna().unique())

filtered_summary = replace_summary.copy()
if selected_class_filter:
    filtered_summary = filtered_summary[filtered_summary['Classification'].isin(selected_class_filter)]
if selected_dept_filter:
    filtered_summary = filtered_summary[filtered_summary['Department'].isin(selected_dept_filter)]

st.dataframe(filtered_summary[['ID', 'Year', 'SimulatedTotalCost', 'CumulativeBudget', 'Classification', 'Department', 'ReplacementCycle', 'ReplacementCondition']])

st.subheader("⚠️ KPI: Classifications Below Required Unit Count")
st.dataframe(deficit_df)

if not deficit_df.empty:
    deficit_chart = deficit_df.copy()
    deficit_chart["Units Short"] = deficit_chart["Required"] - deficit_chart["Available"]

    st.subheader("📉 Deficit Trend by Classification")
    chart_class = px.line(deficit_chart, x='Year', y='Units Short', color='Equipment Classification',
                          title="Deficit Units Per Year by Classification")
    st.plotly_chart(chart_class)

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Replacement Forecast')
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_export.xlsx">Download Excel File</a>'

st.markdown("### 🕅️ Export Forecast")
st.markdown(to_excel(filtered_summary), unsafe_allow_html=True)
