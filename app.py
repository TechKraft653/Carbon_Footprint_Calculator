import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌱 ML Carbon Footprint Calculator", layout="centered")

# Load model and feature columns
model = joblib.load('carbon_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Model metrics from your actual evaluation
MODEL_METRICS = {
    "MAE": 303.77,
    "MSE": 143049.31,
    "R²": 0.90
}

st.markdown(
    "<h1 style='text-align: center; color: #388e3c;'>🌍 Carbon Footprint Calculator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Get a personalized carbon footprint estimate using machine learning.<br>"
    "Fill out the form below and see your results instantly."
    "</div><br>", unsafe_allow_html=True
)

with st.form("input_form"):
    st.markdown("### 👤 Personal & Lifestyle Info")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        diet = st.selectbox("Diet", ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"])
    with col2:
        recycling = st.selectbox("Do you recycle?", ["Yes", "No"])
        vehicle_type = st.selectbox("Vehicle Type", ["Small petrol", "Medium petrol", "Diesel", "Hybrid/Electric", "None"])

    st.markdown("### 🏠 Home & Energy Use")
    col3, col4 = st.columns(2)
    with col3:
        heating = st.selectbox("Heating Energy Source", ["Electricity", "Natural Gas", "Oil", "None"])
        electricity = st.number_input("Annual electricity use (kWh)", min_value=0, value=2000, step=10)
    with col4:
        waste = st.number_input("Annual waste generated (kg)", min_value=0, value=400, step=10)

    st.markdown("### 🚗 Transportation")
    col5, col6 = st.columns(2)
    with col5:
        transport = st.selectbox("Primary Transport", ["Car", "Public Transit", "Bike/Walk"])
    with col6:
        km_driven = st.number_input("Annual kilometers driven", min_value=0, value=5000, step=100)

    submitted = st.form_submit_button("🌱 Calculate My Carbon Footprint")

if submitted:
    # Build input dictionary for all features
    input_dict = {
        'Sex_' + sex: 1,
        'Diet_' + diet: 1,
        'Heating Energy Source_' + heating: 1,
        'Transport_' + transport: 1,
        'Vehicle Type_' + vehicle_type: 1,
        'Recycling_' + recycling: 1,
        'Annual kilometers driven': km_driven,
        'Annual electricity use (kWh)': electricity,
        'Annual waste generated (kg)': waste
    }
    # Ensure all required columns are present, fill missing with 0
    input_data = {col: input_dict.get(col, 0) for col in feature_columns}
    input_df = pd.DataFrame([input_data])

    # Ensure no NaNs and correct data type
    input_df = input_df.fillna(0)
    input_df = input_df.astype(float)

    # Predict
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.markdown(
        f"<h2 style='color:#388e3c;'>🌱 Your Predicted Annual Carbon Footprint:</h2>"
        f"<h1 style='color:#1976d2;'>{prediction:,.2f} kg CO₂e</h1>",
        unsafe_allow_html=True
    )

    # Progress bar relative to a high global footprint (20,000 kg CO₂e)
    st.progress(min(int(prediction/20000*100), 100), text="Relative to high global footprints (20,000 kg CO₂e/year)")

    # --- Analysis Section ---
    st.markdown("## 📈 Footprint Analysis")

    # 1. Band Classification
    if prediction < 4000:
        band = "Low"
        color = "green"
    elif prediction < 8000:
        band = "Average"
        color = "orange"
    else:
        band = "High"
        color = "red"
    st.markdown(f"**Your footprint is classified as:** <span style='color:{color}; font-weight:bold;'>{band}</span>", unsafe_allow_html=True)

    # 2. Comparison Table
    st.markdown("### 🌏 How You Compare")
    comparison = pd.DataFrame({
        "Category": ["Your Footprint", "Global Average", "Developed Country Average"],
        "Value (kg CO₂e/year)": [prediction, 4000, 12000]
    })
    st.table(comparison)

    # 3. Personalized Insights
    st.markdown("### 🔍 Insights")
    insights = []
    if electricity > 3500:
        insights.append("Your electricity use is above average. Consider energy-efficient appliances or renewable sources.")
    if km_driven > 8000:
        insights.append("Your car travel is high. Try carpooling, public transport, or cycling more often.")
    if diet == "Omnivore":
        insights.append("A plant-rich diet can significantly reduce your carbon footprint.")
    if recycling == "No":
        insights.append("Recycling can lower your waste emissions by up to 10%.")
    if not insights:
        insights.append("Great job! Your habits are already climate-friendly. Keep it up!")
    for tip in insights:
        st.write(f"- {tip}")

    # 4. Category-wise Emissions Breakdown (Heuristic Example)
    home_energy = electricity * 0.5 + (200 if heating == "Electricity" else 100 if heating == "Natural Gas" else 150 if heating == "Oil" else 0)
    transport_emission = km_driven * (0.18 if vehicle_type == "Medium petrol" else 0.12 if vehicle_type == "Small petrol" else 0.16 if vehicle_type == "Diesel" else 0.05 if vehicle_type == "Hybrid/Electric" else 0)
    food_emission = 2500 if diet == "Omnivore" else 1700 if diet == "Vegetarian" else 1500 if diet == "Vegan" else 2000
    waste_emission = waste * (0.6 if recycling == "No" else 0.54)  # 10% reduction if recycling

    breakdown_dict = {
        "Home Energy": home_energy,
        "Transportation": transport_emission,
        "Food & Diet": food_emission,
        "Waste": waste_emission
    }
    breakdown_df = pd.DataFrame(list(breakdown_dict.items()), columns=["Category", "Emissions (kg CO₂e)"]).set_index("Category")

    st.markdown("### 📊 Category-wise Emissions Breakdown")
    st.bar_chart(breakdown_df)

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie(breakdown_dict.values(), labels=breakdown_dict.keys(), autopct='%1.1f%%', startangle=90, colors=['#81c784', '#64b5f6', '#ffd54f', '#ff8a65'])
    ax.axis('equal')
    st.pyplot(fig)

    # 5. Actionable Recommendations
    st.markdown("### 💡 Tips to Reduce Your Footprint")
    st.success("- Switch to renewable energy sources\n"
               "- Reduce car travel and use public transport or cycling\n"
               "- Shift towards a more plant-based diet\n"
               "- Recycle and reduce waste")

    # 6. Model Accuracy Section
    st.markdown("---")
    st.markdown("## 📋 Model Accuracy")
    st.markdown(
        "<ul>"
        f"<li><b>MAE</b> (Mean Absolute Error): <span style='color:#1976d2;'>{MODEL_METRICS['MAE']}</span></li>"
        f"<li><b>MSE</b> (Mean Squared Error): <span style='color:#1976d2;'>{MODEL_METRICS['MSE']}</span></li>"
        f"<li><b>R²</b> (Coefficient of Determination): <span style='color:#388e3c;'>{MODEL_METRICS['R²']}</span></li>"
        "</ul>",
        unsafe_allow_html=True
    )
    st.info(
        "• R² = 0.90 means the model explains 90% of the variance in carbon emissions.\n"
        "• MAE ≈ 304 kg CO₂e means predictions are on average within 304 units of the true value."
    )

    st.caption("Emission estimates are approximate and may vary by region. Powered by your trained ML model.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "© 2025 Carbon Footprint Calculator. All rights reserved.<br>"
    "</div>", unsafe_allow_html=True
)
