import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "tuned_random_forest.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")

st.set_page_config(
    page_title="AI Industrial Maintenance Copilot",
    page_icon="🔧",
    layout="wide"
)


st.title("🔧 AI Industrial Maintenance Copilot")

st.write(
    "An AI-powered predictive maintenance system that estimates "
    "Remaining Useful Life (RUL) and assesses machine condition."
)

st.success("✅ Tuned Random Forest model loaded successfully.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Tuned Random Forest")

with col2:
    st.metric("Input Features", len(feature_names))

with col3:
    st.metric("Model R²", "0.5142")


st.header("📊 Machine Sensor Data")

st.write(
    "Enter the current machine sensor measurements and derived "
    "features used by the trained model."
)

default_values = {
    "sensor_14": 8129.95,
    "sensor_14_rolling_mean": 8129.95,
    "sensor_14_rolling_std": 0.0,
    "sensor_14_trend": 0.0,

    "sensor_13": 2388.15,
    "sensor_13_rolling_mean": 2388.15,
    "sensor_13_rolling_std": 0.0,
    "sensor_13_trend": 0.0,

    "sensor_15": 8.4689,
    "sensor_15_rolling_mean": 8.4689,
    "sensor_15_rolling_std": 0.0,
    "sensor_15_trend": 0.0,

    "sensor_11": 47.66,
    "sensor_11_rolling_mean": 47.66,
    "sensor_11_rolling_std": 0.0,
    "sensor_11_trend": 0.0,

    "sensor_4": 1416.15,
    "sensor_4_rolling_mean": 1416.15,
    "sensor_4_rolling_std": 0.0,
    "sensor_4_trend": 0.0
}


input_values = {}

st.subheader("Current Sensor Measurements")

sensor_names = [
    "sensor_14",
    "sensor_13",
    "sensor_15",
    "sensor_11",
    "sensor_4"
]

columns = st.columns(2)

for i, sensor in enumerate(sensor_names):

    with columns[i % 2]:

        input_values[sensor] = st.number_input(
            sensor,
            value=float(default_values.get(sensor, 0.0)),
            format="%.4f",
            key=f"input_{sensor}"
        )


with st.expander("⚙️ Advanced / Derived Features", expanded=False):

    st.write(
        "These features were created during feature engineering "
        "using rolling statistics and degradation trends."
    )

    derived_features = [
        feature for feature in feature_names
        if feature not in sensor_names
    ]

    columns = st.columns(2)

    for i, feature in enumerate(derived_features):

        with columns[i % 2]:

            input_values[feature] = st.number_input(
                feature,
                value=float(default_values.get(feature, 0.0)),
                format="%.4f",
                key=f"input_{feature}"
            )



st.divider()

if st.button(
    "🔮 Predict Remaining Useful Life",
    type="primary",
    use_container_width=True
):

    input_df = pd.DataFrame([input_values])

    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    predicted_rul = model.predict(input_scaled)[0]

    predicted_rul = max(0, predicted_rul)


    st.header("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Remaining Useful Life",
            f"{predicted_rul:.2f} cycles"
        )

    with result_col2:

        if predicted_rul <= 30:
            status = "CRITICAL"

        elif predicted_rul <= 75:
            status = "WARNING"

        elif predicted_rul <= 150:
            status = "MONITOR"

        else:
            status = "NORMAL"

        st.metric(
            "Machine Status",
            status
        )

    st.subheader("🛠️ Maintenance Assessment")

    if status == "CRITICAL":

        st.error(
            "🔴 CRITICAL: Immediate maintenance is recommended. "
            "Inspect the machine and consider taking it out of service."
        )

        recommendation = (
            "Immediate maintenance required. Inspect the machine "
            "and consider taking it out of service."
        )

    elif status == "WARNING":

        st.warning(
            "🟠 WARNING: Maintenance should be scheduled soon."
        )

        recommendation = (
            "Schedule maintenance soon. Inspect important sensor "
            "signals and plan corrective action."
        )

    elif status == "MONITOR":

        st.info(
            "🟡 MONITOR: The machine should be monitored more frequently."
        )

        recommendation = (
            "Increase monitoring frequency and inspect sensor trends "
            "for signs of degradation."
        )

    else:

        st.success(
            "🟢 NORMAL: Machine condition appears normal."
        )

        recommendation = (
            "Continue routine monitoring and scheduled maintenance."
        )


    st.write("### Recommendation")

    st.write(recommendation)


    st.subheader("🔍 Key Factors Influencing Prediction")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    top_features = importance_df.head(5)

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        top_features.set_index("Feature")["Importance"]
    )


    st.subheader("💡 Model Interpretation")

    top_feature = top_features.iloc[0]["Feature"]

    st.write(
        f"The most influential feature in this prediction is "
        f"**{top_feature}**. The model uses the combined sensor "
        f"measurements, rolling statistics, and degradation trends "
        f"to estimate the machine's remaining useful life."
    )

    st.caption(
        "Note: RUL prediction is an estimate produced by the trained "
        "machine-learning model and should support, not replace, "
        "engineering inspection and maintenance decisions."
    )