import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dengue Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}
h1, h2 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

data = pd.DataFrame({
    'Rainfall': [235, 158, 61, 55],
    'Humidity': [73, 70, 76, 80]
})

data['Dengue_Cases'] = (
    1000 * data['Rainfall'] +
    500 * data['Humidity']
)

X = data[['Rainfall', 'Humidity']]
y = data['Dengue_Cases']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

st.title("🌍 Dengue Prediction Dashboard")
st.caption("Predict dengue risk using rainfall and humidity")

st.sidebar.header("📥 Input Parameters")

rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 120.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)

predict = st.sidebar.button("🔍 Predict")

if predict:

    user_data = pd.DataFrame([[rainfall, humidity]],
                             columns=['Rainfall', 'Humidity'])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]
    st.session_state.history.append({
        "Rainfall": rainfall,
        "Humidity": humidity,
        "Predicted_Cases": int(prediction)
    })

    col1, col2, col3 = st.columns(3)

    col1.metric("🌧 Rainfall", f"{rainfall:.1f} mm")
    col2.metric("💧 Humidity", f"{humidity:.1f}%")
    col3.metric("🦟 Dengue Cases", f"{int(prediction)}")

    st.markdown("---")

    if prediction > data['Dengue_Cases'].mean():
        st.error("🔴 High Risk")
    else:
        st.success("🟢 Low to Moderate Risk")

    st.markdown("---")

    st.subheader("📊 Rainfall vs Dengue Cases")

    fig, ax = plt.subplots()
    ax.scatter(data['Rainfall'], data['Dengue_Cases'], label='Data')
    ax.scatter(rainfall, prediction, color='red', s=120, label='Your Input')
    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Dengue Cases")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("🔥 Correlation Heatmap")

    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)

    st.pyplot(fig2)

    st.markdown("---")

    st.subheader("📈 Rainfall Trend")

    fig3, ax3 = plt.subplots()
    ax3.plot(data['Rainfall'], marker='o')
    ax3.set_ylabel("Rainfall")
    ax3.grid()

    st.pyplot(fig3)

    st.markdown("---")

    st.subheader("📊 Insights")

    if rainfall > 150:
        st.write("• High rainfall increases mosquito breeding.")
    elif rainfall < 70:
        st.write("• Low rainfall reduces dengue risk.")
    else:
        st.write("• Moderate rainfall shows moderate risk.")

    if humidity > 75:
        st.write("• High humidity supports mosquito survival.")
    else:
        st.write("• Humidity impact is moderate.")

    st.write("• Combined effect influences dengue outbreaks.")

    st.markdown("---")

    st.subheader("📥 Download Report")

    history_df = pd.DataFrame(st.session_state.history)

    if not history_df.empty:
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Report",
            data=csv,
            file_name="dengue_predictions.csv",
            mime="text/csv"
        )

    st.subheader("📈 Prediction History")

    if len(st.session_state.history) > 1:
        fig4, ax4 = plt.subplots()
        ax4.plot(history_df['Predicted_Cases'], marker='o')
        ax4.set_xlabel("Prediction Index")
        ax4.set_ylabel("Dengue Cases")
        ax4.set_title("Prediction Trend")
        ax4.grid()

        st.pyplot(fig4)
    else:
        st.write("Make multiple predictions to see trend.")