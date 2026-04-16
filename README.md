# Epidemic-Prediction-Using-Environmental-Factors

A machine learning-based web application to predict dengue cases using environmental factors such as rainfall and humidity.

## Live Demo
 https://epidemic-prediction-using-environmental-factors-3wtnug6t8xzqrj.streamlit.app/

## Features

- User input for rainfall and humidity
- Machine learning prediction (Linear Regression)
- Interactive visualizations (scatter, trend, heatmap)
- Risk level indicator (High / Moderate)
- Prediction history tracking
- Download report (CSV)

## Model Details

- Algorithm: Linear Regression
- Features: Rainfall, Humidity
- Target: Dengue Cases
- Performance: R² Score ≈ 0.99

## Dataset

Sample dataset used:
- Rainfall (mm)
- Humidity (%)
- Dengue Cases

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run Locally

```bash
pip install -r requirements.txt
```
streamlit run app.py
