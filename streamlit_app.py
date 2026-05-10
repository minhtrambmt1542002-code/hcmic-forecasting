import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="HCMIC Forecasting", layout="wide")

st.title("📦 HCMIC Warehouse Forecasting & Optimization")

st.write("Upload your EMS warehouse data")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file:

    # =========================
    # READ DATA
    # =========================
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df["MA3"] = df["Demand"].rolling(3).mean()
    df["Lag1"] = df["Demand"].shift(1)

    df = df.dropna()

    st.subheader("⚙️ Feature Engineering")
    st.dataframe(df)

    # =========================
    # FORECAST MODEL
    # =========================
    X = df[["Revenue", "RawMat", "Lag1", "MA3"]]
    y = df["Demand"]

    model = LinearRegression()
    model.fit(X, y)

    df["Forecast"] = model.predict(X)

    st.subheader("📈 Forecast Result")
    st.dataframe(df[["Demand", "Forecast"]])

    # =========================
    # INVENTORY MODEL
    # =========================
    df["Inventory"] = df["Forecast"] * 1.1

    # =========================
    # WAREHOUSE SIZE
    # =========================
    df["WarehouseSize"] = df["Inventory"] * 1.2

    st.subheader("🏢 Warehouse Capacity")
    st.dataframe(df[["Inventory", "WarehouseSize"]])

    # =========================
    # COST FUNCTION
    # =========================
    holding_cost = 2
    warehouse_cost = 1

    df["Cost"] = (
        df["Inventory"] * holding_cost
        + df["WarehouseSize"] * warehouse_cost
    )

    total_cost = df["Cost"].sum()

    st.subheader("💰 Total Cost")
    st.metric("Total Cost", f"{total_cost:,.2f}")

    # =========================
    # FINAL OUTPUT
    # =========================
    st.subheader("✅ Final Output")
    st.dataframe(df)
