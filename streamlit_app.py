import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="HCMIC Forecasting",
    layout="wide"
)

# ======================================================
# TITLE
# ======================================================

st.title("📦 HCMIC Warehouse Forecasting & Optimization")

st.write(
    "Data-driven EMS warehouse forecasting and optimization system"
)

# ======================================================
# FILE UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "Upload EMS Excel File",
    type=["xlsx"]
)

# ======================================================
# MAIN PROCESS
# ======================================================

if uploaded_file:

    # ==================================================
    # READ FILE
    # ==================================================

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    numeric_cols = [
        "ProductionRevenue",
        "RawMaterialInventory",
        "ReceivingTransaction",
        "LocationTransferTransaction",
        "ShippingTransaction",
        "FG_Pallet",
        "RM_Pallet",
        "NoOfBin"
    ]

    for col in numeric_cols:
        df[f"{col}_MA3"] = df[col].rolling(3).mean()
        df[f"{col}_Lag1"] = df[col].shift(1)

    df = df.dropna()

    st.subheader("⚙️ Feature Engineering")
    st.dataframe(df)

    # ==================================================
    # FORECASTING MODELS
    # ==================================================

    st.subheader("📈 Forecasting Results")

    # ==========================================
    # 1. Forecast Production Revenue
    # ==========================================

    model_rev = LinearRegression()

    X_rev = df[[
        "ProductionRevenue_Lag1",
        "ProductionRevenue_MA3"
    ]]

    y_rev = df["ProductionRevenue"]

    model_rev.fit(X_rev, y_rev)

    df["Forecast_ProductionRevenue"] = model_rev.predict(X_rev)

    # ==========================================
    # 2. Forecast Raw Material Inventory
    # ==========================================

    model_rm = LinearRegression()

    X_rm = df[[
        "RawMaterialInventory_Lag1",
        "RawMaterialInventory_MA3"
    ]]

    y_rm = df["RawMaterialInventory"]

    model_rm.fit(X_rm, y_rm)

    df["Forecast_RawMaterialInventory"] = model_rm.predict(X_rm)

    # ==========================================
    # 3. Forecast Receiving Transaction
    # ==========================================

    model_receiving = LinearRegression()

    X_receiving = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_receiving = df["ReceivingTransaction"]

    model_receiving.fit(X_receiving, y_receiving)

    df["Forecast_ReceivingTransaction"] = model_receiving.predict(X_receiving)

    # ==========================================
    # 4. Forecast Shipping Transaction
    # ==========================================

    model_shipping = LinearRegression()

    X_shipping = df[[
        "Forecast_ProductionRevenue"
    ]]

    y_shipping = df["ShippingTransaction"]

    model_shipping.fit(X_shipping, y_shipping)

    df["Forecast_ShippingTransaction"] = model_shipping.predict(X_shipping)

    # ==========================================
    # 5. Forecast Location Transfer
    # ==========================================

    model_transfer = LinearRegression()

    X_transfer = df[[
        "Forecast_ProductionRevenue",
        "Forecast_ReceivingTransaction"
    ]]

    y_transfer = df["LocationTransferTransaction"]

    model_transfer.fit(X_transfer, y_transfer)

    df["Forecast_LocationTransfer"] = model_transfer.predict(X_transfer)

    # ==========================================
    # 6. Forecast FG Pallet
    # ==========================================

    model_fg = LinearRegression()

    X_fg = df[[
        "Forecast_ProductionRevenue",
        "Forecast_ReceivingTransaction"
    ]]

    y_fg = df["FG_Pallet"]

    model_fg.fit(X_fg, y_fg)

    df["Forecast_FG_Pallet"] = model_fg.predict(X_fg)

    # ==========================================
    # 7. Forecast RM Pallet
    # ==========================================

    model_rm_pallet = LinearRegression()

    X_rm_pallet = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_rm_pallet = df["RM_Pallet"]

    model_rm_pallet.fit(X_rm_pallet, y_rm_pallet)

    df["Forecast_RM_Pallet"] = model_rm_pallet.predict(X_rm_pallet)

    # ==========================================
    # 8. Forecast No Of Bin
    # ==========================================

    model_bin = LinearRegression()

    X_bin = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_bin = df["NoOfBin"]

    model_bin.fit(X_bin, y_bin)

    df["Forecast_NoOfBin"] = model_bin.predict(X_bin)

    # ==================================================
    # DERIVED METRICS
    # ==================================================

    df["Forecast_TotalTransaction"] = (
        df["Forecast_ReceivingTransaction"]
        + df["Forecast_ShippingTransaction"]
        + df["Forecast_LocationTransfer"]
    )

    df["Forecast_TotalPallet"] = (
        df["Forecast_FG_Pallet"]
        + df["Forecast_RM_Pallet"]
    )

    # ==================================================
    # WAREHOUSE CAPACITY
    # ==================================================

    df["WarehouseCapacity"] = (
        df["Forecast_TotalPallet"] * 1.2
    )

    # ==================================================
    # COST FUNCTION
    # ==================================================

    holding_cost = 2
    warehouse_cost = 1
    transaction_cost = 0.5

    df["WarehouseCost"] = (
        df["Forecast_TotalPallet"] * holding_cost
        + df["WarehouseCapacity"] * warehouse_cost
        + df["Forecast_TotalTransaction"] * transaction_cost
    )

    total_cost = df["WarehouseCost"].sum()

    # ==================================================
    # KPI SECTION
    # ==================================================

    st.subheader("📊 KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Warehouse Cost",
        f"{total_cost:,.2f}"
    )

    col2.metric(
        "Average Warehouse Capacity",
        f"{df['WarehouseCapacity'].mean():,.2f}"
    )

    col3.metric(
        "Average Total Transaction",
        f"{df['Forecast_TotalTransaction'].mean():,.2f}"
    )

    # ==================================================
    # CHARTS
    # ==================================================

    st.subheader("📈 Forecast Visualization")

    fig = px.line(
        df,
        x="Month",
        y=[
            "ProductionRevenue",
            "Forecast_ProductionRevenue"
        ],
        title="Production Revenue Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="Month",
        y=[
            "Forecast_TotalTransaction"
        ],
        title="Forecast Total Transaction"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        df,
        x="Month",
        y=[
            "WarehouseCapacity"
        ],
        title="Warehouse Capacity"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ==================================================
    # FINAL OUTPUT
    # ==================================================

    st.subheader("✅ Final Forecast Output")

    st.dataframe(df)

