
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HCMIC Forecasting System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📦 HCMIC EMS Forecasting & Warehouse Optimization")

st.write(
    """
    Enterprise EMS Forecasting Planning System
    """
)

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload EMS Forecasting Dataset",
    type=["xlsx"]
)

# =========================================================
# MAIN PROCESS
# =========================================================

if uploaded_file:

    # =====================================================
    # READ DATA
    # =====================================================

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Dataset")

    st.dataframe(df)

    # =====================================================
    # REQUIRED COLUMNS
    # =====================================================

    required_cols = [
        "Month",
        "ProfitCenter",
        "ProductionRevenue",
        "RawMaterialInventory",
        "ReceivingTransaction",
        "LocationTransferTransaction",
        "ShippingTransaction",
        "FG_Pallet",
        "RM_Pallet",
        "NoOfBin"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        st.error(f"Missing Columns: {missing_cols}")

        st.stop()

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

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

    df = df.sort_values(
        ["ProfitCenter", "Month"]
    )

    for col in numeric_cols:

        df[f"{col}_Lag1"] = (
            df.groupby("ProfitCenter")[col]
            .shift(1)
        )

        df[f"{col}_MA3"] = (
            df.groupby("ProfitCenter")[col]
            .transform(
                lambda x: x.rolling(3).mean()
            )
        )

    df = df.dropna()

    st.subheader("⚙️ Feature Engineering")

    st.dataframe(df)

    # =====================================================
    # FORECASTING FUNCTION
    # =====================================================

    def build_forecast(
        target,
        features
    ):

        result = []

        customers = df["ProfitCenter"].unique()

        for customer in customers:

            temp = df[
                df["ProfitCenter"] == customer
            ].copy()

            if len(temp) < 2:
                continue

            X = temp[features]

            y = temp[target]

            model = LinearRegression()

            model.fit(X, y)

            temp[f"Forecast_{target}"] = (
                model.predict(X)
            )

            result.append(temp)

        if len(result) > 0:

            return pd.concat(result)

        else:

            return pd.DataFrame()

    # =====================================================
    # PRIMARY FORECASTS
    # =====================================================

    df_rev = build_forecast(
        target="ProductionRevenue",
        features=[
            "ProductionRevenue_Lag1",
            "ProductionRevenue_MA3"
        ]
    )

    df_rm = build_forecast(
        target="RawMaterialInventory",
        features=[
            "RawMaterialInventory_Lag1",
            "RawMaterialInventory_MA3"
        ]
    )

    # =====================================================
    # MERGE PRIMARY FORECAST
    # =====================================================

    df["Forecast_ProductionRevenue"] = (
        df_rev["Forecast_ProductionRevenue"]
    )

    df["Forecast_RawMaterialInventory"] = (
        df_rm["Forecast_RawMaterialInventory"]
    )

    # =====================================================
    # SECONDARY FORECASTS
    # =====================================================

    # Receiving Transaction

    model_receiving = LinearRegression()

    X_receiving = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_receiving = df[
        "ReceivingTransaction"
    ]

    model_receiving.fit(
        X_receiving,
        y_receiving
    )

    df["Forecast_ReceivingTransaction"] = (
        model_receiving.predict(
            X_receiving
        )
    )

    # Shipping Transaction

    model_shipping = LinearRegression()

    X_shipping = df[[
        "Forecast_ProductionRevenue"
    ]]

    y_shipping = df[
        "ShippingTransaction"
    ]

    model_shipping.fit(
        X_shipping,
        y_shipping
    )

    df["Forecast_ShippingTransaction"] = (
        model_shipping.predict(
            X_shipping
        )
    )

    # Location Transfer

    model_transfer = LinearRegression()

    X_transfer = df[[
        "Forecast_ProductionRevenue",
        "Forecast_ReceivingTransaction"
    ]]

    y_transfer = df[
        "LocationTransferTransaction"
    ]

    model_transfer.fit(
        X_transfer,
        y_transfer
    )

    df["Forecast_LocationTransferTransaction"] = (
        model_transfer.predict(
            X_transfer
        )
    )

    # FG Pallet

    model_fg = LinearRegression()

    X_fg = df[[
        "Forecast_ProductionRevenue",
        "Forecast_ReceivingTransaction"
    ]]

    y_fg = df["FG_Pallet"]

    model_fg.fit(X_fg, y_fg)

    df["Forecast_FG_Pallet"] = (
        model_fg.predict(X_fg)
    )

    # RM Pallet

    model_rm_pallet = LinearRegression()

    X_rm_pallet = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_rm_pallet = df["RM_Pallet"]

    model_rm_pallet.fit(
        X_rm_pallet,
        y_rm_pallet
    )

    df["Forecast_RM_Pallet"] = (
        model_rm_pallet.predict(
            X_rm_pallet
        )
    )

    # Bin

    model_bin = LinearRegression()

    X_bin = df[[
        "Forecast_RawMaterialInventory"
    ]]

    y_bin = df["NoOfBin"]

    model_bin.fit(X_bin, y_bin)

    df["Forecast_NoOfBin"] = (
        model_bin.predict(X_bin)
    )

    # =====================================================
    # DERIVED METRICS
    # =====================================================

    df["Forecast_TotalTransaction"] = (

        df["Forecast_ReceivingTransaction"]

        + df["Forecast_ShippingTransaction"]

        + df["Forecast_LocationTransferTransaction"]

    )

    df["Forecast_NoOfPallet"] = (

        df["Forecast_FG_Pallet"]

        + df["Forecast_RM_Pallet"]

    )

    # =====================================================
    # WAREHOUSE CAPACITY
    # =====================================================

    df["WarehouseCapacity"] = (

        df["Forecast_NoOfPallet"]

        * 1.2

    )

    # =====================================================
    # COST FUNCTION
    # =====================================================

    holding_cost = 2
    warehouse_cost = 1
    transaction_cost = 0.5

    df["WarehouseCost"] = (

        df["Forecast_NoOfPallet"]

        * holding_cost

        + df["WarehouseCapacity"]

        * warehouse_cost

        + df["Forecast_TotalTransaction"]

        * transaction_cost

    )

    # =====================================================
    # KPI SECTION
    # =====================================================

    st.subheader("📊 KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Warehouse Cost",
        f"{df['WarehouseCost'].sum():,.0f}"
    )

    col2.metric(
        "Average Warehouse Capacity",
        f"{df['WarehouseCapacity'].mean():,.0f}"
    )

    col3.metric(
        "Average Transaction",
        f"{df['Forecast_TotalTransaction'].mean():,.0f}"
    )

    # =====================================================
    # PLANNING MATRIX
    # =====================================================

    st.subheader("📋 EMS Forecast Planning Matrix")

    matrix_data = []

    categories = {

        "Production Revenue":
            "Forecast_ProductionRevenue",

        "130000 - Raw Materials Inventory":
            "Forecast_RawMaterialInventory",

        "Receiving transaction":
            "Forecast_ReceivingTransaction",

        "Location transfer transaction":
            "Forecast_LocationTransferTransaction",

        "Shipping transaction":
            "Forecast_ShippingTransaction",

        "Total transaction":
            "Forecast_TotalTransaction",

        "No. of Bin":
            "Forecast_NoOfBin",

        "No. of pallet":
            "Forecast_NoOfPallet",

        "FG Pallet":
            "Forecast_FG_Pallet",

        "Raw Material Pallet":
            "Forecast_RM_Pallet"

    }

    for category, forecast_col in categories.items():

        for customer in df["ProfitCenter"].unique():

            temp = df[
                df["ProfitCenter"] == customer
            ]

            row = {
                "Categories": category,
                "ProfitCenter": customer
            }

            for _, r in temp.iterrows():

                row[r["Month"]] = round(
                    r[forecast_col],
                    0
                )

            matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data)

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    st.subheader("📈 Forecast Visualization")

    fig = px.line(
        df,
        x="Month",
        y="Forecast_ProductionRevenue",
        color="ProfitCenter",
        title="Production Revenue Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # EXPORT
    # =====================================================

    st.subheader("📥 Download Forecast Matrix")

    csv = matrix_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Forecast Matrix CSV",
        data=csv,
        file_name="EMS_Forecast_Matrix.csv",
        mime="text/csv"
    )
