```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    # CLEAN DATA
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

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.fillna(0)

    # =====================================================
    # REMOVE INACTIVE CUSTOMERS
    # =====================================================

    df["TotalActivity"] = (

        df["ProductionRevenue"]

        + df["RawMaterialInventory"]

        + df["ReceivingTransaction"]

        + df["ShippingTransaction"]

    )

    df = df[
        df["TotalActivity"] > 0
    ]

    # =====================================================
    # SORT DATA
    # =====================================================

    df = df.sort_values(
        ["ProfitCenter", "Month"]
    )

    # =====================================================
    # CUSTOMER LIST
    # =====================================================

    customers = df["ProfitCenter"].unique()

    # =====================================================
    # FORECAST ENGINE
    # =====================================================

    forecast_rows = []

    future_months = [
        "Sep26",
        "Oct26",
        "Nov26",
        "Dec26",
        "Jan27",
        "Feb27",
        "Mar27",
        "Apr27",
        "May27",
        "Jun27",
        "Jul27",
        "Aug27"
    ]

    for customer in customers:

        temp = df[
            df["ProfitCenter"] == customer
        ].copy()

        # =============================================
        # SKIP EMPTY
        # =============================================

        if temp.empty:
            continue

        # =============================================
        # LAST VALUES
        # =============================================

        last_rev = (
            temp["ProductionRevenue"]
            .iloc[-1]
        )

        last_rm = (
            temp["RawMaterialInventory"]
            .iloc[-1]
        )

        # =============================================
        # GROWTH RATE
        # =============================================

        if len(temp) >= 2:

            rev_growth = (

                temp["ProductionRevenue"]
                .pct_change()
                .mean()

            )

            rm_growth = (

                temp["RawMaterialInventory"]
                .pct_change()
                .mean()

            )

        else:

            rev_growth = 0.03
            rm_growth = 0.03

        # =============================================
        # CLEAN GROWTH RATE
        # =============================================

        if np.isnan(rev_growth):
            rev_growth = 0.03

        if np.isnan(rm_growth):
            rm_growth = 0.03

        rev_growth = max(
            min(rev_growth, 0.3),
            -0.3
        )

        rm_growth = max(
            min(rm_growth, 0.3),
            -0.3
        )

        # =============================================
        # FORECAST FUTURE
        # =============================================

        for month in future_months:

            # =========================================
            # PRIMARY FORECAST
            # =========================================

            last_rev = (
                last_rev
                * (1 + rev_growth)
            )

            last_rm = (
                last_rm
                * (1 + rm_growth)
            )

            # =========================================
            # OPERATIONAL RATIOS
            # =========================================

            receiving = (
                last_rm * 0.002
            )

            shipping = (
                last_rev * 0.0015
            )

            transfer = (
                receiving * 1.2
            )

            fg_pallet = (
                last_rev / 100000
            )

            rm_pallet = (
                last_rm / 120000
            )

            no_of_bin = (
                last_rm / 50000
            )

            # =========================================
            # DERIVED METRICS
            # =========================================

            total_transaction = (

                receiving

                + shipping

                + transfer

            )

            no_of_pallet = (

                fg_pallet

                + rm_pallet

            )

            warehouse_capacity = (
                no_of_pallet * 1.2
            )

            warehouse_cost = (

                warehouse_capacity * 2

                + total_transaction * 0.5

            )

            # =========================================
            # SAVE FORECAST ROW
            # =========================================

            forecast_rows.append({

                "Month": month,

                "ProfitCenter": customer,

                "ProductionRevenue":
                    round(last_rev, 0),

                "RawMaterialInventory":
                    round(last_rm, 0),

                "ReceivingTransaction":
                    round(receiving, 0),

                "LocationTransferTransaction":
                    round(transfer, 0),

                "ShippingTransaction":
                    round(shipping, 0),

                "TotalTransaction":
                    round(total_transaction, 0),

                "FG_Pallet":
                    round(fg_pallet, 0),

                "RM_Pallet":
                    round(rm_pallet, 0),

                "NoOfPallet":
                    round(no_of_pallet, 0),

                "NoOfBin":
                    round(no_of_bin, 0),

                "WarehouseCapacity":
                    round(warehouse_capacity, 0),

                "WarehouseCost":
                    round(warehouse_cost, 0)

            })

    # =====================================================
    # FORECAST DATAFRAME
    # =====================================================

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    # =====================================================
    # CHECK FORECAST RESULT
    # =====================================================

    if forecast_df.empty:

        st.error(
            "No forecast data generated."
        )

        st.stop()

    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    st.subheader("📊 KPI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Warehouse Cost",
        f"{forecast_df['WarehouseCost'].sum():,.0f}"
    )

    col2.metric(
        "Average Warehouse Capacity",
        f"{forecast_df['WarehouseCapacity'].mean():,.0f}"
    )

    col3.metric(
        "Average Transaction",
        f"{forecast_df['TotalTransaction'].mean():,.0f}"
    )

    # =====================================================
    # PLANNING MATRIX
    # =====================================================

    st.subheader("📋 EMS Forecast Planning Matrix")

    categories = {

        "Production Revenue":
            "ProductionRevenue",

        "130000 - Raw Materials Inventory":
            "RawMaterialInventory",

        "Receiving transaction":
            "ReceivingTransaction",

        "Location transfer transaction":
            "LocationTransferTransaction",

        "Shipping transaction":
            "ShippingTransaction",

        "Total transaction":
            "TotalTransaction",

        "No. of Bin":
            "NoOfBin",

        "No. of pallet":
            "NoOfPallet",

        "FG Pallet":
            "FG_Pallet",

        "Raw Material Pallet":
            "RM_Pallet"

    }

    matrix_data = []

    for category, col_name in categories.items():

        for customer in customers:

            temp = forecast_df[
                forecast_df["ProfitCenter"]
                == customer
            ]

            if temp.empty:
                continue

            row = {

                "Categories": category,

                "ProfitCenter": customer

            }

            for _, r in temp.iterrows():

                row[r["Month"]] = (
                    round(r[col_name], 0)
                )

            matrix_data.append(row)

    matrix_df = pd.DataFrame(
        matrix_data
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    st.subheader("📈 Revenue Forecast")

    fig = px.line(
        forecast_df,
        x="Month",
        y="ProductionRevenue",
        color="ProfitCenter",
        title="Production Revenue Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # FORECAST TABLE
    # =====================================================

    st.subheader("📄 Forecast Dataset")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD
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
```
