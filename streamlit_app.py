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

        if temp.empty:
            continue

        # =================================================
        # MOVING AVERAGE FORECAST
        # =================================================

        forecast_rev = (
            temp["ProductionRevenue"]
            .tail(3)
            .mean()
        )

        forecast_rm = (
            temp["RawMaterialInventory"]
            .tail(3)
            .mean()
        )

        # =================================================
        # STABILIZE IF NAN
        # =================================================

        if np.isnan(forecast_rev):
            forecast_rev = 0

        if np.isnan(forecast_rm):
            forecast_rm = 0

        # =================================================
        # OPERATIONAL RATIOS
        # =================================================

        avg_receiving_ratio = (

            temp["ReceivingTransaction"].sum()

            /

            max(
                temp["RawMaterialInventory"].sum(),
                1
            )

        )

        avg_shipping_ratio = (

            temp["ShippingTransaction"].sum()

            /

            max(
                temp["ProductionRevenue"].sum(),
                1
            )

        )

        avg_transfer_ratio = (

            temp["LocationTransferTransaction"].sum()

            /

            max(
                temp["ReceivingTransaction"].sum(),
                1
            )

        )

        avg_fg_ratio = (

            temp["FG_Pallet"].sum()

            /

            max(
                temp["ProductionRevenue"].sum(),
                1
            )

        )

        avg_rm_ratio = (

            temp["RM_Pallet"].sum()

            /

            max(
                temp["RawMaterialInventory"].sum(),
                1
            )

        )

        avg_bin_ratio = (

            temp["NoOfBin"].sum()

            /

            max(
                temp["RawMaterialInventory"].sum(),
                1
            )

        )

        # =================================================
        # FORECAST FUTURE
        # =================================================

        for month in future_months:

            # =============================================
            # MAIN FORECAST
            # =============================================

            production_revenue = forecast_rev

            raw_material = forecast_rm

            # =============================================
            # OPERATIONAL FORECAST
            # =============================================

            receiving = (

                raw_material

                * avg_receiving_ratio

            )

            shipping = (

                production_revenue

                * avg_shipping_ratio

            )

            transfer = (

                receiving

                * avg_transfer_ratio

            )

            fg_pallet = (

                production_revenue

                * avg_fg_ratio

            )

            rm_pallet = (

                raw_material

                * avg_rm_ratio

            )

            no_of_bin = (

                raw_material

                * avg_bin_ratio

            )

            # =============================================
            # DERIVED METRICS
            # =============================================

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

            # =============================================
            # SAVE FORECAST
            # =============================================

            forecast_rows.append({

                "Month": month,

                "ProfitCenter": customer,

                "ProductionRevenue":
                    round(production_revenue, 0),

                "RawMaterialInventory":
                    round(raw_material, 0),

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
    # CHECK FORECAST
    # =====================================================

    if forecast_df.empty:

        st.error(
            "No forecast generated."
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

    # =====================================================
    # DISPLAY MATRIX
    # =====================================================

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    # =====================================================
    # CHART
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
    # FORECAST DATA
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
