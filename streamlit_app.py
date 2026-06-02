import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# OPTIONAL ADVANCED FORECAST
# =========================================================

try:

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATS_AVAILABLE = True

except:

    STATS_AVAILABLE = False

# =====================================================
# IMPORT VALIDATION METRICS
# =====================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# WAREHOUSE FLEXIBILITY PARAMETERS
# =====================================================

alpha = 0.20
beta = 0.10

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

    with st.spinner("Generating Forecast..."):

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
        # MONTH CONVERSION
        # =====================================================

        try:

            df["MonthDate"] = pd.to_datetime(
                df["Month"],
                format="%b'%y"
            )

        except:

            st.error(
                "Month format invalid. Use format like Mar'26"
            )

            st.stop()

        # =====================================================
        # REMOVE INACTIVE
        # =====================================================

        df["TotalActivity"] = (

            df["RawMaterialInventory"]

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
            ["ProfitCenter", "MonthDate"]
        )

        # =====================================================
        # FEATURE ENGINEERING
        # =====================================================

        df["RM_Lag1"] = (
            df.groupby("ProfitCenter")
            ["RawMaterialInventory"]
            .shift(1)
        )

        df["RM_Lag2"] = (
            df.groupby("ProfitCenter")
            ["RawMaterialInventory"]
            .shift(2)
        )

        df["RM_MA3"] = (
            df.groupby("ProfitCenter")
            ["RawMaterialInventory"]
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1
                ).mean()
            )
        )

        # =====================================================
        # CUSTOMER LIST
        # =====================================================

        customers = df["ProfitCenter"].unique()

        # =====================================================
        # FEATURE ENGINEERING DISPLAY
        # =====================================================

        st.subheader("⚙️ Feature Engineering")

        for customer in customers:

            st.markdown(f"## ProfitCenter: {customer}")

            feature_df = df[
                df["ProfitCenter"] == customer
            ]

            st.dataframe(
                feature_df,
                use_container_width=True
            )

        # =====================================================
        # CUSTOMER ANALYTICS
        # =====================================================

        total_customers = (
            df["ProfitCenter"]
            .nunique()
        )

        active_customers = (
            df[
                df["RawMaterialInventory"] > 0
            ]["ProfitCenter"]
            .nunique()
        )

        inactive_customers = (
            total_customers
            - active_customers
        )

        # =====================================================
        # CUSTOMER SEGMENTATION
        # =====================================================

        customer_summary = (

            df.groupby("ProfitCenter")

            .agg({

                "RawMaterialInventory": "mean",

                "ReceivingTransaction": "mean",

                "NoOfBin": "mean"

            })

            .reset_index()

        )

        customer_summary["DemandSegment"] = np.where(

            customer_summary["RawMaterialInventory"]

            >= customer_summary["RawMaterialInventory"].quantile(0.75),

            "High Inventory",

            np.where(

                customer_summary["RawMaterialInventory"]

                >= customer_summary["RawMaterialInventory"].quantile(0.40),

                "Medium Inventory",

                "Low Inventory"

            )

        )

        # =====================================================
        # VARIANCE ANALYSIS
        # =====================================================

        customer_variance = (

            df.groupby("ProfitCenter")

            .agg({

                "RawMaterialInventory": [
                    "mean",
                    "std"
                ]

            })

        )

        customer_variance.columns = [
            "RM_Mean",
            "RM_STD"
        ]

        customer_variance = (
            customer_variance
            .reset_index()
        )

        # =====================================================
        # COEFFICIENT OF VARIATION
        # =====================================================

        customer_variance["CV"] = (

            customer_variance["RM_STD"]

            /

            customer_variance["RM_Mean"]

        )

        customer_variance["CV"] = (
            customer_variance["CV"]
            .replace([np.inf, -np.inf], 0)
        )

        customer_variance["CV"] = (
            customer_variance["CV"]
            .fillna(0)
        )

        # =====================================================
        # VARIANCE SEGMENT
        # =====================================================

        customer_variance["VarianceSegment"] = np.where(

            customer_variance["CV"] >= 0.50,

            "🔥 Highly Volatile",

            np.where(

                customer_variance["CV"] >= 0.20,

                "⚡ Moderate Variance",

                "🟢 Stable"

            )

        )

        # =====================================================
        # MERGE VARIANCE
        # =====================================================

        customer_summary = customer_summary.merge(

            customer_variance[[
                "ProfitCenter",
                "CV",
                "VarianceSegment"
            ]],

            on="ProfitCenter",

            how="left"

        )

        # =====================================================
        # TREND ANALYSIS
        # =====================================================

        trend_list = []

        for customer in customers:

            temp = df[
                df["ProfitCenter"] == customer
            ].copy()

            temp = temp.sort_values(
                "MonthDate"
            )

            if len(temp) < 2:

                growth_pct = 0

                trend = "Insufficient Data"

            else:

                first_value = (
                    temp["RawMaterialInventory"]
                    .iloc[0]
                )

                last_value = (
                    temp["RawMaterialInventory"]
                    .iloc[-1]
                )

                if first_value == 0:

                    growth_pct = 0

                else:

                    growth_pct = (
                        (
                            last_value
                            - first_value
                        )
                        / first_value
                    ) * 100

                if growth_pct >= 50:

                    trend = "🚀 Strong Increasing"

                elif growth_pct >= 15:

                    trend = "📈 Increasing"

                elif growth_pct >= 5:

                    trend = "🟢 Slight Increasing"

                elif growth_pct <= -50:

                    trend = "🔻 Strong Decreasing"

                elif growth_pct <= -15:

                    trend = "📉 Decreasing"

                elif growth_pct <= -5:

                    trend = "🟠 Slight Decreasing"

                else:

                    trend = "➖ Stable"

            trend_list.append({

                "ProfitCenter": customer,

                "GrowthPercent":
                    round(growth_pct, 2),

                "Trend": trend

            })

        trend_df = pd.DataFrame(
            trend_list
        )

        customer_summary = customer_summary.merge(
            trend_df,
            on="ProfitCenter",
            how="left"
        )

        # =====================================================
        # GENERATE FUTURE MONTHS
        # =====================================================

        last_month = df["MonthDate"].max()

        future_dates = pd.date_range(
            start=last_month + pd.DateOffset(months=1),
            periods=6,
            freq='MS'
        )

        future_months = [
            d.strftime("%b'%y")
            for d in future_dates
        ]

        # =====================================================
        # FORECAST ENGINE
        # =====================================================

        forecast_rows = []

        # =====================================================
        # LOOP CUSTOMER
        # =====================================================

        for customer in customers:

            temp = df[
                df["ProfitCenter"] == customer
            ].copy()

            if len(temp) < 6:
                continue

            # =================================================
            # FORECAST RAW MATERIAL
            # =================================================

            try:

                if STATS_AVAILABLE and len(temp) >= 12:

                    model_rm = ExponentialSmoothing(
                        temp["RawMaterialInventory"],
                        trend='add',
                        seasonal='add',
                        seasonal_periods=12
                    ).fit()

                    forecast_rm = model_rm.forecast(6)

                else:

                    avg_growth = (
                        temp["RawMaterialInventory"]
                        .pct_change()
                        .mean()
                    )

                    avg_growth = np.nan_to_num(
                        avg_growth,
                        nan=0
                    )

                    last_value = (
                        temp["RawMaterialInventory"]
                        .iloc[-1]
                    )

                    forecast_rm = []

                    for i in range(6):

                        next_value = (
                            last_value
                            * (1 + avg_growth)
                        )

                        forecast_rm.append(
                            max(next_value, 0)
                        )

                        last_value = next_value

                    forecast_rm = pd.Series(
                        forecast_rm
                    )

            except:

                forecast_rm = pd.Series(
                    [temp["RawMaterialInventory"].mean()] * 6
                )

            # =================================================
            # CLIP NEGATIVE
            # =================================================

            forecast_rm = forecast_rm.clip(
                lower=0
            )

            # =================================================
            # DERIVE REVENUE FROM RM
            # =================================================

            avg_revenue_ratio = (

                temp["ProductionRevenue"].sum()

                /

                max(
                    temp["RawMaterialInventory"].sum(),
                    1
                )

            )

            forecast_rev = (
                forecast_rm
                * avg_revenue_ratio
            )

            forecast_rev = forecast_rev.clip(
                lower=0
            )

            # =================================================
            # CUSTOMER VARIANCE
            # =================================================

            customer_cv = (

                customer_summary[
                    customer_summary["ProfitCenter"]
                    == customer
                ]["CV"]

                .iloc[0]

            )

            variance_segment = (

                customer_summary[
                    customer_summary["ProfitCenter"]
                    == customer
                ]["VarianceSegment"]

                .iloc[0]

            )

            # =================================================
            # VARIANCE CONTROL
            # =================================================

            if variance_segment == "🔥 Highly Volatile":

                variance_factor = 0.25

            elif variance_segment == "⚡ Moderate Variance":

                variance_factor = 0.12

            else:

                variance_factor = 0.05

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

            for i, month in enumerate(future_months):

                production_revenue = max(
                    forecast_rev.iloc[i],
                    0
                )

                raw_material = max(
                    forecast_rm.iloc[i],
                    0
                )

                # =================================================
                # RANDOM VARIANCE
                # =================================================

                variance_noise = np.random.normal(

                    0,

                    customer_cv * variance_factor

                )

                raw_material = (

                    raw_material

                    * (1 + variance_noise)

                )

                raw_material = max(
                    raw_material,
                    0
                )

                # =================================================
                # OPERATIONAL FORECAST
                # =================================================

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

                total_transaction = (
                    receiving
                    + shipping
                    + transfer
                )

                no_of_pallet = (
                    fg_pallet
                    + rm_pallet
                )

                # =================================================
                # FLEXIBLE WAREHOUSE CAPACITY
                # =================================================

                base_capacity = (
                    no_of_pallet
                )

                warehouse_capacity = (

                    base_capacity

                    * (1 + alpha)

                )

                # =================================================
                # COST COMPONENTS
                # =================================================

                holding_cost = (
                    raw_material * 0.15
                )

                shortage_cost = (
                    max(
                        receiving - raw_material,
                        0
                    ) * 0.30
                )

                capacity_cost = (
                    warehouse_capacity * 2
                )

                transaction_cost = (
                    total_transaction * 0.50
                )

                warehouse_cost = (

                    holding_cost

                    + shortage_cost

                    + capacity_cost

                    + transaction_cost

                )

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

        if forecast_df.empty:

            st.error(
                "No forecast generated."
            )

            st.stop()

        # =====================================================
        # SORT FORECAST MONTH
        # =====================================================

        forecast_df["MonthDate"] = pd.to_datetime(
            forecast_df["Month"],
            format="%b'%y"
        )

        forecast_df = forecast_df.sort_values(
            "MonthDate"
        )

        # =====================================================
        # FORECAST VALIDATION
        # =====================================================

        validation_rows = []

        for customer in customers:

            temp = df[
                df["ProfitCenter"] == customer
            ].copy()

            temp = temp.sort_values(
                "MonthDate"
            )

            if len(temp) < 6:
                continue

            actual = (
                temp["RawMaterialInventory"]
                .iloc[-6:]
                .values
            )

            predicted = (
                temp["RM_MA3"]
                .iloc[-6:]
                .values
            )

            mae = mean_absolute_error(
                actual,
                predicted
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )

            mape = np.mean(

                np.abs(
                    (
                        actual - predicted
                    )

                    /

                    np.where(
                        actual == 0,
                        1,
                        actual
                    )

                )

            ) * 100

            validation_rows.append({

                "ProfitCenter": customer,

                "MAE":
                    round(mae, 2),

                "RMSE":
                    round(rmse, 2),

                "MAPE":
                    round(mape, 2)

            })

        validation_df = pd.DataFrame(
            validation_rows
        )

        # =====================================================
        # DISPLAY VALIDATION
        # =====================================================

        st.subheader(
            "📏 Forecast Accuracy Validation"
        )

        st.dataframe(
            validation_df,
            use_container_width=True
        )

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
        # DISPLAY SEGMENTATION
        # =====================================================

        st.subheader("📊 Customer Demand Segmentation")

        st.dataframe(
            customer_summary,
            use_container_width=True
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

        st.subheader("📈 Raw Material Inventory Forecast")

        fig = px.line(
            forecast_df,
            x="Month",
            y="RawMaterialInventory",
            color="ProfitCenter",
            title="Raw Material Inventory Forecast",
            markers=True
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

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "HCMIC EMS Forecasting & Warehouse Optimization System"
)
