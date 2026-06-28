import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

np.random.seed(42)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error

alpha = 0.20
beta  = 0.10

st.set_page_config(
    page_title="HCMIC Forecasting System",
    layout="wide"
)
st.title("HCMIC EMS Forecasting & Warehouse Optimization")
st.write("Enterprise EMS Forecasting Planning System")

uploaded_file = st.file_uploader(
    "Upload EMS Forecasting Dataset",
    type=["xlsx"]
)

if uploaded_file:
    with st.spinner("Generating Forecast..."):

        df = pd.read_excel(uploaded_file)

        st.subheader("Raw Dataset")
        st.dataframe(df)

        required_cols = [
            "Month", "ProfitCenter", "ProductionRevenue",
            "RawMaterialInventory", "ReceivingTransaction",
            "LocationTransferTransaction", "ShippingTransaction",
            "FG_Pallet", "RM_Pallet", "NoOfBin"
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing Columns: {missing_cols}")
            st.stop()

        numeric_cols = [
            "ProductionRevenue", "RawMaterialInventory",
            "ReceivingTransaction", "LocationTransferTransaction",
            "ShippingTransaction", "FG_Pallet", "RM_Pallet", "NoOfBin"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)

        # FIX 5 (prev): errors="coerce" chong loi locale Windows
        df["MonthDate"] = pd.to_datetime(
            df["Month"], format="%b'%y", errors="coerce"
        )
        if df["MonthDate"].isna().all():
            st.error("Month format invalid. Use format like Mar'26")
            st.stop()
        df = df.dropna(subset=["MonthDate"])

        df["TotalActivity"] = (
            df["RawMaterialInventory"]
            + df["ReceivingTransaction"]
            + df["ShippingTransaction"]
        )
        df = df[df["TotalActivity"] > 0]
        df = df.sort_values(["ProfitCenter", "MonthDate"])

        df["RM_Lag1"] = df.groupby("ProfitCenter")["RawMaterialInventory"].shift(1)
        df["RM_Lag2"] = df.groupby("ProfitCenter")["RawMaterialInventory"].shift(2)
        df["RM_MA3"]  = (
            df.groupby("ProfitCenter")["RawMaterialInventory"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df["RM_STD3"] = (
            df.groupby("ProfitCenter")["RawMaterialInventory"]
            .transform(lambda x: x.rolling(3, min_periods=1).std())
        ).fillna(0)
        df["RM_CV3"] = (
            df["RM_STD3"] / df["RM_MA3"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        customers = df["ProfitCenter"].unique()

        # =====================================================
        # FEATURE ENGINEERING DISPLAY
        # =====================================================
        st.subheader("Feature Engineering")
        for customer in customers:
            st.markdown(f"## ProfitCenter: {customer}")
            st.dataframe(df[df["ProfitCenter"] == customer], use_container_width=True)

        # =====================================================
        # CUSTOMER SEGMENTATION
        # =====================================================
        customer_summary = (
            df.groupby("ProfitCenter")
            .agg({"RawMaterialInventory": "mean", "ReceivingTransaction": "mean", "NoOfBin": "mean"})
            .reset_index()
        )

        customer_variance = (
            df.groupby("ProfitCenter")
            .agg({"RawMaterialInventory": ["mean", "std"]})
        )
        customer_variance.columns = ["RM_Mean", "RM_STD"]
        customer_variance = customer_variance.reset_index()
        customer_variance["CV"] = (
            customer_variance["RM_STD"] / customer_variance["RM_Mean"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        customer_variance["VarianceSegment"] = np.where(
            customer_variance["CV"] >= 0.50, "Highly Volatile",
            np.where(customer_variance["CV"] >= 0.20, "Moderate Variance", "Stable")
        )

        customer_summary = customer_summary.merge(
            customer_variance[["ProfitCenter", "CV", "VarianceSegment"]],
            on="ProfitCenter", how="left"
        )
        customer_summary["DemandSegment"] = np.where(
            customer_summary["CV"] >= 0.50, "Volatile",
            np.where(customer_summary["CV"] >= 0.20, "Moderate", "Stable")
        )

        # =====================================================
        # TREND ANALYSIS
        # =====================================================
        trend_list = []
        for customer in customers:
            temp = df[df["ProfitCenter"] == customer].copy().sort_values("MonthDate")
            if len(temp) < 2:
                growth_pct, trend = 0, "Insufficient Data"
            else:
                first_v = temp["RawMaterialInventory"].iloc[0]
                last_v  = temp["RawMaterialInventory"].iloc[-1]
                growth_pct = ((last_v - first_v) / first_v * 100) if first_v != 0 else 0
                if   growth_pct >= 50:  trend = "Strong Increasing"
                elif growth_pct >= 15:  trend = "Increasing"
                elif growth_pct >= 5:   trend = "Slight Increasing"
                elif growth_pct <= -50: trend = "Strong Decreasing"
                elif growth_pct <= -15: trend = "Decreasing"
                elif growth_pct <= -5:  trend = "Slight Decreasing"
                else:                   trend = "Stable"
            trend_list.append({"ProfitCenter": customer, "GrowthPercent": round(growth_pct, 2), "Trend": trend})

        trend_df = pd.DataFrame(trend_list)
        customer_summary = customer_summary.merge(trend_df, on="ProfitCenter", how="left")

        # =====================================================
        # FUTURE MONTHS
        # =====================================================
        last_date = df["MonthDate"].max()
        future_months = [
            (last_date + pd.DateOffset(months=i + 1)).strftime("%b'%y")
            for i in range(6)
        ]

        # =====================================================
        # HELPER: has_seasonality
        # FIX 2: dung CV > 0.10 thay vi std > 0 (tranh false positive)
        # =====================================================
        def check_seasonality(series):
            if len(series) < 24:
                return False
            cv = series.std() / max(series.mean(), 1)
            return cv > 0.10

        # =====================================================
        # FORECAST LOOP
        # =====================================================
        forecast_rows = []

        for customer in customers:
            temp = df[df["ProfitCenter"] == customer].copy()
            temp = temp.sort_values("MonthDate").reset_index(drop=True)

            if len(temp) < 3:
                continue

            # Raw Data Check
            st.markdown(f"### Raw Data Check: {customer}")
            st.write(temp[[
                "Month", "RawMaterialInventory", "ProductionRevenue",
                "ReceivingTransaction", "LocationTransferTransaction", "ShippingTransaction"
            ]])

            # -------------------------------------------------
            # FORECAST RAW MATERIAL
            # -------------------------------------------------
            has_seasonality = check_seasonality(temp["RawMaterialInventory"])

            try:
                if STATS_AVAILABLE and len(temp) >= 24 and has_seasonality:
                    model_rm = ExponentialSmoothing(
                        temp["RawMaterialInventory"], trend="add",
                        seasonal="add", seasonal_periods=12
                    ).fit()
                    forecast_rm = model_rm.forecast(6)
                    model_used  = "Holt-Winters"

                elif STATS_AVAILABLE and len(temp) >= 12:
                    model_rm = ExponentialSmoothing(
                        temp["RawMaterialInventory"], trend="add", seasonal=None
                    ).fit()
                    forecast_rm = model_rm.forecast(6)
                    model_used  = "Holt Trend"

                else:
                    avg_growth = (
                        temp["RawMaterialInventory"]
                        .pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).mean()
                    )
                    avg_growth  = np.clip(avg_growth, -0.30, 0.30)
                    last_value  = temp["RawMaterialInventory"].iloc[-1]
                    forecast_rm = []
                    for _ in range(6):
                        next_value = last_value * (1 + avg_growth)
                        forecast_rm.append(max(next_value, 0))
                        last_value = next_value
                    forecast_rm = pd.Series(forecast_rm)
                    model_used  = "Average Growth"

            except Exception as e:
                st.warning(f"Forecast fallback [{customer}]: {e}")
                avg_growth = (
                    temp["RawMaterialInventory"]
                    .pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).mean()
                )
                avg_growth  = np.clip(avg_growth, -0.30, 0.30)
                last_value  = temp["RawMaterialInventory"].iloc[-1]
                forecast_rm = []
                for _ in range(6):
                    next_value = last_value * (1 + avg_growth)
                    forecast_rm.append(max(next_value, 0))
                    last_value = next_value
                forecast_rm = pd.Series(forecast_rm)
                model_used  = "Average Growth (Fallback)"

            forecast_rm = forecast_rm.clip(lower=0).reset_index(drop=True)

            # -------------------------------------------------
            # FORECAST REVENUE — rolling 12 thang
            # -------------------------------------------------
            window      = min(12, len(temp))
            avg_rev_12m = temp["ProductionRevenue"].iloc[-window:].mean()
            avg_rm_12m  = temp["RawMaterialInventory"].iloc[-window:].mean()

            if avg_rm_12m > 0:
                forecast_rev = (avg_rev_12m * (forecast_rm / avg_rm_12m)).clip(lower=0)
            else:
                forecast_rev = pd.Series([avg_rev_12m] * 6)
            forecast_rev = forecast_rev.reset_index(drop=True)

            # -------------------------------------------------
            # OPERATIONAL RATIOS
            # -------------------------------------------------
            avg_receiving_ratio = np.clip(
                temp["ReceivingTransaction"].sum() / max(temp["RawMaterialInventory"].sum(), 1), 0, 5)
            avg_shipping_ratio  = np.clip(
                temp["ShippingTransaction"].sum()  / max(temp["ProductionRevenue"].sum(), 1),    0, 5)
            avg_transfer_ratio  = np.clip(
                temp["LocationTransferTransaction"].sum() / max(temp["ReceivingTransaction"].sum(), 1), 0, 5)
            avg_fg_ratio        = np.clip(
                temp["FG_Pallet"].sum() / max(temp["ProductionRevenue"].sum(), 1),       0, 5)
            avg_rm_ratio        = np.clip(
                temp["RM_Pallet"].sum() / max(temp["RawMaterialInventory"].sum(), 1),    0, 5)
            avg_bin_ratio       = np.clip(
                temp["NoOfBin"].sum()   / max(temp["RawMaterialInventory"].sum(), 1),    0, 5)

            # FIX 1: consumption_ratio = RM / Revenue (dung huong)
            consumption_ratio = np.clip(
                temp["RawMaterialInventory"].sum() / max(temp["ProductionRevenue"].sum(), 1), 0, 5)

            # -------------------------------------------------
            # FORECAST FUTURE MONTHS
            # -------------------------------------------------
            inventory = temp["RawMaterialInventory"].iloc[-1]

            for i, month in enumerate(future_months):
                production_revenue = max(forecast_rev.iloc[i], 0)
                raw_material       = max(forecast_rm.iloc[i],  0)

                receiving = raw_material       * avg_receiving_ratio
                shipping  = production_revenue * avg_shipping_ratio
                transfer  = receiving          * avg_transfer_ratio
                fg_pallet = production_revenue * avg_fg_ratio
                rm_pallet = raw_material       * avg_rm_ratio
                no_of_bin = raw_material       * avg_bin_ratio

                total_transaction = receiving + shipping + transfer
                no_of_pallet      = fg_pallet + rm_pallet

                # FIX 1: RM consumption tu production, khong phai shipping FG
                rm_consumed    = production_revenue * consumption_ratio
                inventory_next = max(inventory + receiving - rm_consumed, 0)

                warehouse_capacity = no_of_pallet * (1 + alpha - beta)

                holding_cost     = raw_material * 0.15
                shortage_cost    = max(raw_material - inventory_next, 0) * 0.30
                capacity_cost    = warehouse_capacity * 2
                transaction_cost = total_transaction * 0.50
                warehouse_cost   = holding_cost + shortage_cost + capacity_cost + transaction_cost

                forecast_rows.append({
                    "Month":                       month,
                    "ProfitCenter":                customer,
                    "ForecastModel":               model_used,
                    "ProductionRevenue":           round(production_revenue, 0),
                    "RawMaterialInventory":        round(raw_material,       0),
                    "ReceivingTransaction":        round(receiving,          0),
                    "LocationTransferTransaction": round(transfer,           0),
                    "ShippingTransaction":         round(shipping,           0),
                    "TotalTransaction":            round(total_transaction,  0),
                    "FG_Pallet":                   round(fg_pallet,          0),
                    "RM_Pallet":                   round(rm_pallet,          0),
                    "NoOfPallet":                  round(no_of_pallet,       0),
                    "NoOfBin":                     round(no_of_bin,          0),
                    "WarehouseCapacity":           round(warehouse_capacity, 0),
                    "WarehouseCost":               round(warehouse_cost,     0)
                })

                inventory = inventory_next

        # =====================================================
        # FORECAST DATAFRAME
        # =====================================================
        forecast_df = pd.DataFrame(forecast_rows)
        if forecast_df.empty:
            st.error("No forecast generated.")
            st.stop()

        forecast_df["MonthDate"] = pd.to_datetime(
            forecast_df["Month"], format="%b'%y", errors="coerce"
        )
        forecast_df = forecast_df.sort_values("MonthDate")

        # =====================================================
        # FORECAST VALIDATION — Train/Test Split
        # FIX: use len(train) to decide whether to run validation and skip if train < 12
        # =====================================================
        validation_rows = []

        for customer in customers:
            temp = df[df["ProfitCenter"] == customer].copy()
            temp = temp.sort_values("MonthDate").reset_index(drop=True)

            if len(temp) < 9:
                # still skip extremely small series for forecasting overall, keep this check for forecast loop consistency
                continue

            # prepare train/test
            train  = temp["RawMaterialInventory"].iloc[:-6]
            test   = temp["RawMaterialInventory"].iloc[-6:]
            # debug: show counts as requested
            st.write(customer, len(temp), len(train))

            # require minimum train length for validation (12 months). If train < 12 => skip validation
            if len(train) < 12:
                st.info(f"Skipping validation for {customer}: insufficient training length (train={len(train)} < 12).")
                continue

            actual = test.values

            # FIX 2: CV > 0.10 cho validation
            has_seasonality_val = check_seasonality(train)

            try:
                # use len(train) thresholds (model fits on train)
                if STATS_AVAILABLE and len(train) >= 24 and has_seasonality_val:
                    model_val = ExponentialSmoothing(
                        train, trend="add", seasonal="add", seasonal_periods=12
                    ).fit()
                    predicted = model_val.forecast(6).values
                    val_model = "Holt-Winters"

                elif STATS_AVAILABLE and len(train) >= 12:
                    model_val = ExponentialSmoothing(
                        train, trend="add", seasonal=None
                    ).fit()
                    predicted = model_val.forecast(6).values
                    val_model = "Holt Trend"

                else:
                    avg_growth = (
                        train.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).mean()
                    )
                    avg_growth = np.clip(avg_growth, -0.30, 0.30)
                    last_value = train.iloc[-1]
                    predicted  = []
                    for _ in range(6):
                        next_value = last_value * (1 + avg_growth)
                        predicted.append(max(next_value, 0))
                        last_value = next_value
                    predicted = np.array(predicted)
                    val_model = "Average Growth"

            except Exception as e:
                st.warning(f"Validation fallback [{customer}]: {e}")
                avg_growth = (
                    train.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).mean()
                )
                avg_growth = np.clip(avg_growth, -0.30, 0.30)
                last_value = train.iloc[-1]
                predicted  = []
                for _ in range(6):
                    next_value = last_value * (1 + avg_growth)
                    predicted.append(max(next_value, 0))
                    last_value = next_value
                predicted = np.array(predicted)
                val_model = "Average Growth (Fallback)"

            min_len   = min(len(actual), len(predicted))
            actual    = actual[-min_len:]
            predicted = predicted[-min_len:]

            mae  = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(
                np.abs((actual - predicted) / np.where(actual == 0, 1, actual))
            ) * 100

            validation_rows.append({
                "ProfitCenter": customer,
                "ModelUsed":    val_model,
                "MAE":          round(mae,  2),
                "RMSE":         round(rmse, 2),
                "MAPE (%)":     round(mape, 2)
            })

        validation_df = pd.DataFrame(validation_rows)
        st.subheader("Forecast Accuracy Validation (Train/Test Split)")
        st.dataframe(validation_df, use_container_width=True)

        # =====================================================
        # MODEL SUMMARY PER CUSTOMER
        # =====================================================
        st.subheader("Forecast Model Selection Summary")
        model_summary = (
            forecast_df[["ProfitCenter", "ForecastModel"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        st.dataframe(model_summary, use_container_width=True)

        # =====================================================
        # KPI DASHBOARD
        # =====================================================
        st.subheader("KPI Dashboard")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Warehouse Cost",      f"{forecast_df['WarehouseCost'].sum():,.0f}")
        col2.metric("Average Warehouse Capacity", f"{forecast_df['WarehouseCapacity'].mean():,.0f}")
        col3.metric("Average Transaction",        f"{forecast_df['TotalTransaction'].mean():,.0f}")

        # =====================================================
        # CUSTOMER DEMAND SEGMENTATION
        # =====================================================
        st.subheader("Customer Demand Segmentation")
        st.dataframe(customer_summary, use_container_width=True)

        # =====================================================
        # PLANNING MATRIX
        # =====================================================
        st.subheader("EMS Forecast Planning Matrix")

        categories = {
            "Production Revenue":               "ProductionRevenue",
            "130000 - Raw Materials Inventory": "RawMaterialInventory",
            "Receiving transaction":            "ReceivingTransaction",
            "Location transfer transaction":    "LocationTransferTransaction",
            "Shipping transaction":             "ShippingTransaction",
            "Total transaction":                "TotalTransaction",
            "No. of Bin":                       "NoOfBin",
            "No. of pallet":                    "NoOfPallet",
            "FG Pallet":                        "FG_Pallet",
            "Raw Material Pallet":              "RM_Pallet"
        }

        matrix_data = []
        for category, col_name in categories.items():
            for customer in customers:
                temp_f = forecast_df[forecast_df["ProfitCenter"] == customer]
                if temp_f.empty:
                    continue
                row = {"Categories": category, "ProfitCenter": customer}
                for _, r in temp_f.iterrows():
                    row[r["Month"]] = round(r[col_name], 0)
                matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data)
        st.dataframe(matrix_df, use_container_width=True)

        # =====================================================
        # CHART
        # =====================================================
        st.subheader("Raw Material Inventory Forecast")
        fig = px.line(
            forecast_df, x="Month", y="RawMaterialInventory",
            color="ProfitCenter", title="Raw Material Inventory Forecast", markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # FORECAST DATASET
        # =====================================================
        st.subheader("Forecast Dataset")
        st.dataframe(forecast_df, use_container_width=True)

        # =====================================================
        # DOWNLOAD — FIX 3: bao gom ForecastModel trong CSV
        # =====================================================
        st.subheader("Download")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            # FIX 3: merge ForecastModel vao matrix
            model_map  = model_summary.set_index("ProfitCenter")["ForecastModel"]
            matrix_with_model = matrix_df.copy()
            matrix_with_model.insert(
                2, "ForecastModel",
                matrix_with_model["ProfitCenter"].map(model_map)
            )
            csv_matrix = matrix_with_model.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Forecast Matrix CSV",
                data=csv_matrix,
                file_name="EMS_Forecast_Matrix.csv",
                mime="text/csv"
            )

        with col_dl2:
            # Export toan bo forecast_df (co ForecastModel column)
            csv_full = forecast_df.drop(columns=["MonthDate"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Forecast Dataset CSV",
                data=csv_full,
                file_name="EMS_Forecast_Full.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("HCMIC EMS Forecasting & Warehouse Optimization System")
