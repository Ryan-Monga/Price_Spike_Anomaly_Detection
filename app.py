"""
Price Spike Anomaly Detection for Pump-and-Dump Schemes
Streamlit Surveillance Dashboard and Analytics Application.
"""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    DEFAULT_ASSETS,
    SUPPORTED_PERIODS,
    CORE_FEATURE_COLUMNS,
    BaselineConfig,
    OneClassSVMConfig,
    LSTMAutoencoderConfig,
    HybridDetectorConfig
)
from src.model_manager import ModelManager

# Page Configuration
st.set_page_config(
    page_title="Market Surveillance | Pump-and-Dump Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial Surveillance Terminal Aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #131a29;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-badge-anomaly {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-normal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131a29;
        border-radius: 6px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "manager" not in st.session_state:
    st.session_state.manager = ModelManager()

if "surveillance_df" not in st.session_state:
    st.session_state.surveillance_df = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "last_experiment" not in st.session_state:
    st.session_state.last_experiment = None

manager: ModelManager = st.session_state.manager

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("🛡️ Surveillance Hub")
    st.caption("Price Spike & Volume Anomaly Detection")
    st.markdown("---")

    st.subheader("Asset & Timeframe")
    asset_mode = st.radio("Asset Selection", ["Presets", "Custom Ticker"], horizontal=True)
    if asset_mode == "Presets":
        selected_ticker = st.selectbox("Select Asset", DEFAULT_ASSETS, index=0)
    else:
        selected_ticker = st.text_input("Enter Ticker (e.g. NVDA, SOL-USD)", value="AAPL")

    selected_period = st.selectbox("Historical Horizon", SUPPORTED_PERIODS, index=3)
    force_refresh = st.checkbox("Force Live Refresh", value=False, help="Bypass local cache and query Yahoo Finance directly.")

    if st.button("📥 Load & Preprocess Data", use_container_width=True, type="primary"):
        with st.spinner(f"Ingesting market data for {selected_ticker}..."):
            try:
                features_df, meta = manager.load_dataset(
                    ticker=selected_ticker,
                    period=selected_period,
                    force_refresh=force_refresh
                )
                # Run initial surveillance prediction
                st.session_state.surveillance_df = manager.predict_surveillance()
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(features_df)} observations for {selected_ticker} ({meta.get('source', 'unknown')})")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    st.markdown("---")
    st.subheader("Surveillance Model Status")
    status = manager.get_model_status()
    st.write(f"• **Baseline**: {status['baseline']['status']}")
    st.write(f"• **One-Class SVM**: {status['one_class_svm']['status']}")
    st.write(f"• **LSTM Autoencoder**: {status['lstm_autoencoder']['status']}")
    st.write(f"• **Hybrid Threshold**: `{status['hybrid']['threshold']}`")

    st.markdown("---")
    st.caption(
        "⚖️ **Legal Notice**: Identified alerts represent statistical & algorithmic deviations. "
        "They constitute model-derived risk indicators, not confirmed legal proof of market manipulation."
    )

# Auto-load default dataset on first launch if not loaded
if not st.session_state.data_loaded:
    try:
        features_df, meta = manager.load_dataset(ticker="AAPL", period="1y", force_refresh=False)
        st.session_state.surveillance_df = manager.predict_surveillance()
        st.session_state.data_loaded = True
    except Exception:
        pass

surv_df = st.session_state.surveillance_df

# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================
tab_overview, tab_market, tab_investigate, tab_train, tab_eval, tab_data = st.tabs([
    "📊 Overview",
    "📈 Market Analysis",
    "🔍 Anomaly Investigation",
    "🧠 Model Training",
    "🧪 Evaluation Lab",
    "📁 Data & Reports"
])

# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    if surv_df is None or len(surv_df) == 0:
        st.info("Please load market data from the sidebar to begin surveillance.")
    else:
        latest = surv_df.iloc[-1]
        latest_date = surv_df.index[-1].strftime("%Y-%m-%d")
        total_obs = len(surv_df)
        total_anomalies = int(surv_df["is_anomaly"].sum())
        anomaly_rate = (total_anomalies / total_obs) * 100.0

        meta = manager.data_metadata
        source_label = meta.get("source", "historical")
        is_cached = meta.get("is_cached", False)
        is_synthetic = meta.get("is_synthetic", False)

        # Header metadata badges
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.subheader(f"Surveillance Overview: {manager.current_ticker}")
            st.caption(f"Last Available Observation: **{latest_date}** (Daily Resolution)")
        with c2:
            if is_synthetic:
                st.warning("⚠️ Offline Demo Data")
            elif is_cached:
                st.info("💾 Cached Historical Data")
            else:
                st.success("🌐 Live Yahoo Finance Data")
        with c3:
            status_text = latest["detection_status"]
            if status_text == "ANOMALY":
                st.markdown('<div class="status-badge-anomaly">⚠️ ANOMALY DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-normal">✅ NORMAL REGIME</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Top 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Latest Closing Price</div>
                <div class="metric-value">${latest['Close']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            ret_val = latest["simple_return"] * 100.0
            color = "#10b981" if ret_val >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Latest Daily Return</div>
                <div class="metric-value" style="color: {color};">{ret_val:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            rvol = latest["relative_volume"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Relative Volume</div>
                <div class="metric-value">{rvol:.2f}x</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            h_score = latest["hybrid_score"]
            thresh = manager.hybrid_detector.threshold
            score_color = "#ef4444" if h_score >= thresh else "#38bdf8"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Hybrid Anomaly Risk</div>
                <div class="metric-value" style="color: {score_color};">{h_score:.3f} <span style="font-size: 0.9rem; color: #94a3b8;">/ {thresh:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Period Surveillance Summary
        st.markdown("### Surveillance Activity Summary")
        col_summary1, col_summary2 = st.columns([1, 2])
        with col_summary1:
            st.metric("Total Period Observations", f"{total_obs} days")
            st.metric("Total Detected Anomalies", f"{total_anomalies} events")
            st.metric("Period Anomaly Frequency", f"{anomaly_rate:.2f}%")

        with col_summary2:
            st.write("**Latest Evidence Log:**")
            st.info(latest["alert_explanation"])
            
            # Mini score breakdown
            st.write("**Component Risk Contributions:**")
            comp_cols = st.columns(3)
            with comp_cols[0]:
                st.write(f"Statistical Score: `{latest['stat_score']:.3f}`")
            with comp_cols[1]:
                svm_s = f"{latest['svm_score']:.3f}" if not pd.isna(latest['svm_score']) else "N/A (Untrained)"
                st.write(f"One-Class SVM: `{svm_s}`")
            with comp_cols[2]:
                lstm_s = f"{latest['lstm_score']:.3f}" if not pd.isna(latest['lstm_score']) else "N/A (Untrained)"
                st.write(f"LSTM Autoencoder: `{lstm_s}`")

# ------------------------------------------------------------------------------
# TAB 2: MARKET ANALYSIS
# ------------------------------------------------------------------------------
with tab_market:
    if surv_df is None or len(surv_df) == 0:
        st.info("Please load market data from the sidebar.")
    else:
        st.subheader("Interactive Market Surveillance Timeline")
        st.caption("Hover over points to inspect prices, volume, returns, and model anomaly risk scores.")

        # Multi-panel Plotly Chart
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.38, 0.22, 0.20, 0.20],
            subplot_titles=[
                "Asset Closing Price & Detected Anomaly Flags",
                "Trading Volume & Baseline Average",
                "Daily Simple Return & Volatility Bands (±2.5σ)",
                "Hybrid Anomaly Score vs Surveillance Threshold"
            ]
        )

        dates = surv_df.index
        anomalies_mask = surv_df["is_anomaly"]
        thresh = manager.hybrid_detector.threshold

        # Row 1: Closing Price
        fig.add_trace(
            go.Scatter(
                x=dates, y=surv_df["Close"],
                name="Close Price",
                line=dict(color="#38bdf8", width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: $%{y:.2f}"
            ),
            row=1, col=1
        )

        # Anomaly markers on Price
        if anomalies_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[anomalies_mask],
                    y=surv_df.loc[anomalies_mask, "Close"],
                    mode="markers",
                    name="Anomaly Flagged",
                    marker=dict(color="#ef4444", size=9, symbol="circle-open-dot", line=dict(width=2, color="#ef4444")),
                    hovertemplate="🚨 ANOMALY<br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}"
                ),
                row=1, col=1
            )

        # Row 2: Volume
        vol_colors = ["#ef4444" if a else "#475569" for a in anomalies_mask]
        fig.add_trace(
            go.Bar(
                x=dates, y=surv_df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}"
            ),
            row=2, col=1
        )

        # Row 3: Daily Return & Volatility Bands
        mean_ret = surv_df["rolling_mean_30"]
        band_up = mean_ret + 2.5 * surv_df["volatility_30"]
        band_dn = mean_ret - 2.5 * surv_df["volatility_30"]

        fig.add_trace(
            go.Scatter(
                x=dates, y=surv_df["simple_return"],
                name="Daily Return",
                line=dict(color="#a855f7", width=1.5),
                hovertemplate="Return: %{y:.2%}"
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=band_up,
                name="+2.5σ Band",
                line=dict(color="#64748b", width=1, dash="dot"),
                hoverinfo="skip"
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=band_dn,
                name="-2.5σ Band",
                line=dict(color="#64748b", width=1, dash="dot"),
                hoverinfo="skip"
            ),
            row=3, col=1
        )

        # Row 4: Hybrid Anomaly Score & Threshold
        fig.add_trace(
            go.Scatter(
                x=dates, y=surv_df["hybrid_score"],
                name="Hybrid Score",
                line=dict(color="#00d2ff", width=2),
                hovertemplate="Hybrid Score: %{y:.3f}"
            ),
            row=4, col=1
        )
        # Threshold line
        fig.add_trace(
            go.Scatter(
                x=[dates.min(), dates.max()],
                y=[thresh, thresh],
                name=f"Threshold ({thresh:.2f})",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
                hoverinfo="skip"
            ),
            row=4, col=1
        )

        fig.update_layout(
            height=850,
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#131a29",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ANOMALY INVESTIGATION
# ------------------------------------------------------------------------------
with tab_investigate:
    if surv_df is None or len(surv_df) == 0:
        st.info("Please load market data from the sidebar.")
    else:
        st.subheader("🔍 Suspicious Event Investigation & Evidence Drill-Down")
        st.caption("Inspect individual flagged events, review technical features, and inspect the surrounding market context.")

        anomaly_rows = surv_df[surv_df["is_anomaly"]].copy()
        
        if len(anomaly_rows) == 0:
            st.success("No anomalous events identified in the current period under threshold rules.")
        else:
            st.write(f"Identified **{len(anomaly_rows)}** suspicious market observations:")

            # Prepare clean summary table
            display_cols = [
                "Close", "simple_return", "relative_volume", "return_zscore",
                "stat_score", "svm_score", "lstm_score", "hybrid_score", "alert_explanation"
            ]
            clean_table = anomaly_rows[display_cols].copy()
            clean_table["simple_return"] = clean_table["simple_return"].map(lambda x: f"{x:+.2%}")
            clean_table["relative_volume"] = clean_table["relative_volume"].map(lambda x: f"{x:.2f}x")
            clean_table["return_zscore"] = clean_table["return_zscore"].map(lambda x: f"{x:+.2f}")
            clean_table["stat_score"] = clean_table["stat_score"].map(lambda x: f"{x:.3f}")
            clean_table["svm_score"] = clean_table["svm_score"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
            clean_table["lstm_score"] = clean_table["lstm_score"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
            clean_table["hybrid_score"] = clean_table["hybrid_score"].map(lambda x: f"{x:.3f}")

            st.dataframe(clean_table, use_container_width=True)

            # Deep-dive selection
            st.markdown("---")
            st.subheader("Event Deep-Dive & Local Context Window")
            
            selected_date = st.selectbox(
                "Select Flagged Date to Inspect",
                options=[d.strftime("%Y-%m-%d") for d in anomaly_rows.index]
            )
            
            if selected_date:
                dt_key = pd.to_datetime(selected_date)
                sel_row = surv_df.loc[dt_key]

                col_dt1, col_dt2 = st.columns([1, 1])
                with col_dt1:
                    st.write("**Event Technical Diagnostics:**")
                    st.write(f"• **Close Price:** `${sel_row['Close']:.2f}`")
                    st.write(f"• **Daily Return:** `{sel_row['simple_return']:+.2%}`")
                    st.write(f"• **Return Z-Score:** `{sel_row['return_zscore']:+.2f}`")
                    st.write(f"• **Relative Volume:** `{sel_row['relative_volume']:.2f}x` normal baseline")
                    st.write(f"• **High-Low Range:** `{sel_row['hl_range']:.2%}` of price")

                with col_dt2:
                    st.write("**Model Detection Evidence:**")
                    st.info(sel_row["alert_explanation"])

                # Surrounding 14-day context window chart
                idx_pos = surv_df.index.get_loc(dt_key)
                start_w = max(0, idx_pos - 7)
                end_w = min(len(surv_df), idx_pos + 8)
                window_df = surv_df.iloc[start_w:end_w]

                w_fig = go.Figure()
                w_fig.add_trace(go.Candlestick(
                    x=window_df.index,
                    open=window_df["Open"],
                    high=window_df["High"],
                    low=window_df["Low"],
                    close=window_df["Close"],
                    name="Local Price Window"
                ))
                # Add vertical marker on target date
                w_fig.add_vline(x=dt_key, line_width=2, line_dash="dash", line_color="#ef4444")
                w_fig.update_layout(
                    title=f"Local Context (±7 Days) Around {selected_date}",
                    template="plotly_dark",
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#131a29",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(w_fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MODEL TRAINING
# ------------------------------------------------------------------------------
with tab_train:
    st.subheader("🧠 Machine Learning Model Training & Calibration")
    st.caption("Train One-Class SVM and PyTorch LSTM Autoencoder strictly on the training partition.")

    if manager.train_df is None:
        st.info("Please load market data from the sidebar to initialize training splits.")
    else:
        split_meta = manager.split_metadata
        if split_meta:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Training Split (70%)", f"{split_meta.train_size} days", f"{split_meta.train_dates[0]} to {split_meta.train_dates[1]}")
            with sc2:
                st.metric("Validation Split (15%)", f"{split_meta.val_size} days", f"{split_meta.val_dates[0]} to {split_meta.val_dates[1]}")
            with sc3:
                st.metric("Test Split (15%)", f"{split_meta.test_size} days", f"{split_meta.test_dates[0]} to {split_meta.test_dates[1]}")

        st.markdown("---")
        train_col1, train_col2 = st.columns(2)

        # One-Class SVM Trainer
        with train_col1:
            st.markdown("### 1. One-Class SVM")
            st.caption("Unsupervised kernel boundary detector for feature distributions.")
            
            svm_nu = st.slider("Contamination Parameter (nu)", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                               help="Upper bound on fraction of training outliers.")
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)

            if st.button("🚀 Train One-Class SVM", use_container_width=True):
                with st.spinner("Fitting One-Class SVM on training observations..."):
                    try:
                        cfg = OneClassSVMConfig(nu=svm_nu, kernel=svm_kernel)
                        svm_meta = manager.train_one_class_svm(cfg)
                        st.session_state.surveillance_df = manager.predict_surveillance()
                        st.success(f"One-Class SVM trained successfully on {svm_meta['train_observations']} observations.")
                    except Exception as e:
                        st.error(f"SVM Training Failed: {e}")

        # PyTorch LSTM Autoencoder Trainer
        with train_col2:
            st.markdown("### 2. PyTorch LSTM Autoencoder")
            st.caption("Deep temporal sequence model flagging abnormal reconstruction errors.")

            lstm_seq_len = st.slider("Sequence Lookback (days)", min_value=10, max_value=45, value=25, step=5)
            lstm_epochs = st.slider("Training Epochs", min_value=5, max_value=40, value=20, step=5)
            lstm_hidden = st.selectbox("Hidden Dimension", [16, 32, 64], index=1)
            lstm_lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.001)

            if st.button("🚀 Train LSTM Autoencoder", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(epoch, total_epochs, tr_loss, v_loss):
                    progress_bar.progress(epoch / total_epochs)
                    status_text.text(f"Epoch {epoch}/{total_epochs} — Train Loss: {tr_loss:.5f} | Val Loss: {v_loss:.5f}")

                with st.spinner("Training LSTM Autoencoder on CPU/GPU..."):
                    try:
                        cfg = LSTMAutoencoderConfig(
                            sequence_length=lstm_seq_len,
                            hidden_dim=lstm_hidden,
                            latent_dim=max(8, lstm_hidden // 2),
                            learning_rate=lstm_lr,
                            epochs=lstm_epochs
                        )
                        lstm_meta = manager.train_lstm_autoencoder(cfg, progress_callback=update_progress)
                        st.session_state.surveillance_df = manager.predict_surveillance()
                        st.success("LSTM Autoencoder trained successfully!")
                    except Exception as e:
                        st.error(f"LSTM Training Failed: {e}")

        # Training loss curve visualization
        if manager.lstm_detector and len(manager.lstm_detector.training_history["train_loss"]) > 0:
            st.markdown("---")
            st.subheader("LSTM Autoencoder Convergence Curves")
            history = manager.lstm_detector.training_history
            epochs_x = list(range(1, len(history["train_loss"]) + 1))

            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(x=epochs_x, y=history["train_loss"], name="Training MSE Loss", line=dict(color="#38bdf8", width=2)))
            if history.get("val_loss") and len(history["val_loss"]) > 0:
                loss_fig.add_trace(go.Scatter(x=epochs_x, y=history["val_loss"], name="Validation MSE Loss", line=dict(color="#f59e0b", width=2, dash="dash")))

            loss_fig.update_layout(
                title="Reconstruction Error over Training Epochs",
                xaxis_title="Epoch",
                yaxis_title="Mean Squared Error",
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#131a29",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(loss_fig, use_container_width=True)

        # Hybrid Ensemble Weights & Threshold Tuning
        st.markdown("---")
        st.subheader("3. Hybrid Detector Weighting & Validation Calibration")
        st.caption("Adjust component score weights and calibrate decision threshold on validation data.")

        hw1, hw2, hw3 = st.columns(3)
        with hw1:
            w_stat = st.slider("Statistical Weight (w1)", 0.0, 1.0, 0.333, step=0.05)
        with hw2:
            w_svm = st.slider("One-Class SVM Weight (w2)", 0.0, 1.0, 0.333, step=0.05)
        with hw3:
            w_lstm = st.slider("LSTM Autoencoder Weight (w3)", 0.0, 1.0, 0.334, step=0.05)

        manager.hybrid_detector.set_weights(w_stat, w_svm, w_lstm)

        thresh_col1, thresh_col2 = st.columns([2, 1])
        with thresh_col1:
            curr_thresh = manager.hybrid_detector.threshold
            new_thresh = st.slider("Anomaly Decision Threshold", 0.1, 0.95, float(curr_thresh), step=0.01)
            manager.hybrid_detector.set_threshold(new_thresh)
        with thresh_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎯 Calibrate on Validation (95th %ile)", use_container_width=True):
                try:
                    calibrated = manager.calibrate_threshold(percentile=95.0)
                    st.session_state.surveillance_df = manager.predict_surveillance()
                    st.success(f"Calibrated threshold set to: {calibrated}")
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

# ------------------------------------------------------------------------------
# TAB 5: EVALUATION LAB
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("🧪 Synthetic Anomaly Benchmark & Evaluation Lab")
    st.caption("Inject controlled market manipulation scenarios into the test split to measure precision, recall, F1, F4, and FPR.")

    if manager.test_df is None:
        st.info("Please load market data from the sidebar to initialize test datasets.")
    else:
        st.warning(
            "⚠️ **Benchmark Notice**: Results evaluate detection fidelity against controlled synthetic scenarios. "
            "They do not establish real-world accuracy on unverified market schemes."
        )

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            scenario = st.selectbox(
                "Anomaly Scenario",
                options=[
                    "pump_and_dump_reversal",
                    "sudden_upward_shock",
                    "sudden_downward_shock",
                    "volume_churn",
                    "extreme_return_no_volume",
                    "high_volatility_normal"
                ],
                format_func=lambda x: {
                    "pump_and_dump_reversal": "1. Multi-Day Pump & Dump Reversal",
                    "sudden_upward_shock": "2. Sudden Upward Shock + Volume Surge",
                    "sudden_downward_shock": "3. Sudden Downward Dump + Panic Volume",
                    "volume_churn": "4. Heavy Volume Churn (Flat Price)",
                    "extreme_return_no_volume": "5. Extreme Return (No Volume)",
                    "high_volatility_normal": "6. Benign High Volatility (False Positive Test)"
                }.get(x, x)
            )
        with ec2:
            duration = st.slider("Injection Duration (days)", min_value=1, max_value=6, value=3)
        with ec3:
            strength = st.slider("Scenario Severity Multiplier", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

        if st.button("🔬 Run Benchmark Experiment", type="primary", use_container_width=True):
            with st.spinner("Injecting controlled scenario and evaluating surveillance detectors..."):
                try:
                    mod_df, comp_df, record, metrics = manager.run_synthetic_experiment(
                        scenario=scenario,
                        duration=duration,
                        strength=strength,
                        seed=42
                    )
                    st.session_state.last_experiment = {
                        "mod_df": mod_df,
                        "comp_df": comp_df,
                        "record": record,
                        "metrics": metrics
                    }
                    st.success("Experiment completed successfully!")
                except Exception as e:
                    st.error(f"Experiment execution failed: {e}")

        # Render Experiment Results if available
        if st.session_state.last_experiment is not None:
            exp = st.session_state.last_experiment
            record = exp["record"]
            comp_df = exp["comp_df"]
            metrics = exp["metrics"]

            st.markdown("---")
            st.subheader(f"Experiment Audit: {record.scenario}")
            st.write(f"• **Event Window:** `{record.start_timestamp}` to `{record.end_timestamp}` ({record.duration_days} days)")
            st.write(f"• **Audit Details:** {record.description}")
            st.write(f"• **Ground-Truth Classification:** `{'ANOMALY (1)' if record.ground_truth_label == 1 else 'BENIGN BENCHMARK (0)'}`")

            # Comparative Metrics Table
            st.markdown("### Model Performance Comparison")
            st.dataframe(comp_df, use_container_width=True)

            # Metrics Breakdown Cards
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Hybrid Precision", f"{metrics.precision:.2%}")
            with mc2:
                st.metric("Hybrid Recall", f"{metrics.recall:.2%}")
            with mc3:
                st.metric("F1-Score", f"{metrics.f1_score:.3f}")
            with mc4:
                st.metric("F4-Score (Recall Prioritized)", f"{metrics.f4_score:.3f}")

            # Confusion Matrix Visualization
            st.markdown("### Confusion Matrix (Hybrid Detector)")
            cm_col1, cm_col2 = st.columns([1, 1])
            with cm_col1:
                cm_data = [
                    [metrics.true_negatives, metrics.false_positives],
                    [metrics.false_negatives, metrics.true_positives]
                ]
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm_data,
                    x=["Predicted Normal", "Predicted Anomaly"],
                    y=["Actual Normal", "Actual Anomaly"],
                    colorscale="Blues",
                    text=[
                        [f"TN: {metrics.true_negatives}", f"FP: {metrics.false_positives}"],
                        [f"FN: {metrics.false_negatives}", f"TP: {metrics.true_positives}"]
                    ],
                    texttemplate="%{text}",
                    textfont={"size": 15}
                ))
                cm_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#131a29",
                    height=280,
                    margin=dict(l=40, r=40, t=30, b=40)
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            with cm_col2:
                st.write("**Operational Surveillance Notes:**")
                st.write(f"• **False Positive Rate:** `{metrics.false_positive_rate:.2%}`")
                delay_str = f"{metrics.detection_delay_mean} days" if metrics.detection_delay_mean is not None else "Undetected"
                st.write(f"• **Average Detection Delay:** `{delay_str}`")
                st.write(f"• **Formula (F4):** Recall is weighted 4x as heavily as precision to minimize missed manipulative attacks.")

            # Optional GAN Augmentation Experiment
            st.markdown("---")
            st.subheader("🧬 Experimental Feature: Generative Sequence Augmentation (GAN)")
            st.caption("Evaluate whether augmenting baseline training data with a 1D sequence GAN improves surveillance metrics.")
            
            if st.button("🚀 Run GAN Augmentation Experiment", use_container_width=True):
                with st.spinner("Training sequence GAN on baseline sequences and comparing performance..."):
                    try:
                        gan_df = manager.run_gan_augmentation_experiment(
                            scenario=scenario,
                            duration=duration,
                            strength=strength
                        )
                        st.dataframe(gan_df, use_container_width=True)
                        st.caption("Notice: GAN-generated sequences expand baseline coverage, which can lower false positives in high-volatility regimes.")
                    except Exception as e:
                        st.error(f"GAN experiment failed: {e}")

# ------------------------------------------------------------------------------
# TAB 6: DATA & REPORTS
# ------------------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Data Management & Surveillance Reports")
    st.caption("Inspect raw data, engineered causal features, and export forensic results.")

    if surv_df is None or len(surv_df) == 0:
        st.info("No data currently loaded.")
    else:
        # Previews
        p1, p2 = st.tabs(["Raw Market Data", "Engineered Features Matrix"])
        with p1:
            st.dataframe(manager.raw_df, use_container_width=True)
        with p2:
            st.dataframe(manager.features_df[CORE_FEATURE_COLUMNS], use_container_width=True)

        st.markdown("---")
        st.subheader("Export Forensic Records")
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv_pred = surv_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_pred,
                file_name=f"{manager.current_ticker}_surveillance_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        with c_exp2:
            if st.session_state.last_experiment is not None:
                eval_csv = st.session_state.last_experiment["comp_df"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Benchmark Evaluation CSV",
                    data=eval_csv,
                    file_name=f"{manager.current_ticker}_benchmark_evaluation.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.caption("Run a benchmark experiment in the Evaluation Lab to export evaluation results.")

        st.markdown("---")
        st.subheader("Persistent SQLite Experiment History")
        past_df = manager.storage.get_past_experiments(limit=25)
        if len(past_df) > 0:
            st.dataframe(past_df, use_container_width=True)
        else:
            st.caption("No past experiments recorded in SQLite yet. Run an experiment in the Evaluation Lab to log history.")
