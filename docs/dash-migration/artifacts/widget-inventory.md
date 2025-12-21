# Widget Inventory (Streamlit)

## Keyed widgets (raw key list)
438:    run_top = st.button("Run Backtest", type="primary", use_container_width=True, key="run_top")
452:        key="preset_selector"
477:            key="w_exchange"
479:        symbol = st.text_input("Symbol", key="w_symbol")
485:            key="w_timeframe"
506:        bb_len = st.number_input("BB length", 5, 200, key="w_bb_len")
507:        bb_std = st.number_input("BB std dev", 1.0, 4.0, key="w_bb_std", step=0.1)
512:            key="w_bb_basis_type"
516:        kc_ema_len = st.number_input("KC EMA/SMA length (mid)", 5, 200, key="w_kc_ema_len")
517:        kc_atr_len = st.number_input("KC ATR length", 5, 200, key="w_kc_atr_len")
518:        kc_mult = st.number_input("KC ATR multiplier", 0.5, 5.0, key="w_kc_mult", step=0.1)
519:        kc_mid_is_ema = st.checkbox("KC mid uses EMA (uncheck = SMA)", key="w_kc_mid_is_ema")
523:        rsi_len_30m = st.number_input("RSI length", 5, 100, key="w_rsi_len_30m")
528:            key="w_rsi_smoothing_type"
530:        rsi_ma_len = st.number_input("RSI MA length", 2, 100, key="w_rsi_ma_len")
535:            key="w_rsi_ma_type"
541:        rsi_min = st.number_input("RSI minimum (entry)", 0, 100, key="w_rsi_min")
542:        rsi_ma_min = st.number_input("RSI MA minimum (entry)", 0, 100, key="w_rsi_ma_min")
545:            key="w_use_rsi_relation",
551:            key="w_rsi_relation"
559:            key="w_entry_band_mode",
567:            key="w_exit_channel"
573:            key="w_exit_level",
578:        cash = st.number_input("Starting cash", 100, 1_000_000_000, 10_000, 100, key="w_cash")
579:        commission = st.number_input("Commission (fraction)", 0.0, 0.01, 0.001, 0.0001, key="w_commission")
586:            key="w_trade_mode",
589:        use_stop = st.checkbox("Enable stop loss", key="w_use_stop")
595:            key="w_stop_mode",
603:                key="w_stop_pct"
611:                key="w_stop_atr_mult"
615:        use_trailing = st.checkbox("Enable trailing stop", key="w_use_trailing")
620:            key="w_trail_pct"
627:            key="w_max_bars_in_trade"
634:            key="w_daily_loss_limit"
641:            key="w_risk_per_trade_pct"
649:                                          key="w_max_leverage")
652:                                                    key="w_maintenance_margin_pct")
655:                enable_max_margin_util = st.checkbox("Limit margin utilization?", value=False, key="w_enable_max_margin_util")
658:                                                        key="w_max_margin_utilization")
667:    run_bottom = st.button("Run Backtest", type="primary", use_container_width=True, key="run_bottom")
791:            show_candles = st.checkbox("Show Candlesticks", value=True, key="show_candles")
793:            lock_rsi_y = st.checkbox("Lock RSI Y-axis (0-100)", value=True, key="lock_rsi_y")
1045:        mode = st.radio("Select Tool", ["Parameter Optimization", "Strategy Discovery", "Leaderboard", "Pattern Recognition"], horizontal=True, key="analysis_mode")
1063:                    key="opt_rsi_range",
1073:                    key="opt_stop_range",
1083:                    key="opt_band_range",
1094:                    key="opt_validation_mode"
1102:                    key="opt_metric",
1111:                    key="opt_grid_steps",
1116:                include_entry_modes = st.checkbox("Test all entry band modes", value=True, key="opt_include_entry_modes")
1117:                include_exit_levels = st.checkbox("Test both exit levels (mid/lower)", value=False, key="opt_include_exit_levels")
1124:                    key="opt_min_trades",
1132:                    key="opt_min_pf"
1137:                    key="opt_min_wr"
1142:                    key="opt_max_dd"
1147:                    key="opt_min_total_ret"
1152:                    wf_train_days = st.number_input("Train window (days)", min_value=30, max_value=3650, value=180, step=30, key="wf_train_days")
1153:                    wf_test_days = st.number_input("Test window (days)", min_value=7, max_value=365, value=30, step=7, key="wf_test_days")
1154:                    wf_max_folds = st.number_input("Max folds (0 = all)", min_value=0, max_value=200, value=8, step=1, key="wf_max_folds")
1340:                    key="disc_n_workers",
1344:                disc_use_parallel = st.checkbox("Enable Parallel Processing", value=True, key="disc_use_parallel")
1347:                disc_skip_tested = st.checkbox("Skip tested combinations", value=True, key="disc_skip_tested")
1353:                disc_rsi_range = st.slider("RSI Min Range", 60, 82, (68, 74), key="disc_rsi_range")
1354:                disc_rsi_ma_range = st.slider("RSI MA Min Range", 58, 80, (66, 72), key="disc_rsi_ma_range")
1355:                disc_band_range = st.slider("Band Mult Range", 1.5, 2.8, (1.9, 2.1), key="disc_band_range")
1359:                disc_leverage_options = st.multiselect("Leverage", [2.0, 3.0, 5.0, 10.0, 20.0], [2.0, 5.0, 10.0], key="disc_leverage_options")
1360:                disc_risk_options = st.multiselect("Risk %", [0.5, 1.0, 1.5, 2.0, 3.0], [0.5, 1.0, 2.0], key="disc_risk_options")
1361:                disc_include_atr = st.checkbox("Include ATR Stops", value=True, key="disc_include_atr")
1362:                disc_include_trailing = st.checkbox("Include Trailing", value=True, key="disc_include_trailing")
1366:                disc_min_return = st.number_input("Min Return %", -100.0, 100.0, 0.0, key="disc_min_return")
1367:                disc_max_dd = st.number_input("Max Drawdown %", 1.0, 50.0, 20.0, key="disc_max_dd")
1368:                disc_min_trades = st.number_input("Min Trades", 5, 100, 10, key="disc_min_trades")
1369:                disc_min_pf = st.number_input("Min Profit Factor", 0.5, 3.0, 1.0, key="disc_min_pf")
1398:            if st.button("🔬 Run Strategy Discovery", type="primary", key="run_discovery_btn"):
1468:                        key="lb_sort_by"
1471:                    lb_top_n = st.selectbox("Show top", [10, 25, 50, 100], key="lb_top_n")
1540:                                if st.button("📥 Load This Strategy", key="load_strategy_btn"):
1563:            if st.button("🔄 Analyze Patterns", key="analyze_patterns_btn"):

## Unkeyed inputs
- st.date_input: Date range (uses tuple value from session_state)

## Output widgets
- st.plotly_chart
- st.dataframe
- st.download_button
- st.info / st.warning / st.error / st.success
- st.markdown / st.write / st.caption

## Layout containers
- st.sidebar
- st.expander
- st.tabs
- st.columns
- st.divider

## Session state keys (non-widget + widget-related)
session_state.dirty_params
session_state.discovery_results
session_state.discovery_running
session_state.get
session_state.last_params
session_state.preset_params
session_state.results
session_state.run_ready
session_state.selected_preset
session_state.selected_trade
session_state.selected_winning_strategy
session_state.w_bb_basis_type
session_state.w_end_date
session_state.w_entry_band_mode
session_state.w_exchange
session_state.w_exit_channel
session_state.w_exit_level
session_state.w_rsi_ma_type
session_state.w_rsi_relation
session_state.w_rsi_smoothing_type
session_state.w_start_date
session_state.w_stop_mode
session_state.w_timeframe
session_state.w_trade_mode
- Additional widget keys are listed above in the keyed widget list.
