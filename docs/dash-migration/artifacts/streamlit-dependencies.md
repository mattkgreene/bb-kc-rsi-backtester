# Streamlit Dependencies

## Streamlit-specific APIs in use
225:st.set_page_config(page_title="BB+KC+RSI Backtester", layout="wide")
229:if "selected_trade" not in st.session_state:
230:    st.session_state.selected_trade = None
231:if "run_ready" not in st.session_state:
232:    st.session_state.run_ready = False
233:if "last_params" not in st.session_state:
234:    st.session_state.last_params = None
235:if "results" not in st.session_state:
236:    st.session_state.results = None
237:if "dirty_params" not in st.session_state:
238:    st.session_state.dirty_params = False
239:if "selected_preset" not in st.session_state:
240:    st.session_state.selected_preset = "Custom"
241:if "preset_params" not in st.session_state:
242:    st.session_state.preset_params = DEFAULT_PRESET.copy()
243:if "discovery_running" not in st.session_state:
244:    st.session_state.discovery_running = False
245:if "discovery_results" not in st.session_state:
246:    st.session_state.discovery_results = None
247:if "selected_winning_strategy" not in st.session_state:
248:    st.session_state.selected_winning_strategy = None
303:    if key not in st.session_state:
304:        st.session_state[key] = default_val
349:                st.session_state[actual_key] = transform(value)
353:                    st.session_state[widget_key] = value
363:    if st.session_state.selected_preset != "Custom":
364:        preset = STRATEGY_PRESETS.get(st.session_state.selected_preset, {})
366:    return st.session_state.preset_params.get(key, default)
370:@st.cache_data(show_spinner=False)
381:@st.cache_data(show_spinner=False)
434:with st.sidebar:
438:    run_top = st.button("Run Backtest", type="primary", use_container_width=True, key="run_top")
448:    selected_preset_idx = st.selectbox(
456:    if selected_preset != st.session_state.selected_preset:
457:        st.session_state.selected_preset = selected_preset
459:            st.session_state.preset_params = STRATEGY_PRESETS[selected_preset].copy()
461:            sync_widgets_to_preset(st.session_state.preset_params)
462:        st.session_state.dirty_params = True
471:    with st.expander("Data & Timeframe", expanded=True):
473:        exchange = st.selectbox(
476:            index=exchange_options.index(st.session_state.w_exchange) if st.session_state.w_exchange in exchange_options else 3,
479:        symbol = st.text_input("Symbol", key="w_symbol")
481:        timeframe = st.selectbox(
484:            index=timeframe_options.index(st.session_state.w_timeframe) if st.session_state.w_timeframe in timeframe_options else 0,
490:        date_range = st.date_input(
492:            value=(st.session_state.w_start_date, st.session_state.w_end_date),
497:            st.session_state.w_start_date = start_date
498:            st.session_state.w_end_date = end_date
500:            start_date = st.session_state.w_start_date
501:            end_date = st.session_state.w_end_date
504:    with st.expander("Indicators (BB, KC, RSI)"):
506:        bb_len = st.number_input("BB length", 5, 200, key="w_bb_len")
507:        bb_std = st.number_input("BB std dev", 1.0, 4.0, key="w_bb_std", step=0.1)
509:        bb_basis_type = st.selectbox(
511:            index=bb_basis_options.index(st.session_state.w_bb_basis_type),
516:        kc_ema_len = st.number_input("KC EMA/SMA length (mid)", 5, 200, key="w_kc_ema_len")
517:        kc_atr_len = st.number_input("KC ATR length", 5, 200, key="w_kc_atr_len")
518:        kc_mult = st.number_input("KC ATR multiplier", 0.5, 5.0, key="w_kc_mult", step=0.1)
519:        kc_mid_is_ema = st.checkbox("KC mid uses EMA (uncheck = SMA)", key="w_kc_mid_is_ema")
523:        rsi_len_30m = st.number_input("RSI length", 5, 100, key="w_rsi_len_30m")
525:        rsi_smoothing_type = st.selectbox(
527:            index=rsi_smoothing_options.index(st.session_state.w_rsi_smoothing_type),
530:        rsi_ma_len = st.number_input("RSI MA length", 2, 100, key="w_rsi_ma_len")
532:        rsi_ma_type = st.selectbox(
534:            index=rsi_ma_options.index(st.session_state.w_rsi_ma_type),
539:    with st.expander("Entry & Exit Logic"):
541:        rsi_min = st.number_input("RSI minimum (entry)", 0, 100, key="w_rsi_min")
542:        rsi_ma_min = st.number_input("RSI MA minimum (entry)", 0, 100, key="w_rsi_ma_min")
543:        use_rsi_relation = st.checkbox(
548:        rsi_relation = st.selectbox(
550:            index=rsi_relation_options.index(st.session_state.w_rsi_relation),
555:        entry_band_mode = st.selectbox(
558:            index=entry_band_options.index(st.session_state.w_entry_band_mode),
564:        exit_channel = st.selectbox(
566:            index=exit_channel_options.index(st.session_state.w_exit_channel),
570:        exit_level = st.selectbox(
572:            index=exit_level_options.index(st.session_state.w_exit_level),
577:    with st.expander("Risk Management & Capital"):
578:        cash = st.number_input("Starting cash", 100, 1_000_000_000, 10_000, 100, key="w_cash")
579:        commission = st.number_input("Commission (fraction)", 0.0, 0.01, 0.001, 0.0001, key="w_commission")
582:        trade_mode = st.selectbox(
585:            index=trade_mode_options.index(st.session_state.w_trade_mode) if st.session_state.w_trade_mode in trade_mode_options else 0,
589:        use_stop = st.checkbox("Enable stop loss", key="w_use_stop")
591:        stop_mode = st.selectbox(
594:            index=stop_mode_options.index(st.session_state.w_stop_mode) if st.session_state.w_stop_mode in stop_mode_options else 0,
599:            stop_pct = st.number_input(
607:            stop_atr_mult = st.number_input(
615:        use_trailing = st.checkbox("Enable trailing stop", key="w_use_trailing")
616:        trail_pct = st.number_input(
623:        max_bars_in_trade = st.number_input(
630:        daily_loss_limit = st.number_input(
637:        risk_per_trade_pct = st.number_input(
647:            max_leverage = st.number_input("Max leverage", 1.0, 125.0, 
650:            maintenance_margin_pct = st.number_input("Maintenance margin %", 0.1, 50.0, 
655:                enable_max_margin_util = st.checkbox("Limit margin utilization?", value=False, key="w_enable_max_margin_util")
656:                max_margin_utilization = st.number_input("Max margin utilization %", 10.0, 100.0, 
667:    run_bottom = st.button("Run Backtest", type="primary", use_container_width=True, key="run_bottom")
726:if st.session_state.last_params is not None and st.session_state.last_params != params_now:
727:    st.session_state.dirty_params = True
732:    with st.spinner("Fetching data and running backtest..."):
739:            st.stop()
741:        st.session_state.results = {"df": df, "stats": stats, "ds": ds, "trades": trades, "equity_curve": equity_curve}
742:        st.session_state.last_params = params_now
743:        st.session_state.dirty_params = False
744:        st.session_state.selected_trade = None
745:        st.session_state.run_ready = True
750:if st.session_state.get("run_ready", False) and st.session_state.results is not None:
752:    if st.session_state.dirty_params:
755:    results = st.session_state.results
763:    tab_dashboard, tab_trades, tab_analysis = st.tabs(["📊 Dashboard", "📝 Trades & Diagnostics", "🔬 Analysis & Discovery"])
778:        c1, c2, c3, c4, c5, c6 = st.columns(6)
789:        chart_col1, chart_col2 = st.columns([1, 1])
791:            show_candles = st.checkbox("Show Candlesticks", value=True, key="show_candles")
793:            lock_rsi_y = st.checkbox("Lock RSI Y-axis (0-100)", value=True, key="lock_rsi_y")
803:        selected_trade = st.session_state.get("selected_trade")
962:        st.plotly_chart(fig, use_container_width=True)
979:            col1, col2 = st.columns([2, 1])
1019:                    st.session_state.selected_trade = sel[0]
1029:                        st.dataframe(diag, use_container_width=True, height=400)
1032:                        st.download_button(
1045:        mode = st.radio("Select Tool", ["Parameter Optimization", "Strategy Discovery", "Leaderboard", "Pattern Recognition"], horizontal=True, key="analysis_mode")
1053:            opt_col1, opt_col2 = st.columns(2)
1059:                rsi_range = st.slider(
1068:                stop_range = st.slider(
1078:                band_range = st.slider(
1090:                opt_validation_mode = st.radio(
1098:                opt_metric = st.selectbox(
1107:                grid_steps = st.slider(
1116:                include_entry_modes = st.checkbox("Test all entry band modes", value=True, key="opt_include_entry_modes")
1117:                include_exit_levels = st.checkbox("Test both exit levels (mid/lower)", value=False, key="opt_include_exit_levels")
1120:                min_trades = st.number_input(
1129:                min_pf = st.number_input(
1134:                min_wr = st.number_input(
1139:                max_dd = st.number_input(
1144:                min_total_ret = st.number_input(
1152:                    wf_train_days = st.number_input("Train window (days)", min_value=30, max_value=3650, value=180, step=30, key="wf_train_days")
1153:                    wf_test_days = st.number_input("Test window (days)", min_value=7, max_value=365, value=30, step=7, key="wf_test_days")
1154:                    wf_max_folds = st.number_input("Max folds (0 = all)", min_value=0, max_value=200, value=8, step=1, key="wf_max_folds")
1194:            if st.button("🚀 Run Optimization", type="primary"):
1198:                progress_bar = st.progress(0, text="Starting optimization...")
1199:                status_text = st.empty()
1205:                with st.spinner("Running grid search..."):
1237:                                st.dataframe(fold_results, use_container_width=True)
1250:                                st.download_button(
1285:                                st.dataframe(
1300:                                anal_col1, anal_col2 = st.columns(2)
1313:                                st.download_button(
1332:            perf_col1, perf_col2, perf_col3 = st.columns(3)
1336:                disc_n_workers = st.slider(
1344:                disc_use_parallel = st.checkbox("Enable Parallel Processing", value=True, key="disc_use_parallel")
1347:                disc_skip_tested = st.checkbox("Skip tested combinations", value=True, key="disc_skip_tested")
1349:            disc_col1, disc_col2, disc_col3 = st.columns(3)
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
1413:                progress_bar = st.progress(0, text="Starting discovery...")
1440:                    st.session_state.discovery_results = disc_result
1456:                lb_c1, lb_c2, lb_c3, lb_c4 = st.columns(4)
1463:                lb_sort_col1, lb_sort_col2 = st.columns(2)
1465:                    lb_sort_by = st.selectbox(
1471:                    lb_top_n = st.selectbox("Show top", [10, 25, 50, 100], key="lb_top_n")
1526:                                param_col1, param_col2 = st.columns(2)
1540:                                if st.button("📥 Load This Strategy", key="load_strategy_btn"):
1542:                                    st.session_state.preset_params.update(selected_strategy.params)
1543:                                    st.session_state.selected_preset = "Custom"
1544:                                    st.session_state.dirty_params = True
1545:                                    st.session_state.selected_winning_strategy = selected_strategy
1549:                                    st.rerun()
1553:                    st.download_button(
1563:            if st.button("🔄 Analyze Patterns", key="analyze_patterns_btn"):
1564:                with st.spinner("Analyzing winning strategies..."):

## Replacement mapping (Dash)
- st.session_state -> dcc.Store (client) or server-side cache; use callback inputs/outputs for state sync
- st.cache_data -> flask-caching or diskcache for data/backtest memoization
- st.sidebar -> html.Div with fixed column or dbc.Offcanvas/Sidebar (if using Dash Bootstrap Components)
- st.expander -> dbc.Collapse or custom details/summary
- st.tabs -> dcc.Tabs
- st.columns -> html.Div with CSS grid or dbc.Row/Col
- st.rerun -> trigger callback updates; use dcc.Interval or callbacks with state
- st.stop -> dash.exceptions.PreventUpdate
- st.spinner/progress/empty -> dcc.Loading + progress component
- st.download_button -> dcc.Download
- st.plotly_chart -> dcc.Graph
- st.dataframe -> dash_table.DataTable
- st.selectbox/radio/slider/number_input/text_input/checkbox/multiselect/date_input -> dcc.Dropdown/dcc.RadioItems/dcc.Slider/dcc.Input/dcc.Checklist/dcc.DatePickerRange
