# BB + KC + RSI Short Strategy Backtester

A comprehensive backtesting application for a **short-only mean-reversion strategy** that combines Bollinger Bands (BB), Keltner Channels (KC), and RSI indicators. Built with Streamlit for an interactive UI and supports multiple cryptocurrency exchanges via CCXT.

## Strategy Overview

This backtester implements a **short-selling mean-reversion strategy** based on the following logic:

### Entry Conditions (Short)
1. **Band Touch**: Price touches upper Bollinger Band and/or Keltner Channel (configurable: Either, KC only, BB only, or Both)
2. **RSI Threshold**: RSI (30-minute resampled) >= minimum threshold (default: 70)
3. **RSI MA Threshold**: RSI Moving Average >= minimum threshold (default: 70)
4. **RSI Relation**: Optional comparison between RSI and RSI MA (e.g., RSI >= RSI_MA)

### Exit Conditions
- **Signal Exit**: Price drops to BB/KC mid or lower band
- **Stop Loss**: Fixed percentage or ATR-based stop
- **Trailing Stop**: Dynamic stop that follows price movement
- **Time Stop**: Maximum bars in trade limit
- **Liquidation**: Margin/Futures mode liquidation when equity threshold breached

## Architecture

```
bb-kc-rsi-backtester/
├── app/
│   ├── backtest/
│   │   └── engine.py      # Core backtesting logic, strategy loop, stats
│   ├── core/
│   │   ├── data.py        # OHLCV data fetching via CCXT
│   │   ├── indicators.py  # Technical indicators (BB, KC, RSI, ATR)
│   │   ├── presets.py     # Strategy preset configurations
│   │   └── utils.py       # Shared utilities
│   ├── optimization/
│   │   └── grid_search.py # Parameter optimization via grid search
│   └── ui/
│       └── app.py         # Streamlit dashboard
├── docs/
│   ├── STRATEGIES.md      # Strategy preset documentation
│   └── OPTIMIZATION.md    # Optimization guide
├── docker/
│   └── streamlit.Dockerfile
├── requirements/
│   ├── base.txt           # Core dependencies
│   └── streamlit.txt      # UI dependencies
├── docker-compose.yml
├── railway.toml           # Railway deployment config
└── README.md
```

## Strategy Presets

The backtester includes 8 pre-configured strategy presets optimized for different trading objectives:

| Preset | Focus | Risk Level | Description |
|--------|-------|------------|-------------|
| **Conservative** | High win rate | Low | Strict entry conditions (both bands), tight stops |
| **Low Drawdown** | Capital preservation | Very Low | Minimizes maximum drawdown |
| **Aggressive** | High profit factor | High | Looser entries, ATR stops, trailing |
| **High Profit Factor** | Maximize PF | Medium | Optimized for best risk/reward |
| **Scalping** | Quick trades | Medium | Short holding, mid-band exits |
| **Swing** | Large moves | Medium | Longer holds, lower band exits |
| **Momentum Burst** | Extreme RSI | Medium | Very high RSI threshold (80+) |
| **Mean Reversion** | Classic BB | Medium | Standard BB mean reversion |

### Using Presets

1. Select a preset from the "Strategy Preset" dropdown at the top of the sidebar
2. Parameters are automatically populated
3. Modify individual parameters if desired
4. Run the backtest

For detailed preset descriptions, see [docs/STRATEGIES.md](docs/STRATEGIES.md).

## Parameter Optimization

The backtester includes a grid search optimizer to find optimal parameter configurations:

### Quick Start

1. Run a backtest with your current parameters
2. Expand the "Parameter Optimization" section
3. Set parameter ranges to test
4. Choose optimization metric (profit_factor recommended)
5. Click "Run Optimization"
6. Review top configurations

### Optimization Metrics

- **profit_factor**: Gross profit / gross loss (recommended)
- **sharpe_ratio**: Risk-adjusted return
- **sortino_ratio**: Downside-adjusted return
- **win_rate**: Percentage of winning trades

### Avoiding Overfitting

- Use at least 6 months of historical data
- Require minimum 30+ trades for statistical significance
- Validate results on out-of-sample data
- Prefer stable parameter regions over isolated peaks

For detailed optimization guidance, see [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

### Data Flow

```
User Input → Streamlit UI → Data Fetcher (CCXT) → Exchange API
                               ↓
                          Raw OHLCV Data
                               ↓
                      Indicator Calculation (BB, KC, RSI)
                               ↓
                      Backtest Engine (Strategy Loop)
                               ↓
                      Stats + Trades + Dataset
                               ↓
                      Charts + Metrics + Trade Table
```

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bb-kc-rsi-backtester
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/base.txt -r requirements/streamlit.txt
   ```

4. **Run the application**
   ```bash
   cd app
   streamlit run ui/app.py
   ```

### Docker

```bash
docker compose up --build
```

The application will be available at `http://localhost:8501`

## Configuration Parameters

### Data Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| Exchange | Data source (coinbase, kraken, gemini, bitstamp, binanceus) | bitstamp |
| Symbol | Trading pair | BTC/USD |
| Timeframe | Candle interval (30m, 1h, 4h, 1d) | 30m |
| Date Range | Backtest window | 2023-10-01 to today |

### Bollinger Bands
| Parameter | Description | Default |
|-----------|-------------|---------|
| BB Length | Period for moving average | 20 |
| BB Std Dev | Standard deviation multiplier | 2.0 |
| BB Basis Type | Moving average type (SMA/EMA) | SMA |

### Keltner Channel
| Parameter | Description | Default |
|-----------|-------------|---------|
| KC EMA Length | Period for mid-line | 20 |
| KC ATR Length | ATR calculation period | 14 |
| KC Multiplier | ATR multiplier for bands | 2.0 |
| KC Mid Type | Mid-line type (EMA/SMA) | EMA |

### RSI (30-minute Resampled)
| Parameter | Description | Default |
|-----------|-------------|---------|
| RSI Length | RSI calculation period | 14 |
| RSI Smoothing | Smoothing type (EMA/SMA/RMA) | EMA |
| RSI MA Length | Moving average of RSI period | 10 |
| RSI MA Type | RSI MA smoothing type | SMA |

### Entry Conditions
| Parameter | Description | Default |
|-----------|-------------|---------|
| RSI Minimum | Minimum RSI for entry | 70 |
| RSI MA Minimum | Minimum RSI MA for entry | 70 |
| Use RSI Relation | Enable RSI vs RSI MA comparison | true |
| RSI Relation | Comparison operator (<, <=, >, >=) | >= |
| Entry Band Mode | Which band must be touched (Either, KC, BB, Both) | Either |

### Exit Conditions
| Parameter | Description | Default |
|-----------|-------------|---------|
| Exit Channel | Channel for exit signal (BB/KC) | BB |
| Exit Level | Exit at mid or lower band | mid |

### Risk Management
| Parameter | Description | Default |
|-----------|-------------|---------|
| Starting Cash | Initial capital | $10,000 |
| Commission | Trading fee (fraction) | 0.001 |
| Trade Mode | Simple (1x) or Margin/Futures | Simple |
| Enable Stop Loss | Use stop loss | false |
| Stop Type | Fixed % or ATR-based | Fixed % |
| Fixed Stop % | Stop loss percentage | 2.0% |
| ATR Stop Multiplier | ATR multiplier for stop | 2.0 |
| Enable Trailing Stop | Use trailing stop | false |
| Trailing Stop % | Trail distance | 1.0% |
| Max Bars in Trade | Time-based exit | 100 |
| Daily Loss Limit % | Max daily loss before stopping | 3.0% |
| Risk Per Trade % | Equity risked per trade | 1.0% |

### Margin/Futures Settings (when enabled)
| Parameter | Description | Default |
|-----------|-------------|---------|
| Max Leverage | Maximum leverage allowed | 5.0x |
| Maintenance Margin % | Liquidation threshold | 0.5% |
| Max Margin Utilization % | Cap on margin usage | 70% |

## Output Metrics

### Performance Statistics
- **Trades**: Total number of trades executed
- **Win Rate**: Percentage of profitable trades
- **Avg Return**: Mean return per trade
- **Median Return**: Median return per trade
- **Best/Worst Return**: Extremes of trade returns
- **Profit Factor**: Gross profit / Gross loss
- **Total Equity Return**: Overall portfolio return
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside-adjusted return

### Trade Details
Each trade includes:
- Entry/Exit time and bar index
- Entry/Exit prices
- Trade side (Short)
- Exit reason (signal_exit, stop_loss, time_stop, liquidation)
- Size and notional value
- Realized P&L
- R-multiple (profit/risk ratio)
- Effective leverage
- Margin utilization

## Deployment

### Railway

The application includes a `railway.toml` configuration for easy deployment:

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Dockerfile
3. The app will be deployed with health checks enabled

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Server port | 8501 |
| PYTHONPATH | Python module path | /workspace/app |

## Dependencies

### Core (`requirements/base.txt`)
- pandas - Data manipulation
- numpy - Numerical operations
- ccxt - Exchange connectivity
- backtesting - Backtest utilities
- python-dotenv - Environment management
- bokeh - Visualization support
- matplotlib - Plotting

### UI (`requirements/streamlit.txt`)
- streamlit - Web application framework
- plotly - Interactive charts
- streamlit-aggrid - Advanced data tables

## Usage Examples

### Basic Backtest
1. Select exchange and symbol (e.g., BTC/USD on Bitstamp)
2. Choose timeframe (30m recommended for this strategy)
3. Set date range for historical data
4. Adjust BB/KC/RSI parameters or use defaults
5. Click "Run Backtest"
6. Review charts, metrics, and trade table

### Using Strategy Presets
1. Select a preset from the dropdown (e.g., "High Profit Factor")
2. Review the preset description and parameters
3. Optionally customize individual parameters
4. Click "Run Backtest" to see performance
5. Compare different presets to find what works best

### Risk-Adjusted Backtest
1. Enable "Margin / Futures" trade mode
2. Set max leverage (e.g., 5x)
3. Enable stop loss with ATR-based stops
4. Set risk per trade to 1-2%
5. Enable daily loss limit
6. Run backtest to see leverage-adjusted returns

### Parameter Optimization Workflow
1. Run initial backtest with a preset (e.g., "Conservative")
2. Open the "Parameter Optimization" expander
3. Set parameter ranges:
   - RSI Min: 65-75
   - Stop %: 1.5-3.0%
   - Band Mult: 1.8-2.2
4. Select "profit_factor" as optimization target
5. Click "Run Optimization"
6. Review top configurations in results table
7. Apply best configuration by manually adjusting parameters
8. Validate on a different date range

### Finding Optimal Configuration
1. Start with "Aggressive" preset for profit-focused trading
2. Run optimization with these ranges:
   - RSI Min: 62-72 (lower = more trades)
   - Entry Mode: Test all (Either, KC, BB, Both)
   - Stop %: 2.0-4.0%
3. Look for configurations with:
   - Profit Factor > 1.5
   - At least 20 trades
   - Max Drawdown < 20%
4. Test the best config on a different time period

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- CCXT library for exchange connectivity
- Streamlit for the interactive UI framework
- Plotly for charting capabilities
