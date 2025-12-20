# OHLCV Cache Schema

Database: data/market_data.db
Table: ohlcv_cache

Columns:
- exchange TEXT
- symbol TEXT
- timeframe TEXT
- ts INTEGER (epoch ms)
- open REAL
- high REAL
- low REAL
- close REAL
- volume REAL

Primary Key:
- (exchange, symbol, timeframe, ts)

Indices:
- idx_ohlcv_lookup on (exchange, symbol, timeframe, ts)
