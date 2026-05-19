"""
Portfolio-Level Backtester for Dual-Strategy (Momentum + Mean Reversion) System

Improvements over the previous revision:
  1. auto_adjust=True so splits/dividends don't trigger phantom signals/stops.
  2. Momentum fills require the breakout level to actually trade (no phantom fills).
  3. STOP_GAP and STOP_LOSS both pay the wider stop-slippage penalty.
  4. Stale-bar equity falls back to the last available close (not entry price);
     positions whose data goes missing for too long are force-closed.
  5. Survivorship-bias warning printed up front (current Wikipedia constituents).
  6. yfinance batch download with explicit failure logging.
  7. Liquidity cap: position size is capped at a fraction of 20-day $ volume.
  8. Cost-sensitivity: optionally re-run with frictions disabled to size the edge.
     Walk-forward stub provided for parameter robustness testing.
  9. Trading-day-based time stop, normalized signal-strength ranking, safer tz
     handling, optional CSV trade log.

Usage:
    python portfolio_backtester.py
"""

import io
import logging
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Be specific instead of muting everything.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='yfinance')

logger = logging.getLogger('portfolio_backtester')


# ----------------------------- Config -----------------------------

@dataclass
class PortfolioConfig:
    # Capital and risk
    starting_capital: float = 50_000.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.20
    max_positions: int = 5

    # Momentum params
    momentum_lookback: int = 20
    breakout_threshold: float = 0.005
    rvol_threshold: float = 1.8
    momentum_atr_mult: float = 2.0
    momentum_target_atr_mult: float = 4.0
    momentum_max_gap_pct: float = 0.05  # skip if open is >5% above breakout

    # Mean reversion params
    zscore_lookback: int = 20
    zscore_entry: float = -2.0
    zscore_exit: float = 0.0
    reversion_atr_mult: float = 1.5
    low_volume_threshold: float = 0.7
    rsi_threshold: float = 30.0
    reversion_max_hold_days: int = 10  # tighter time stop for fades

    # Filters
    min_price: float = 5.0
    max_price: float = 500.0
    max_hold_days: int = 30  # trading days, not calendar days
    cooldown_days: int = 5   # trading days

    # Regime / trend filters
    use_market_regime: bool = True
    market_regime_symbol: str = 'SPY'
    market_regime_sma: int = 200
    reversion_requires_uptrend: bool = True

    # Trading costs
    apply_costs: bool = True
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    slippage_bps: float = 5.0
    stop_slippage_bps: float = 15.0

    # Liquidity / data hygiene
    liquidity_cap_pct: float = 0.01      # cap order at 1% of 20-day $ volume
    max_missing_bars: int = 5            # force-close after N missing bars
    auto_adjust: bool = True             # adjust for splits & dividends

    # Analytics
    risk_free_rate_annual: float = 0.0   # subtracted from daily returns for Sharpe
    trade_log_path: Optional[str] = None # if set, writes a CSV trade log


# ----------------------------- Data fetching -----------------------------

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone if present; idempotent."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_all_data(symbols: list, start_date: str, end_date: str,
                   auto_adjust: bool = True, chunk_size: int = 100) -> tuple[dict, list]:
    """Batch-fetch OHLCV using yfinance.download in chunks. Returns (data, failed)."""
    print(f"Fetching {len(symbols)} symbols from {start_date} to {end_date}...")
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # yf.download handles batching internally but very large requests still time out.
    # Chunking gives us better resilience and progress visibility.
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            df = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                group_by='ticker',
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"  Chunk {i // chunk_size + 1} failed entirely: {e}")
            failed.extend(chunk)
            continue

        if df is None or df.empty:
            failed.extend(chunk)
            continue

        # Single-ticker chunks return a flat DataFrame; multi-ticker returns a
        # MultiIndex with the ticker as the outer level.
        if len(chunk) == 1:
            sym = chunk[0]
            sub = df.dropna(how='all')
            if len(sub) >= 50 and all(c in sub.columns for c in cols):
                data[sym] = _normalize_index(sub[cols].copy())
            else:
                failed.append(sym)
        else:
            top_level = df.columns.get_level_values(0).unique()
            for sym in chunk:
                if sym not in top_level:
                    failed.append(sym)
                    continue
                try:
                    sub = df[sym].dropna(how='all')
                except KeyError:
                    failed.append(sym)
                    continue
                if len(sub) < 50 or not all(c in sub.columns for c in cols):
                    failed.append(sym)
                    continue
                data[sym] = _normalize_index(sub[cols].copy())

        print(f"  Progress: {min(i + chunk_size, len(symbols))}/{len(symbols)} "
              f"(ok={len(data)}, failed={len(failed)})")

    print(f"  Successfully fetched {len(data)} symbols. {len(failed)} failed.")
    if failed:
        # Log a small sample so the user can see which names dropped out.
        sample = ', '.join(failed[:10])
        more = '' if len(failed) <= 10 else f' (+{len(failed) - 10} more)'
        print(f"  Failed sample: {sample}{more}")
    return data, failed


def fetch_single(symbol: str, start_date: str, end_date: str,
                 auto_adjust: bool = True) -> Optional[pd.DataFrame]:
    """Single-symbol fetch (used for SPY)."""
    try:
        df = yf.Ticker(symbol).history(start=start_date, end=end_date,
                                       auto_adjust=auto_adjust)
        if df.empty or len(df) < 50:
            return None
        df = _normalize_index(df)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


# ----------------------------- Indicators -----------------------------

def calculate_indicators(df: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    df = df.copy()

    tr0 = (df['High'] - df['Low']).abs()
    tr1 = (df['High'] - df['Close'].shift(1)).abs()
    tr2 = (df['Low'] - df['Close'].shift(1)).abs()
    df['tr'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    df['atr_14'] = df['tr'].rolling(14).mean()

    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()

    # RSI-14 (Wilder's)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['high_lookback'] = df['High'].rolling(config.momentum_lookback).max()
    df['breakout_level'] = df['high_lookback'].shift(1) * (1 + config.breakout_threshold)
    df['avg_volume_20'] = df['Volume'].rolling(20).mean()
    df['rvol'] = df['Volume'] / df['avg_volume_20']
    # Liquidity proxy used for order-size cap.
    df['dollar_volume_20'] = (df['Close'] * df['Volume']).rolling(20).mean()

    df['momentum_signal'] = (
        (df['High'] > df['breakout_level']) &
        (df['rvol'] > config.rvol_threshold) &
        (df['Close'] > df['sma_20'])
    )

    df['mean_20'] = df['Close'].rolling(config.zscore_lookback).mean()
    df['std_20'] = df['Close'].rolling(config.zscore_lookback).std()
    df['zscore'] = np.where(df['std_20'] > 0, (df['Close'] - df['mean_20']) / df['std_20'], 0)
    df['low_volume'] = df['Volume'] < (df['avg_volume_20'] * config.low_volume_threshold)

    reversion_base = (
        (df['zscore'] < config.zscore_entry) &
        df['low_volume'] &
        (df['rsi_14'] < config.rsi_threshold)
    )
    if config.reversion_requires_uptrend:
        reversion_base = reversion_base & (df['Close'] > df['sma_200'])
    df['reversion_signal'] = reversion_base

    return df


def build_market_regime(spy_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    sma = spy_df['Close'].rolling(config.market_regime_sma).mean()
    return (spy_df['Close'] > sma).rename('regime_on')


# ----------------------------- Trade and Pending -----------------------------

@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_loss: float
    target: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    exit_reason: Optional[str] = None
    bars_held: int = 0
    missing_bars: int = 0
    last_known_close: float = 0.0  # for stale-data equity marking

    @property
    def position_value(self) -> float:
        return self.shares * self.entry_price


@dataclass
class PendingEntry:
    symbol: str
    strategy: str
    signal_date: pd.Timestamp
    signal_close: float
    breakout_level: float
    atr: float
    mean_20: float
    rvol: float
    zscore: float
    dollar_volume_20: float
    signal_strength: float  # normalized cross-strategy (z-units above threshold)


# ----------------------------- Cost helpers -----------------------------

def apply_slippage(price: float, side: str, bps: float, enabled: bool = True) -> float:
    if not enabled or bps <= 0:
        return price
    adj = price * (bps / 10_000.0)
    return price + adj if side == 'buy' else price - adj


def commission_for(shares: int, config: PortfolioConfig) -> float:
    if not config.apply_costs:
        return 0.0
    return max(config.commission_min, shares * config.commission_per_share)


# ----------------------------- Position sizing -----------------------------

def size_position(equity: float, entry: float, stop: float,
                  dollar_volume_20: float, config: PortfolioConfig) -> int:
    if stop >= entry or entry <= 0:
        return 0
    risk_amt = equity * config.risk_per_trade
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    shares = int(risk_amt / risk_per_share)

    # Cap at max position size relative to account equity.
    max_value_acct = equity * config.max_position_pct
    if shares * entry > max_value_acct:
        shares = int(max_value_acct / entry)

    # Cap at fraction of average daily $ volume to model real liquidity.
    if dollar_volume_20 and dollar_volume_20 > 0 and config.liquidity_cap_pct > 0:
        max_value_liq = dollar_volume_20 * config.liquidity_cap_pct
        if shares * entry > max_value_liq:
            shares = int(max_value_liq / entry)

    return max(0, shares)


# ----------------------------- Backtester -----------------------------

class PortfolioBacktester:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.cash = config.starting_capital
        self.open_positions: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []
        self.pending_entries: list[PendingEntry] = []
        # cooldown stores the trading-day INDEX of the stop-out, not the date.
        self.cooldown: dict[str, int] = {}
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._date_to_idx: dict[pd.Timestamp, int] = {}

    # ---- main run ----

    def run(self, data: dict, start_date: str, end_date: str,
            spy_df: Optional[pd.DataFrame] = None) -> dict:

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        print("Computing indicators...")
        indicators = {sym: calculate_indicators(df, self.config) for sym, df in data.items()}

        regime = None
        if self.config.use_market_regime and spy_df is not None:
            regime = build_market_regime(spy_df, self.config)

        # Use SPY's calendar as the master trading calendar when available — it's
        # a clean US-equity calendar. Fall back to the union if not.
        if spy_df is not None:
            all_dates = sorted(d for d in spy_df.index if start_dt <= d <= end_dt)
        else:
            all_dates = sorted(set().union(*[set(ind.index) for ind in indicators.values()]))
            all_dates = [d for d in all_dates if start_dt <= d <= end_dt]
        self._date_to_idx = {d: i for i, d in enumerate(all_dates)}

        print(f"Simulating {len(all_dates)} trading days across {len(indicators)} symbols...")

        for date in all_dates:
            self._process_pending_entries(date, indicators)
            self._manage_positions(date, indicators)
            regime_on = True
            if regime is not None and date in regime.index:
                regime_on = bool(regime.loc[date])
            self._generate_signals(date, indicators, regime_on)
            self._record_equity(date, indicators)

        self._force_close_all(all_dates[-1] if all_dates else end_dt, indicators)

        return self._summary(start_dt, end_dt, spy_df)

    # ---- step handlers ----

    def _process_pending_entries(self, date: pd.Timestamp, indicators: dict):
        if not self.pending_entries:
            return

        self.pending_entries.sort(key=lambda p: p.signal_strength, reverse=True)

        for entry in self.pending_entries:
            if entry.symbol in self.open_positions:
                continue
            if len(self.open_positions) >= self.config.max_positions:
                break
            ind = indicators.get(entry.symbol)
            if ind is None or date not in ind.index:
                continue
            today = ind.loc[date]
            if pd.isna(today['Open']) or pd.isna(today.get('High')):
                continue

            if entry.strategy == 'MOMENTUM':
                # The breakout level must actually trade today. Either the open
                # is already above it (gap-up), or the day's high reaches it.
                if today['Open'] >= entry.breakout_level:
                    raw_entry = today['Open']
                elif today['High'] >= entry.breakout_level:
                    raw_entry = entry.breakout_level
                else:
                    # Buy-stop never triggered.
                    continue

                # Don't chase wild gap-ups.
                if today['Open'] > entry.breakout_level * (1 + self.config.momentum_max_gap_pct):
                    continue

                stop = raw_entry - entry.atr * self.config.momentum_atr_mult
                target = raw_entry + entry.atr * self.config.momentum_target_atr_mult
            else:  # REVERSION
                raw_entry = today['Open']
                stop = raw_entry - entry.atr * self.config.reversion_atr_mult
                target = entry.mean_20

            fill_price = apply_slippage(raw_entry, 'buy', self.config.slippage_bps,
                                        enabled=self.config.apply_costs)
            equity = self._current_equity(date, indicators)
            shares = size_position(equity, fill_price, stop,
                                   entry.dollar_volume_20, self.config)
            if shares <= 0:
                continue
            cost = shares * fill_price
            comm = commission_for(shares, self.config)
            if cost + comm > self.cash:
                continue

            self.cash -= (cost + comm)
            trade = Trade(
                symbol=entry.symbol,
                strategy=entry.strategy,
                entry_date=date,
                entry_price=fill_price,
                shares=shares,
                stop_loss=stop,
                target=target,
                commission=comm,
                last_known_close=fill_price,
            )
            self.open_positions[entry.symbol] = trade

        # Pending signals are stale after one bar.
        self.pending_entries = []

    def _manage_positions(self, date: pd.Timestamp, indicators: dict):
        to_close = []
        for sym, pos in self.open_positions.items():
            ind = indicators.get(sym)
            if ind is None or date not in ind.index:
                pos.missing_bars += 1
                if pos.missing_bars >= self.config.max_missing_bars:
                    # Force-close at last known close to avoid silently holding
                    # a symbol that may have delisted/halted.
                    fill = apply_slippage(pos.last_known_close, 'sell',
                                          self.config.stop_slippage_bps,
                                          enabled=self.config.apply_costs)
                    comm = commission_for(pos.shares, self.config)
                    self.cash += pos.shares * fill - comm
                    pos.exit_date = date
                    pos.exit_price = fill
                    pos.commission += comm
                    pos.pnl = (fill - pos.entry_price) * pos.shares - pos.commission
                    pos.exit_reason = 'DATA_GAP'
                    to_close.append(sym)
                continue

            bar = ind.loc[date]
            if pd.isna(bar['Close']):
                pos.missing_bars += 1
                continue

            pos.missing_bars = 0
            pos.last_known_close = float(bar['Close'])
            pos.bars_held += 1

            exit_price = None
            exit_reason = None

            # Order matters: gap-throughs first.
            if bar['Open'] <= pos.stop_loss:
                exit_price = bar['Open']
                exit_reason = 'STOP_GAP'
            elif bar['Low'] <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = 'STOP_LOSS'
            elif bar['Open'] >= pos.target:
                exit_price = bar['Open']
                exit_reason = 'TARGET_GAP'
            elif bar['High'] >= pos.target:
                exit_price = pos.target
                exit_reason = 'TARGET_HIT'
            elif pos.strategy == 'REVERSION' and bar['zscore'] >= self.config.zscore_exit:
                exit_price = bar['Close']
                exit_reason = 'MEAN_REACHED'
            else:
                # Trading-day-based time stop, with tighter limit for reversion.
                limit = (self.config.reversion_max_hold_days
                         if pos.strategy == 'REVERSION'
                         else self.config.max_hold_days)
                if pos.bars_held >= limit:
                    exit_price = bar['Close']
                    exit_reason = 'TIME_EXIT'

            if exit_price is not None:
                # Both gap-through and intraday stops slip more than normal exits.
                slip_bps = (self.config.stop_slippage_bps
                            if exit_reason in ('STOP_LOSS', 'STOP_GAP')
                            else self.config.slippage_bps)
                fill = apply_slippage(exit_price, 'sell', slip_bps,
                                      enabled=self.config.apply_costs)
                comm = commission_for(pos.shares, self.config)
                self.cash += pos.shares * fill - comm
                pos.exit_date = date
                pos.exit_price = fill
                pos.commission += comm
                pos.pnl = (fill - pos.entry_price) * pos.shares - pos.commission
                pos.exit_reason = exit_reason
                if exit_reason in ('STOP_LOSS', 'STOP_GAP'):
                    self.cooldown[sym] = self._date_to_idx.get(date, 0)
                to_close.append(sym)

        for sym in to_close:
            self.closed_trades.append(self.open_positions.pop(sym))

    def _generate_signals(self, date: pd.Timestamp, indicators: dict, regime_on: bool):
        cur_idx = self._date_to_idx.get(date, 0)
        for sym, ind in indicators.items():
            if date not in ind.index:
                continue
            if sym in self.open_positions:
                continue
            cooldown_idx = self.cooldown.get(sym)
            if cooldown_idx is not None and (cur_idx - cooldown_idx) < self.config.cooldown_days:
                continue

            bar = ind.loc[date]
            if pd.isna(bar['Close']) or pd.isna(bar['atr_14']):
                continue
            if bar['Close'] < self.config.min_price or bar['Close'] > self.config.max_price:
                continue

            atr = float(bar['atr_14'])
            if atr <= 0:
                continue

            dvol = float(bar['dollar_volume_20']) if not pd.isna(bar['dollar_volume_20']) else 0.0

            if bar['momentum_signal'] and regime_on:
                # Strength in z-style units: how far above the rvol threshold (in std-equivalents).
                strength = (float(bar['rvol']) - self.config.rvol_threshold)
                self.pending_entries.append(PendingEntry(
                    symbol=sym, strategy='MOMENTUM', signal_date=date,
                    signal_close=float(bar['Close']),
                    breakout_level=float(bar['breakout_level']),
                    atr=atr, mean_20=float(bar['mean_20']),
                    rvol=float(bar['rvol']), zscore=float(bar['zscore']),
                    dollar_volume_20=dvol,
                    signal_strength=strength,
                ))
            elif bar['reversion_signal']:
                # Strength: how many extra std deviations beyond the entry threshold.
                strength = abs(float(bar['zscore'])) - abs(self.config.zscore_entry)
                self.pending_entries.append(PendingEntry(
                    symbol=sym, strategy='REVERSION', signal_date=date,
                    signal_close=float(bar['Close']), breakout_level=0.0,
                    atr=atr, mean_20=float(bar['mean_20']),
                    rvol=float(bar['rvol']), zscore=float(bar['zscore']),
                    dollar_volume_20=dvol,
                    signal_strength=strength,
                ))

    def _record_equity(self, date: pd.Timestamp, indicators: dict):
        self.equity_curve.append((date, self._current_equity(date, indicators)))

    def _current_equity(self, date: pd.Timestamp, indicators: dict) -> float:
        equity = self.cash
        for sym, pos in self.open_positions.items():
            ind = indicators.get(sym)
            if ind is None or date not in ind.index:
                # Use last known close, not entry price — entry-price fallback
                # silently masks losses on stale data.
                equity += pos.shares * pos.last_known_close
                continue
            close = ind.loc[date, 'Close']
            if pd.isna(close):
                equity += pos.shares * pos.last_known_close
            else:
                equity += pos.shares * float(close)
        return equity

    def _force_close_all(self, last_date: pd.Timestamp, indicators: dict):
        for sym, pos in list(self.open_positions.items()):
            ind = indicators.get(sym)
            close = pos.last_known_close
            exit_date = last_date
            if ind is not None:
                last_avail = ind.index[ind.index <= last_date]
                if len(last_avail) > 0:
                    exit_date = last_avail[-1]
                    last_close = ind.loc[exit_date, 'Close']
                    if not pd.isna(last_close):
                        close = float(last_close)
            fill = apply_slippage(close, 'sell', self.config.slippage_bps,
                                  enabled=self.config.apply_costs)
            comm = commission_for(pos.shares, self.config)
            self.cash += pos.shares * fill - comm
            pos.exit_date = exit_date
            pos.exit_price = fill
            pos.commission += comm
            pos.pnl = (fill - pos.entry_price) * pos.shares - pos.commission
            pos.exit_reason = 'END_OF_DATA'
            self.closed_trades.append(pos)
        self.open_positions = {}

    # ---- analytics ----

    def _summary(self, start_dt, end_dt, spy_df) -> dict:
        if not self.closed_trades:
            return {'error': 'No trades executed'}

        eq_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity']).set_index('date')
        eq_df['returns'] = eq_df['equity'].pct_change().fillna(0)
        ending_equity = eq_df['equity'].iloc[-1] if len(eq_df) else self.cash
        total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl is not None)

        wins = [t for t in self.closed_trades if t.pnl is not None and t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl is not None and t.pnl <= 0]
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

        peak = eq_df['equity'].cummax()
        dd = (peak - eq_df['equity']) / peak
        max_dd = dd.max() if len(dd) else 0

        # Sharpe with optional risk-free deduction.
        rf_daily = (1 + self.config.risk_free_rate_annual) ** (1 / 252) - 1
        excess = eq_df['returns'] - rf_daily
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

        years = (end_dt - start_dt).days / 365.25
        cagr = (ending_equity / self.config.starting_capital) ** (1 / years) - 1 if years > 0 else 0

        mom_trades = [t for t in self.closed_trades if t.strategy == 'MOMENTUM']
        rev_trades = [t for t in self.closed_trades if t.strategy == 'REVERSION']

        result = {
            'starting_capital': self.config.starting_capital,
            'ending_capital': round(ending_equity, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round((ending_equity / self.config.starting_capital - 1) * 100, 2),
            'cagr_pct': round(cagr * 100, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'total_trades': len(self.closed_trades),
            'win_rate_pct': round(len(wins) / len(self.closed_trades) * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
            'momentum_trades': len(mom_trades),
            'momentum_win_rate_pct': round(sum(1 for t in mom_trades if t.pnl > 0) / len(mom_trades) * 100, 2) if mom_trades else 0,
            'reversion_trades': len(rev_trades),
            'reversion_win_rate_pct': round(sum(1 for t in rev_trades if t.pnl > 0) / len(rev_trades) * 100, 2) if rev_trades else 0,
            'equity_curve': eq_df,
            'trades': self.closed_trades,
        }

        if spy_df is not None and len(spy_df) > 0:
            spy_window = spy_df.loc[start_dt:end_dt]
            if len(spy_window) > 1:
                spy_ret = spy_window['Close'].iloc[-1] / spy_window['Close'].iloc[0] - 1
                result['spy_buyhold_return_pct'] = round(spy_ret * 100, 2)

        if self.config.trade_log_path:
            self._write_trade_log(self.config.trade_log_path)

        return result

    def _write_trade_log(self, path: str):
        rows = []
        for t in self.closed_trades:
            rows.append({
                'symbol': t.symbol,
                'strategy': t.strategy,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'shares': t.shares,
                'stop_loss': t.stop_loss,
                'target': t.target,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'bars_held': t.bars_held,
                'commission': t.commission,
                'pnl': t.pnl,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Trade log written to {path}")


# ----------------------------- Universe helpers -----------------------------

def get_midcap_tickers() -> list:
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=15).read()
        tables = pd.read_html(io.StringIO(html.decode('utf-8')))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [str(t).replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"  Error fetching midcaps: {e}")
        return []


def get_nasdaq_tickers() -> list:
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=15).read()
        tables = pd.read_html(io.StringIO(html.decode('utf-8')))
        for t in tables:
            if 'Ticker' in t.columns and len(t) > 50:
                return [str(tick).replace('.', '-') for tick in t['Ticker'].tolist()]
            if 'Symbol' in t.columns and len(t) > 50:
                return [str(tick).replace('.', '-') for tick in t['Symbol'].tolist()]
    except Exception as e:
        print(f"  Error fetching nasdaq-100: {e}")
    return []


def print_survivorship_warning():
    print("=" * 70)
    print("SURVIVORSHIP-BIAS WARNING")
    print("=" * 70)
    print("This backtest uses the *current* S&P 400 / Nasdaq-100 constituents")
    print("from Wikipedia. Stocks that were delisted, demoted, or went bankrupt")
    print("during the test window are NOT included. Reported CAGR / Sharpe will")
    print("be biased upward (especially for the momentum sleeve). For research-")
    print("grade results, use a point-in-time constituent dataset.")
    print("=" * 70 + "\n")


# ----------------------------- Reporting -----------------------------

def print_results(result: dict, title: str = "PORTFOLIO BACKTEST RESULTS"):
    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Starting capital:        ${result['starting_capital']:>12,.2f}")
    print(f"Ending capital:          ${result['ending_capital']:>12,.2f}")
    print(f"Total P&L:               ${result['total_pnl']:>12,.2f}")
    print(f"Total return:            {result['total_return_pct']:>12.2f}%")
    print(f"CAGR:                    {result['cagr_pct']:>12.2f}%")
    print(f"Sharpe (annualized):     {result['sharpe']:>12.2f}")
    print(f"Max drawdown:            {result['max_drawdown_pct']:>12.2f}%")
    if 'spy_buyhold_return_pct' in result:
        print(f"SPY buy-and-hold return: {result['spy_buyhold_return_pct']:>12.2f}%")

    print("\n" + "-" * 70)
    print("TRADE STATISTICS")
    print("-" * 70)
    print(f"Total trades:            {result['total_trades']:>12}")
    print(f"Win rate:                {result['win_rate_pct']:>12.2f}%")
    print(f"Profit factor:           {result['profit_factor']:>12.2f}")
    print(f"Avg win:                 ${result['avg_win']:>12,.2f}")
    print(f"Avg loss:                ${result['avg_loss']:>12,.2f}")

    print("\n" + "-" * 70)
    print("BY STRATEGY")
    print("-" * 70)
    print(f"Momentum trades:         {result['momentum_trades']:>12} (win rate {result['momentum_win_rate_pct']:.1f}%)")
    print(f"Reversion trades:        {result['reversion_trades']:>12} (win rate {result['reversion_win_rate_pct']:.1f}%)")

    print("\n" + "-" * 70)
    print("EXIT REASON BREAKDOWN")
    print("-" * 70)
    reasons = {}
    for t in result['trades']:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<20} {count:>6}")


def print_cost_comparison(with_costs: dict, without_costs: dict):
    print("\n" + "=" * 70)
    print("COST SENSITIVITY: WITH vs WITHOUT FRICTIONS")
    print("=" * 70)

    def row(label, key, fmt='{:>10.2f}'):
        a = with_costs.get(key, float('nan'))
        b = without_costs.get(key, float('nan'))
        try:
            delta = b - a
            print(f"  {label:<28} with={fmt.format(a)}  none={fmt.format(b)}  edge={fmt.format(delta)}")
        except Exception:
            pass

    row('Total return (%)', 'total_return_pct')
    row('CAGR (%)', 'cagr_pct')
    row('Sharpe', 'sharpe')
    row('Profit factor', 'profit_factor')
    row('Max drawdown (%)', 'max_drawdown_pct')


# ----------------------------- Walk-forward stub -----------------------------

def walk_forward_run(data: dict, spy_df: Optional[pd.DataFrame],
                     base_config: PortfolioConfig,
                     start: str, end: str,
                     train_months: int = 18, test_months: int = 6) -> list:
    """
    Walk-forward harness skeleton.

    For each (train, test) window, you would:
      1. Run a parameter search over `base_config` on the train window.
      2. Apply the best parameters to the test window.
      3. Stitch together the OOS equity curves.

    This stub just runs the base config on each test window so the plumbing
    works. Replace the inner loop with a real grid/Bayesian search.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    out = []
    cur = start_dt + pd.DateOffset(months=train_months)
    while cur < end_dt:
        train_start = cur - pd.DateOffset(months=train_months)
        test_start = cur
        test_end = min(cur + pd.DateOffset(months=test_months), end_dt)
        # TODO: parameter search on [train_start, test_start) -> best_cfg
        best_cfg = base_config
        bt = PortfolioBacktester(best_cfg)
        result = bt.run(data, start_date=str(test_start.date()),
                        end_date=str(test_end.date()), spy_df=spy_df)
        out.append({
            'train_start': train_start, 'test_start': test_start, 'test_end': test_end,
            'result': result,
        })
        cur = test_end
    return out


# ----------------------------- Main -----------------------------

def main():
    print("\n" + "=" * 70)
    print("PORTFOLIO BACKTESTER  DUAL STRATEGY (REVISED)")
    print("=" * 70)

    config = PortfolioConfig(starting_capital=50_000.0,
                             trade_log_path='trades.csv')

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
    fetch_start = (datetime.now() - timedelta(days=3 * 365 + 250)).strftime('%Y-%m-%d')

    print_survivorship_warning()

    print("Building universe...")
    universe = sorted(set(get_midcap_tickers() + get_nasdaq_tickers()))
    if not universe:
        universe = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'META', 'GOOGL', 'AMZN',
                    'TSLA', 'CROX', 'DECK', 'WMS', 'LECO', 'TOL', 'PEN',
                    'PSTG', 'NOVT', 'HALO', 'ARMK', 'BWA']
    print(f"  Universe size: {len(universe)}")
    # universe = universe[:150]  # uncomment to trim runtime

    data, failed = fetch_all_data(universe, fetch_start, end_date,
                                  auto_adjust=config.auto_adjust)
    print("Fetching SPY for benchmark and regime...")
    spy = fetch_single('SPY', fetch_start, end_date, auto_adjust=config.auto_adjust)

    # Run with realistic costs.
    bt = PortfolioBacktester(config)
    result = bt.run(data, start_date=start_date, end_date=end_date, spy_df=spy)
    print_results(result, "PORTFOLIO BACKTEST RESULTS (with costs)")

    # Cost-sensitivity: run again with frictions disabled to size the edge.
    cfg_no_costs = PortfolioConfig(**{**config.__dict__, 'apply_costs': False,
                                      'trade_log_path': None})
    bt2 = PortfolioBacktester(cfg_no_costs)
    result_no_costs = bt2.run(data, start_date=start_date, end_date=end_date, spy_df=spy)
    print_results(result_no_costs, "PORTFOLIO BACKTEST RESULTS (no frictions)")

    print_cost_comparison(result, result_no_costs)


if __name__ == '__main__':
    main()
