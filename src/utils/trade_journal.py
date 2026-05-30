"""Standalone trade-journal and post-trade review utilities.

This module is intentionally independent from the main hedge-fund workflow. It
can be imported by notebooks, scripts, or a future review agent without changing
existing trading logic.

Purpose
-------
Record and evaluate model signals after entry:
- entry price
- highest/lowest price after entry
- T+3/T+5/T+10 returns
- stop-loss / take-profit hit status
- whether the original signal improved or failed

The functions below do not fetch market data. Pass observed prices into them so
that the same logic can be used with A-shares, US stocks, crypto, or manual paper
trading records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable, Literal


TradeSide = Literal["long", "short"]
TradeOutcome = Literal[
    "take_profit_2_hit",
    "take_profit_1_hit",
    "stop_loss_hit",
    "open",
    "expired_positive",
    "expired_negative",
    "expired_flat",
]


@dataclass
class TradeRecord:
    ticker: str
    entry_date: str
    entry_price: float
    side: TradeSide = "long"
    quantity: int = 0
    signal: str | None = None
    confidence: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    max_holding_days: int | None = None
    notes: str | None = None


@dataclass
class PriceObservation:
    trade_date: str
    close: float
    high: float | None = None
    low: float | None = None


@dataclass
class TradeReview:
    ticker: str
    entry_date: str
    last_date: str
    side: TradeSide
    entry_price: float
    last_close: float
    highest_price: float
    lowest_price: float
    max_return_pct: float
    max_drawdown_pct: float
    last_return_pct: float
    t3_return_pct: float | None
    t5_return_pct: float | None
    t10_return_pct: float | None
    stop_loss_hit: bool
    take_profit_1_hit: bool
    take_profit_2_hit: bool
    outcome: TradeOutcome
    review_note: str


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _pct_return(entry_price: float, exit_price: float, side: TradeSide) -> float:
    if entry_price <= 0:
        return 0.0
    if side == "long":
        return (exit_price / entry_price - 1) * 100
    return (entry_price / exit_price - 1) * 100


def _days_between(start: str, end: str) -> int:
    return (_parse_date(end) - _parse_date(start)).days


def review_trade(trade: TradeRecord, observations: Iterable[PriceObservation]) -> TradeReview:
    """Evaluate a single paper trade or real trade against observed prices.

    Parameters
    ----------
    trade:
        The original model signal or executed trade.
    observations:
        Daily observations after entry. Each item should include close price and
        optionally high/low prices. If high/low are omitted, close is used.
    """
    ordered = sorted(list(observations), key=lambda item: item.trade_date)
    if not ordered:
        raise ValueError("At least one price observation is required")

    highs = [obs.high if obs.high is not None else obs.close for obs in ordered]
    lows = [obs.low if obs.low is not None else obs.close for obs in ordered]
    closes = [obs.close for obs in ordered]

    highest_price = max(highs)
    lowest_price = min(lows)
    last_close = closes[-1]
    last_date = ordered[-1].trade_date

    if trade.side == "long":
        max_return_pct = _pct_return(trade.entry_price, highest_price, trade.side)
        max_drawdown_pct = _pct_return(trade.entry_price, lowest_price, trade.side)
        stop_loss_hit = trade.stop_loss is not None and lowest_price <= trade.stop_loss
        take_profit_1_hit = trade.take_profit_1 is not None and highest_price >= trade.take_profit_1
        take_profit_2_hit = trade.take_profit_2 is not None and highest_price >= trade.take_profit_2
    else:
        max_return_pct = _pct_return(trade.entry_price, lowest_price, trade.side)
        max_drawdown_pct = _pct_return(trade.entry_price, highest_price, trade.side)
        stop_loss_hit = trade.stop_loss is not None and highest_price >= trade.stop_loss
        take_profit_1_hit = trade.take_profit_1 is not None and lowest_price <= trade.take_profit_1
        take_profit_2_hit = trade.take_profit_2 is not None and lowest_price <= trade.take_profit_2

    last_return_pct = _pct_return(trade.entry_price, last_close, trade.side)

    def return_on_or_after(day_count: int) -> float | None:
        candidates = [obs for obs in ordered if _days_between(trade.entry_date, obs.trade_date) >= day_count]
        if not candidates:
            return None
        return round(_pct_return(trade.entry_price, candidates[0].close, trade.side), 4)

    holding_days = _days_between(trade.entry_date, last_date)
    expired = trade.max_holding_days is not None and holding_days >= trade.max_holding_days

    if take_profit_2_hit:
        outcome: TradeOutcome = "take_profit_2_hit"
        review_note = "Excellent signal: second take-profit was reached."
    elif take_profit_1_hit:
        outcome = "take_profit_1_hit"
        review_note = "Good signal: first take-profit was reached; trailing stop should be monitored."
    elif stop_loss_hit:
        outcome = "stop_loss_hit"
        review_note = "Failed or mistimed signal: stop-loss was reached."
    elif expired:
        if last_return_pct > 1:
            outcome = "expired_positive"
            review_note = "Signal expired with positive return."
        elif last_return_pct < -1:
            outcome = "expired_negative"
            review_note = "Signal expired with negative return."
        else:
            outcome = "expired_flat"
            review_note = "Signal expired mostly flat."
    else:
        outcome = "open"
        review_note = "Trade still open; continue tracking against the risk plan."

    return TradeReview(
        ticker=trade.ticker,
        entry_date=trade.entry_date,
        last_date=last_date,
        side=trade.side,
        entry_price=trade.entry_price,
        last_close=last_close,
        highest_price=highest_price,
        lowest_price=lowest_price,
        max_return_pct=round(max_return_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        last_return_pct=round(last_return_pct, 4),
        t3_return_pct=return_on_or_after(3),
        t5_return_pct=return_on_or_after(5),
        t10_return_pct=return_on_or_after(10),
        stop_loss_hit=stop_loss_hit,
        take_profit_1_hit=take_profit_1_hit,
        take_profit_2_hit=take_profit_2_hit,
        outcome=outcome,
        review_note=review_note,
    )


def review_to_dict(review: TradeReview) -> dict:
    return asdict(review)
