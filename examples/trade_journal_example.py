"""Example: paper-trade review loop using src.utils.trade_journal.

Run from the repository root after checking out the xinyu-risk-managed-portfolio branch:

    python examples/trade_journal_example.py

This example uses manual observations. Replace the sample prices with your own
paper-trading record, e.g. T+3/T+5/T+10 closes from your brokerage app.
"""

from src.utils.trade_journal import PriceObservation, TradeRecord, review_to_dict, review_trade


trade = TradeRecord(
    ticker="002281.SZ",
    entry_date="2026-04-20",
    entry_price=124.00,
    side="long",
    quantity=100,
    signal="bullish",
    confidence=66,
    stop_loss=114.08,
    take_profit_1=141.86,
    take_profit_2=153.76,
    max_holding_days=10,
    notes="Example based on a model signal. Replace with your actual paper trade.",
)

observations = [
    PriceObservation(trade_date="2026-04-21", close=128.00, high=130.00, low=123.50),
    PriceObservation(trade_date="2026-04-24", close=142.00, high=145.00, low=126.00),
    PriceObservation(trade_date="2026-04-27", close=156.00, high=158.00, low=140.00),
    PriceObservation(trade_date="2026-05-05", close=180.00, high=185.00, low=150.00),
]

review = review_trade(trade, observations)
print(review_to_dict(review))
