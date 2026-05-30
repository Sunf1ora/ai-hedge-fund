import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")

    # Deterministic risk plan added after the LLM decision.
    # These fields make the output more usable for paper-trading and prevent the
    # model from only saying buy/sell without an exit discipline.
    entry_price: float | None = Field(default=None, description="Reference entry price used for the plan")
    stop_loss: float | None = Field(default=None, description="Invalidation price for the trade")
    take_profit_1: float | None = Field(default=None, description="First partial take-profit price")
    take_profit_2: float | None = Field(default=None, description="Second/extended take-profit price")
    trailing_stop_pct: float | None = Field(default=None, description="Trailing stop percentage after take_profit_1 is reached")
    max_holding_days: int | None = Field(default=None, description="Maximum holding window for the signal")
    risk_notes: str | None = Field(default=None, description="Short deterministic risk note")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates risk-managed trade plans for multiple tickers."""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Compress analyst signals to {sig, conf}
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        agent_id=agent_id,
        state=state,
    )
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()},
                             "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def compute_allowed_actions(
        tickers: list[str],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
) -> dict[str, dict[str, int]]:
    """Compute allowed actions and max quantities for each ticker deterministically."""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_requirement = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(
            ticker,
            {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0},
        )
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)

        # Start with zeros
        actions = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # Long side
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # Short side
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_qty > 0:
            if margin_requirement <= 0.0:
                # If margin requirement is zero or unset, only cap by max_qty
                max_short = max_qty
            else:
                available_margin = max(0.0, (equity / margin_requirement) - margin_used)
                max_short_margin = int(available_margin // price)
                max_short = max(0, min(max_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short

        # Hold always valid
        actions["hold"] = 0

        # Prune zero-capacity actions to reduce tokens, keep hold
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """Keep only {agent: {sig, conf}} and drop empty agents."""
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


def _normalize_confidence(confidence: float | int | None) -> float:
    """Return confidence as a 0-100 float, accepting either 0-1 or 0-100 inputs."""
    if confidence is None:
        return 50.0
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 50.0
    if 0 <= value <= 1:
        value *= 100
    return max(0.0, min(value, 100.0))


def compute_signal_score(ticker_signals: dict[str, dict]) -> dict[str, float | int | str]:
    """Convert analyst votes into a deterministic score in [-1, 1].

    This is intentionally simple and transparent. It is not meant to replace the
    analysts; it is a guardrail that prevents a weak, noisy bullish vote from
    becoming an oversized buy order.
    """
    signal_values = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    weighted_sum = 0.0
    total_weight = 0.0
    votes: list[float] = []

    for payload in ticker_signals.values():
        sig = payload.get("sig") or payload.get("signal")
        if sig not in signal_values:
            continue
        conf = _normalize_confidence(payload.get("conf") if "conf" in payload else payload.get("confidence"))
        weight = max(conf / 100.0, 0.05)
        value = signal_values[sig]
        weighted_sum += value * weight
        total_weight += weight
        votes.append(value)

    if total_weight == 0:
        return {"score": 0.0, "agreement": 0.0, "vote_count": 0, "direction": "neutral"}

    score = weighted_sum / total_weight
    direction = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
    if direction == "neutral":
        agreement = votes.count(0.0) / len(votes) if votes else 0.0
    else:
        target = 1.0 if direction == "bullish" else -1.0
        agreement = sum(1 for vote in votes if vote == target) / len(votes)

    return {
        "score": float(score),
        "agreement": float(agreement),
        "vote_count": len(votes),
        "direction": direction,
    }


def _build_exit_plan(action: str, price: float, signal_score: float, confidence: int) -> dict[str, float | int | str | None]:
    """Create deterministic stop-loss, take-profit, trailing-stop, and holding-window fields."""
    if price <= 0 or action not in {"buy", "short"}:
        return {
            "entry_price": price if price > 0 else None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "trailing_stop_pct": None,
            "max_holding_days": None,
            "risk_notes": "No exit plan required for non-entry action",
        }

    strength = abs(signal_score)
    # Weaker signals get tighter stops and shorter review windows.
    if strength < 0.35 or confidence < 60:
        stop_pct = 0.06
        max_days = 5
        note = "Weak edge: tight stop and fast review"
    elif strength < 0.60 or confidence < 75:
        stop_pct = 0.08
        max_days = 10
        note = "Moderate edge: standard swing-trade plan"
    else:
        stop_pct = 0.10
        max_days = 20
        note = "Strong edge: wider stop with trailing exit"

    reward_1 = stop_pct * 1.8
    reward_2 = stop_pct * 3.0
    trailing_stop_pct = max(0.05, stop_pct * 0.75)

    if action == "buy":
        return {
            "entry_price": round(price, 4),
            "stop_loss": round(price * (1 - stop_pct), 4),
            "take_profit_1": round(price * (1 + reward_1), 4),
            "take_profit_2": round(price * (1 + reward_2), 4),
            "trailing_stop_pct": round(trailing_stop_pct, 4),
            "max_holding_days": max_days,
            "risk_notes": note,
        }

    return {
        "entry_price": round(price, 4),
        "stop_loss": round(price * (1 + stop_pct), 4),
        "take_profit_1": round(price * (1 - reward_1), 4),
        "take_profit_2": round(price * (1 - reward_2), 4),
        "trailing_stop_pct": round(trailing_stop_pct, 4),
        "max_holding_days": max_days,
        "risk_notes": note,
    }


def _post_process_decisions(
    decisions: dict[str, PortfolioDecision],
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    allowed_actions_full: dict[str, dict[str, int]],
) -> dict[str, PortfolioDecision]:
    """Apply deterministic quality gates, quantity caps, and exit plans.

    This layer is the main fix for the previous weakness: the LLM can propose a
    trade, but weak/low-agreement signals are downgraded to hold, and every entry
    receives a stop-loss/take-profit plan.
    """
    processed: dict[str, PortfolioDecision] = {}

    for ticker in tickers:
        decision = decisions.get(
            ticker,
            PortfolioDecision(action="hold", quantity=0, confidence=0, reasoning="Missing decision; default hold"),
        )
        allowed = allowed_actions_full.get(ticker, {"hold": 0})
        score_info = compute_signal_score(signals_by_ticker.get(ticker, {}))
        score = float(score_info["score"])
        agreement = float(score_info["agreement"])
        vote_count = int(score_info["vote_count"])
        confidence = int(_normalize_confidence(decision.confidence))

        # Enforce allowed action and max quantity.
        if decision.action not in allowed:
            decision = PortfolioDecision(action="hold", quantity=0, confidence=confidence, reasoning="Action not allowed; downgraded to hold")
        elif decision.action != "hold":
            decision.quantity = max(0, min(int(decision.quantity), int(allowed.get(decision.action, 0))))
            if decision.quantity == 0:
                decision.action = "hold"
                decision.reasoning = "Quantity unavailable; hold"

        # Quality gate: a directional entry needs signal strength and agreement.
        if decision.action == "buy":
            if score < 0.20 or agreement < 0.45 or confidence < 55 or vote_count < 2:
                decision = PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=min(confidence, 50),
                    reasoning="Bullish edge too weak; hold",
                )
        elif decision.action == "short":
            if score > -0.20 or agreement < 0.45 or confidence < 55 or vote_count < 2:
                decision = PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=min(confidence, 50),
                    reasoning="Bearish edge too weak; hold",
                )

        # Size cap: weaker signals should not use the full risk limit.
        if decision.action in {"buy", "short"}:
            max_allowed = int(allowed.get(decision.action, 0))
            strength = abs(score)
            if strength < 0.35:
                size_multiplier = 0.33
            elif strength < 0.60:
                size_multiplier = 0.60
            else:
                size_multiplier = 1.00
            capped_qty = int(max_allowed * size_multiplier)
            decision.quantity = max(0, min(decision.quantity, capped_qty))
            if decision.quantity == 0:
                decision.action = "hold"
                decision.reasoning = "Signal too weak for minimum position; hold"

        plan = _build_exit_plan(
            action=decision.action,
            price=float(current_prices.get(ticker, 0.0)),
            signal_score=score,
            confidence=confidence,
        )
        decision.entry_price = plan["entry_price"]
        decision.stop_loss = plan["stop_loss"]
        decision.take_profit_1 = plan["take_profit_1"]
        decision.take_profit_2 = plan["take_profit_2"]
        decision.trailing_stop_pct = plan["trailing_stop_pct"]
        decision.max_holding_days = plan["max_holding_days"]
        decision.risk_notes = plan["risk_notes"]
        decision.confidence = confidence

        processed[ticker] = decision

    return processed


def generate_trading_decision(
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        agent_id: str,
        state: AgentState,
) -> PortfolioManagerOutput:
    """Get decisions from the LLM, then enforce deterministic risk and exit rules."""

    # Deterministic constraints
    allowed_actions_full = compute_allowed_actions(tickers, current_prices, max_shares, portfolio)

    # Pre-fill pure holds to avoid sending them to the LLM at all
    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        # If only 'hold' key exists, there is no trade possible
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=100, reasoning="No valid trade available"
            )
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(
            decisions=_post_process_decisions(
                decisions=prefilled_decisions,
                tickers=tickers,
                signals_by_ticker=signals_by_ticker,
                current_prices=current_prices,
                allowed_actions_full=allowed_actions_full,
            )
        )

    # Build compact payloads only for tickers sent to LLM
    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    # Minimal prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a portfolio manager.\n"
                "Inputs per ticker: analyst signals and allowed actions with max qty (already validated).\n"
                "Pick one allowed action per ticker and a quantity ≤ the max. "
                "Prefer hold when signals are mixed or weak. "
                "Keep reasoning very concise (max 100 chars). No cash or margin math. Return JSON only."
            ),
            (
                "human",
                "Signals:\n{signals}\n\n"
                "Allowed:\n{allowed}\n\n"
                "Format:\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}"
            ),
        ]
    )

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
    }
    prompt = template.invoke(prompt_data)

    # Default factory fills remaining tickers as hold if the LLM fails
    def create_default_portfolio_output():
        # start from prefilled
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=0, reasoning="Default decision: hold"
            )
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # Merge prefilled holds with LLM results
    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)

    return PortfolioManagerOutput(
        decisions=_post_process_decisions(
            decisions=merged,
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            allowed_actions_full=allowed_actions_full,
        )
    )
