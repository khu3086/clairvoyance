"""Clairvoyance MCP server.

Exposes the FastAPI service's capabilities as MCP tools so Claude (or any
MCP-compatible client) can query the model conversationally:

    "What are today's top recommendations?"
    "What's the model's view on NVDA?"
    "Which symbols does the system track?"

Architecture: this server is a thin HTTP client over the FastAPI service.
The FastAPI service must be running for these tools to work.

Run with:
    python -m mcp_server.server

Or wire into Claude Desktop via claude_desktop_config.json — see README_STEP4.md.
"""
import os
import logging

import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr — MCP uses stdout for protocol messages, so
# ANY logging to stdout would corrupt the stream.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = os.getenv("CLAIRVOYANCE_API_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("CLAIRVOYANCE_TIMEOUT", "30"))

mcp = FastMCP("clairvoyance")


def _api_get(path: str, params: dict | None = None) -> dict:
    """Call the FastAPI service. Raises with a friendly message on failure."""
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not reach Clairvoyance API at {API_URL}. "
            f"Make sure uvicorn is running: `uvicorn api.main:app --port 8000`"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API returned {e.response.status_code}: {e.response.text}") from e


# ----------------------------- TOOLS ----------------------------------------

@mcp.tool()
def get_recommendations(n: int = 5) -> dict:
    """Get the top-N predicted outperformers across the tracked universe.

    Returns a ranked list of stocks the ML model predicts are most likely to
    beat the S&P 500 (SPY) by at least 0.5% over the next 5 trading days.
    Each recommendation includes the symbol, company name, sector, predicted
    probability, and the date of the most recent bar used.

    Args:
        n: How many top picks to return (1–50). Default 5.
    """
    if not 1 <= n <= 50:
        raise ValueError("n must be between 1 and 50")
    return _api_get("/recommend", {"n": n})


@mcp.tool()
def predict_symbol(symbol: str) -> dict:
    """Get the model's outperformance probability for a specific stock.

    Returns the probability that `symbol` beats SPY by ≥0.5% over the next
    5 trading days, plus sector/industry metadata.

    Args:
        symbol: Stock ticker, e.g. "AAPL", "MSFT", "NVDA". Case-insensitive.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    try:
        return _api_get(f"/predict/{symbol}")
    except RuntimeError as e:
        if "404" in str(e):
            return {
                "error": (
                    f"'{symbol}' is not in the tracked universe, or there isn't "
                    f"enough recent history to make a prediction."
                )
            }
        raise


@mcp.tool()
def list_tracked_symbols() -> dict:
    """List every security currently tracked by the system.

    Use this to discover what symbols are available before calling
    predict_symbol or interpreting recommendations.
    """
    symbols = _api_get("/symbols")
    return {"count": len(symbols), "symbols": symbols}


@mcp.tool()
def get_model_info() -> dict:
    """Get metadata about the currently loaded ML model.

    Returns the artifact name, training metrics (test AUC, accuracy),
    feature columns the model uses, hyperparameters, and the timestamp
    when the model was saved. Useful for understanding what version of
    the model is producing today's recommendations.
    """
    return _api_get("/models/current")


@mcp.tool()
def check_api_health() -> dict:
    """Verify the underlying Clairvoyance API is reachable and a model is loaded.

    Returns the API status. If the FastAPI service isn't running, this
    explains how to start it.
    """
    try:
        return _api_get("/health")
    except RuntimeError as e:
        return {"status": "unreachable", "error": str(e), "api_url": API_URL}


@mcp.tool()
def explain_prediction(symbol: str) -> str:
    """Generate a natural-language explanation of the model's view on a symbol.

    Returns a short paragraph describing the prediction with sector context
    and a sentiment label. Designed to feed cleanly into a conversation rather
    than as raw JSON.

    Args:
        symbol: Stock ticker, e.g. "AAPL".
    """
    pred = predict_symbol(symbol)
    if isinstance(pred, dict) and pred.get("error"):
        return f"Can't explain {symbol}: {pred['error']}"

    prob = pred["prob"]
    name = pred.get("name") or pred["symbol"]
    sector = pred.get("sector") or "Unknown sector"
    as_of = pred.get("as_of", "unknown date")

    if prob >= 0.55:
        sentiment = "Strong outperform"
    elif prob >= 0.50:
        sentiment = "Moderate outperform"
    elif prob >= 0.45:
        sentiment = "Neutral, leaning cautious"
    else:
        sentiment = "Likely underperform"

    return (
        f"{name} ({pred['symbol']}, {sector}). "
        f"As of {as_of[:10]}, the model assigns a {prob:.1%} probability that "
        f"{pred['symbol']} beats SPY by ≥0.5% over the next 5 trading days. "
        f"Sentiment label: {sentiment}. "
        f"(For reference: the base rate of positive labels in training is ~44%, "
        f"so a probability above 0.50 is meaningfully above random.)"
    )


# ----------------------------- ENTRYPOINT -----------------------------------

if __name__ == "__main__":
    # Default transport is stdio, which is what Claude Desktop expects.
    # For HTTP transport (e.g. for testing with MCP Inspector via SSE):
    #   mcp.run(transport="streamable-http")
    logger.info("Starting Clairvoyance MCP server (stdio). API URL: %s", API_URL)
    mcp.run()
