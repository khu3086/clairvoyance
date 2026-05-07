# Step 4 — MCP Server

Exposes the Clairvoyance recommendation engine as MCP tools so Claude (or any MCP-compatible client) can query it conversationally:

> *"What are today's top investment recommendations?"*
> *"What does the model think about NVDA?"*
> *"Which stocks are tracked?"*
> *"What model version is currently loaded?"*

## Architecture

```
┌──────────────┐     stdio      ┌─────────────┐     HTTP      ┌──────────────┐
│ Claude       │ ◄─────────────►│ MCP server  │ ────────────► │ FastAPI      │
│ Desktop      │   tool calls   │ server.py   │  /recommend   │ uvicorn:8000 │
└──────────────┘                └─────────────┘  /predict/... └──────┬───────┘
                                                                     │
                                                              ┌──────▼───────┐
                                                              │ MongoDB      │
                                                              │ + model.pt   │
                                                              └──────────────┘
```

The MCP server is a **thin client** — the real work happens in the FastAPI service. This means:
- FastAPI must be running for the MCP tools to work
- Same model is reused; no double-loading
- Architecture is layered: data → model → API → MCP client

## Tools exposed

| Tool | What it does |
|---|---|
| `get_recommendations(n)` | Top-N predicted outperformers |
| `predict_symbol(symbol)` | Probability for one ticker |
| `list_tracked_symbols()` | Universe of tracked stocks |
| `get_model_info()` | Loaded model's metrics, hyperparams |
| `check_api_health()` | Diagnostic — is the FastAPI reachable? |
| `explain_prediction(symbol)` | Natural-language summary |

## Install

```powershell
cd C:\Users\khush\OneDrive\Documents\clairvoyance
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(This adds the `mcp` package. Other deps already present.)

## Test the server standalone

The MCP server uses **stdio** as its transport — it talks over standard input/output, which means you can't test it with `curl`. Two ways to test:

### Option 1: MCP Inspector (visual UI)

The Inspector is a Node.js tool. It takes one command and gives you a clickable web UI for every tool:

```powershell
# In one terminal: start the FastAPI service
cd C:\Users\khush\OneDrive\Documents\clairvoyance
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000

# In a second terminal: start the inspector pointed at the MCP server
cd C:\Users\khush\OneDrive\Documents\clairvoyance
.\.venv\Scripts\Activate.ps1
npx @modelcontextprotocol/inspector python -m mcp_server.server
```

The inspector will print a URL like `http://localhost:6274` — open it in your browser. You'll see all six tools, can fill in arguments, click "Call Tool", and inspect the response.

This is the recommended way to verify the MCP server works before wiring it into Claude.

### Option 2: Quick sanity-check the import

```powershell
python -c "from mcp_server.server import mcp; print('OK, tools:', [t for t in mcp._tool_manager.list_tools()])"
```

If this prints a list of tools without error, the server module loads correctly.

## Wire it into Claude Desktop

This is the demo: chat with Claude and have it pull live recommendations from your model.

### 1. Install Claude Desktop

Download from claude.ai (Windows version). Sign in with your Anthropic account.

### 2. Locate the config file

Windows path:
```
%APPDATA%\Claude\claude_desktop_config.json
```

Open it in Notepad:
```powershell
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
```

If the file doesn't exist, Notepad will offer to create it.

### 3. Paste this config

(Adjust paths if your project lives elsewhere.)

```json
{
  "mcpServers": {
    "clairvoyance": {
      "command": "C:\\Users\\khush\\OneDrive\\Documents\\clairvoyance\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:\\Users\\khush\\OneDrive\\Documents\\clairvoyance",
      "env": {
        "CLAIRVOYANCE_API_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

If you already have other MCP servers configured, **merge** the `clairvoyance` entry into the existing `mcpServers` object — don't replace the whole file.

The double backslashes (`\\`) are required because JSON strings escape backslashes. The `"command"` key points to the *Python in your venv*, not your global Python — that's what gives the MCP server access to `httpx`, `mcp`, etc.

### 4. Restart Claude Desktop

Fully quit (right-click the tray icon → Quit) and reopen.

### 5. Verify

In a new chat, look for the tools icon (🔧 or 🪛 depending on version) — click it. You should see "clairvoyance" listed with six tools. If not, click "View Logs" and check for errors.

### 6. Demo it

Start uvicorn first:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000
```

Then in Claude Desktop, ask things like:

> What are the top 5 stocks the Clairvoyance model recommends today?

> What's the model's view on NVDA, and how does that compare to AAPL?

> Which sectors have the most picks in the top 10 right now?

> What model version is currently loaded, and what was its test AUC?

Claude will pick the right tool, call it, and synthesize the JSON into natural-language answers.

## Troubleshooting

**"Could not reach Clairvoyance API"** — uvicorn isn't running. Start it in another terminal.

**Tools don't show up in Claude Desktop** — Check the logs (Help → View Logs in Claude Desktop). Common fixes:
- Wrong Python path in `command` (must point to your venv's python.exe)
- JSON syntax error in `claude_desktop_config.json` (validate with an online JSON linter)
- Forgot to fully quit Claude Desktop before reopening

**`ModuleNotFoundError: No module named 'mcp_server'`** — `cwd` in the config is wrong. Should point to the project root where the `mcp_server/` folder lives.

**`mcp` package not found on install** — make sure pip is up to date: `python -m pip install --upgrade pip`, then retry.

## Why this is impressive in a portfolio

Most ML portfolio projects stop at "I have a notebook" or "I have a dashboard." Very few have:

1. A model trained with proper time-series methodology
2. A serving API
3. **An MCP server that lets Claude itself reason about the model's outputs**

The third point is genuinely novel and signals you're paying attention to where AI tooling is heading. In an interview, you can demo it in 30 seconds: open Claude Desktop, ask a question, get a real answer powered by your model.

## What's next

Step 5: React dashboard, calling the same FastAPI endpoints. The MCP server and dashboard will coexist as two different frontends over the same backend — exactly the pattern you'd see in a production system.
