# 📈 Deep Reinforcement Learning for Indian Index Options Trading

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![RL-Framework](https://img.shields.io/badge/Stable--Baselines3-MaskablePPO-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29.1-orange.svg)](https://gymnasium.farama.org/)
[![Market](https://img.shields.io/badge/Market-NSE%2FBSE%20India-red.svg)](https://www.nseindia.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An enterprise-grade, multi-modal **Reinforcement Learning (RL)** system engineered for automated **index options trading** on the Indian Stock Market (NSE / BSE). 

The system trains an autonomous trading agent using **Maskable Proximal Policy Optimization (MaskablePPO)** with custom 1D-CNN temporal feature extractors, continuous risk-adjusted reward shaping, Black-Scholes Greeks calculations, microsecond-level state management, and multi-source sentiment/macro signal integration.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Data & Feature Engineering Pipeline](#-data--feature-engineering-pipeline)
  - [Supported Symbols & Indices](#supported-symbols--indices)
  - [3D Data Tensor Architecture](#3d-data-tensor-architecture)
  - [Black-Scholes Options Pricing & Greeks](#black-scholes-options-pricing--greeks)
  - [Macro & Sentiment Signal Merging](#macro--sentiment-signal-merging)
- [Reinforcement Learning Environment](#-reinforcement-learning-environment)
  - [Action Space & Masking](#action-space--masking)
  - [Observation Vector Construction](#observation-vector-construction)
  - [Intra-Candle Execution & Risk Management](#intra-candle-execution--risk-management)
  - [Reward Function Engineering](#reward-function-engineering)
- [Neural Network Architecture](#-neural-network-architecture)
- [Evaluation & Benchmarking Suite](#-evaluation--benchmarking-suite)
- [Reporting Engine](#-reporting-engine)
- [Frontend Dashboard & AI Strategist](#-frontend-dashboard--ai-strategist)
- [Quick Start & Setup](#-quick-start--setup)
- [Execution & Pipeline Usage](#-execution--pipeline-usage)
- [Configuration Reference](#-configuration-reference)
- [Cloud Infrastructure & Deployment](#-cloud-infrastructure--deployment)
- [Testing](#-testing)
- [License](#-license)

---

## 🚀 Key Features

- **Multi-Symbol Slot Randomization**: Monitors up to **6 index symbols concurrently** per episode with dynamic slot allocation, enabling seamless multi-asset training without overfitting to static symbol orderings.
- **MaskablePPO Engine**: Utilizes invalid action masking (`sb3-contrib`) to mathematically prevent execution of impossible or illegal trades (e.g., exiting unowned positions, over-leveraging capital, trading sub-minimum option values).
- **Temporal 1D-CNN Feature Extractor**: Custom PyTorch neural backbone (`Trading1DCNN`) processing rolling 30-candle temporal windows across 8 technical channels to extract market momentum, regime transitions, and volatility patterns.
- **Black-Scholes Greeks Vectorization**: High-performance vectorized calculation of Delta, Gamma, Theta, Vega, and Put Greeks across 2 expiries (Current & Next Week) and 5 strikes (-2 to +2 offsets).
- **Intra-Candle Execution Engine**: High-fidelity trade simulation matching intra-candle Stop-Loss (SL) and Take-Profit (TP) fills with dynamic slippage modeling (`SIM_AGGRESSION=0.4`) and SEBI peak-margin allocation limits.
- **Graduated Friction & Commission Model**: Realistic Indian market execution penalty structure with multi-tier turnover costs (penalizes high-frequency scalping, rewards patient holding beyond 30 minutes).
- **Multi-Modal Signal Fusion**: Concatenates price action with 11 global macroeconomic indices (Dow Jones, FTSE, Gold, etc. lagged by 1 day to eliminate look-ahead bias), VADER financial news sentiment, and economic event triggers.
- **Comprehensive Evaluation & Benchmarking**: Multi-regime validation framework (6-Month OoS, 12-Month OoS, In-Sample, Zero-Shot Transfer on unseen symbols) evaluated against Buy & Hold, Short Straddle, EMA-50/200 Crossover, and Cash baselines.
- **Automated Analysis & Multi-Tab Excel Reporting**: Auto-parses TensorBoard logs (`auto_analyzer.py`) and compiles multi-sheet Excel reports featuring equity curves, trade logs, drawdown charts, and risk-adjusted metrics (Sharpe, Sortino, Profit Factor).
- **Cloud-Ready & Containerized**: Dockerized pipeline with Google Cloud Build integration, Workload Identity Federation (WIF) setup, and Cloud Run job deployment scripts.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Master Orchestrator                                  │
│                             (run_pipeline.py)                                    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
    ┌────────────────────────────────────┼────────────────────────────────────┐
    ▼                                    ▼                                    ▼
┌───────────────────────┐   ┌─────────────────────────┐   ┌───────────────────────────┐
│     Phase 1: TRAIN    │   │   Phase 2: EVALUATE     │   │     Phase 3: REPORT       │
│  (train_ppo.py)       │   │  (evaluate_ppo.py)      │   │  (excel_reporting.py)     │
└───────────┬───────────┘   └────────────┬────────────┘   └─────────────┬─────────────┘
            │                            │                              │
            ▼                            ▼                              ▼
┌───────────────────────┐   ┌─────────────────────────┐   ┌───────────────────────────┐
│ TradingEnv + StateMgr │   │ Multi-Regime Backtesting│   │ Multi-Tab Excel Portfolio │
│ (1D-CNN + Masked PPO) │   │ vs. 4 Benchmark Strat.  │   │ Report (Equity, Drawdown) │
└───────────────────────┘   └─────────────────────────┘   └───────────────────────────┘
```

### End-to-End Data Flow

```
                      ┌──────────────────────┐
                      │   Upstox API v3      │
                      │  (5-min OHLCV Data)  │
                      └──────────┬───────────┘
                                 │
                                 ▼
┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│ News Scrapers    │──>│ SQLite Database     │──>│ Preprocessing    │
│ (Moneycontrol/ET)│   │ (Sentiment & Macro) │   │ (Greeks & Regimes│
└──────────────────┘   └─────────────────────┘   └────────┬─────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ 3D Data Tensor (NumPy)    │
                                            │ (Symbols x Time x 95 Feat)│
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ TradingEnv (Gymnasium)    │
                                            │ Action Masking + Reward   │
                                            └─────────────┬─────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ MaskablePPO Neural Agent  │
                                            └───────────────────────────┘
```

---

## 📁 Repository Structure

```
RL-Options-Trading/
├── backend/
│   ├── env/                        # Gymnasium RL Trading Environment
│   │   ├── trading_env.py          # Core environment (Observation, Action Masking, Step logic)
│   │   └── state_manager.py        # Vectorized portfolio & position management (NumPy)
│   ├── train/                      # Training, Evaluation & Optimization Suite
│   │   ├── ppo_config.py           # Global hyperparameters, environment constants & paths
│   │   ├── train_ppo.py            # MaskablePPO training loop with callbacks & guards
│   │   ├── evaluate_ppo.py         # Multi-regime evaluation suite & benchmark engine
│   │   ├── feature_extractors.py   # Custom Trading1DCNN PyTorch feature extractor
│   │   ├── metrics.py              # Quantitative financial metrics (Sharpe, Sortino, MaxDD)
│   │   ├── tune_ppo.py             # Optuna hyperparameter search pipeline
│   │   ├── reporting.py            # Evaluation output formatters & summary printers
│   │   └── auto_analyzer.py        # Automated TensorBoard log & metric analyzer
│   ├── db/                         # Auxiliary Database Layer
│   │   ├── local_db_manager.py     # SQLite manager for news, macro, and economic events
│   │   └── migrate_parquets.py     # Parquet database migration script
│   ├── sentiment/                  # Multi-Modal Sentiment & Macro Signal Engine
│   │   ├── sentiment_engine.py     # VADER financial sentiment scoring engine
│   │   ├── news_aggregator.py      # Multi-source scraper (Moneycontrol, Economic Times)
│   │   ├── news_pipeline.py        # Automated news fetching and pipeline execution
│   │   ├── rss_backfiller.py       # RSS historical feed backfiller
│   │   ├── archive_filler.py       # News archive backfill utility
│   │   ├── batch_scorer.py         # Batch sentiment score calculator
│   │   ├── event_manager.py        # Scheduled economic event manager
│   │   └── macro_pipeline.py       # Global macro market indicator synchronization
│   ├── reporting/                  # Excel Report Generator
│   │   └── excel_reporting.py      # Multi-sheet Excel workbook generator (XlsxWriter)
│   ├── models/                     # Checkpoints (.zip) & normalization vectors (.pkl)
│   └── run_pipeline.py             # Master CLI orchestrator (Train → Evaluate → Report)
├── data/
│   ├── historical/                 # 5-minute OHLCV candles stored in Parquet format
│   ├── pipelines/
│   │   ├── upstox_pipeline.py      # Upstox API v3 historical sync engine (10-day chunks)
│   │   ├── upstox_auth.py          # OAuth authentication & token refresh flow
│   │   ├── db_manager.py           # Parquet file reader/writer interface
│   │   └── discover_indices.py     # Market symbol and index discovery utility
│   ├── preprocessing/
│   │   ├── greeks.py               # Vectorized Black-Scholes Greeks calculator
│   │   └── normalization.py        # Feature scaling and z-score utilities
│   └── trading_data.db             # Local SQLite database (Sentiment, Macro, Events)
├── docs/                           # In-depth technical documentation
│   ├── 00_overview.md              # High-level architecture guide
│   ├── 01_glossary.md              # Financial and RL terminology glossary
│   ├── 02_data_pipeline.md         # Data pipeline & feature tensor specification
│   ├── 03_environment.md           # TradingEnv, state manager & reward logic
│   └── 04_training_evaluation.md   # Training loop, evaluation regimes & hyperparameter tuning
├── notebooks/                      # Exploratory research & analysis notebooks
├── reports/                        # Output directory for Excel reports and eval pickles
├── ppo_trading_tensorboard/        # TensorBoard training logs & metrics
├── Dockerfile                      # Production container spec (Python 3.10-slim)
├── cloudbuild.yaml                 # GCP Cloud Build configuration
├── create_jobs.ps1                 # PowerShell script for GCP Cloud Run jobs setup
├── deploy.ps1                      # PowerShell script for GCP deployment
├── setup_wif.ps1                   # Workload Identity Federation (WIF) setup script
├── bench_vectorized.py             # Performance benchmark script for environment state
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive project overview
```

---

## 📊 Data & Feature Engineering Pipeline

### Supported Symbols & Indices

The pipeline synchronizes intraday **5-minute OHLCV data** directly via the Upstox API v3:

| Symbol Name | Exchange | Segment | Lot Size | Weekly Expiry Day | Role in System |
|---|---|---|---|---|---|
| **Nifty 50** | NSE | INDEX_OPTION | 50 | Thursday | Core Training Asset |
| **Nifty Bank** | NSE | INDEX_OPTION | 15 | Wednesday | Core Training Asset |
| **Nifty Fin Service** | NSE | INDEX_OPTION | 40 | Tuesday | Core Training Asset |
| **Nifty Midcap Select** | NSE | INDEX_OPTION | 75 | Monday | Core Training Asset |
| **SENSEX** | BSE | INDEX_OPTION | 10 | Friday | Core Training Asset |
| **Nifty Next 50** | NSE | INDEX_OPTION | 25 | Friday | Training / Validation Asset |
| **SENSEX50** | BSE | INDEX_OPTION | 15 | Friday | Zero-Shot Transfer Test Asset |
| **Nifty IT** | NSE | INDEX | N/A | N/A | Sectoral Context (Observation-Only) |

### 3D Data Tensor Architecture

To avoid Pandas DataFrame overhead during RL training steps, all historical market data, indicators, Greeks, and external signals are stacked into a **3D NumPy Tensor**:

$$\text{Shape} = (N_{\text{symbols}}, N_{\text{timesteps}}, N_{\text{features}}) \quad \text{e.g., } (6, 120\,000, 95)$$

#### Feature Mapping Breakdown (95 Columns per Timestep):
- **Columns `[0 - 9]` (Price & Microstructure)**: Open, High, Low, Close, Volume, Open Interest (OI), Days to Expiry (DTE), Day of Week (0–4), Is Expiry Day (0/1), Timestamp.
- **Columns `[10 - 69]` (Vectorized Options Greeks)**: Calculated across **2 Expiries** ($e_0$ = Current Week, $e_1$ = Next Week) $\times$ **5 Strike Offsets** ($-2, -1, 0, +1, +2$) $\times$ **6 Features** ($\Delta, \Gamma, \Theta, \text{Vega}, \text{Put } \Delta, \text{Put } \Theta$).
- **Columns `[70 - 74]` (Technical Indicators)**: RSI(14), ATR(14), Annualized Index Volatility (EWM $\sigma$), EMA(50), EMA(200).
- **Columns `[75 - 78]` (Theoretical ATM Prices)**: $e_0$ Call Price, $e_0$ Put Price, $e_1$ Call Price, $e_1$ Put Price.
- **Columns `[79 - 95]` (Macro & Sentiment Context)**: 11 Global Macro Indices returns (lagged 1 day), Positive/Negative Sentiment Scores (lagged 1 day), Headline Count, Economic Event Max Impact (1–3), Is Event Day, Volatility Percentile (252-day), and Nifty IT 1-day Return.

### Black-Scholes Options Pricing & Greeks

The preprocessing module (`data/preprocessing/greeks.py`) computes Black-Scholes Greeks dynamically for calls and puts:

$$d_1 = \frac{\ln(S / K) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

$$\Delta_{\text{Call}} = N(d_1), \quad \Delta_{\text{Put}} = N(d_1) - 1$$

$$\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}, \quad \text{Vega} = S N'(d_1) \sqrt{T}$$

$$\Theta_{\text{Call}} = -\frac{S N'(d_1) \sigma}{2 \sqrt{T}} - r K e^{-r T} N(d_2)$$

### Macro & Sentiment Signal Merging

To guarantee **zero look-ahead bias**:
- Macro index log returns (Dow Jones, FTSE, Gold, Brent Crude, GIFT Nifty, etc.) and VADER sentiment scores are **lagged by 1 trading day** ($t-1$). Yesterday's macro closing state is the latest available state when the market opens today.
- Scheduled economic calendar events (e.g., RBI Monetary Policy, Inflation Releases) use same-day timestamps because event release times are known in advance.

---

## 🤖 Reinforcement Learning Environment

The environment (`backend/env/trading_env.py`) adheres strictly to the **Gymnasium** API.

### Action Space & Masking

The action space is defined as a **MultiDiscrete** array across 6 symbol slots:

$$\text{Action Space} = \text{MultiDiscrete}([4, 4] \times 6) \quad \text{(12 total dimensions)}$$

For each active slot:
1. **Trade Action (Dim 0)**: `0 = Hold`, `1 = Buy Call`, `2 = Buy Put`, `3 = Exit Position`.
2. **Risk Profile (Dim 1)**: `0 = Tight (SL -5% / TP +8%)`, `1 = Regular (SL -10% / TP +15%)`, `2 = Aggressive (SL -15% / TP +25%)`, `3 = None (Default for Empty/Hold)`.

#### Action Masking Rules (`action_masks()`):
- **Empty Slot**: Disables `Exit`; forces Risk Profile to `None`.
- **Occupied Slot**: Disables `Buy Call` and `Buy Put`.
- **Insufficient Capital**: Disables all `Buy` actions if available capital < `MIN_TRADE_VALUE` (₹250,000).
- **Overtrading Safeguard**: Disables `Buy` actions if daily trades ≥ `MAX_TRADES_PER_DAY` (10).
- **Penny Option Protection**: Disables `Buy` actions if option premium < `MIN_OPTION_PRICE` (₹5).
- **Inactive Slot**: Forces `Hold` action.

### Observation Vector Construction

At each timestep, `_get_obs()` constructs a flat observation vector of size **1,905 features**:

```
Observation Vector Layout (1905 Floats)
├── Portfolio Context [0 - 1]: (Capital Ratio, Cash Ratio)
├── Slot Segments [2 - 1885] (6 Slots x 314 Features):
│   ├── Temporal History: 30 lookback candles x 8 channels (OHLC norm, Vol log, OI log, RSI, IV) = 240
│   ├── Options Greeks: 60 scaled Greek values across expiries/strikes
│   ├── ATM Option Prices: 4 spot-normalized premiums
│   └── Slot Metadata: ATR/spot ratio, EMA ratios, DTE, Position PnL %, Hold duration
├── Global External Signals [1886 - 1903]: 18 features (Macro returns, Sentiment, Event flags)
└── Volatility Regime Flag [1904]: Current Market Regime (0 = Low Vol, 1 = Med Vol, 2 = High Vol)
```

### Intra-Candle Execution & Risk Management

To prevent unrealistic execution assumptions during 5-minute candles, intra-candle highs/lows are checked against Stop-Loss (SL) and Take-Profit (TP) boundaries:

$$\text{Worst Price} = P_{\text{entry}} \times \left(1 + \Delta \cdot \frac{S_{\text{low}} - S_{\text{entry}}}{S_{\text{entry}}}\right)$$

$$\text{Best Price} = P_{\text{entry}} \times \left(1 + \Delta \cdot \frac{S_{\text{high}} - S_{\text{entry}}}{S_{\text{entry}}}\right)$$

If an SL or TP threshold is breached intra-candle, the exit fill incorporates slippage via `SIM_AGGRESSION` (0.4):

$$P_{\text{exit}} = P_{\text{SL}} + (P_{\text{worst}} - P_{\text{SL}}) \times \text{Slippage}$$

### Reward Function Engineering

The step reward function $R_t$ balances profitability against capital protection:

$$R_t = R_{\text{PnL}} + R_{\text{Drawdown}} + R_{\text{Entry}} + R_{\text{Patience}} + R_{\text{Vol}} + R_{\text{Stale}}$$

1. **PnL Reward**: Upon trade exit, $R_{\text{PnL}} = \frac{\Delta \text{Value}}{\text{Entry Value}} \times \text{Reward Scale}$.
2. **Drawdown Penalty**: 
   - If Drawdown $> 3\%$ (Soft Threshold): Penalty proportional to drawdown $\times 2.0$.
   - If Drawdown $> 10\%$ (Hard Threshold): Heavy escalating exponential penalty.
3. **Entry Friction**: $-0.0003$ per new trade to penalize excessive churning.
4. **Patience Bonus**: $+0.001$ per step for holding profitable positions beyond 3 candles ($15\text{ mins}$).
5. **Volatility Bonus**: $+15\%$ reward multiplier when operating in high-volatility regimes.
6. **Graduated Commissions**:
   - Hold $< 2$ steps ($< 10\text{ mins}$): Scalping commission of ₹40 / lot.
   - Hold $< 6$ steps ($< 30\text{ mins}$): Reduced commission of ₹20 / lot.
   - Hold $\ge 6$ steps ($\ge 30\text{ mins}$): ₹0 commission.

---

## 🧠 Neural Network Architecture

The model uses a custom dual-head network architecture built on top of `Stable-Baselines3`:

```
                           Input Observation (1905 Vector)
                                         │
                                         ▼
                      ┌────────────────────────────────────┐
                      │    Trading1DCNN Feature Extractor  │
                      │  - Conv1D (8 -> 32, kernel=3)      │
                      │  - Conv1D (32 -> 64, kernel=3)     │
                      │  - AdaptiveAvgPool1d(1)            │
                      │  - Linear (64 -> 128) + ReLU       │
                      │  - Dense Feature Projection        │
                      └──────────────────┬─────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │        Policy Head        │               │        Value Head         │
      │   MultiDiscrete Actions   │               │   Scalar Value V(s)       │
      │  net_arch: pi=[256, 128]  │               │  net_arch: vf=[512, 256]  │
      └───────────────────────────┘               └───────────────────────────┘
```

- **Policy Network (`pi`)**: Layers `[256, 128]` mapping extracted features to masked action probabilities.
- **Value Network (`vf`)**: Scaled up to `[512, 256]` layers to enhance value estimation accuracy and maximize explained variance (EV > 0.85).

---

## 📈 Evaluation & Benchmarking Suite

The evaluation engine (`backend/train/evaluate_ppo.py`) tests trained agents across four distinct market regimes:

1. **Standard Out-of-Sample (6-Month)**: Evaluates generalization on recent unseen historical data.
2. **Full Out-of-Sample (12-Month)**: Long-horizon stability test covering multiple market trends.
3. **In-Sample Validation**: Underfitting/overfitting baseline comparison against training data.
4. **Zero-Shot Transfer**: Evaluates agent performance on un-trained index symbols (e.g., SENSEX50).

### Benchmark Comparisons

Every RL run is evaluated side-by-side against 4 automated baseline strategies:
- **Buy & Hold**: Long index position held from start to end of period.
- **Short Straddle**: Sells ATM Call + Put at start of week, collects option premium, manages at expiry.
- **EMA Crossover**: Technical strategy entering long when EMA(50) > EMA(200) and exiting when below.
- **Cash Baseline**: Risk-free benchmark maintaining 100% cash balance.

### Financial Metrics Computed

- **Cumulative Return (%)**
- **Annualized Sharpe Ratio** (Risk-Free Rate = 5%)
- **Sortino Ratio** (Downside deviation risk-adjusted)
- **Maximum Drawdown (%)** & Peak-to-Trough Duration
- **Win Rate (%)** & Profit Factor (Total Wins / Total Losses)
- **Average Win / Average Loss Ratio**
- **Average Trade Hold Duration (Candles)**

---

## 📑 Reporting Engine

The reporting engine (`backend/reporting/excel_reporting.py`) automatically compiles backtest evaluations into an interactive multi-tab Excel workbook:

- 📊 **Summary Tab**: Executive summary of portfolio statistics, Sharpe/Sortino ratios, and model metadata.
- 📈 **Equity Curve Tab**: High-resolution chart depicting portfolio net asset value (NAV) over time vs benchmarks.
- 📝 **Trade Log Tab**: Exhaustive log of every executed trade (Entry/Exit price, spot levels, PnL, duration, reason for exit).
- 📅 **Daily PnL Tab**: Daily aggregated returns and win/loss breakdown.
- 📉 **Drawdown Analysis Tab**: Peak-to-trough drawdown visualization and recovery metrics.
- ⚖️ **Benchmark Comparison Tab**: Tabular comparative analysis against Buy & Hold, Straddle, and EMA strategies.

---

## 🎨 Frontend Dashboard & AI Strategist

The project includes a React 19 monitoring interface designed for high-density visualization:

- **Technology Stack**: React 19, TypeScript, Vite, Tailwind CSS v4, shadcn/ui, Recharts.
- **Backend-for-Frontend (BFF)**: Node.js Express server (`server.ts`) proxying REST endpoints (`/api/dashboard/*`) to the Python RL backend.
- **Gemini 3.1 Pro Integration**: Integrated **AI Strategist** chat interface (`AIStrategist.tsx`) utilizing the `@google/genai` SDK for market regime analysis and natural language portfolio queries.

---

## ⚙️ Quick Start & Setup

### Prerequisites

- **Python**: Version `3.10` or higher
- **Node.js**: Version `18+` (Optional, required for Frontend Dashboard)
- **Upstox API Credentials**: Required for live data ingestion (`UPSTOX_API_KEY`, `UPSTOX_API_SECRET`)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/msking18/RL-Options-Trading.git
   cd RL-Options-Trading
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   UPSTOX_API_KEY=your_upstox_api_key
   UPSTOX_API_SECRET=your_upstox_api_secret
   UPSTOX_REDIRECT_URI=https://127.0.0.1:5000/
   GEMINI_API_KEY=your_gemini_api_key
   ```

---

## 🕹️ Execution & Pipeline Usage

### 1. Synchronize Historical Market Data
Fetch historical 5-minute OHLCV candles from Upstox:
```bash
python -m data.pipelines.upstox_pipeline
```

### 2. Run the Full Master Pipeline
Executes **Training → Multi-Regime Evaluation → Excel Report Generation**:
```bash
python -m backend.run_pipeline --timesteps 3000000
```

#### Orchestrator Options:
```bash
python -m backend.run_pipeline --help
# Options:
#  --timesteps INT       Total training timesteps (default: 2000000)
#  --skip-train          Skip training and use existing checkpoint
#  --skip-eval           Skip evaluation phase
#  --skip-report         Skip Excel report generation
#  --force-fresh         Train from scratch (ignores existing weights)
#  --model-source TEXT   Model checkpoint to evaluate: "best_ev", "early_stop", or "latest"
```

### 3. Run Individual Phases

#### Phase A: Model Training
```bash
python -m backend.train.train_ppo --total-timesteps 3000000
```

#### Phase B: Model Evaluation
```bash
python -m backend.train.evaluate_ppo --model-source best_ev
```

#### Phase C: Hyperparameter Tuning (Optuna)
```bash
python -m backend.train.tune_ppo --n-trials 50
```

#### Phase D: Generate Excel Report
```bash
python -m backend.reporting.excel_reporting --input reports/evaluations/eval_results_latest.pkl --output reports/portfolio_report.xlsx
```

#### Phase E: Launch TensorBoard
```bash
tensorboard --logdir=ppo_trading_tensorboard
```

---

## ⚙️ Configuration Reference

All environment rules, reward weights, and training hyperparameters are centralized in `backend/train/ppo_config.py`:

```python
# Portfolio Parameters
INITIAL_CAPITAL = 5_000_000       # ₹50 Lakh starting capital
MIN_TRADE_VALUE = 250_000         # Min trade size (~5% of portfolio)
MIN_OPTION_PRICE = 5              # Reject options priced below ₹5
MAX_TRADES_PER_DAY = 10           # Overtrading hard cap
TOTAL_SLOTS = 6                   # Max concurrent symbol slots
LOOKBACK_WINDOW = 30              # Temporal history length (30 x 5-min candles = 2.5 hrs)

# PPO Hyperparameters
LEARNING_RATE = 1e-4
N_STEPS = 4096                    # Rollout buffer size
BATCH_SIZE = 256
N_EPOCHS = 6
GAMMA = 0.995                     # High discount factor for options holding
GAE_LAMBDA = 0.92
CLIP_RANGE = 0.15
ENT_COEF = 0.008                  # Policy entropy exploration coefficient
VF_COEF = 1.0

# Reward Weights
REWARD_SCALE = 1.0
ENTRY_PENALTY = 0.0003
PATIENCE_BONUS = 0.001
DRAWDOWN_THRESHOLD_SOFT = 0.03    # 3% drawdown trigger
DRAWDOWN_THRESHOLD_HARD = 0.10    # 10% drawdown trigger
DRAWDOWN_PENALTY_MULTIPLIER = 2.0
SIM_AGGRESSION = 0.4
VOLATILITY_EXPANSION_BONUS = 0.15
```

---

## ☁️ Cloud Infrastructure & Deployment

The codebase includes Google Cloud Platform (GCP) deployment scripts for automated training on GCP Cloud Run Jobs:

- **Dockerfile**: Optimized multi-stage Python 3.10-slim container image.
- **Artifact Registry & Cloud Build**: Build container images via `cloudbuild.yaml`:
  ```bash
  gcloud builds submit --config=cloudbuild.yaml
  ```
- **Workload Identity Federation (WIF)**: Passwordless Cloud Build authentication configured via `setup_wif.ps1`.
- **Automated Cloud Job Launchers**: PowerShell automation scripts (`deploy.ps1`, `create_jobs.ps1`) for launching high-memory GPU/CPU cloud training jobs.

---

## 🧪 Testing

Execute unit tests and environment verification checks:

```bash
# Run Gymnasium environment compliance test
python -m unittest tests/test_env.py

# Run backend execution pipeline smoke test
python -m unittest backend/tests/smoke_test.py

# Benchmark state manager vectorization performance
python bench_vectorized.py
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built for Quantitative Trading & Reinforcement Learning Research.</i>
</p>
