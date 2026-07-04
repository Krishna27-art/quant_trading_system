import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell,
  LineChart, Line, AreaChart, Area, PieChart, Pie,
} from "recharts";
import {
  LayoutDashboard, Radio, LineChart as LineChartIcon, FlaskConical, Cpu,
  ClipboardList, Settings, Search, Bell, ChevronDown, Star, TrendingUp,
  TrendingDown, ArrowUpRight, ArrowDownRight, Info, Filter, X, Command,
  Target, ShieldAlert, Clock, Gauge, Layers, Zap, CircleDot, ChevronRight,
  SlidersHorizontal, BarChart3, Activity, BrainCircuit,
} from "lucide-react";

/* ============================================================================
   DESIGN TOKENS
   Institutional dark terminal. Base: slate-950/900. Signature accent: amber
   (a nod to the amber-phosphor heritage of Bloomberg-era terminals) used only
   for AI/prediction-confidence elements. Cyan is the secondary data-ink color
   for chart overlays. Emerald/rose carry strict buy/sell semantics only.
   All classes below are stock Tailwind utilities (no arbitrary values).
============================================================================ */
const INK = {
  amber: "#fbbf24",
  cyan: "#22d3ee",
  emerald: "#34d399",
  rose: "#fb7185",
  violet: "#a78bfa",
  grid: "#1e293b",
  axis: "#64748b",
};

/* ============================================================================
   STATIC UNIVERSE
============================================================================ */
const STOCKS = [
  { symbol: "RELIANCE", name: "Reliance Industries", sector: "Energy" },
  { symbol: "TCS", name: "Tata Consultancy Services", sector: "IT" },
  { symbol: "HDFCBANK", name: "HDFC Bank", sector: "Financials" },
  { symbol: "INFY", name: "Infosys", sector: "IT" },
  { symbol: "ICICIBANK", name: "ICICI Bank", sector: "Financials" },
  { symbol: "BHARTIARTL", name: "Bharti Airtel", sector: "Telecom" },
  { symbol: "SBIN", name: "State Bank of India", sector: "Financials" },
  { symbol: "ITC", name: "ITC Ltd", sector: "FMCG" },
  { symbol: "LT", name: "Larsen & Toubro", sector: "Industrials" },
  { symbol: "AXISBANK", name: "Axis Bank", sector: "Financials" },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank", sector: "Financials" },
  { symbol: "MARUTI", name: "Maruti Suzuki", sector: "Auto" },
  { symbol: "TATAMOTORS", name: "Tata Motors", sector: "Auto" },
  { symbol: "SUNPHARMA", name: "Sun Pharma", sector: "Pharma" },
  { symbol: "ADANIENT", name: "Adani Enterprises", sector: "Diversified" },
  { symbol: "HINDUNILVR", name: "Hindustan Unilever", sector: "FMCG" },
];

const MODELS = [
  { name: "LightGBM-v4", acc: 0.671, roc: 0.742, f1: 0.61, brier: 0.19, status: "healthy" },
  { name: "CatBoost-v3", acc: 0.658, roc: 0.729, f1: 0.60, brier: 0.20, status: "healthy" },
  { name: "XGBoost-v6", acc: 0.644, roc: 0.711, f1: 0.58, brier: 0.21, status: "drift" },
  { name: "Transformer-v2", acc: 0.683, roc: 0.751, f1: 0.63, brier: 0.18, status: "healthy" },
  { name: "Ensemble", acc: 0.701, roc: 0.768, f1: 0.65, brier: 0.17, status: "healthy" },
];

/* ============================================================================
   SEEDED RNG + DATA GENERATION
   Deterministic per symbol so charts don't reshuffle on every re-render.
============================================================================ */
function hashSeed(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}

function generateSeries(symbol, days = 220) {
  const rand = hashSeed(symbol + "::series");
  const basePrice = 200 + rand() * 3000;
  let price = basePrice;
  const out = [];
  const drift = (rand() - 0.45) * 0.0015;
  const vol = 0.012 + rand() * 0.014;
  const today = new Date();
  for (let i = days; i >= 0; i--) {
    const shock = (rand() - 0.5) * 2 * vol;
    price = Math.max(10, price * (1 + drift + shock));
    const open = price * (1 + (rand() - 0.5) * 0.006);
    const high = Math.max(open, price) * (1 + rand() * 0.01);
    const low = Math.min(open, price) * (1 - rand() * 0.01);
    const volume = Math.round(500000 + rand() * 4500000);
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    out.push({ date: d.toISOString().slice(0, 10), open, high, low, close: price, volume });
  }
  return out;
}

function ema(series, period) {
  const k = 2 / (period + 1);
  let prev = series[0].close;
  return series.map((d, i) => {
    prev = i === 0 ? d.close : d.close * k + prev * (1 - k);
    return prev;
  });
}

function vwapSeries(series) {
  let cumPV = 0, cumV = 0;
  return series.map((d) => {
    const typical = (d.high + d.low + d.close) / 3;
    cumPV += typical * d.volume;
    cumV += d.volume;
    return cumPV / cumV;
  });
}

function bollinger(series, period = 20, mult = 2) {
  return series.map((_, i) => {
    if (i < period - 1) return { upper: null, lower: null, mid: null };
    const win = series.slice(i - period + 1, i + 1).map((d) => d.close);
    const mean = win.reduce((a, b) => a + b, 0) / period;
    const variance = win.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    const sd = Math.sqrt(variance);
    return { upper: mean + mult * sd, lower: mean - mult * sd, mid: mean };
  });
}

function rsiSeries(series, period = 14) {
  const out = new Array(series.length).fill(null);
  let gains = 0, losses = 0;
  for (let i = 1; i < series.length; i++) {
    const change = series[i].close - series[i - 1].close;
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i <= period) {
      gains += gain; losses += loss;
      if (i === period) {
        const rs = gains / period / ((losses / period) || 1e-6);
        out[i] = 100 - 100 / (1 + rs);
      }
    } else {
      gains = (gains * (period - 1) + gain) / period;
      losses = (losses * (period - 1) + loss) / period;
      const rs = gains / (losses || 1e-6);
      out[i] = 100 - 100 / (1 + rs);
    }
  }
  return out;
}

function macdSeries(series) {
  const emaFast = ema(series, 12);
  const emaSlow = ema(series, 26);
  const macdLine = emaFast.map((v, i) => v - emaSlow[i]);
  const k = 2 / (9 + 1);
  let prevSignal = macdLine[0];
  const signal = macdLine.map((v, i) => {
    prevSignal = i === 0 ? v : v * k + prevSignal * (1 - k);
    return prevSignal;
  });
  return macdLine.map((v, i) => ({ macd: v, signal: signal[i], hist: v - signal[i] }));
}

const REASON_BANK = [
  "Price reclaimed the 20-day EMA with rising volume",
  "RSI recovering from oversold with bullish divergence",
  "MACD histogram flipped positive on daily timeframe",
  "Institutional block deals detected over last 3 sessions",
  "Relative strength vs sector index turned positive",
  "Options chain shows call OI buildup near current strike",
  "Delivery volume percentage above 30-day average",
  "Positive news sentiment score over trailing 5 sessions",
  "Price breaking out of a 6-week consolidation range",
  "Sector rotation flows turning favorable this week",
  "Earnings revision trend positive over last 2 quarters",
  "Low realized volatility ahead of scheduled catalyst",
];

const FEATURE_BANK = [
  "20D Momentum", "RSI(14)", "MACD Histogram", "Volume Z-Score",
  "Sector Relative Strength", "Options PCR", "FII Net Flow",
  "News Sentiment Score", "Volatility Regime", "Price-to-VWAP Gap",
  "Delivery %", "Earnings Surprise",
];

function generatePrediction(symbol, lastClose) {
  const rand = hashSeed(symbol + "::pred");
  const probability = 0.52 + rand() * 0.4; // 0.52 - 0.92
  const confidence = 0.5 + rand() * 0.45;
  const entry = lastClose * (1 + (rand() - 0.5) * 0.006);
  const stopLoss = entry * (1 - (0.02 + rand() * 0.035));
  const t1 = entry * (1 + (0.02 + rand() * 0.02));
  const t2 = entry * (1 + (0.045 + rand() * 0.03));
  const t3 = entry * (1 + (0.08 + rand() * 0.05));
  const expectedReturn = ((t2 - entry) / entry) * 100;
  const riskReward = (t2 - entry) / (entry - stopLoss);
  const holdingDays = Math.round(4 + rand() * 22);
  const score = Math.round((probability * 0.5 + confidence * 0.3 + Math.min(riskReward / 4, 1) * 0.2) * 100);
  const model = MODELS[Math.floor(rand() * MODELS.length)].name;

  const shuffledReasons = [...REASON_BANK].sort(() => rand() - 0.5).slice(0, 4);
  const shuffledFeatures = [...FEATURE_BANK]
    .sort(() => rand() - 0.5)
    .slice(0, 8)
    .map((f) => {
      const shap = (rand() - 0.4) * 0.24;
      return { feature: f, importance: Math.abs(shap) + rand() * 0.05, shap };
    })
    .sort((a, b) => b.importance - a.importance);

  return {
    symbol, direction: "BUY", entry, stopLoss, targets: [t1, t2, t3],
    probability, confidence, expectedReturn, riskReward, holdingDays,
    predictionScore: score, modelVersion: model,
    generatedAt: new Date(Date.now() - Math.floor(rand() * 3) * 3600 * 1000),
    reasons: shuffledReasons, featureImportance: shuffledFeatures,
  };
}

function generateHistory(symbol) {
  const rand = hashSeed(symbol + "::history");
  const rows = [];
  const n = 10 + Math.floor(rand() * 6);
  for (let i = 0; i < n; i++) {
    const outcomeRoll = rand();
    const outcome = outcomeRoll > 0.62 ? "target" : outcomeRoll > 0.24 ? "stop" : "open";
    const profitPct = outcome === "target" ? 3 + rand() * 9 : outcome === "stop" ? -(1.5 + rand() * 3) : (rand() - 0.4) * 4;
    const d = new Date();
    d.setDate(d.getDate() - (n - i) * 6 - Math.floor(rand() * 4));
    rows.push({
      id: `${symbol}-${i}`, date: d.toISOString().slice(0, 10),
      predicted: 100 + rand() * 20, actual: 100 + rand() * 20 + profitPct,
      outcome, profitPct, holdingDays: 4 + Math.floor(rand() * 20),
      model: MODELS[Math.floor(rand() * MODELS.length)].name,
    });
  }
  return rows;
}

/* ============================================================================
   MOCK API LAYER
   Shaped to mirror what a real fetch() to quant_trading_system's backend
   would return. Swap the bodies of these functions for real endpoint calls
   (e.g. GET /api/predictions, GET /api/stocks/:symbol/ohlcv) and every
   consumer component below keeps working unchanged.
============================================================================ */
const _seriesCache = new Map();
const mockApi = {
  async getOHLCV(symbol) {
    if (!_seriesCache.has(symbol)) _seriesCache.set(symbol, generateSeries(symbol));
    await sleep(120);
    return _seriesCache.get(symbol);
  },
  async getPrediction(symbol) {
    const series = await this.getOHLCV(symbol);
    await sleep(90);
    return generatePrediction(symbol, series[series.length - 1].close);
  },
  async getPredictionHistory(symbol) {
    await sleep(100);
    return generateHistory(symbol);
  },
  async getDashboard() {
    await sleep(180);
    const preds = await Promise.all(STOCKS.map(async (s) => {
      const series = await this.getOHLCV(s.symbol);
      const p = generatePrediction(s.symbol, series[series.length - 1].close);
      return { ...s, ...p, lastClose: series[series.length - 1].close, prevClose: series[series.length - 2].close };
    }));
    return preds.sort((a, b) => b.predictionScore - a.predictionScore);
  },
};
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

/* ============================================================================
   FORMATTERS
============================================================================ */
const fmtINR = (n) => "₹" + n.toLocaleString("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const fmtPct = (n, digits = 1) => `${n >= 0 ? "+" : ""}${n.toFixed(digits)}%`;
const fmtCompact = (n) => new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(n);

/* ============================================================================
   UI ATOMS
============================================================================ */
function Card({ children, className = "", title, action, dense = false }) {
  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-lg flex flex-col min-h-0 ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 shrink-0">
          <h3 className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">{title}</h3>
          {action}
        </div>
      )}
      <div className={dense ? "p-2" : "p-4"}>{children}</div>
    </div>
  );
}

function Badge({ children, tone = "slate" }) {
  const tones = {
    slate: "bg-slate-800 text-slate-300 border-slate-700",
    emerald: "bg-emerald-950 text-emerald-400 border-emerald-800",
    rose: "bg-rose-950 text-rose-400 border-rose-800",
    amber: "bg-amber-950 text-amber-400 border-amber-800",
    cyan: "bg-cyan-950 text-cyan-400 border-cyan-800",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border ${tones[tone]}`}>
      {children}
    </span>
  );
}

function Stat({ label, value, sub, tone }) {
  const toneClass = tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-slate-100";
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wide text-slate-500">{label}</span>
      <span className={`text-sm font-mono font-semibold ${toneClass}`}>{value}</span>
      {sub && <span className="text-[10px] text-slate-500">{sub}</span>}
    </div>
  );
}

function Skeleton({ className }) {
  return <div className={`animate-pulse bg-slate-800/70 rounded ${className}`} />;
}

function ProbabilityRing({ value, size = 84 }) {
  const r = (size - 10) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value, 0), 1);
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#1e293b" strokeWidth="7" fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={r} stroke={INK.amber} strokeWidth="7" fill="none"
          strokeDasharray={c} strokeDashoffset={c * (1 - pct)} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-mono font-bold text-amber-400">{(pct * 100).toFixed(0)}%</span>
        <span className="text-[9px] text-slate-500 uppercase tracking-wide">Prob.</span>
      </div>
    </div>
  );
}

/* ============================================================================
   TOP NAV + MARKET TICKER
============================================================================ */
function useJitteringIndices() {
  const [indices, setIndices] = useState([
    { name: "NIFTY 50", value: 24812.35, base: 24812.35 },
    { name: "BANK NIFTY", value: 52340.1, base: 52340.1 },
    { name: "SENSEX", value: 81544.2, base: 81544.2 },
    { name: "INDIA VIX", value: 13.42, base: 13.42, isVix: true },
    { name: "USD/INR", value: 85.64, base: 85.64 },
    { name: "GOLD", value: 71230, base: 71230 },
    { name: "CRUDE", value: 6842, base: 6842 },
  ]);
  useEffect(() => {
    const id = setInterval(() => {
      setIndices((prev) =>
        prev.map((idx) => {
          const delta = (Math.random() - 0.5) * (idx.base * 0.0008);
          return { ...idx, value: idx.value + delta };
        })
      );
    }, 2200);
    return () => clearInterval(id);
  }, []);
  return indices;
}

function MarketTicker() {
  const indices = useJitteringIndices();
  return (
    <div className="hidden lg:flex items-center gap-5 px-4 overflow-hidden border-l border-slate-800 h-full">
      {indices.map((idx) => {
        const chg = ((idx.value - idx.base) / idx.base) * 100;
        const up = idx.isVix ? chg < 0 : chg >= 0;
        return (
          <div key={idx.name} className="flex items-center gap-1.5 whitespace-nowrap">
            <span className="text-[10px] text-slate-500 font-medium">{idx.name}</span>
            <span className="text-xs font-mono text-slate-200">
              {idx.value >= 1000 ? idx.value.toFixed(0) : idx.value.toFixed(2)}
            </span>
            <span className={`text-[10px] font-mono flex items-center ${up ? "text-emerald-400" : "text-rose-400"}`}>
              {up ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
              {Math.abs(chg).toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TopNav({ query, setQuery, onSelectSymbol, page }) {
  const [showResults, setShowResults] = useState(false);
  const results = query
    ? STOCKS.filter(
        (s) => s.symbol.toLowerCase().includes(query.toLowerCase()) || s.name.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 6)
    : [];
  return (
    <header className="h-14 border-b border-slate-800 bg-slate-950/95 backdrop-blur flex items-center shrink-0 sticky top-0 z-30">
      <div className="w-56 h-full flex items-center gap-2 px-4 border-r border-slate-800 shrink-0">
        <div className="w-6 h-6 rounded bg-amber-400 flex items-center justify-center">
          <Zap size={14} className="text-slate-950" strokeWidth={2.5} />
        </div>
        <span className="font-semibold text-slate-100 text-sm tracking-tight">QUANTIS</span>
        <Badge tone="slate">RESEARCH</Badge>
      </div>

      <div className="relative flex items-center px-4 w-72 shrink-0">
        <Search size={14} className="absolute left-6 text-slate-500" />
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setShowResults(true); }}
          onFocus={() => setShowResults(true)}
          onBlur={() => setTimeout(() => setShowResults(false), 150)}
          placeholder="Search symbol or company…"
          className="w-full bg-slate-900 border border-slate-800 rounded-md pl-8 pr-10 py-1.5 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-amber-500/50"
        />
        <kbd className="absolute right-6 text-[9px] text-slate-500 border border-slate-700 rounded px-1 py-0.5">/</kbd>
        {showResults && results.length > 0 && (
          <div className="absolute top-11 left-4 w-72 bg-slate-900 border border-slate-800 rounded-md shadow-2xl overflow-hidden z-40">
            {results.map((s) => (
              <button
                key={s.symbol}
                onMouseDown={() => { onSelectSymbol(s.symbol); setQuery(""); setShowResults(false); }}
                className="w-full text-left px-3 py-2 hover:bg-slate-800 flex items-center justify-between"
              >
                <div>
                  <div className="text-xs font-mono text-slate-100">{s.symbol}</div>
                  <div className="text-[10px] text-slate-500">{s.name}</div>
                </div>
                <Badge tone="slate">{s.sector}</Badge>
              </button>
            ))}
          </div>
        )}
      </div>

      <MarketTicker />

      <div className="ml-auto flex items-center gap-3 px-4 shrink-0">
        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-medium">
          <CircleDot size={12} className="animate-pulse" /> MARKET OPEN
        </div>
        <button className="text-slate-400 hover:text-slate-200 relative">
          <Bell size={16} />
          <span className="absolute -top-1 -right-1 w-1.5 h-1.5 rounded-full bg-amber-400" />
        </button>
        <div className="w-7 h-7 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-[10px] text-slate-300 font-semibold">
          QA
        </div>
      </div>
    </header>
  );
}

/* ============================================================================
   SIDEBAR
============================================================================ */
const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, ready: true },
  { id: "live-signals", label: "Live Signals", icon: Radio, ready: true },
  { id: "stock-detail", label: "Stock Research", icon: LineChartIcon, ready: true },
  { id: "research", label: "Research", icon: FlaskConical, ready: false },
  { id: "models", label: "Models", icon: Cpu, ready: false },
  { id: "evaluation", label: "Evaluation", icon: ClipboardList, ready: false },
  { id: "settings", label: "Settings", icon: Settings, ready: false },
];

function Sidebar({ page, setPage }) {
  return (
    <aside className="w-56 shrink-0 border-r border-slate-800 bg-slate-950 flex flex-col py-3">
      <nav className="flex flex-col gap-0.5 px-2">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = page === item.id;
          return (
            <button
              key={item.id}
              disabled={!item.ready}
              onClick={() => item.ready && setPage(item.id)}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors
                ${active ? "bg-amber-400/10 text-amber-400 border border-amber-500/20" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200 border border-transparent"}
                ${!item.ready ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              <Icon size={15} />
              {item.label}
              {!item.ready && <span className="ml-auto text-[8px] text-slate-600 uppercase">soon</span>}
            </button>
          );
        })}
      </nav>
      <div className="mt-auto px-3 pt-3 border-t border-slate-800 mx-2">
        <div className="text-[9px] text-slate-600 leading-relaxed">
          Research & analytics only.<br />Not investment advice. No order execution.
        </div>
      </div>
    </aside>
  );
}

/* ============================================================================
   DASHBOARD PAGE
============================================================================ */
function ChangeCell({ last, prev }) {
  const chg = ((last - prev) / prev) * 100;
  const up = chg >= 0;
  return (
    <span className={`font-mono text-xs flex items-center gap-0.5 ${up ? "text-emerald-400" : "text-rose-400"}`}>
      {up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
      {fmtPct(chg)}
    </span>
  );
}

function TopSignalsTable({ data, onOpen }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
            <th className="text-left font-medium py-1.5">Symbol</th>
            <th className="text-right font-medium py-1.5">LTP</th>
            <th className="text-right font-medium py-1.5">Chg</th>
            <th className="text-right font-medium py-1.5">Prob.</th>
            <th className="text-right font-medium py-1.5">Exp. Return</th>
            <th className="text-right font-medium py-1.5">Score</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 8).map((row) => (
            <tr
              key={row.symbol}
              onClick={() => onOpen(row.symbol)}
              className="border-b border-slate-900 hover:bg-slate-800/50 cursor-pointer"
            >
              <td className="py-1.5">
                <div className="font-mono text-slate-100 font-medium">{row.symbol}</div>
                <div className="text-[9px] text-slate-500">{row.sector}</div>
              </td>
              <td className="text-right font-mono text-slate-300">{fmtINR(row.lastClose)}</td>
              <td className="text-right"><ChangeCell last={row.lastClose} prev={row.prevClose} /></td>
              <td className="text-right font-mono text-amber-400">{(row.probability * 100).toFixed(0)}%</td>
              <td className="text-right font-mono text-emerald-400">{fmtPct(row.expectedReturn)}</td>
              <td className="text-right">
                <span className="font-mono text-slate-200 font-semibold">{row.predictionScore}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RankedStocksList({ data, onOpen }) {
  const max = Math.max(...data.map((d) => d.predictionScore));
  return (
    <div className="flex flex-col gap-1.5">
      {data.slice(0, 10).map((row, i) => (
        <button key={row.symbol} onClick={() => onOpen(row.symbol)} className="flex items-center gap-2 group text-left">
          <span className="text-[10px] text-slate-600 w-4 font-mono">{i + 1}</span>
          <span className="text-xs font-mono text-slate-200 w-24 group-hover:text-amber-400 shrink-0">{row.symbol}</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-amber-600 to-amber-400 rounded-full" style={{ width: `${(row.predictionScore / max) * 100}%` }} />
          </div>
          <span className="text-[10px] font-mono text-slate-400 w-7 text-right">{row.predictionScore}</span>
        </button>
      ))}
    </div>
  );
}

function ProbabilityDistribution({ data }) {
  const buckets = Array.from({ length: 8 }, (_, i) => ({ range: `${50 + i * 5}-${55 + i * 5}`, count: 0 }));
  data.forEach((d) => {
    const idx = Math.min(7, Math.max(0, Math.floor((d.probability * 100 - 50) / 5)));
    buckets[idx].count += 1;
  });
  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={buckets} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
        <XAxis dataKey="range" tick={{ fontSize: 9, fill: INK.axis }} axisLine={{ stroke: INK.grid }} tickLine={false} />
        <YAxis tick={{ fontSize: 9, fill: INK.axis }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11, borderRadius: 6 }} labelStyle={{ color: "#e2e8f0" }} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {buckets.map((_, i) => <Cell key={i} fill={i > 4 ? INK.amber : "#475569"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function SectorHeatmap({ data }) {
  const bySector = {};
  data.forEach((d) => {
    if (!bySector[d.sector]) bySector[d.sector] = [];
    bySector[d.sector].push(((d.lastClose - d.prevClose) / d.prevClose) * 100);
  });
  const sectors = Object.entries(bySector).map(([name, vals]) => ({
    name, avg: vals.reduce((a, b) => a + b, 0) / vals.length, count: vals.length,
  }));
  const maxAbs = Math.max(...sectors.map((s) => Math.abs(s.avg)), 1);
  return (
    <div className="grid grid-cols-3 gap-1.5">
      {sectors.map((s) => {
        const intensity = Math.min(Math.abs(s.avg) / maxAbs, 1);
        const bg = s.avg >= 0
          ? `rgba(52,211,153,${0.15 + intensity * 0.55})`
          : `rgba(251,113,133,${0.15 + intensity * 0.55})`;
        return (
          <div key={s.name} style={{ background: bg }} className="rounded-md p-2 flex flex-col justify-between h-16 border border-slate-800/50">
            <span className="text-[9px] text-slate-200 font-medium leading-tight">{s.name}</span>
            <span className="text-xs font-mono font-semibold text-slate-100">{fmtPct(s.avg)}</span>
          </div>
        );
      })}
    </div>
  );
}

function MarketBreadth() {
  const rand = useMemo(() => hashSeed("breadth" + new Date().toDateString()), []);
  const advances = Math.round(1100 + rand() * 500);
  const declines = Math.round(1800 - advances);
  const data = [{ name: "Advances", value: advances }, { name: "Declines", value: declines }];
  return (
    <div className="flex items-center gap-4">
      <ResponsiveContainer width={90} height={90}>
        <PieChart>
          <Pie data={data} dataKey="value" innerRadius={28} outerRadius={42} startAngle={90} endAngle={-270}>
            <Cell fill={INK.emerald} />
            <Cell fill={INK.rose} />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="flex flex-col gap-2 text-xs">
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-400" /> Advances <span className="ml-auto font-mono text-slate-200">{advances}</span></div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-rose-400" /> Declines <span className="ml-auto font-mono text-slate-200">{declines}</span></div>
        <div className="text-[10px] text-slate-500">A/D Ratio: {(advances / declines).toFixed(2)}</div>
      </div>
    </div>
  );
}

function AccuracySummary() {
  const spark = useMemo(() => {
    const rand = hashSeed("acc-spark");
    let v = 60;
    return Array.from({ length: 30 }, (_, i) => {
      v += (rand() - 0.45) * 3;
      v = Math.max(50, Math.min(78, v));
      return { i, v };
    });
  }, []);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end gap-3">
        <span className="text-2xl font-mono font-bold text-slate-100">67.8%</span>
        <span className="text-[11px] text-emerald-400 mb-1 font-mono">+1.4% (30d)</span>
      </div>
      <ResponsiveContainer width="100%" height={44}>
        <AreaChart data={spark}>
          <defs>
            <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={INK.amber} stopOpacity={0.4} />
              <stop offset="100%" stopColor={INK.amber} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey="v" stroke={INK.amber} strokeWidth={1.5} fill="url(#accGrad)" />
        </AreaChart>
      </ResponsiveContainer>
      <div className="grid grid-cols-3 gap-2 pt-1 border-t border-slate-800 mt-1">
        <Stat label="Hit Rate" value="67.8%" />
        <Stat label="Avg RR" value="2.3x" />
        <Stat label="Predictions" value="1,248" />
      </div>
    </div>
  );
}

function FeatureImportancePanel() {
  const feats = useMemo(() => {
    const rand = hashSeed("global-feat");
    return FEATURE_BANK.map((f) => ({ feature: f, importance: 0.05 + rand() * 0.3 }))
      .sort((a, b) => b.importance - a.importance).slice(0, 6);
  }, []);
  const max = Math.max(...feats.map((f) => f.importance));
  return (
    <div className="flex flex-col gap-1.5">
      {feats.map((f) => (
        <div key={f.feature} className="flex items-center gap-2">
          <span className="text-[10px] text-slate-400 w-32 truncate">{f.feature}</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-cyan-400/70 rounded-full" style={{ width: `${(f.importance / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function ModelHealth() {
  return (
    <div className="grid grid-cols-1 gap-1.5">
      {MODELS.map((m) => (
        <div key={m.name} className="flex items-center justify-between px-2 py-1.5 rounded bg-slate-800/40">
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${m.status === "healthy" ? "bg-emerald-400" : "bg-amber-400"}`} />
            <span className="text-[11px] text-slate-200 font-mono">{m.name}</span>
          </div>
          <span className="text-[10px] font-mono text-slate-400">Acc {(m.acc * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function RecentPredictions({ data, onOpen }) {
  return (
    <div className="flex flex-col gap-1">
      {data.slice(0, 6).map((row) => (
        <button key={row.symbol} onClick={() => onOpen(row.symbol)} className="flex items-center justify-between px-1 py-1 hover:bg-slate-800/40 rounded text-left">
          <div className="flex items-center gap-2">
            <Badge tone="emerald">BUY</Badge>
            <span className="text-xs font-mono text-slate-200">{row.symbol}</span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono">{row.holdingDays}d hold</span>
          <span className="text-[10px] font-mono text-amber-400">{(row.probability * 100).toFixed(0)}%</span>
        </button>
      ))}
    </div>
  );
}

function Dashboard({ data, loading, onOpen }) {
  if (loading || !data) {
    return (
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-40" />)}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card title="Today's Top Buy Signals" className="col-span-8 row-span-2">
        <TopSignalsTable data={data} onOpen={onOpen} />
      </Card>
      <Card title="Top Ranked Stocks" className="col-span-4 row-span-2">
        <RankedStocksList data={data} onOpen={onOpen} />
      </Card>

      <Card title="Prediction Accuracy" className="col-span-4"><AccuracySummary /></Card>
      <Card title="Market Breadth" className="col-span-4"><MarketBreadth /></Card>
      <Card title="Model Health" className="col-span-4"><ModelHealth /></Card>

      <Card title="Probability Distribution" className="col-span-4"><ProbabilityDistribution data={data} /></Card>
      <Card title="Sector Heatmap" className="col-span-4"><SectorHeatmap data={data} /></Card>
      <Card title="Global Feature Importance" className="col-span-4"><FeatureImportancePanel /></Card>

      <Card title="Recent Predictions" className="col-span-6"><RecentPredictions data={data} onOpen={onOpen} /></Card>
      <Card title="News Sentiment" className="col-span-6">
        <div className="flex flex-col gap-1.5 text-[11px]">
          {["Positive FII flows lift banking pack sentiment", "IT services commentary steady into earnings season", "Auto sector sentiment mixed on input cost concerns"].map((n, i) => (
            <div key={i} className="flex items-start gap-2">
              <Badge tone={i === 2 ? "amber" : "emerald"}>{i === 2 ? "MIXED" : "POS"}</Badge>
              <span className="text-slate-400">{n}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ============================================================================
   LIVE SIGNALS PAGE
============================================================================ */
function LiveSignals({ dashboardData, onOpen }) {
  const [feed, setFeed] = useState([]);
  const [minProb, setMinProb] = useState(0.5);
  const [sortKey, setSortKey] = useState("time");
  const counterRef = useRef(0);

  useEffect(() => {
    if (!dashboardData) return;
    const id = setInterval(() => {
      const rand = Math.random();
      const stock = dashboardData[Math.floor(rand * dashboardData.length)];
      counterRef.current += 1;
      setFeed((prev) => [
        { ...stock, id: counterRef.current, time: new Date() },
        ...prev,
      ].slice(0, 40));
    }, 3200);
    return () => clearInterval(id);
  }, [dashboardData]);

  const filtered = feed.filter((f) => f.probability >= minProb);
  const sorted = [...filtered].sort((a, b) => {
    if (sortKey === "time") return b.time - a.time;
    if (sortKey === "prob") return b.probability - a.probability;
    if (sortKey === "score") return b.predictionScore - a.predictionScore;
    return 0;
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 rounded-lg px-4 py-3">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium">
          <Radio size={14} className="animate-pulse" /> LIVE FEED
        </div>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <SlidersHorizontal size={12} className="text-slate-500" />
          <span className="text-[10px] text-slate-500 whitespace-nowrap">Min prob. {(minProb * 100).toFixed(0)}%</span>
          <input type="range" min="0.5" max="0.9" step="0.01" value={minProb} onChange={(e) => setMinProb(parseFloat(e.target.value))} className="w-full accent-amber-400" />
        </div>
        <div className="ml-auto flex items-center gap-1">
          {["time", "prob", "score"].map((k) => (
            <button key={k} onClick={() => setSortKey(k)} className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${sortKey === k ? "bg-amber-400/10 text-amber-400 border border-amber-500/30" : "text-slate-500 hover:text-slate-300"}`}>
              {k}
            </button>
          ))}
        </div>
        <span className="text-[10px] text-slate-500 font-mono">{sorted.length} signals</span>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
                <th className="text-left font-medium py-2">Time</th>
                <th className="text-left font-medium py-2">Symbol</th>
                <th className="text-left font-medium py-2">Sector</th>
                <th className="text-right font-medium py-2">Entry</th>
                <th className="text-right font-medium py-2">Stop</th>
                <th className="text-right font-medium py-2">Target 2</th>
                <th className="text-right font-medium py-2">Prob.</th>
                <th className="text-right font-medium py-2">Confidence</th>
                <th className="text-right font-medium py-2">RR</th>
                <th className="text-right font-medium py-2">Score</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 && (
                <tr><td colSpan={10} className="text-center text-slate-600 py-8 text-[11px]">Waiting for signals that match your filter…</td></tr>
              )}
              {sorted.map((row) => (
                <tr key={row.id} onClick={() => onOpen(row.symbol)} className="border-b border-slate-900 hover:bg-slate-800/50 cursor-pointer animate-[fadeIn_0.3s_ease]">
                  <td className="py-2 font-mono text-slate-500">{row.time.toLocaleTimeString("en-IN", { hour12: false })}</td>
                  <td className="py-2 font-mono text-slate-100 font-medium">{row.symbol}</td>
                  <td className="py-2"><Badge tone="slate">{row.sector}</Badge></td>
                  <td className="py-2 text-right font-mono text-slate-300">{fmtINR(row.entry)}</td>
                  <td className="py-2 text-right font-mono text-rose-400">{fmtINR(row.stopLoss)}</td>
                  <td className="py-2 text-right font-mono text-emerald-400">{fmtINR(row.targets[1])}</td>
                  <td className="py-2 text-right font-mono text-amber-400">{(row.probability * 100).toFixed(0)}%</td>
                  <td className="py-2 text-right font-mono text-slate-300">{(row.confidence * 100).toFixed(0)}%</td>
                  <td className="py-2 text-right font-mono text-slate-300">{row.riskReward.toFixed(2)}x</td>
                  <td className="py-2 text-right font-mono text-slate-100 font-semibold">{row.predictionScore}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ============================================================================
   STOCK DETAIL — CHART
============================================================================ */
function CandlestickChart({ series, prediction, overlays, timeframe }) {
  const w = 900, h = 320, padTop = 12, padBottom = 8, volH = 56;
  const sliceMap = { "3M": 66, "6M": 132, "1Y": 220 };
  const visible = series.slice(-sliceMap[timeframe]);
  const closes = visible.map((d) => d.close);
  const highs = visible.map((d) => d.high);
  const lows = visible.map((d) => d.low);

  const predVals = prediction ? [prediction.entry, prediction.stopLoss, ...prediction.targets] : [];
  let hi = Math.max(...highs, ...predVals);
  let lo = Math.min(...lows, ...predVals);
  const pad = (hi - lo) * 0.06;
  hi += pad; lo -= pad;

  const emaFast = overlays.ema20 ? ema(visible, 20) : null;
  const emaSlow = overlays.ema50 ? ema(visible, 50) : null;
  const vwap = overlays.vwap ? vwapSeries(visible) : null;
  const bb = overlays.bollinger ? bollinger(visible) : null;

  const priceH = h - volH - padTop - padBottom;
  const x = (i) => (i / (visible.length - 1)) * w;
  const y = (v) => padTop + priceH * (1 - (v - lo) / (hi - lo));
  const maxVol = Math.max(...visible.map((d) => d.volume));
  const yVol = (v) => h - (v / maxVol) * (volH - 4);
  const candleW = Math.max(1.5, (w / visible.length) * 0.6);

  const linePath = (vals) => vals.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto select-none" preserveAspectRatio="none">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <line key={f} x1={0} x2={w} y1={padTop + priceH * f} y2={padTop + priceH * f} stroke="#1e293b" strokeWidth="1" />
      ))}

      {bb && (
        <>
          <path d={linePath(bb.map((b) => b.upper ?? lo))} stroke="#475569" strokeWidth="1" fill="none" strokeDasharray="2,2" />
          <path d={linePath(bb.map((b) => b.lower ?? lo))} stroke="#475569" strokeWidth="1" fill="none" strokeDasharray="2,2" />
        </>
      )}

      {visible.map((d, i) => {
        const up = d.close >= d.open;
        return (
          <g key={i}>
            <line x1={x(i)} x2={x(i)} y1={y(d.high)} y2={y(d.low)} stroke={up ? INK.emerald : INK.rose} strokeWidth="1" />
            <rect
              x={x(i) - candleW / 2} y={y(Math.max(d.open, d.close))}
              width={candleW} height={Math.max(1, Math.abs(y(d.open) - y(d.close)))}
              fill={up ? INK.emerald : INK.rose}
            />
            <rect x={x(i) - candleW / 2} y={h - 2 - (d.volume / maxVol) * (volH - 4)} width={candleW} height={(d.volume / maxVol) * (volH - 4)} fill={up ? "#34d39955" : "#fb718555"} />
          </g>
        );
      })}

      {emaFast && <path d={linePath(emaFast)} stroke={INK.cyan} strokeWidth="1.5" fill="none" />}
      {emaSlow && <path d={linePath(emaSlow)} stroke={INK.violet} strokeWidth="1.5" fill="none" />}
      {vwap && <path d={linePath(vwap)} stroke={INK.amber} strokeWidth="1.2" fill="none" strokeDasharray="4,2" />}

      {prediction && [
        { v: prediction.entry, color: "#e2e8f0", label: "ENTRY" },
        { v: prediction.stopLoss, color: INK.rose, label: "SL" },
        { v: prediction.targets[0], color: INK.emerald, label: "T1" },
        { v: prediction.targets[1], color: INK.emerald, label: "T2" },
        { v: prediction.targets[2], color: INK.emerald, label: "T3" },
      ].map((m) => (
        <g key={m.label}>
          <line x1={0} x2={w} y1={y(m.v)} y2={y(m.v)} stroke={m.color} strokeWidth="1" strokeDasharray="4,3" opacity="0.85" />
          <rect x={w - 66} y={y(m.v) - 8} width={64} height={14} fill={m.color} opacity="0.15" />
          <text x={w - 4} y={y(m.v) + 3} textAnchor="end" fontSize="9" fontFamily="monospace" fill={m.color}>
            {m.label} {m.v.toFixed(1)}
          </text>
        </g>
      ))}
    </svg>
  );
}

function OscillatorChart({ series, timeframe, mode }) {
  const w = 900, h = 90;
  const sliceMap = { "3M": 66, "6M": 132, "1Y": 220 };
  const visible = series.slice(-sliceMap[timeframe]);
  const x = (i) => (i / (visible.length - 1)) * w;

  if (mode === "rsi") {
    const rsi = rsiSeries(visible);
    const valid = rsi.map((v) => v ?? 50);
    const y = (v) => h - (v / 100) * h;
    const path = valid.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto" preserveAspectRatio="none">
        <line x1={0} x2={w} y1={y(70)} y2={y(70)} stroke="#475569" strokeDasharray="3,2" strokeWidth="1" />
        <line x1={0} x2={w} y1={y(30)} y2={y(30)} stroke="#475569" strokeDasharray="3,2" strokeWidth="1" />
        <path d={path} stroke={INK.cyan} strokeWidth="1.5" fill="none" />
        <text x={4} y={10} fontSize="9" fill="#64748b" fontFamily="monospace">RSI(14)</text>
      </svg>
    );
  }
  const macd = macdSeries(visible);
  const vals = macd.map((m) => m.hist);
  const maxAbs = Math.max(...vals.map(Math.abs), 0.01);
  const y = (v) => h / 2 - (v / maxAbs) * (h / 2 - 6);
  const macdLine = macd.map((m) => m.macd);
  const sigLine = macd.map((m) => m.signal);
  const maxLineAbs = Math.max(...macdLine.map(Math.abs), ...sigLine.map(Math.abs), 0.01);
  const yLine = (v) => h / 2 - (v / maxLineAbs) * (h / 2 - 6);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto" preserveAspectRatio="none">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke="#334155" strokeWidth="1" />
      {vals.map((v, i) => (
        <rect key={i} x={x(i) - 1.5} y={Math.min(y(v), h / 2)} width={3} height={Math.abs(y(v) - h / 2)} fill={v >= 0 ? INK.emerald : INK.rose} opacity="0.6" />
      ))}
      <path d={macdLine.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${yLine(v)}`).join(" ")} stroke={INK.cyan} strokeWidth="1.2" fill="none" />
      <path d={sigLine.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${yLine(v)}`).join(" ")} stroke={INK.amber} strokeWidth="1.2" fill="none" />
      <text x={4} y={10} fontSize="9" fill="#64748b" fontFamily="monospace">MACD(12,26,9)</text>
    </svg>
  );
}

/* ============================================================================
   STOCK DETAIL — PREDICTION CARD + TABS
============================================================================ */
function PredictionCard({ prediction, loading }) {
  if (loading || !prediction) {
    return <Card title="AI Prediction"><Skeleton className="h-64" /></Card>;
  }
  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <Badge tone="emerald"><TrendingUp size={11} /> BUY SIGNAL</Badge>
        <span className="text-[10px] text-slate-500 font-mono">{prediction.modelVersion}</span>
      </div>
      <div className="flex items-center gap-4 mb-4">
        <ProbabilityRing value={prediction.probability} />
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 flex-1">
          <Stat label="Confidence" value={`${(prediction.confidence * 100).toFixed(0)}%`} />
          <Stat label="Exp. Return" value={fmtPct(prediction.expectedReturn)} tone="up" />
          <Stat label="Risk:Reward" value={`${prediction.riskReward.toFixed(2)}x`} />
          <Stat label="Hold Period" value={`${prediction.holdingDays}d`} />
        </div>
      </div>

      <div className="border-t border-slate-800 pt-3 flex flex-col gap-2 mb-4">
        <PriceRow label="Entry" value={prediction.entry} tone="slate" icon={Target} />
        <PriceRow label="Stop Loss" value={prediction.stopLoss} tone="rose" icon={ShieldAlert} />
        <PriceRow label="Target 1" value={prediction.targets[0]} tone="emerald" icon={ArrowUpRight} />
        <PriceRow label="Target 2" value={prediction.targets[1]} tone="emerald" icon={ArrowUpRight} />
        <PriceRow label="Target 3" value={prediction.targets[2]} tone="emerald" icon={ArrowUpRight} />
      </div>

      <div className="border-t border-slate-800 pt-3">
        <div className="flex items-center gap-1.5 mb-2 text-[10px] uppercase tracking-wide text-slate-500 font-semibold">
          <BrainCircuit size={12} /> Why this signal
        </div>
        <ul className="flex flex-col gap-1.5">
          {prediction.reasons.map((r, i) => (
            <li key={i} className="text-[11px] text-slate-400 flex items-start gap-1.5">
              <CircleDot size={9} className="text-amber-400 mt-0.5 shrink-0" /> {r}
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t border-slate-800 mt-3 pt-2 flex items-center justify-between text-[10px] text-slate-600 font-mono">
        <span>Prediction Score</span>
        <span className="text-slate-200 font-semibold">{prediction.predictionScore}/100</span>
      </div>
    </Card>
  );
}

function PriceRow({ label, value, tone, icon: Icon }) {
  const toneClass = { slate: "text-slate-200", rose: "text-rose-400", emerald: "text-emerald-400" }[tone];
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] text-slate-500 flex items-center gap-1.5"><Icon size={11} /> {label}</span>
      <span className={`text-xs font-mono font-semibold ${toneClass}`}>{fmtINR(value)}</span>
    </div>
  );
}

function ExplainabilityTab({ prediction }) {
  const max = Math.max(...prediction.featureImportance.map((f) => f.importance));
  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <h4 className="text-[11px] uppercase tracking-wide text-slate-500 mb-3 font-semibold">SHAP Feature Contributions</h4>
        <div className="flex flex-col gap-2">
          {prediction.featureImportance.map((f) => (
            <div key={f.feature} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 w-36 truncate">{f.feature}</span>
              <div className="flex-1 h-4 relative bg-slate-800/50 rounded overflow-hidden">
                <div
                  className={`h-full absolute top-0 ${f.shap >= 0 ? "bg-emerald-400/70 left-1/2" : "bg-rose-400/70 right-1/2"}`}
                  style={{ width: `${(Math.abs(f.shap) / max) * 50}%` }}
                />
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600" />
              </div>
              <span className={`text-[10px] font-mono w-12 text-right ${f.shap >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {f.shap >= 0 ? "+" : ""}{f.shap.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h4 className="text-[11px] uppercase tracking-wide text-slate-500 mb-3 font-semibold">Confidence Explanation</h4>
        <p className="text-xs text-slate-400 leading-relaxed mb-4">
          The model's confidence reflects agreement across the ensemble and the historical reliability of similar
          feature patterns. Higher confidence indicates the current setup closely resembles training examples with
          consistent outcomes.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Model Version" value={prediction.modelVersion} />
          <Stat label="Prediction Score" value={`${prediction.predictionScore}/100`} />
          <Stat label="Confidence" value={`${(prediction.confidence * 100).toFixed(1)}%`} />
          <Stat label="Generated" value={prediction.generatedAt.toLocaleTimeString("en-IN", { hour12: false })} />
        </div>
      </div>
    </div>
  );
}

function HistoryTab({ history, loading }) {
  if (loading) return <Skeleton className="h-48" />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
            <th className="text-left font-medium py-2">Date</th>
            <th className="text-right font-medium py-2">Predicted</th>
            <th className="text-right font-medium py-2">Actual</th>
            <th className="text-right font-medium py-2">Outcome</th>
            <th className="text-right font-medium py-2">P/L %</th>
            <th className="text-right font-medium py-2">Holding</th>
            <th className="text-right font-medium py-2">Model</th>
          </tr>
        </thead>
        <tbody>
          {history.map((row) => (
            <tr key={row.id} className="border-b border-slate-900">
              <td className="py-2 font-mono text-slate-400">{row.date}</td>
              <td className="py-2 text-right font-mono text-slate-400">{row.predicted.toFixed(1)}</td>
              <td className="py-2 text-right font-mono text-slate-200">{row.actual.toFixed(1)}</td>
              <td className="py-2 text-right">
                <Badge tone={row.outcome === "target" ? "emerald" : row.outcome === "stop" ? "rose" : "slate"}>
                  {row.outcome === "target" ? "TARGET HIT" : row.outcome === "stop" ? "STOP HIT" : "OPEN"}
                </Badge>
              </td>
              <td className={`py-2 text-right font-mono ${row.profitPct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>{fmtPct(row.profitPct)}</td>
              <td className="py-2 text-right font-mono text-slate-400">{row.holdingDays}d</td>
              <td className="py-2 text-right font-mono text-slate-500">{row.model}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StockDetail({ symbol, watchlist, toggleWatchlist }) {
  const [series, setSeries] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState("6M");
  const [tab, setTab] = useState("overview");
  const [overlays, setOverlays] = useState({ ema20: true, ema50: true, vwap: false, bollinger: false });
  const [oscMode, setOscMode] = useState("rsi");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([mockApi.getOHLCV(symbol), mockApi.getPrediction(symbol), mockApi.getPredictionHistory(symbol)]).then(
      ([s, p, h]) => { if (!cancelled) { setSeries(s); setPrediction(p); setHistory(h); setLoading(false); } }
    );
    return () => { cancelled = true; };
  }, [symbol]);

  const meta = STOCKS.find((s) => s.symbol === symbol);
  const last = series ? series[series.length - 1] : null;
  const prev = series ? series[series.length - 2] : null;

  const toggleOverlay = (key) => setOverlays((o) => ({ ...o, [key]: !o[key] }));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <button onClick={() => toggleWatchlist(symbol)}>
          <Star size={18} className={watchlist.has(symbol) ? "fill-amber-400 text-amber-400" : "text-slate-600"} />
        </button>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-slate-100 font-mono">{symbol}</h2>
            <Badge tone="slate">{meta?.sector}</Badge>
          </div>
          <span className="text-[11px] text-slate-500">{meta?.name}</span>
        </div>
        {last && (
          <div className="ml-4 flex items-center gap-2">
            <span className="text-xl font-mono font-bold text-slate-100">{fmtINR(last.close)}</span>
            <ChangeCell last={last.close} prev={prev.close} />
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          {["3M", "6M", "1Y"].map((tf) => (
            <button key={tf} onClick={() => setTimeframe(tf)} className={`text-[10px] px-2.5 py-1 rounded font-medium ${timeframe === tf ? "bg-amber-400/10 text-amber-400 border border-amber-500/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}>
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-8 flex flex-col gap-3">
          <Card>
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              {[
                { key: "ema20", label: "EMA 20", color: "text-cyan-400" },
                { key: "ema50", label: "EMA 50", color: "text-violet-400" },
                { key: "vwap", label: "VWAP", color: "text-amber-400" },
                { key: "bollinger", label: "Bollinger", color: "text-slate-400" },
              ].map((ov) => (
                <button
                  key={ov.key} onClick={() => toggleOverlay(ov.key)}
                  className={`text-[10px] px-2 py-1 rounded border font-medium flex items-center gap-1 ${overlays[ov.key] ? `border-slate-700 bg-slate-800 ${ov.color}` : "border-transparent text-slate-600"}`}
                >
                  <span className="w-2 h-0.5 bg-current inline-block" /> {ov.label}
                </button>
              ))}
              <div className="ml-auto flex items-center gap-2 text-[10px] text-slate-500">
                <Badge tone="emerald">ENTRY / SL / T1-T3 overlay live</Badge>
              </div>
            </div>
            {loading || !series ? <Skeleton className="h-80" /> : (
              <CandlestickChart series={series} prediction={prediction} overlays={overlays} timeframe={timeframe} />
            )}
          </Card>
          <Card>
            <div className="flex items-center gap-2 mb-1">
              {["rsi", "macd"].map((m) => (
                <button key={m} onClick={() => setOscMode(m)} className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${oscMode === m ? "bg-slate-800 text-slate-100" : "text-slate-500"}`}>{m}</button>
              ))}
            </div>
            {loading || !series ? <Skeleton className="h-24" /> : <OscillatorChart series={series} timeframe={timeframe} mode={oscMode} />}
          </Card>

          <Card className="p-0">
            <div className="flex border-b border-slate-800 px-2 overflow-x-auto">
              {[
                { id: "overview", label: "Overview" },
                { id: "technicals", label: "Technicals" },
                { id: "explainability", label: "Explainability" },
                { id: "history", label: "History" },
              ].map((t) => (
                <button
                  key={t.id} onClick={() => setTab(t.id)}
                  className={`px-3 py-2.5 text-xs font-medium border-b-2 -mb-px ${tab === t.id ? "border-amber-400 text-amber-400" : "border-transparent text-slate-500 hover:text-slate-300"}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="p-4">
              {tab === "overview" && prediction && (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="52W High" value={fmtINR(Math.max(...series.map((d) => d.high)))} />
                  <Stat label="52W Low" value={fmtINR(Math.min(...series.map((d) => d.low)))} />
                  <Stat label="Avg Volume" value={fmtCompact(series.reduce((a, d) => a + d.volume, 0) / series.length)} />
                  <Stat label="Prediction Score" value={`${prediction.predictionScore}/100`} />
                  <Stat label="Model" value={prediction.modelVersion} />
                  <Stat label="Sector" value={meta?.sector} />
                </div>
              )}
              {tab === "technicals" && (
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <Stat label="RSI(14)" value={rsiSeries(series).filter(Boolean).slice(-1)[0]?.toFixed(1) ?? "—"} />
                  <Stat label="EMA20" value={fmtINR(ema(series, 20).slice(-1)[0])} />
                  <Stat label="EMA50" value={fmtINR(ema(series, 50).slice(-1)[0])} />
                  <Stat label="VWAP" value={fmtINR(vwapSeries(series).slice(-1)[0])} />
                  <Stat label="Bollinger Upper" value={fmtINR(bollinger(series).slice(-1)[0].upper)} />
                  <Stat label="Bollinger Lower" value={fmtINR(bollinger(series).slice(-1)[0].lower)} />
                </div>
              )}
              {tab === "explainability" && prediction && <ExplainabilityTab prediction={prediction} />}
              {tab === "history" && <HistoryTab history={history || []} loading={loading} />}
            </div>
          </Card>
        </div>

        <div className="col-span-4">
          <PredictionCard prediction={prediction} loading={loading} />
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   APP ROOT
============================================================================ */
export default function QuantTerminal() {
  const [page, setPage] = useState("dashboard");
  const [symbol, setSymbol] = useState("RELIANCE");
  const [query, setQuery] = useState("");
  const [watchlist, setWatchlist] = useState(new Set());
  const [dashboardData, setDashboardData] = useState(null);
  const [loadingDash, setLoadingDash] = useState(true);

  useEffect(() => {
    mockApi.getDashboard().then((d) => { setDashboardData(d); setLoadingDash(false); });
  }, []);

  const openStock = useCallback((sym) => { setSymbol(sym); setPage("stock-detail"); }, []);
  const toggleWatchlist = useCallback((sym) => {
    setWatchlist((prev) => { const next = new Set(prev); next.has(sym) ? next.delete(sym) : next.add(sym); return next; });
  }, []);

  return (
    <div className="h-screen w-full bg-slate-950 text-slate-200 flex flex-col font-sans overflow-hidden">
      <style>{`@keyframes fadeIn { from { opacity: 0; background-color: rgba(251,191,36,0.08);} to { opacity: 1; } }`}</style>
      <TopNav query={query} setQuery={setQuery} onSelectSymbol={openStock} page={page} />
      <div className="flex flex-1 min-h-0">
        <Sidebar page={page} setPage={setPage} />
        <main className="flex-1 overflow-y-auto p-5">
          <div className="flex items-center gap-2 mb-4 text-[11px] text-slate-500">
            <span className="capitalize">{page.replace("-", " ")}</span>
            <ChevronRight size={12} />
            <span className="text-slate-300">{page === "stock-detail" ? symbol : "Overview"}</span>
            <span className="ml-auto flex items-center gap-1.5">
              <Info size={11} /> Research & analytics only — not investment advice
            </span>
          </div>

          {page === "dashboard" && <Dashboard data={dashboardData} loading={loadingDash} onOpen={openStock} />}
          {page === "live-signals" && <LiveSignals dashboardData={dashboardData} onOpen={openStock} />}
          {page === "stock-detail" && <StockDetail symbol={symbol} watchlist={watchlist} toggleWatchlist={toggleWatchlist} />}
        </main>
      </div>
    </div>
  );
}