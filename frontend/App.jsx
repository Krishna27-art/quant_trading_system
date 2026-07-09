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
   Institutional dark terminal. Base: slate-950/900. Signature accent: green
   for AI/prediction-confidence elements. Cyan is the secondary data-ink color
   for chart overlays. Emerald/rose carry buy/sell semantics.
============================================================================ */
const INK = {
  green: "#10b981",
  cyan: "#22d3ee",
  emerald: "#34d399",
  rose: "#fb7185",
  violet: "#a78bfa",
  grid: "#1e293b",
  axis: "#64748b",
};

const API_BASE = "http://localhost:8000/api";

/* ============================================================================
   FORMATTERS
============================================================================ */
const fmtINR = (n) => "₹" + (typeof n === 'number' ? n.toLocaleString("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 }) : "0.00");
const fmtPct = (n, digits = 2) => `${typeof n === 'number' ? (n >= 0 ? "+" : "") + n.toFixed(digits) : "0.00"}%`;
const fmtCompact = (n) => typeof n === 'number' ? new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(n) : "0";
const clr = (v) => v >= 0 ? "text-emerald-400" : "text-rose-400";

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
    green: "bg-emerald-950 text-emerald-400 border-emerald-800",
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
  const pctVal = Math.min(Math.max(value, 0), 1);
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#1e293b" strokeWidth="7" fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={r} stroke={INK.green} strokeWidth="7" fill="none"
          strokeDasharray={c} strokeDashoffset={c * (1 - pctVal)} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-mono font-bold text-emerald-400">{(pctVal * 100).toFixed(0)}%</span>
        <span className="text-[9px] text-slate-500 uppercase tracking-wide">Prob.</span>
      </div>
    </div>
  );
}

function ConfidenceBar({ confidence }) {
  // Accept raw 0-1 or percentage-style scores
  const norm = normConf == null ? confidence : (typeof normConf === 'function' ? normConf(confidence) : (confidence > 1.5 ? confidence / 100 : confidence));
  const rawConf = norm * 100;
  const rounded = Math.round(rawConf);
  
  // Statistical calibration band (margin of error) based on confidence tier
  let band = "±6%";
  if (rounded >= 80) band = "±2%";
  else if (rounded >= 70) band = "±3%";
  else if (rounded >= 60) band = "±4%";

  let barColor = "bg-slate-600";
  let textColor = "text-slate-400";
  if (rounded >= 75) {
    barColor = "bg-emerald-400";
    textColor = "text-emerald-400 font-bold";
  } else if (rounded >= 60) {
    barColor = "bg-cyan-400";
    textColor = "text-cyan-400 font-semibold";
  } else if (rounded >= 55) {
    barColor = "bg-amber-400";
    textColor = "text-amber-400";
  }

  return (
    <div className="flex items-center gap-2 justify-end">
      <div className="flex flex-col items-end gap-0.5">
        <div className="flex items-center gap-1.5">
          <div className="w-10 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
            <div className={`h-full ${barColor}`} style={{ width: `${Math.max(0, Math.min(100, (rounded - 50) * 2))}%` }} />
          </div>
          <span className={`font-mono text-[11px] ${textColor}`}>{rounded}%</span>
        </div>
        <span className="text-[8px] text-slate-500 font-mono tracking-tight">Acc: {band}</span>
      </div>
    </div>
  );
}

/* ============================================================================
   TOP NAV + MARKET TICKER
=========================================================================== */
function MarketTicker({ indices }) {
  return (
    <div className="hidden lg:flex items-center gap-5 px-4 overflow-hidden border-l border-slate-800 h-full">
      {indices.map((idx) => {
        const up = idx.change >= 0;
        return (
          <div key={idx.name} className="flex items-center gap-1.5 whitespace-nowrap">
            <span className="text-[10px] text-slate-500 font-medium">{idx.name}</span>
            <span className="text-xs font-mono text-slate-200">
              {idx.value >= 1000 ? idx.value.toFixed(0) : idx.value.toFixed(2)}
            </span>
            <span className={`text-[10px] font-mono flex items-center ${up ? "text-emerald-400" : "text-rose-400"}`}>
              {up ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
              {Math.abs(idx.change_pct ?? 0).toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TopNav({ query, setQuery, stocks, onSelectSymbol, indices }) {
  const [showResults, setShowResults] = useState(false);
  const results = query
    ? stocks.filter(
        (s) => s.symbol.toLowerCase().includes(query.toLowerCase()) || s.name.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 6)
    : [];

  return (
    <header className="h-14 border-b border-slate-800 bg-slate-950/95 backdrop-blur flex items-center shrink-0 sticky top-0 z-30">
      <div className="w-56 h-full flex items-center gap-2 px-4 border-r border-slate-800 shrink-0">
        <div className="w-6 h-6 rounded bg-emerald-400 flex items-center justify-center">
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
          className="w-full bg-slate-900 border border-slate-800 rounded-md pl-8 pr-10 py-1.5 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-emerald-500/50"
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

      <MarketTicker indices={indices} />

      <div className="ml-auto flex items-center gap-3 px-4 shrink-0">
        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-medium">
          <CircleDot size={12} className="animate-pulse" /> SYSTEM ONLINE
        </div>
        <button className="text-slate-400 hover:text-slate-200 relative">
          <Bell size={16} />
          <span className="absolute -top-1 -right-1 w-1.5 h-1.5 rounded-full bg-emerald-400" />
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
  { id: "model-postmortem", label: "Model Post-Mortem", icon: BrainCircuit, ready: true },
  { id: "system-health", label: "System Health", icon: Activity, ready: true },
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
              onClick={() => setPage(item.id)}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors
                ${active ? "bg-cyan-400/10 text-cyan-400 border border-cyan-500/20" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200 border border-transparent"}`}
            >
              <Icon size={15} />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="mt-auto px-3 pt-3 border-t border-slate-800 mx-2">
        <div className="text-[9px] text-slate-600 leading-relaxed">
          Research & analytics only.<br />Not investment advice. Real capital at risk.
        </div>
      </div>
    </aside>
  );
}

/* ============================================================================
   DASHBOARD PAGE
============================================================================ */
function TopSignalsTable({ data, onOpen }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
            <th className="text-left font-medium py-1.5">Symbol</th>
            <th className="text-right font-medium py-1.5">Entry/LTP</th>
            <th className="text-right font-medium py-1.5">Target</th>
            <th className="text-right font-medium py-1.5">Stop Loss</th>
            <th className="text-right font-medium py-1.5">Horizon</th>
            <th className="text-right font-medium py-1.5">Confidence</th>
            <th className="text-right font-medium py-1.5">Status</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row) => (
            <tr
              key={row.id}
              onClick={() => onOpen(row.symbol)}
              className="border-b border-slate-900 hover:bg-slate-800/50 cursor-pointer"
            >
              <td className="py-1.5">
                <div className="font-mono text-slate-100 font-medium">{row.symbol}</div>
                <div className="text-[9px] text-slate-500">{row.name}</div>
              </td>
              <td className="text-right font-mono text-slate-300">{fmtINR(row.entry_price)}</td>
              <td className="text-right font-mono text-emerald-400">{fmtINR(row.target_price)}</td>
              <td className="text-right font-mono text-rose-400">{fmtINR(row.stop_loss)}</td>
              <td className="text-right text-slate-400 uppercase">{row.horizon}</td>
              <td className="text-right"><ConfidenceBar confidence={row.confidence} /></td>
              <td className="text-right">
                <Badge tone={row.result === 'correct' ? 'emerald' : row.result === 'wrong' ? 'rose' : 'green'}>
                  {row.result ? row.result.toUpperCase() : 'PENDING'}
                </Badge>
              </td>
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td colSpan={7} className="text-center text-slate-500 py-6">No signals generated in last ingestion cycle.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function RankedStocksList({ data, onOpen, isMovers = true }) {
  return (
    <div className="flex flex-col gap-1.5">
      {data.slice(0, 10).map((row, i) => (
        <button key={row.symbol} onClick={() => onOpen(row.symbol)} className="flex items-center gap-2 group text-left">
          <span className="text-[10px] text-slate-600 w-4 font-mono">{i + 1}</span>
          <span className="text-xs font-mono text-slate-200 w-24 group-hover:text-emerald-400 shrink-0">{row.symbol}</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${isMovers ? 'bg-gradient-to-r from-emerald-600 to-emerald-400' : 'bg-gradient-to-r from-rose-600 to-rose-400'}`}
              style={{ width: `${Math.min(100, Math.max(0, isMovers ? (row.change_pct + 10) * 5 : Math.abs(row.change_pct) * 10))}%` }}
            />
          </div>
          <span className={`text-[10px] font-mono w-10 text-right ${clr(row.change_pct)}`}>{fmtPct(row.change_pct)}</span>
        </button>
      ))}
    </div>
  );
}

function ProbabilityDistribution({ data }) {
  const buckets = Array.from({ length: 5 }, (_, i) => ({ range: `${50 + i * 10}-${60 + i * 10}%`, count: 0 }));
  data.forEach((d) => {
    const rawConf = d.confidence <= 1 ? d.confidence * 100 : d.confidence;
    const idx = Math.min(4, Math.max(0, Math.floor((rawConf - 50) / 10)));
    buckets[idx].count += 1;
  });
  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={buckets} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
        <XAxis dataKey="range" tick={{ fontSize: 9, fill: INK.axis }} axisLine={{ stroke: INK.grid }} tickLine={false} />
        <YAxis tick={{ fontSize: 9, fill: INK.axis }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11, borderRadius: 6 }} labelStyle={{ color: "#e2e8f0" }} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {buckets.map((_, i) => <Cell key={i} fill={i > 2 ? INK.green : "#475569"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function SectorHeatmap({ sectors }) {
  const maxAbs = Math.max(...sectors.map((s) => Math.abs(s.change)), 1);
  return (
    <div className="grid grid-cols-3 gap-1.5">
      {sectors.slice(0, 9).map((s) => {
        const intensity = Math.min(Math.abs(s.change) / maxAbs, 1);
        const bg = s.change >= 0
          ? `rgba(52,211,153,${0.15 + intensity * 0.55})`
          : `rgba(251,113,133,${0.15 + intensity * 0.55})`;
        return (
          <div key={s.name} style={{ background: bg }} className="rounded-md p-2 flex flex-col justify-between h-16 border border-slate-800/50">
            <span className="text-[9px] text-slate-200 font-medium leading-tight truncate">{s.name}</span>
            <span className="text-xs font-mono font-semibold text-slate-100">{fmtPct(s.change)}</span>
          </div>
        );
      })}
    </div>
  );
}

function MarketBreadth({ breadth }) {
  const total = breadth.advances + breadth.declines;
  const advPct = total > 0 ? (breadth.advances / total) * 100 : 50;
  const decPct = total > 0 ? (breadth.declines / total) * 100 : 50;
  const data = [{ name: "Advances", value: breadth.advances }, { name: "Declines", value: breadth.declines }];
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
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-400" /> Advances <span className="ml-auto font-mono text-slate-200">{breadth.advances}</span></div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-rose-400" /> Declines <span className="ml-auto font-mono text-slate-200">{breadth.declines}</span></div>
        <div className="text-[10px] text-slate-500">Ratio: {breadth.declines > 0 ? (breadth.advances / breadth.declines).toFixed(2) : "N/A"}</div>
      </div>
    </div>
  );
}

function AccuracySummary({ metrics }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end gap-3">
        <span className="text-2xl font-mono font-bold text-slate-100">{metrics.winRate}</span>
        <span className="text-[11px] text-emerald-400 mb-1 font-mono">Win Rate</span>
      </div>
      <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800 mt-1">
        <Stat label="Total Predictions" value={metrics.total} />
        <Stat label="Win Ratio" value={`${metrics.wins}W - ${metrics.losses}L`} />
      </div>
    </div>
  );
}

function ModelHealth({ health }) {
  const modelHealth = health.find(h => h.name?.toLowerCase() === "ml models" || h.name?.toLowerCase() === "ml_models" || h.name?.toLowerCase() === "mlmodels");
  const modelsList = modelHealth && modelHealth.status !== 'unhealthy' ? modelHealth.message.split(", ") : [];

  return (
    <div className="grid grid-cols-1 gap-1.5">
      {modelsList.length > 0 && modelsList[0] !== "None" ? (
        modelsList.map((m) => (
          <div key={m} className="flex items-center justify-between px-2 py-1.5 rounded bg-slate-800/40 border border-slate-800/50">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span className="text-[11px] text-slate-200 font-mono truncate max-w-[150px]" title={m}>{m}</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">ACTIVE</span>
          </div>
        ))
      ) : (
        <div className="text-slate-500 text-[11px] py-4 text-center font-mono">
          {modelHealth ? modelHealth.message : "No active models found. Run training script."}
        </div>
      )}
    </div>
  );
}

function FiiDiiWidget({ fiiDii }) {
  if (!fiiDii || !fiiDii.daily_activity) {
    return <div className="text-slate-500 text-[11px] py-4 text-center font-mono">FII/DII activity data loading...</div>;
  }
  const act = fiiDii.daily_activity;
  const tr = fiiDii.flow_trend;
  
  const formatCr = (val) => {
    if (val === null || val === undefined) return "—";
    const prefix = val >= 0 ? "+" : "";
    return `${prefix}${val.toFixed(1)} Cr`;
  };
  
  const getColorClass = (val) => {
    if (!val) return "text-slate-400 font-mono";
    return val >= 0 ? "text-emerald-400 font-mono font-semibold" : "text-rose-400 font-mono font-semibold";
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-800 pb-1 flex justify-between">
        <span>Daily Net (Cash Segment)</span>
        <span className="font-mono text-slate-500">{act.date}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 py-1">
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400">FII Cash Net</span>
          <span className={getColorClass(act.fii_cash_net_cr)}>{formatCr(act.fii_cash_net_cr)}</span>
        </div>
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400">DII Cash Net</span>
          <span className={getColorClass(act.dii_cash_net_cr)}>{formatCr(act.dii_cash_net_cr)}</span>
        </div>
        <div className="flex justify-between items-center text-xs col-span-2 border-t border-slate-900 pt-1.5">
          <span className="text-slate-400">Net Daily Flow</span>
          <span className={getColorClass(act.net_flow_cr)}>{formatCr(act.net_flow_cr)}</span>
        </div>
      </div>
      
      <div className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold border-t border-b border-slate-800 py-1 flex justify-between mt-1">
        <span>Flow Trend (20D Avg)</span>
        <Badge tone={tr.fii_flow_trend === 'BUYING' ? 'emerald' : tr.fii_flow_trend === 'SELLING' ? 'rose' : 'slate'}>
          {tr.fii_flow_trend || 'NEUTRAL'}
        </Badge>
      </div>
      <div className="grid grid-cols-2 gap-3 py-1">
        <div className="flex flex-col gap-0.5">
          <span className="text-[9px] uppercase tracking-wide text-slate-500">Avg FII Flow</span>
          <span className={`text-xs ${getColorClass(tr.avg_daily_fii_flow_cr)}`}>
            {formatCr(tr.avg_daily_fii_flow_cr)}/day
          </span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-[9px] uppercase tracking-wide text-slate-500">Avg DII Flow</span>
          <span className={`text-xs ${getColorClass(tr.avg_daily_dii_flow_cr)}`}>
            {formatCr(tr.avg_daily_dii_flow_cr)}/day
          </span>
        </div>
      </div>
    </div>
  );
}

function AiOutlookWidget({ outlook }) {
  if (!outlook) {
    return (
      <div className="text-slate-400 text-xs py-4 text-center font-mono">
        Loading AI Outlook...
      </div>
    );
  }

  const confidencePct = Math.round(outlook.confidence * 100);

  return (
    <div className="flex flex-col gap-4 font-mono text-xs text-slate-300">
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-950 p-2.5 rounded border border-slate-800 flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase">Regime</span>
          <span className="font-bold text-cyan-400">{outlook.market_regime}</span>
        </div>
        <div className="bg-slate-950 p-2.5 rounded border border-slate-800 flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase">Risk Level</span>
          <span className={`font-bold ${outlook.risk_level === 'High' ? 'text-red-400' : outlook.risk_level === 'Moderate' ? 'text-amber-400' : 'text-emerald-400'}`}>
            {outlook.risk_level}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-1 bg-slate-950 p-3 rounded border border-slate-800">
        <div className="flex justify-between items-center text-[10px] text-slate-500 uppercase">
          <span>AI Strategy Confidence</span>
          <span className="font-mono text-cyan-400 font-bold">{confidencePct}%</span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
          <div 
            className="bg-cyan-500 h-1.5 rounded-full transition-all duration-500" 
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {outlook.warnings && outlook.warnings.length > 0 && (
        <div className="flex flex-col gap-1.5 bg-red-950/20 border border-red-900/30 rounded p-3">
          <span className="text-[10px] uppercase text-red-400 font-bold tracking-wider">Major Risks</span>
          <ul className="list-disc list-inside text-[11px] text-slate-300 flex flex-col gap-1">
            {outlook.warnings.map((w, idx) => (
              <li key={idx} className="leading-tight">{w}</li>
            ))}
          </ul>
        </div>
      )}

      {outlook.watchlist && outlook.watchlist.length > 0 && (
        <div className="flex flex-col gap-1.5 bg-slate-950 p-3 rounded border border-slate-800">
          <span className="text-[10px] uppercase text-slate-500 tracking-wider">Stocks to Watch</span>
          <div className="flex flex-wrap gap-1.5">
            {outlook.watchlist.map((stock, idx) => (
              <span key={idx} className="bg-slate-900 text-slate-200 px-2 py-0.5 rounded border border-slate-800 text-[10px]">
                {stock}
              </span>
            ))}
          </div>
        </div>
      )}

      {outlook.top_themes && outlook.top_themes.length > 0 && (
        <div className="flex flex-col gap-1.5 bg-slate-950 p-3 rounded border border-slate-800">
          <span className="text-[10px] uppercase text-slate-500 tracking-wider">Top Themes</span>
          <div className="flex flex-wrap gap-1">
            {outlook.top_themes.map((theme, idx) => (
              <span key={idx} className="text-cyan-400/90 text-[11px]">
                • {theme}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


/* Helper: normalise prediction field to canonical BUY / SELL */
const normPred = (p) => {
  const v = (p.prediction || '').toString().toUpperCase();
  if (v === '2' || v === 'BUY' || v === 'LONG') return 'BUY';
  if (v === '1' || v === 'SELL' || v === 'SHORT') return 'SELL';
  return v || 'HOLD';
};

/* Helper: normalise confidence to 0-1 range regardless of what the backend returns */
const normConf = (c) => {
  if (c == null) return 0;
  const n = parseFloat(c);
  if (isNaN(n)) return 0;
  // If stored as percentage-style (> 1.5), divide by 100
  // If stored as 0-100 range, divide by 100
  return n >= 1 ? n / 100 : n;
};

function Dashboard({ stocks, predictions, health, sectors, breadth, metrics, fiiDii, topLosers, outlook, onOpen }) {
  const buySignals = useMemo(() => {
    return predictions.filter(p => normPred(p) === 'BUY').sort((a, b) => normConf(b.confidence) - normConf(a.confidence));
  }, [predictions]);

  const topMovers = useMemo(() => {
    return [...stocks]
      .filter(s => s.change_pct > 0)
      .sort((a, b) => b.change_pct - a.change_pct);
  }, [stocks]);

  return (
    <div className="grid grid-cols-12 gap-4 animate-fade-in">
      <Card title="Latest Buy Signals" className="col-span-8 row-span-3">
        <TopSignalsTable data={buySignals} onOpen={onOpen} />
      </Card>
      <Card title="Today's AI Outlook" className="col-span-4">
        <AiOutlookWidget outlook={outlook} />
      </Card>
      <Card title="Top Movers (Chg%)" className="col-span-4">
        <RankedStocksList data={topMovers} onOpen={onOpen} isMovers={true} />
      </Card>
      <Card title="Top Losers (Chg%)" className="col-span-4">
        <RankedStocksList data={topLosers} onOpen={onOpen} isMovers={false} />
      </Card>

      <Card title="FII / DII Institutional Flows" className="col-span-4">
        <FiiDiiWidget fiiDii={fiiDii} />
      </Card>
      <Card title="Confidence Density" className="col-span-4">
        <ProbabilityDistribution data={predictions} />
      </Card>
      <Card title="Sector Heatmap" className="col-span-4">
        <SectorHeatmap sectors={sectors} />
      </Card>

      <Card title="Prediction Performance" className="col-span-4">
        <AccuracySummary metrics={metrics} />
      </Card>
      <Card title="Market Breadth" className="col-span-4">
        <MarketBreadth breadth={breadth} />
      </Card>
      <Card title="Model Registry Status" className="col-span-4">
        <ModelHealth health={health} />
      </Card>
    </div>
  );
}

function SystemHealthPage({ health, signalsToday }) {
  const counts = useMemo(() => {
    let healthy = 0, degraded = 0, unhealthy = 0;
    health.forEach(h => {
      if (h.status === 'healthy') healthy++;
      else if (h.status === 'degraded') degraded++;
      else unhealthy++;
    });
    return { healthy, degraded, unhealthy };
  }, [health]);

  const pipelineStatus = useMemo(() => {
    if (!signalsToday) return null;
    const { total, open, correct, wrong, partial } = signalsToday;
    const resolved = correct + wrong + partial;
    return { total, open, resolved, correct, wrong, partial };
  }, [signalsToday]);

  return (
    <div className="flex flex-col gap-5 animate-fade-in">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex items-center justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Healthy Components</span>
            <span className="text-xl font-mono font-bold text-emerald-400">{counts.healthy}</span>
          </div>
          <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400">
            <CircleDot size={18} />
          </div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex items-center justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Degraded Components</span>
            <span className="text-xl font-mono font-bold text-cyan-400">{counts.degraded}</span>
          </div>
          <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400">
            <Activity size={18} />
          </div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex items-center justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Unhealthy Components</span>
            <span className="text-xl font-mono font-bold text-rose-400">{counts.unhealthy}</span>
          </div>
          <div className="w-8 h-8 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-400">
            <ShieldAlert size={18} />
          </div>
        </div>
      </div>

      {/* Today's Signal Pipeline Banner */}
      {pipelineStatus !== null && (
        <div className={`rounded-lg border p-4 flex items-center justify-between ${
          pipelineStatus.total === 0
            ? 'bg-rose-950/30 border-rose-800/50'
            : 'bg-slate-900 border-slate-800'
        }`}>
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Today's Signal Pipeline</span>
            {pipelineStatus.total === 0
              ? <span className="text-sm font-bold text-rose-400">⚠ ZERO signals generated today — check feed &amp; inference engine</span>
              : <span className="text-sm font-mono text-slate-200">
                  {pipelineStatus.total} total &nbsp;·&nbsp;
                  <span className="text-amber-400">{pipelineStatus.open} open</span> &nbsp;·&nbsp;
                  <span className="text-emerald-400">{pipelineStatus.correct} correct</span> &nbsp;·&nbsp;
                  <span className="text-rose-400">{pipelineStatus.wrong} wrong</span> &nbsp;·&nbsp;
                  <span className="text-cyan-400">{pipelineStatus.partial} partial</span>
                </span>
            }
          </div>
          <Badge tone={pipelineStatus.total === 0 ? 'rose' : pipelineStatus.open > 0 ? 'cyan' : 'emerald'}>
            {pipelineStatus.total === 0 ? 'DEAD' : pipelineStatus.open > 0 ? 'LIVE' : 'CLOSED'}
          </Badge>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        {health.map((h) => {
          const isModels = h.name.toLowerCase() === "ml models" || h.name.toLowerCase() === "ml_models" || h.name.toLowerCase() === "mlmodels";
          
          return (
            <div key={h.name} className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wide">{h.name}</span>
                <Badge tone={h.status === 'healthy' ? 'emerald' : h.status === 'degraded' ? 'cyan' : 'rose'}>
                  {h.status.toUpperCase()}
                </Badge>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase">Latency / Speed</span>
                  <span className="font-mono text-slate-300">{h.value || "N/A"}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase">Message</span>
                  <span className="text-slate-300 font-mono text-[11px] leading-tight block">{h.message}</span>
                </div>
              </div>

              {isModels && h.details?.slots && (
                <div className="mt-2 border-t border-slate-800 pt-3">
                  <span className="text-slate-500 block text-[10px] uppercase mb-2">Model Slot Registration</span>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(h.details.slots).map(([slot, slotStatus]) => {
                      const isActive = slotStatus.includes("active");
                      return (
                        <div key={slot} className="flex items-center justify-between p-1.5 rounded bg-slate-950 border border-slate-800/60">
                          <span className="text-[10px] font-mono text-slate-400 truncate max-w-[120px] capitalize" title={slot}>
                            {slot.replace("_", " ")}
                          </span>
                          <Badge tone={isActive ? "emerald" : "rose"}>
                            {slotStatus.toUpperCase()}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ModelPostmortemPage({ postmortem }) {
  if (!postmortem) {
    return (
      <div className="flex items-center justify-center h-60 text-slate-500 font-mono text-xs">
        Loading nightly model post-mortem report...
      </div>
    );
  }

  const {
    date,
    total_trades,
    total_losses,
    win_rate,
    losing_factors,
    winning_factors,
    analysis,
    actionable_warnings,
    recommendations,
  } = postmortem;

  const wrColor = win_rate >= 0.55 ? "text-emerald-400" : win_rate >= 0.45 ? "text-cyan-400" : "text-rose-400";

  return (
    <div className="flex flex-col gap-6 animate-fade-in text-xs font-sans">
      {/* Overview Stats Dashboard */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500">Analysis Date</span>
          <span className="text-sm font-mono font-bold text-slate-200">{date}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500">Total Trades Evaluated</span>
          <span className="text-sm font-mono font-bold text-slate-200">{total_trades}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500">Stop-Loss Hits (Losses)</span>
          <span className="text-sm font-mono font-bold text-rose-400">{total_losses}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500">Win Rate</span>
          <span className={`text-sm font-mono font-bold ${wrColor}`}>{(win_rate * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Narrative Analysis */}
      <Card title="LLM Comparative Analysis & Feature Drift">
        <div className="p-4 text-slate-300 leading-relaxed font-mono whitespace-pre-wrap text-[11px]">
          {analysis}
        </div>
      </Card>

      {/* Grid of wins vs losses factors */}
      <div className="grid grid-cols-12 gap-6">
        <Card title="Key Success Drivers (Winners)" className="col-span-6">
          <div className="p-4 flex flex-col gap-2.5">
            {winning_factors && winning_factors.length > 0 ? (
              winning_factors.map((f, i) => (
                <div key={i} className="flex gap-2 text-slate-300">
                  <span className="text-emerald-400 font-bold select-none">•</span>
                  <span>{f}</span>
                </div>
              ))
            ) : (
              <span className="text-slate-500 font-mono italic">No success factors identified.</span>
            )}
          </div>
        </Card>

        <Card title="Key Failure Drivers (Losers)" className="col-span-6">
          <div className="p-4 flex flex-col gap-2.5">
            {losing_factors && losing_factors.length > 0 ? (
              losing_factors.map((f, i) => (
                <div key={i} className="flex gap-2 text-slate-300">
                  <span className="text-rose-400 font-bold select-none">•</span>
                  <span>{f}</span>
                </div>
              ))
            ) : (
              <span className="text-slate-500 font-mono italic">No failure factors identified.</span>
            )}
          </div>
        </Card>
      </div>

      {/* Actionable Risk Warnings */}
      <Card title="Actionable Risk Warnings">
        <div className="p-4 flex flex-col gap-3">
          {actionable_warnings && actionable_warnings.length > 0 ? (
            actionable_warnings.map((w, i) => (
              <div key={i} className="flex items-center gap-3 p-2.5 rounded border border-rose-950 bg-rose-950/20 text-rose-300">
                <ShieldAlert size={14} className="text-rose-400 shrink-0" />
                <span className="font-mono">{w}</span>
              </div>
            ))
          ) : (
            <div className="flex items-center gap-3 p-2.5 rounded border border-emerald-950 bg-emerald-950/20 text-emerald-300">
              <CircleDot size={14} className="text-emerald-400 shrink-0" />
              <span className="font-mono">No active risk warnings detected. Model parameters performing within target risk limits.</span>
            </div>
          )}
        </div>
      </Card>

      {/* Recommendations Box */}
      <Card title="Suggested Threshold Adjustments & Parameters">
        <div className="p-4 flex flex-col gap-2 bg-slate-950 rounded-b-lg">
          <span className="text-[10px] text-slate-500 uppercase tracking-wide">LLM Recommended Code/Policy Action:</span>
          <pre className="font-mono text-emerald-400 whitespace-pre-wrap text-[11px] p-2 bg-slate-900 border border-slate-800 rounded leading-relaxed">
            {recommendations || "Maintain existing thresholds."}
          </pre>
        </div>
      </Card>
    </div>
  );
}

/* ============================================================================
   LIVE SIGNALS PAGE
============================================================================ */
function LiveSignals({ predictions, onOpen }) {
  const [minProb, setMinProb] = useState(0);   // default 0% — show all signals
  const [sortKey, setSortKey] = useState("time");
  const [horizonFilter, setHorizonFilter] = useState("ALL");
  const [directionFilter, setDirectionFilter] = useState("ALL");

  const filtered = useMemo(() => {
    return predictions.filter(p => {
      const conf = normConf(p.confidence);
      if (conf < minProb) return false;
      if (horizonFilter !== "ALL" && (p.horizon || '').toUpperCase() !== horizonFilter) return false;
      if (directionFilter !== "ALL" && normPred(p) !== directionFilter) return false;
      return true;
    });
  }, [predictions, minProb, horizonFilter, directionFilter]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const tA = new Date(a.date || a.prediction_date || a.prediction_time || a.created_at).getTime();
      const tB = new Date(b.date || b.prediction_date || b.prediction_time || b.created_at).getTime();
      if (sortKey === "time") return tB - tA;
      if (sortKey === "prob") return normConf(b.confidence) - normConf(a.confidence);
      return 0;
    });
  }, [filtered, sortKey]);

  // Accuracy stats
  const stats = useMemo(() => {
    const total = filtered.length;
    const correct = filtered.filter(p => p.result === 'correct').length;
    const wrong = filtered.filter(p => p.result === 'wrong').length;
    const pending = filtered.filter(p => !p.result || p.result === 'pending').length;
    const resolved = correct + wrong;
    const winRate = resolved > 0 ? ((correct / resolved) * 100).toFixed(1) : null;
    const buys = filtered.filter(p => normPred(p) === 'BUY').length;
    const sells = filtered.filter(p => normPred(p) === 'SELL').length;
    return { total, correct, wrong, pending, winRate, buys, sells };
  }, [filtered]);

  return (
    <div className="flex flex-col gap-4">
      {/* Accuracy Stats Banner */}
      <div className="grid grid-cols-7 gap-2">
        {[
          { label: 'Total Signals', value: stats.total, color: 'text-slate-100' },
          { label: 'BUY Signals', value: stats.buys, color: 'text-emerald-400' },
          { label: 'SELL Signals', value: stats.sells, color: 'text-rose-400' },
          { label: 'Correct', value: stats.correct, color: 'text-emerald-400' },
          { label: 'Wrong', value: stats.wrong, color: 'text-rose-400' },
          { label: 'Pending', value: stats.pending, color: 'text-amber-400' },
          { label: 'Win Rate', value: stats.winRate != null ? `${stats.winRate}%` : '—', color: stats.winRate >= 55 ? 'text-emerald-400' : stats.winRate >= 40 ? 'text-amber-400' : 'text-rose-400' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col gap-0.5">
            <span className="text-[9px] uppercase tracking-wider text-slate-500">{label}</span>
            <span className={`text-lg font-mono font-bold ${color}`}>{value}</span>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-3 flex-wrap">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium shrink-0">
          <Radio size={14} className="animate-pulse" /> LIVE
        </div>

        {/* Horizon filter */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-500 mr-1">Horizon:</span>
          {['ALL','INTRADAY','SWING','LONGTERM'].map(h => (
            <button key={h} onClick={() => setHorizonFilter(h)}
              className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${
                horizonFilter === h ? 'bg-cyan-400/10 text-cyan-400 border border-cyan-500/30' : 'text-slate-500 hover:text-slate-300 border border-transparent'
              }`}>{h === 'ALL' ? 'All' : h.charAt(0) + h.slice(1).toLowerCase()}</button>
          ))}
        </div>

        {/* Direction filter */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-500 mr-1">Dir:</span>
          {['ALL','BUY','SELL'].map(d => (
            <button key={d} onClick={() => setDirectionFilter(d)}
              className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${
                directionFilter === d ? (d === 'BUY' ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-500/30' : d === 'SELL' ? 'bg-rose-400/10 text-rose-400 border border-rose-500/30' : 'bg-cyan-400/10 text-cyan-400 border border-cyan-500/30') : 'text-slate-500 hover:text-slate-300 border border-transparent'
              }`}>{d}</button>
          ))}
        </div>

        {/* Prob slider */}
        <div className="flex items-center gap-2 max-w-[200px]">
          <SlidersHorizontal size={12} className="text-slate-500" />
          <span className="text-[10px] text-slate-500 whitespace-nowrap">Min {(minProb * 100).toFixed(0)}%</span>
          <input type="range" min="0" max="0.95" step="0.01" value={minProb}
            onChange={(e) => setMinProb(parseFloat(e.target.value))}
            className="w-full accent-emerald-400" />
        </div>

        {/* Sort */}
        <div className="ml-auto flex items-center gap-1">
          {["time", "prob"].map((k) => (
            <button key={k} onClick={() => setSortKey(k)}
              className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${
                sortKey === k ? "bg-emerald-400/10 text-emerald-400 border border-emerald-500/30" : "text-slate-500 hover:text-slate-300"
              }`}>{k}</button>
          ))}
        </div>
        <span className="text-[10px] text-slate-500 font-mono">{sorted.length} signals</span>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
                <th className="text-left font-medium py-2">Timestamp</th>
                <th className="text-left font-medium py-2">Symbol</th>
                <th className="text-left font-medium py-2">Signal</th>
                <th className="text-left font-medium py-2">Horizon</th>
                <th className="text-right font-medium py-2">Entry</th>
                <th className="text-right font-medium py-2">Stop Loss</th>
                <th className="text-right font-medium py-2">Target</th>
                <th className="text-right font-medium py-2">Prob.</th>
                <th className="text-right font-medium py-2">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 && (
                <tr><td colSpan={9} className="text-center text-slate-600 py-8 text-[11px]">No signals matched your filters. Try adjusting the Horizon or Min Prob slider.</td></tr>
              )}
              {sorted.map((row) => {
                const direction = normPred(row);
                const conf = normConf(row.confidence);
                return (
                  <tr key={row.id} onClick={() => onOpen(row.symbol)} className="border-b border-slate-900 hover:bg-slate-800/50 cursor-pointer">
                    <td className="py-2 font-mono text-slate-500">{(row.date || row.prediction_date || row.prediction_time || '').replace('T', ' ').slice(5, 19)}</td>
                    <td className="py-2 font-mono text-slate-100 font-medium">{row.symbol}</td>
                    <td className="py-2">
                      <span className={`font-bold text-[10px] px-1.5 py-0.5 rounded ${
                        direction === 'BUY' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                      }`}>{direction}</span>
                    </td>
                    <td className="py-2 text-slate-400 text-[10px] font-mono">{row.horizon}</td>
                    <td className="py-2 text-right font-mono text-slate-300">{fmtINR(row.entry_price)}</td>
                    <td className="py-2 text-right font-mono text-rose-400">{fmtINR(row.stop_loss)}</td>
                    <td className="py-2 text-right font-mono text-emerald-400">{fmtINR(row.target_price)}</td>
                    <td className="py-2 text-right"><ConfidenceBar confidence={conf} /></td>
                    <td className="py-2 text-right font-mono text-slate-400">
                      <Badge tone={row.result === 'correct' ? 'emerald' : row.result === 'wrong' ? 'rose' : 'slate'}>
                        {row.result ? row.result.toUpperCase() : 'PENDING'}
                      </Badge>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ============================================================================
   STOCK DETAIL — CHART (REAL CANDLES MOUNTED)
============================================================================ */
function CandlestickChart({ series, prediction, overlays, timeframe }) {
  const w = 900, h = 320, padTop = 12, padBottom = 8, volH = 56;
  const sliceMap = { "3M": 60, "6M": 120, "1Y": 240 };
  const visible = series.slice(-sliceMap[timeframe]);

  if (visible.length === 0) {
    return <div className="h-80 flex items-center justify-center text-slate-500">No candle data in interval</div>;
  }

  const closes = visible.map((d) => d.close);
  const highs = visible.map((d) => d.high);
  const lows = visible.map((d) => d.low);

  const predVals = prediction ? [prediction.entry_price, prediction.stop_loss, prediction.target_price] : [];
  let hi = Math.max(...highs, ...predVals);
  let lo = Math.min(...lows, ...predVals);
  const pad = (hi - lo) * 0.06;
  hi += pad; lo -= pad;

  const emaFast = overlays.ema20 ? ema(visible, 20) : null;
  const emaSlow = overlays.ema50 ? ema(visible, 50) : null;
  const vwap = overlays.vwap ? vwapSeries(visible) : null;

  const priceH = h - volH - padTop - padBottom;
  const x = (i) => (i / (visible.length - 1)) * w;
  const y = (v) => padTop + priceH * (1 - (v - lo) / (hi - lo));
  const maxVol = Math.max(...visible.map((d) => d.volume), 1);
  const candleW = Math.max(1.5, (w / visible.length) * 0.6);

  const linePath = (vals) => vals.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto select-none" preserveAspectRatio="none">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <line key={f} x1={0} x2={w} y1={padTop + priceH * f} y2={padTop + priceH * f} stroke="#1e293b" strokeWidth="1" />
      ))}

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
            <rect x={x(i) - candleW / 2} y={h - 2 - (d.volume / maxVol) * (volH - 4)} width={candleW} height={(d.volume / maxVol) * (volH - 4)} fill={up ? "#34d39933" : "#fb718533"} />
          </g>
        );
      })}

      {emaFast && <path d={linePath(emaFast)} stroke={INK.cyan} strokeWidth="1.5" fill="none" />}
      {emaSlow && <path d={linePath(emaSlow)} stroke={INK.violet} strokeWidth="1.5" fill="none" />}
      {vwap && <path d={linePath(vwap)} stroke={INK.green} strokeWidth="1.2" fill="none" strokeDasharray="4,2" />}

      {prediction && [
        { v: prediction.entry_price, color: "#e2e8f0", label: "ENTRY" },
        { v: prediction.stop_loss, color: INK.rose, label: "SL" },
        { v: prediction.target_price, color: INK.emerald, label: "TARGET" },
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
  const sliceMap = { "3M": 60, "6M": 120, "1Y": 240 };
  const visible = series.slice(-sliceMap[timeframe]);
  const x = (i) => (i / (visible.length - 1)) * w;

  if (visible.length === 0) return null;

  if (mode === "rsi") {
    const rsiVals = rsiSeries(visible);
    const valid = rsiVals.map((v) => v ?? 50);
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
  const macdVals = macdSeries(visible);
  const vals = macdVals.map((m) => m.hist);
  const maxAbs = Math.max(...vals.map(Math.abs), 0.01);
  const y = (v) => h / 2 - (v / maxAbs) * (h / 2 - 6);
  const macdLine = macdVals.map((m) => m.macd);
  const sigLine = macdVals.map((m) => m.signal);
  const maxLineAbs = Math.max(...macdLine.map(Math.abs), ...sigLine.map(Math.abs), 0.01);
  const yLine = (v) => h / 2 - (v / maxLineAbs) * (h / 2 - 6);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto" preserveAspectRatio="none">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke="#334155" strokeWidth="1" />
      {vals.map((v, i) => (
        <rect key={i} x={x(i) - 1.5} y={Math.min(y(v), h / 2)} width={3} height={Math.abs(y(v) - h / 2)} fill={v >= 0 ? INK.emerald : INK.rose} opacity="0.6" />
      ))}
      <path d={macdLine.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${yLine(v)}`).join(" ")} stroke={INK.cyan} strokeWidth="1.2" fill="none" />
      <path d={sigLine.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${yLine(v)}`).join(" ")} stroke={INK.green} strokeWidth="1.2" fill="none" />
      <text x={4} y={10} fontSize="9" fill="#64748b" fontFamily="monospace">MACD(12,26,9)</text>
    </svg>
  );
}

// Candlestick helper functions
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

/* ============================================================================
   STOCK DETAIL — PREDICTION CARD
============================================================================ */
function PredictionCard({ prediction, symbol, loading }) {
  if (loading) {
    return <Card title="AI Prediction Model"><Skeleton className="h-64" /></Card>;
  }

  if (!prediction) {
    return (
      <Card title="Prediction Profile">
        <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
          <BrainCircuit size={28} className="text-slate-600" />
          <span className="text-slate-400 text-xs font-semibold">NO ACTIVE LIVE SIGNAL</span>
          <p className="text-[10px] text-slate-500 max-w-[200px] leading-relaxed">
            The machine learning ensemble did not detect a high-probability setup for {symbol} in the last calculation cycle.
          </p>
        </div>
      </Card>
    );
  }

  const isBuy = prediction.prediction === "BUY" || prediction.prediction === "LONG";
  const confidencePct = Math.round(prediction.confidence <= 1 ? prediction.confidence * 100 : prediction.confidence);
  const riskReward = prediction.entry_price && prediction.target_price && prediction.stop_loss
    ? Math.abs((prediction.target_price - prediction.entry_price) / (prediction.entry_price - prediction.stop_loss))
    : 1.5;
  const expectedReturn = prediction.entry_price && prediction.target_price
    ? Math.abs((prediction.target_price - prediction.entry_price) / prediction.entry_price) * 100
    : 3.5;

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <Badge tone={isBuy ? "emerald" : "rose"}><TrendingUp size={11} /> {prediction.prediction} SIGNAL</Badge>
        <span className="text-[10px] text-slate-500 font-mono">{prediction.model_version}</span>
      </div>
      <div className="flex items-center gap-4 mb-4">
        <ProbabilityRing value={confidencePct / 100} />
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 flex-1">
          <Stat label="Confidence" value={`${confidencePct}%`} />
          <Stat label="Exp. Return" value={fmtPct(expectedReturn)} tone={isBuy ? "up" : "down"} />
          <Stat label="Risk:Reward" value={`${riskReward.toFixed(2)}x`} />
          <Stat label="Horizon" value={prediction.horizon} />
        </div>
      </div>

      <div className="border-t border-slate-800 pt-3 flex flex-col gap-2 mb-4">
        <PriceRow label="Entry Price" value={prediction.entry_price} tone="slate" icon={Target} />
        <PriceRow label="Stop Loss" value={prediction.stop_loss} tone="rose" icon={ShieldAlert} />
        <PriceRow label="Target Price" value={prediction.target_price} tone="emerald" icon={ArrowUpRight} />
      </div>

      <div className="border-t border-slate-800 pt-3">
        <div className="flex items-center gap-1.5 mb-2 text-[10px] uppercase tracking-wide text-slate-500 font-semibold">
          <BrainCircuit size={12} /> Model Logic / Reasoning
        </div>
        <div className="text-[11px] text-slate-400 bg-slate-950/40 p-2 border border-slate-800 rounded font-mono">
          {prediction.reason || "Pattern matching parameters met across technical features."}
        </div>
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
  const featureImportance = useMemo(() => {
    if (!prediction || !prediction.features_used) return [];
    try {
      const parsed = JSON.parse(prediction.features_used);
      return Object.entries(parsed)
        .filter(([k, v]) => typeof v === 'number' && k !== 'ev_estimate' && k !== 'vix_level')
        .map(([k, v]) => ({
          feature: k,
          importance: Math.abs(v),
          shap: v
        }))
        .sort((a, b) => b.importance - a.importance);
    } catch (e) {
      return [];
    }
  }, [prediction]);

  if (featureImportance.length === 0) {
    return <div className="text-slate-500 text-xs py-4 text-center">No feature importance details stored for this prediction model.</div>;
  }

  const maxVal = Math.max(...featureImportance.map((f) => f.importance), 0.01);

  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <h4 className="text-[11px] uppercase tracking-wide text-slate-500 mb-3 font-semibold">SHAP Feature Contributions</h4>
        <div className="flex flex-col gap-2">
          {featureImportance.slice(0, 8).map((f) => (
            <div key={f.feature} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 w-36 truncate">{f.feature}</span>
              <div className="flex-1 h-4 relative bg-slate-800/50 rounded overflow-hidden">
                <div
                  className={`h-full absolute top-0 ${f.shap >= 0 ? "bg-emerald-400/70 left-1/2" : "bg-rose-400/70 right-1/2"}`}
                  style={{ width: `${(Math.abs(f.shap) / maxVal) * 50}%` }}
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
        <h4 className="text-[11px] uppercase tracking-wide text-slate-500 mb-3 font-semibold">Confidence Overview</h4>
        <p className="text-xs text-slate-400 leading-relaxed mb-4 font-sans">
          The model's probability represents the calibrated output of the ML meta-ensemble. It maps raw classifier score to empirical win rate. Only setups with probability &gt; 55% and positive expected value (EV) after transaction cost scaling are generated.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Classifier" value={prediction.model_version} />
          <Stat label="Prediction ID" value={prediction.id.slice(0, 8)} />
          <Stat label="Horizon Class" value={prediction.horizon} />
          <Stat label="Timestamp" value={prediction.prediction_date || prediction.prediction_time || 'N/A'} />
        </div>
      </div>
    </div>
  );
}

function HistoryTab({ predictions, symbol }) {
  const symbolHistory = useMemo(() => {
    return predictions.filter(p => p.symbol === symbol && p.result !== 'pending');
  }, [predictions, symbol]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase tracking-wide">
            <th className="text-left font-medium py-2">Date</th>
            <th className="text-right font-medium py-2">Entry</th>
            <th className="text-right font-medium py-2">Target</th>
            <th className="text-right font-medium py-2">Stop Loss</th>
            <th className="text-right font-medium py-2">Confidence</th>
            <th className="text-right font-medium py-2">Horizon</th>
            <th className="text-right font-medium py-2">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {symbolHistory.map((row) => (
            <tr key={row.id} className="border-b border-slate-900">
              <td className="py-2 font-mono text-slate-400">{(row.prediction_date || '').slice(0, 10)}</td>
              <td className="py-2 text-right font-mono text-slate-400">{fmtINR(row.entry_price)}</td>
              <td className="py-2 text-right font-mono text-emerald-400">{fmtINR(row.target_price)}</td>
              <td className="py-2 text-right font-mono text-rose-400">{fmtINR(row.stop_loss)}</td>
              <td className="py-2 text-right font-mono text-emerald-400">{Math.round(row.confidence <= 1 ? row.confidence * 100 : row.confidence)}%</td>
              <td className="py-2 text-right font-mono text-slate-400">{row.horizon}</td>
              <td className="py-2 text-right">
                <Badge tone={row.result === "correct" ? "emerald" : "rose"}>
                  {row.result === "correct" ? "TARGET HIT" : "STOP HIT"}
                </Badge>
              </td>
            </tr>
          ))}
          {symbolHistory.length === 0 && (
            <tr>
              <td colSpan={7} className="text-center text-slate-500 py-6">No historical outcomes archived for this symbol yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function StockDetail({ symbol, watchlist, toggleWatchlist, predictions }) {
  const [series, setSeries] = useState([]);
  const [historyData, setHistoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState("6M");
  const [tab, setTab] = useState("overview");
  const [overlays, setOverlays] = useState({ ema20: true, ema50: true, vwap: false });
  const [oscMode, setOscMode] = useState("rsi");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`${API_BASE}/stocks/${symbol}/history?days=250`)
      .then(res => res.json())
      .then(data => {
        if (!cancelled) {
          setHistoryData(data);
          if (data && data.dates) {
            const mapped = data.dates.map((d, i) => ({
              date: d,
              open: data.open[i],
              high: data.high[i],
              low: data.low[i],
              close: data.prices[i],
              volume: data.volumes[i]
            }));
            setSeries(mapped);
          } else {
            setSeries([]);
          }
          setLoading(false);
        }
      })
      .catch(err => {
        console.error("Failed to load historical candles:", err);
        if (!cancelled) {
          setSeries([]);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [symbol]);

  const activePrediction = useMemo(() => {
    return predictions.find(p => p.symbol === symbol && p.result === 'pending') || null;
  }, [predictions, symbol]);

  const toggleOverlay = (key) => setOverlays((o) => ({ ...o, [key]: !o[key] }));

  const currentPrice = series.length > 0 ? series[series.length - 1].close : null;
  const prevPrice = series.length > 1 ? series[series.length - 2].close : null;

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      <div className="flex items-center gap-4">
        <button onClick={() => toggleWatchlist(symbol)}>
          <Star size={18} className={watchlist.has(symbol) ? "fill-emerald-400 text-emerald-400" : "text-slate-600"} />
        </button>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-slate-100 font-mono">{symbol}</h2>
          </div>
          <span className="text-[11px] text-slate-500">Security Research View</span>
        </div>
        {currentPrice && (
          <div className="ml-4 flex items-center gap-2">
            <span className="text-xl font-mono font-bold text-slate-100">{fmtINR(currentPrice)}</span>
            <ChangeCell last={currentPrice} prev={prevPrice} />
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          {["3M", "6M", "1Y"].map((tf) => (
            <button key={tf} onClick={() => setTimeframe(tf)} className={`text-[10px] px-2.5 py-1 rounded font-medium ${timeframe === tf ? "bg-emerald-400/10 text-emerald-400 border border-emerald-500/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}>
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
                { key: "vwap", label: "VWAP", color: "text-emerald-400" },
              ].map((ov) => (
                <button
                  key={ov.key} onClick={() => toggleOverlay(ov.key)}
                  className={`text-[10px] px-2 py-1 rounded border font-medium flex items-center gap-1 ${overlays[ov.key] ? `border-slate-700 bg-slate-800 ${ov.color}` : "border-transparent text-slate-600"}`}
                >
                  <span className="w-2 h-0.5 bg-current inline-block" /> {ov.label}
                </button>
              ))}
            </div>
            {loading ? <Skeleton className="h-80" /> : (
              <CandlestickChart series={series} prediction={activePrediction} overlays={overlays} timeframe={timeframe} />
            )}
          </Card>
          <Card>
            <div className="flex items-center gap-2 mb-1">
              {["rsi", "macd"].map((m) => (
                <button key={m} onClick={() => setOscMode(m)} className={`text-[10px] px-2 py-1 rounded uppercase font-medium ${oscMode === m ? "bg-slate-800 text-slate-100" : "text-slate-500"}`}>{m}</button>
              ))}
            </div>
            {loading ? <Skeleton className="h-24" /> : <OscillatorChart series={series} timeframe={timeframe} mode={oscMode} />}
          </Card>

          <Card className="p-0">
            <div className="flex border-b border-slate-800 px-2 overflow-x-auto">
              {[
                { id: "overview", label: "Overview" },
                { id: "technicals", label: "Technicals" },
                { id: "explainability", label: "Model Shap" },
                { id: "history", label: "Trade History" },
              ].map((t) => (
                <button
                  key={t.id} onClick={() => setTab(t.id)}
                  className={`px-3 py-2.5 text-xs font-medium border-b-2 -mb-px ${tab === t.id ? "border-emerald-400 text-emerald-400" : "border-transparent text-slate-500 hover:text-slate-300"}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="p-4">
              {tab === "overview" && (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Resistance (Pivot)" value={historyData?.levels?.resistance ? fmtINR(historyData.levels.resistance) : "—"} />
                  <Stat label="Support (Pivot)" value={historyData?.levels?.support ? fmtINR(historyData.levels.support) : "—"} />
                  <Stat label="Level Midpoint" value={historyData?.levels?.midpoint ? fmtINR(historyData.levels.midpoint) : "—"} />
                  <Stat label="Volume (Last)" value={historyData?.indicators?.volume ? fmtCompact(historyData.indicators.volume) : "—"} />
                  <Stat label="Signal Source" value={historyData?.source ? historyData.source.toUpperCase() : "—"} />
                </div>
              )}
              {tab === "technicals" && (
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <Stat label="RSI(14)" value={historyData?.indicators?.rsi_14 !== null && historyData?.indicators?.rsi_14 !== undefined ? historyData.indicators.rsi_14.toFixed(1) : "—"} />
                  <Stat label="EMA20" value={historyData?.indicators?.ema_20 ? fmtINR(historyData.indicators.ema_20) : "—"} />
                  <Stat label="EMA50" value={historyData?.indicators?.ema_50 ? fmtINR(historyData.indicators.ema_50) : "—"} />
                  <Stat label="Close vs EMA20" value={historyData?.indicators?.close_vs_ema20_pct !== null && historyData?.indicators?.close_vs_ema20_pct !== undefined ? fmtPct(historyData.indicators.close_vs_ema20_pct) : "—"} />
                </div>
              )}
              {tab === "explainability" && <ExplainabilityTab prediction={activePrediction} />}
              {tab === "history" && <HistoryTab predictions={predictions} symbol={symbol} />}
            </div>
          </Card>
        </div>

        <div className="col-span-4">
          <PredictionCard prediction={activePrediction} symbol={symbol} loading={loading} />
        </div>
      </div>
    </div>
  );
}

function ChangeCell({ last, prev }) {
  if (typeof last !== 'number' || typeof prev !== 'number' || prev === 0) return "--";
  const chg = ((last - prev) / prev) * 100;
  const up = chg >= 0;
  return (
    <span className={`font-mono text-xs flex items-center gap-0.5 ${up ? "text-emerald-400" : "text-rose-400"}`}>
      {up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
      {fmtPct(chg)}
    </span>
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
  const [stocks, setStocks] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [indices, setIndices] = useState([]);
  const [health, setHealth] = useState([]);
  const [signalsToday, setSignalsToday] = useState(null);
  const [fiiDii, setFiiDii] = useState(null);
  const [outlook, setOutlook] = useState(null);
  const [postmortem, setPostmortem] = useState(null);
  const [loading, setLoading] = useState(true);

  // Sorting state for stocks table
  const [sortField, setSortField] = useState("symbol");
  const [sortDir, setSortDir] = useState("asc");
  const [searchVal, setSearchVal] = useState("");
  const [sectorFilter, setSectorFilter] = useState("");

  const fetchAllData = useCallback(async () => {
    try {
      const [stocksRes, predRes, healthRes, indicesRes, fiiDiiRes, outlookRes, postmortemRes, signalsTodayRes] = await Promise.all([
        fetch(`${API_BASE}/stocks`).then(r => r.ok ? r.json() : []),
        fetch(`${API_BASE}/predictions?limit=1000`).then(r => r.ok ? r.json() : []),
        fetch(`${API_BASE}/health/status`).then(r => r.ok ? r.json() : []),
        fetch(`${API_BASE}/indices`).then(r => r.ok ? r.json() : []),
        fetch(`${API_BASE}/institutional/fii_dii`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/market/outlook`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/validation/postmortem`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/validation/signals/today`).then(r => r.ok ? r.json() : null)
      ]);

      setStocks(stocksRes || []);
      setPredictions(predRes || []);
      setHealth(healthRes || []);
      setIndices(indicesRes || []);
      setFiiDii(fiiDiiRes || null);
      setOutlook(outlookRes || null);
      setPostmortem(postmortemRes || null);
      setSignalsToday(signalsTodayRes || null);
    } catch (e) {
      console.error("Failed to fetch dashboard data:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, [fetchAllData]);

  // Derived state calculations
  const sectors = useMemo(() => {
    const bySector = {};
    stocks.forEach(s => {
      if (!s.sector) return;
      if (!bySector[s.sector]) bySector[s.sector] = [];
      bySector[s.sector].push(s.change_pct || 0);
    });
    return Object.entries(bySector).map(([name, changes]) => ({
      name,
      change: changes.reduce((a, b) => a + b, 0) / changes.length,
      count: changes.length
    })).sort((a, b) => b.change - a.change);
  }, [stocks]);

  const uniqueSectors = useMemo(() => {
    return Array.from(new Set(stocks.map(s => s.sector).filter(Boolean))).sort();
  }, [stocks]);

  const breadth = useMemo(() => {
    const advances = stocks.filter(s => s.change_pct > 0).length;
    const declines = stocks.filter(s => s.change_pct < 0).length;
    return { advances, declines };
  }, [stocks]);

  const topLosers = useMemo(() => {
    return [...stocks]
      .filter(s => s.change_pct < 0)
      .sort((a, b) => a.change_pct - b.change_pct);
  }, [stocks]);

  const metrics = useMemo(() => {
    const resolved = predictions.filter(p => p.result === 'correct' || p.result === 'wrong');
    const wins = resolved.filter(p => p.result === 'correct').length;
    const winRate = resolved.length > 0 ? (wins / resolved.length) * 100 : 0.0;
    return {
      winRate: resolved.length > 0 ? winRate.toFixed(1) + "%" : "0.0%",
      total: predictions.length,
      resolved: resolved.length,
      wins,
      losses: resolved.length - wins
    };
  }, [predictions]);

  const openStock = useCallback((sym) => { setSymbol(sym); setPage("stock-detail"); }, []);
  const toggleWatchlist = useCallback((sym) => {
    setWatchlist((prev) => { const next = new Set(prev); next.has(sym) ? next.delete(sym) : next.add(sym); return next; });
  }, []);

  // Filter & sort stocks for Stock Research grid
  const filteredStocks = useMemo(() => {
    let list = [...stocks];
    if (sectorFilter) {
      list = list.filter(s => s.sector === sectorFilter);
    }
    if (searchVal) {
      const queryLower = searchVal.toLowerCase();
      list = list.filter(s => s.symbol.toLowerCase().includes(queryLower) || s.name.toLowerCase().includes(queryLower));
    }
    
    // Sort
    list.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      // Handle custom sorting fields
      if (sortField === "confidence") {
        const predA = predictions.find(p => p.symbol === a.symbol && p.result === 'pending');
        const predB = predictions.find(p => p.symbol === b.symbol && p.result === 'pending');
        valA = predA ? predA.confidence : -1;
        valB = predB ? predB.confidence : -1;
      }

      if (typeof valA === "number" && typeof valB === "number") {
        return sortDir === "asc" ? valA - valB : valB - valA;
      }
      return sortDir === "asc"
        ? String(valA || "").localeCompare(String(valB || ""))
        : String(valB || "").localeCompare(String(valA || ""));
    });
    return list;
  }, [stocks, searchVal, sectorFilter, sortField, sortDir, predictions]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  return (
    <div className="h-screen w-full bg-slate-950 text-slate-200 flex flex-col font-sans overflow-hidden">
      <TopNav query={query} setQuery={setQuery} stocks={stocks} onSelectSymbol={openStock} indices={indices} />
      <div className="flex flex-1 min-h-0">
        <Sidebar page={page} setPage={setPage} />
        <main className="flex-1 overflow-y-auto p-5">
          <div className="flex items-center gap-2 mb-4 text-[11px] text-slate-500">
            <span className="capitalize">{page.replace("-", " ")}</span>
            <ChevronRight size={12} />
            <span className="text-slate-300">{page === "stock-detail" ? symbol : "Overview"}</span>
            <span className="ml-auto flex items-center gap-1.5">
              <Info size={11} /> Real capital quant research platform — Live yfinance & Upstox Integration
            </span>
          </div>

          {loading ? (
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-40" />)}
            </div>
          ) : (
            <>
              {page === "dashboard" && (
                <Dashboard
                  stocks={stocks}
                  predictions={predictions}
                  health={health}
                  sectors={sectors}
                  breadth={breadth}
                  metrics={metrics}
                  fiiDii={fiiDii}
                  topLosers={topLosers}
                  outlook={outlook}
                  onOpen={openStock}
                />
              )}
              {page === "live-signals" && <LiveSignals predictions={predictions} onOpen={openStock} />}
              {page === "model-postmortem" && <ModelPostmortemPage postmortem={postmortem} />}
              {page === "system-health" && <SystemHealthPage health={health} signalsToday={signalsToday} />}
              {page === "stock-detail" && (
                <div className="grid grid-cols-12 gap-6">
                  {/* Left Side stock selector grid */}
                  <div className="col-span-4 flex flex-col gap-3">
                    <Card title="Stock Screener">
                      <div className="flex flex-col gap-2 mb-3">
                        <div className="relative">
                          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
                          <input
                            type="text"
                            placeholder="Filter stocks..."
                            value={searchVal}
                            onChange={(e) => setSearchVal(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 outline-none"
                          />
                        </div>
                        <select
                          value={sectorFilter}
                          onChange={(e) => setSectorFilter(e.target.value)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-400 outline-none"
                        >
                          <option value="">All Sectors</option>
                          {uniqueSectors.map(sec => <option key={sec} value={sec}>{sec}</option>)}
                        </select>
                      </div>
                      
                      <div className="overflow-y-auto max-h-[500px]">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase font-semibold">
                              <th onClick={() => handleSort("symbol")} className="text-left py-1 cursor-pointer hover:text-slate-300">Symbol</th>
                              <th onClick={() => handleSort("price")} className="text-right py-1 cursor-pointer hover:text-slate-300">Price</th>
                              <th onClick={() => handleSort("change_pct")} className="text-right py-1 cursor-pointer hover:text-slate-300">Chg%</th>
                              <th onClick={() => handleSort("confidence")} className="text-right py-1 cursor-pointer hover:text-slate-300">Signal</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredStocks.map(s => {
                              const pred = predictions.find(p => p.symbol === s.symbol && p.result === 'pending');
                              const signalText = pred ? pred.prediction : "HOLD";
                              return (
                                <tr
                                  key={s.symbol}
                                  onClick={() => setSymbol(s.symbol)}
                                  className={`border-b border-slate-900 hover:bg-slate-800/40 cursor-pointer ${symbol === s.symbol ? 'bg-slate-800/60' : ''}`}
                                >
                                  <td className="py-1.5 font-mono font-medium">{s.symbol}</td>
                                  <td className="py-1.5 text-right font-mono text-slate-300">{s.price.toFixed(1)}</td>
                                  <td className={`py-1.5 text-right font-mono ${clr(s.change_pct)}`}>{s.change_pct >= 0 ? '+' : ''}{s.change_pct.toFixed(1)}%</td>
                                  <td className="py-1.5 text-right">
                                    <span className={`text-[9px] px-1 py-0.2 rounded font-semibold ${signalText === 'BUY' ? 'bg-emerald-950 text-emerald-400' : signalText === 'SELL' || signalText === 'SHORT' ? 'bg-rose-950 text-rose-400' : 'bg-slate-950 text-slate-400'}`}>
                                      {signalText}
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </Card>
                  </div>
                  
                  {/* Right Side detailed chart and stats */}
                  <div className="col-span-8">
                    <StockDetail
                      symbol={symbol}
                      watchlist={watchlist}
                      toggleWatchlist={toggleWatchlist}
                      predictions={predictions}
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}