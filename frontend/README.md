# IERM Frontend

A rebuilt frontend for the Indian Equity Research Machine — React 18 + TypeScript + Vite + Tailwind, built for a live trading terminal, not a demo dashboard.

## Why this rebuild

The prior frontend only called 7 of 66 backend routes and had no live data path (no WebSocket, no reconnect logic, no loading/error/empty states). This version fixes the structural gaps:

- **Every screen has a real loading, error, and empty state.** Nothing renders blank or throws to a white screen — see `ErrorBoundary.tsx` and each component's `isLoading`/`isError` branches.
- **Live data has a real transport.** `lib/websocket.ts` is a reconnecting WebSocket client with exponential backoff; `hooks/useSignals.ts` merges WS pushes into the React Query cache so polling is a safety net, not the primary path.
- **One typed API surface.** `lib/api.ts` is the only place that knows backend URLs. Point `endpoints` at your real FastAPI/Flask routes and every hook picks it up — no scattered fetch calls.
- **Runs before the backend is fully wired.** Set `VITE_USE_MOCKS=true` and the whole UI works on deterministic mock data, so frontend and backend work can proceed in parallel.
- **Scalable structure.** Domain hooks (`useSignals`, `usePositions`, `useRisk`, `usePerformance`) own their own caching/polling intervals; components stay presentational; adding a new screen is: new type → new hook → new page → route.

## Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Runs at `http://localhost:5173`. The dev server proxies `/api` and `/ws` to `VITE_API_TARGET` (default `http://localhost:8000`) — point that at wherever your backend actually runs.

To run with no backend at all:

```bash
# in .env.local
VITE_USE_MOCKS=true
```

## Wiring to your actual backend

Everything backend-facing lives in two files:

1. **`src/lib/api.ts`** — the `endpoints` object lists every REST path the frontend expects (`/signals/live`, `/positions`, `/risk/snapshot`, `/performance/summary`, `/market/status`, `/risk/kill-switch`). Update these to match your real router paths, and adjust the response shapes in `src/types/index.ts` if your backend's JSON differs.
2. **`src/lib/websocket.ts`** — expects JSON frames of shape `{ type: string, payload: unknown }`. `useSignals.ts` currently listens for a `signal.new` event; wire this to whatever your `oms_signals` Redis stream publishes once it has a producer in the live prediction path.

## Structure

```
src/
  types/        domain types (Signal, Position, RiskSnapshot, ...)
  lib/          api client, websocket client, mock data
  store/        zustand — connection status, live signal buffer, UI state
  hooks/        one hook per domain, each owns its own polling/caching
  components/
    layout/     Sidebar, Topbar, Layout shell
    ui/         Card, Badge, Button, Skeleton, EmptyState — no business logic
    dashboard/  StatCard, SignalFeed, PositionsTable, RiskPanel, EquityCurve, MarketPulse
  pages/        one page per route, composed from dashboard/ components
```

## Design

Dark trading-terminal theme — near-black charcoal (`ink-900`), tabular-mono numerals on every financial figure (`.num` utility), saffron as the single accent color used only for emphasis (confidence scores, brand mark, primary actions), and teal/rose in place of default green/red for bullish/bearish so it doesn't read as a generic admin template. The signature element is `MarketPulse.tsx` — the scrolling live-signal confidence strip under the top bar, the one persistent "pulse" of the ML engine.

## Next steps

- Add auth (route guard + token refresh) once the backend exposes a login endpoint.
- Add an orders/trade-history page once OMS order records are exposed via REST.
- Consider virtualizing `PositionsTable` and `SignalFeed` if position/signal counts grow past ~100 rows.
