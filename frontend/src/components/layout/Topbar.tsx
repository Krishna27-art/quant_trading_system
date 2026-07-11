import { Menu } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '@/store/useStore'
import { useMarketStatus } from '@/hooks/usePerformance'
import { useClock } from '@/hooks/useClock'
import { fmt } from '@/utils/format'

const STATUS_LABEL: Record<string, string> = {
  connected: 'Live',
  connecting: 'Connecting…',
  disconnected: 'Reconnecting…',
}

const STATUS_DOT: Record<string, string> = {
  connected: 'bg-bull-500',
  connecting: 'bg-warn-500',
  disconnected: 'bg-bear-500',
}

export function Topbar() {
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  const connectionStatus = useStore((s) => s.connectionStatus)
  const { data: market } = useMarketStatus()
  const now = useClock()

  return (
    <header className="h-14 shrink-0 border-b border-ink-600 bg-ink-850/80 backdrop-blur flex items-center px-4 gap-4">
      <button
        onClick={toggleSidebar}
        className="p-1.5 rounded-md text-mist-500 hover:text-mist-100 hover:bg-ink-700"
        aria-label="Toggle sidebar"
      >
        <Menu className="h-4 w-4" />
      </button>

      <div className="flex items-center gap-2 text-xs">
        <span
          className={clsx(
            'h-1.5 w-1.5 rounded-full',
            STATUS_DOT[connectionStatus],
            connectionStatus === 'connected' && 'animate-pulse-dot'
          )}
        />
        <span className="text-mist-500 num">{STATUS_LABEL[connectionStatus]}</span>
      </div>

      {market && (
        <div className="flex items-center gap-4 ml-2 text-xs">
          <span
            className={clsx(
              'px-1.5 py-0.5 rounded border text-[11px] font-mono uppercase tracking-wide',
              market.isOpen
                ? 'border-bull-500/30 text-bull-400 bg-bull-500/10'
                : 'border-ink-500 text-mist-500 bg-ink-700'
            )}
          >
            {market.session.replace('_', ' ')}
          </span>
          <span className="num text-mist-300">
            NIFTY 50{' '}
            <span className={market.nifty50Change >= 0 ? 'text-bull-400' : 'text-bear-400'}>
              {market.nifty50Change >= 0 ? '+' : ''}
              {fmt(market.nifty50Change, 2)}%
            </span>
          </span>
          <span className="num text-mist-300">
            SENSEX{' '}
            <span className={market.sensexChange >= 0 ? 'text-bull-400' : 'text-bear-400'}>
              {market.sensexChange >= 0 ? '+' : ''}
              {fmt(market.sensexChange, 2)}%
            </span>
          </span>
        </div>
      )}

      <div className="ml-auto text-xs text-mist-700 num">
        {now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })} IST
      </div>
    </header>
  )
}
