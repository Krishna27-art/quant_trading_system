import { NavLink } from 'react-router-dom'
import { Activity, LayoutGrid, LineChart, ListOrdered, ShieldAlert } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '@/store/useStore'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutGrid, end: true },
  { to: '/signals', label: 'Signals', icon: Activity },
  { to: '/positions', label: 'Positions', icon: ListOrdered },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/performance', label: 'Performance', icon: LineChart },
]

export function Sidebar() {
  const collapsed = useStore((s) => s.sidebarCollapsed)

  return (
    <aside
      className={clsx(
        'h-screen shrink-0 border-r border-ink-600 bg-ink-850 flex flex-col transition-[width] duration-200',
        collapsed ? 'w-16' : 'w-56'
      )}
    >
      <div className="h-14 flex items-center gap-2 px-4 border-b border-ink-600">
        <div className="h-6 w-6 rounded bg-saffron-500 flex items-center justify-center text-[11px] font-bold text-ink-950 shrink-0">
          I
        </div>
        {!collapsed && <span className="text-sm font-semibold tracking-wide">IERM</span>}
      </div>

      <nav className="flex-1 py-3 px-2 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-saffron-500/10 text-saffron-400 border border-saffron-500/20'
                  : 'text-mist-500 hover:text-mist-100 hover:bg-ink-700 border border-transparent'
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {!collapsed && (
        <div className="p-3 border-t border-ink-600 text-[11px] text-mist-700 leading-relaxed">
          Indian Equity Research Machine
          <br />
          Institutional signal engine
        </div>
      )}
    </aside>
  )
}
