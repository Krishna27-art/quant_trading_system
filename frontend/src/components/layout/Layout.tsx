import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'
import { MarketPulse } from '@/components/dashboard/MarketPulse'

export function Layout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-ink-900">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar />
        <MarketPulse />
        <main className="flex-1 overflow-y-auto p-5">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
