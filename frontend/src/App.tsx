import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import Dashboard from '@/pages/Dashboard'
import Signals from '@/pages/Signals'
import Positions from '@/pages/Positions'
import Risk from '@/pages/Risk'
import Performance from '@/pages/Performance'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="signals" element={<Signals />} />
            <Route path="positions" element={<Positions />} />
            <Route path="risk" element={<Risk />} />
            <Route path="performance" element={<Performance />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
