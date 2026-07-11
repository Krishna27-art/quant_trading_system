import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { mockMarketStatus, mockPerformance } from '@/lib/mockData'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

export function usePerformance() {
  return useQuery({
    queryKey: ['performance'],
    queryFn: USE_MOCKS ? async () => mockPerformance() : api.getPerformanceSummary,
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
}

export function useMarketStatus() {
  return useQuery({
    queryKey: ['marketStatus'],
    queryFn: USE_MOCKS ? async () => mockMarketStatus() : api.getMarketStatus,
    refetchInterval: 10_000,
    staleTime: 5_000,
  })
}
