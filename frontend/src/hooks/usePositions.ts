import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { mockPositions } from '@/lib/mockData'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

export function usePositions() {
  return useQuery({
    queryKey: ['positions'],
    queryFn: USE_MOCKS ? async () => mockPositions() : api.getPositions,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
}
