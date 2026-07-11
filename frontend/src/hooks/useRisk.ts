import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { mockRisk } from '@/lib/mockData'
import type { RiskSnapshot } from '@/types'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

export function useRisk() {
  return useQuery({
    queryKey: ['risk'],
    queryFn: USE_MOCKS ? async () => mockRisk() : api.getRiskSnapshot,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
}

export function useKillSwitch() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (active: boolean) => {
      if (USE_MOCKS) return { ...mockRisk(), killSwitchActive: active }
      return api.toggleKillSwitch(active)
    },
    onSuccess: (data: RiskSnapshot) => {
      queryClient.setQueryData(['risk'], data)
    },
  })
}
