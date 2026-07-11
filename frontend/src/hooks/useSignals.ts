import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { api } from '@/lib/api'
import { mockSignals } from '@/lib/mockData'
import { LiveSocket } from '@/lib/websocket'
import { useStore } from '@/store/useStore'
import type { Signal } from '@/types'

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

export function useSignals() {
  const queryClient = useQueryClient()
  const setConnectionStatus = useStore((s) => s.setConnectionStatus)
  const pushLiveSignal = useStore((s) => s.pushLiveSignal)
  const socketRef = useRef<LiveSocket | null>(null)

  const query = useQuery({
    queryKey: ['signals'],
    queryFn: USE_MOCKS ? async () => mockSignals() : api.getSignals,
    // Polling is the safety net; the WS push below keeps things live in
    // between polls without waiting for the next interval.
    refetchInterval: 15_000,
    staleTime: 5_000,
  })

  useEffect(() => {
    if (USE_MOCKS) {
      setConnectionStatus('connected')
      return
    }
    const socket = new LiveSocket({
      path: '/ws/signals',
      onStatusChange: setConnectionStatus,
    })
    socketRef.current = socket
    socket.connect()

    const unsubscribe = socket.on<Signal>('signal.new', (signal) => {
      pushLiveSignal(signal)
      queryClient.setQueryData<Signal[]>(['signals'], (prev) => {
        if (!prev) return [signal]
        return [signal, ...prev.filter((s) => s.id !== signal.id)].slice(0, 100)
      })
    })

    return () => {
      unsubscribe()
      socket.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return query
}
