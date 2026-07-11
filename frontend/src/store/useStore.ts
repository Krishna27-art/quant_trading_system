import { create } from 'zustand'
import type { ConnectionStatus, Signal } from '@/types'

interface AppState {
  connectionStatus: ConnectionStatus
  setConnectionStatus: (status: ConnectionStatus) => void

  liveSignals: Signal[]
  pushLiveSignal: (signal: Signal) => void

  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useStore = create<AppState>((set) => ({
  connectionStatus: 'connecting',
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  liveSignals: [],
  pushLiveSignal: (signal) =>
    set((state) => ({
      liveSignals: [signal, ...state.liveSignals].slice(0, 50),
    })),

  sidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}))
