import { Component, ErrorInfo, ReactNode } from 'react'
import { AlertOctagon } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Wire this to your logging/monitoring backend.
    console.error('IERM frontend crash:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="h-screen w-screen flex items-center justify-center bg-ink-900">
          <div className="panel p-6 max-w-sm text-center">
            <AlertOctagon className="h-6 w-6 text-bear-400 mx-auto mb-3" />
            <p className="text-sm font-medium text-mist-100">Something broke in the terminal.</p>
            <p className="text-xs text-mist-500 mt-1 num">{this.state.error.message}</p>
            <button
              className="mt-4 text-xs px-3 py-1.5 rounded-md bg-ink-700 border border-ink-500 hover:bg-ink-600"
              onClick={() => window.location.reload()}
            >
              Reload
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
