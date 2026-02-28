/** Error boundary — catches render errors and shows a retry panel. */

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.props.onError?.(error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="flex flex-col items-center justify-center h-full bg-zinc-800 rounded p-6 gap-3">
          <p className="text-sm text-red-400">Something went wrong</p>
          <p className="text-xs text-zinc-500 max-w-xs text-center">
            {this.state.error?.message ?? 'An unexpected error occurred'}
          </p>
          <button
            onClick={this.handleRetry}
            className="px-3 py-1.5 rounded text-xs font-medium bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-colors"
          >
            Retry
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
