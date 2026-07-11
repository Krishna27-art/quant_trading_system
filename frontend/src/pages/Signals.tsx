import { SignalFeed } from '@/components/dashboard/SignalFeed'

export default function Signals() {
  return (
    <div className="max-w-3xl">
      <div className="mb-4">
        <h1 className="text-lg font-semibold">Live Signals</h1>
        <p className="text-sm text-mist-500">
          Every setup that cleared the model's win-probability threshold this session.
        </p>
      </div>
      <SignalFeed />
    </div>
  )
}
