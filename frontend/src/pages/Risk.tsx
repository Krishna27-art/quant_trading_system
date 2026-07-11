import { RiskPanel } from '@/components/dashboard/RiskPanel'

export default function Risk() {
  return (
    <div className="max-w-xl">
      <div className="mb-4">
        <h1 className="text-lg font-semibold">Risk Governance</h1>
        <p className="text-sm text-mist-500">Exposure limits, margin utilization, and the trading kill switch.</p>
      </div>
      <RiskPanel />
    </div>
  )
}
