import { PositionsTable } from '@/components/dashboard/PositionsTable'

export default function Positions() {
  return (
    <div>
      <div className="mb-4">
        <h1 className="text-lg font-semibold">Open Positions</h1>
        <p className="text-sm text-mist-500">Live mark-to-market from the order management system.</p>
      </div>
      <PositionsTable />
    </div>
  )
}
