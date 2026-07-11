import { LucideIcon } from 'lucide-react'

export function EmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: LucideIcon
  title: string
  description: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Icon className="h-8 w-8 text-mist-700 mb-3" strokeWidth={1.5} />
      <p className="text-sm font-medium text-mist-300">{title}</p>
      <p className="text-xs text-mist-500 mt-1 max-w-xs">{description}</p>
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <p className="text-sm font-medium text-bear-400">Couldn't load this data</p>
      <p className="text-xs text-mist-500 mt-1 max-w-xs num">{message}</p>
    </div>
  )
}
