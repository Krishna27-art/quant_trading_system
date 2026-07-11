import { HTMLAttributes, ReactNode } from 'react'
import clsx from 'clsx'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  noPadding?: boolean
}

export function Card({ children, className, noPadding, ...rest }: CardProps) {
  return (
    <div className={clsx('panel', !noPadding && 'p-4', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: string
  subtitle?: string
  action?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between mb-3">
      <div>
        <h2 className="text-sm font-semibold text-mist-100 tracking-wide">{title}</h2>
        {subtitle && <p className="text-xs text-mist-500 mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}
