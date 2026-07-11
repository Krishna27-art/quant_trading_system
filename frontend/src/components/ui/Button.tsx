import { ButtonHTMLAttributes } from 'react'
import clsx from 'clsx'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}

export function Button({ variant = 'ghost', size = 'md', className, ...rest }: ButtonProps) {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
        size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-2 text-sm',
        variant === 'primary' && 'bg-saffron-500 text-ink-950 hover:bg-saffron-400',
        variant === 'ghost' && 'bg-ink-700 text-mist-100 border border-ink-500 hover:bg-ink-600',
        variant === 'danger' && 'bg-bear-600 text-mist-100 hover:bg-bear-500',
        className
      )}
      {...rest}
    />
  )
}
