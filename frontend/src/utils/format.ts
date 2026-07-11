export function fmt(n: number | undefined | null, decimals = 1, fallback = '—'): string {
  return typeof n === 'number' && !Number.isNaN(n) ? n.toFixed(decimals) : fallback;
}
