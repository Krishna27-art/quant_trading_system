/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Base surfaces — near-black charcoal-blue, not pure black
        ink: {
          950: '#080A0F',
          900: '#0A0E14',
          850: '#0D121A',
          800: '#131824',
          700: '#1A2130',
          600: '#232937',
          500: '#2E3648',
        },
        // Text
        mist: {
          100: '#E6E9EF',
          300: '#B7BECC',
          500: '#8B93A7',
          700: '#5B6478',
        },
        // Signature accent — saffron, used sparingly (brand / highlight only)
        saffron: {
          400: '#FFB35C',
          500: '#FF9F43',
          600: '#E6842A',
        },
        // Bullish / bearish — deliberately not stock green/red
        bull: {
          400: '#5EE6B0',
          500: '#3DDC97',
          600: '#28B87D',
        },
        bear: {
          400: '#FF7C93',
          500: '#F0466B',
          600: '#D22E52',
        },
        warn: {
          500: '#F5C453',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', '"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        panel: '0 0 0 1px rgba(255,255,255,0.04), 0 8px 24px -8px rgba(0,0,0,0.5)',
      },
      animation: {
        'ticker': 'ticker 40s linear infinite',
        'pulse-dot': 'pulse-dot 2s ease-in-out infinite',
      },
      keyframes: {
        ticker: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.35 },
        },
      },
    },
  },
  plugins: [],
}
