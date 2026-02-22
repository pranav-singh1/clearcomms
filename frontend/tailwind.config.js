/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        defense: {
          900: '#050505',
          800: '#111111',
          700: '#1a1a1a',
          600: '#222222',
          accent: '#2b5b84',
          accentHover: '#3775aa',
          text: '#ededed',
          muted: '#8a8a8a',
          border: '#2a2a2a'
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}