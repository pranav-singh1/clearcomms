import { Hero } from './components/Hero';
import { Problem } from './components/Problem';
import { Pipeline } from './components/Pipeline';
import { Demo } from './components/Demo';
import { Features } from './components/Features';
import { Performance } from './components/Performance';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-defense-900 text-defense-text flex flex-col font-sans">
      <header className="border-b border-defense-border bg-defense-800/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-bold tracking-tighter text-white">CLEARCOMMS<span className="text-defense-accent">_</span></div>
          <nav className="hidden md:flex gap-6 text-sm text-defense-muted font-mono uppercase tracking-widest">
            <a href="#problem" className="hover:text-white transition-colors">Problem</a>
            <a href="#pipeline" className="hover:text-white transition-colors">Architecture</a>
            <a href="#demo" className="hover:text-white transition-colors">System Demo</a>
            <a href="#features" className="hover:text-white transition-colors">Features</a>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-12 flex flex-col gap-24">
        <Hero />
        <Problem />
        <Pipeline />
        <Demo />
        <Features />
        <Performance />
      </main>
      <Footer />
    </div>
  );
}