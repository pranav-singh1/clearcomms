export function Footer() {
  return (
    <footer className="bg-defense-800 border-t border-defense-border py-8 mt-auto">
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4 text-xs font-mono text-defense-muted">
        <div>
          &copy; {new Date().getFullYear()} CLEARCOMMS SYS_
        </div>
        <div className="flex gap-6">
          <a href="#" className="hover:text-white transition-colors">Documentation</a>
          <a href="https://github.com/qualcomm-ai" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">GitHub Repository</a>
          <span>V_1.0.0</span>
        </div>
      </div>
    </footer>
  );
}