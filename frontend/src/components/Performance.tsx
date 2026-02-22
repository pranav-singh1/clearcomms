export function Performance() {
  return (
    <section id="performance" className="py-12 border-b border-defense-border/50">
      <div className="mb-10">
        <h2 className="text-sm font-mono text-defense-accent uppercase tracking-widest mb-2">05 / Telemetry</h2>
        <h3 className="text-3xl font-semibold text-white">Execution Metrics</h3>
      </div>
      <div className="grid sm:grid-cols-3 gap-6">
        <div className="bg-defense-800 border border-defense-border p-6 flex flex-col items-center text-center">
          <div className="font-mono text-xs text-defense-muted uppercase mb-4">Avg Latency</div>
          <div className="text-4xl font-light text-white mb-1">{"<"} 2.4s</div>
          <div className="text-xs text-defense-accent font-mono">per transmission</div>
        </div>
        <div className="bg-defense-800 border border-defense-border p-6 flex flex-col items-center text-center">
          <div className="font-mono text-xs text-defense-muted uppercase mb-4">Realtime Factor (RTF)</div>
          <div className="text-4xl font-light text-white mb-1">0.12x</div>
          <div className="text-xs text-defense-accent font-mono">faster than realtime</div>
        </div>
        <div className="bg-defense-800 border border-defense-border p-6 flex flex-col items-center text-center">
          <div className="font-mono text-xs text-defense-muted uppercase mb-4">Power Target</div>
          <div className="text-4xl font-light text-white mb-1">Burst</div>
          <div className="text-xs text-defense-accent font-mono">HTP High Power Saver</div>
        </div>
      </div>
    </section>
  );
}