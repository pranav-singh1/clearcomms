export function Problem() {
  return (
    <section id="problem" className="py-12 border-b border-defense-border/50">
      <div className="grid md:grid-cols-3 gap-8">
        <div className="col-span-1">
          <h2 className="text-sm font-mono text-defense-accent uppercase tracking-widest mb-2">01 / The Challenge</h2>
          <h3 className="text-3xl font-semibold text-white">Tactical Communication Limits</h3>
        </div>
        <div className="col-span-2 grid sm:grid-cols-2 gap-8">
          <div className="flex flex-col gap-3 p-6 bg-defense-800 border border-defense-border">
            <svg className="w-6 h-6 text-defense-muted mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="square" strokeLinejoin="miter" strokeWidth={1.5} d="M3 10h18M3 14h18M12 3v18" />
            </svg>
            <h4 className="text-lg font-medium text-white">High Noise Environments</h4>
            <p className="text-sm text-defense-muted leading-relaxed">
              Standard ASR models fail in radio-degraded environments involving squelch, static, and heavy background noise.
            </p>
          </div>
          <div className="flex flex-col gap-3 p-6 bg-defense-800 border border-defense-border">
            <svg className="w-6 h-6 text-defense-muted mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="square" strokeLinejoin="miter" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h4 className="text-lg font-medium text-white">Critical Information Loss</h4>
            <p className="text-sm text-defense-muted leading-relaxed">
              Manual transcription delays intelligence. High-stress scenarios result in dropped details and inaccurate reporting.
            </p>
          </div>
          <div className="flex flex-col gap-3 p-6 bg-defense-800 border border-defense-border sm:col-span-2">
            <svg className="w-6 h-6 text-defense-muted mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="square" strokeLinejoin="miter" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
            <h4 className="text-lg font-medium text-white">Denied Environments</h4>
            <p className="text-sm text-defense-muted leading-relaxed">
              Cloud-reliant APIs are inaccessible in the field. Systems must operate air-gapped on edge hardware without external connectivity.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}