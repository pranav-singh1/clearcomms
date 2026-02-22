export function Features() {
  const items = [
    {
      title: "100% Air-Gapped",
      desc: "All processing occurs strictly on local hardware. Zero external API calls, ensuring complete operational security.",
    },
    {
      title: "Hardware Optimized",
      desc: "ONNX Whisper models natively accelerated on Qualcomm NPU hardware via QNN execution provider.",
    },
    {
      title: "Structured Intelligence",
      desc: "LLaMA-based entity extraction converts raw transcripts into standardized JSON action reports instantly.",
    },
    {
      title: "Tactical Interface",
      desc: "Distraction-free environment tailored for analysts. Review assist, playback, and immediate data export.",
    }
  ];

  return (
    <section id="features" className="py-12 border-b border-defense-border/50">
      <div className="mb-10">
        <h2 className="text-sm font-mono text-defense-accent uppercase tracking-widest mb-2">04 / Core Capabilities</h2>
        <h3 className="text-3xl font-semibold text-white">System Features</h3>
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        {items.map((feat, idx) => (
          <div key={idx} className="bg-defense-800 border border-defense-border p-8 flex flex-col gap-3 group hover:border-defense-accent transition-colors">
            <h4 className="text-lg font-medium text-white flex items-center gap-3">
              <span className="font-mono text-defense-muted text-sm tracking-tighter group-hover:text-defense-accent transition-colors">
                {String(idx + 1).padStart(2, '0')}
              </span>
              {feat.title}
            </h4>
            <p className="text-sm text-defense-muted leading-relaxed pl-8">
              {feat.desc}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}