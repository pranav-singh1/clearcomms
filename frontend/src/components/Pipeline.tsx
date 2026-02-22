export function Pipeline() {
  const steps = [
    { id: "A", name: "RAW AUDIO", desc: "UHF/VHF Capture" },
    { id: "B", name: "PREPROCESS", desc: "Bandpass + Gate" },
    { id: "C", name: "WHISPER", desc: "On-device NPU ASR" },
    { id: "D", name: "LLaMA", desc: "Entity Extraction" },
    { id: "E", name: "OUTPUT", desc: "Structured JSON" }
  ];

  return (
    <section id="pipeline" className="py-12 border-b border-defense-border/50">
      <div className="mb-10">
        <h2 className="text-sm font-mono text-defense-accent uppercase tracking-widest mb-2">02 / System Architecture</h2>
        <h3 className="text-3xl font-semibold text-white">Offline Processing Pipeline</h3>
      </div>
      <div className="relative flex flex-col md:flex-row gap-4 items-center justify-between bg-defense-800 p-8 border border-defense-border overflow-hidden">
        {/* Connecting line */}
        <div className="hidden md:block absolute top-1/2 left-0 w-full h-[1px] bg-defense-border -z-0"></div>
        
        {steps.map((step, idx) => (
          <div key={idx} className="relative z-10 flex flex-col items-center gap-3 bg-defense-900 border border-defense-border p-4 w-40 text-center">
            <div className="w-8 h-8 rounded-none border border-defense-accent text-defense-accent flex items-center justify-center font-mono text-xs mb-2">
              {step.id}
            </div>
            <div className="font-mono text-xs font-bold text-white tracking-wider">{step.name}</div>
            <div className="text-[10px] text-defense-muted uppercase">{step.desc}</div>
          </div>
        ))}
      </div>
    </section>
  );
}