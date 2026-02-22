export function Hero() {
  return (
    <section className="flex flex-col md:flex-row items-center justify-between gap-12 pt-16 pb-8 border-b border-defense-border/50">
      <div className="flex-1 flex flex-col gap-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-defense-800 border border-defense-border text-xs font-mono text-defense-muted uppercase tracking-widest self-start">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          System Online
        </div>
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white leading-tight">
          ClearComms<span className="text-defense-accent">_</span>
        </h1>
        <p className="text-xl text-defense-muted max-w-xl leading-relaxed">
          Offline AI that converts noisy radio into accurate transcripts and structured incident reports.
        </p>
        <div className="text-sm font-mono text-defense-muted border-l-2 border-defense-accent pl-4 py-1">
          Runs fully on device. No internet required.
        </div>
        <div className="flex items-center gap-4 pt-4">
          <a href="#demo" className="px-6 py-3 bg-white text-black font-semibold text-sm hover:bg-gray-200 transition-colors">
            View System Demo
          </a>
          <a href="https://github.com/qualcomm-ai" target="_blank" rel="noreferrer" className="px-6 py-3 bg-defense-800 text-white font-semibold text-sm border border-defense-border hover:bg-defense-700 transition-colors">
            View Source
          </a>
        </div>
      </div>
      <div className="flex-1 w-full bg-defense-800 border border-defense-border aspect-video p-4 flex flex-col shadow-2xl relative overflow-hidden">
        {/* Abstract UI placeholder */}
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-defense-accent to-transparent opacity-50"></div>
        <div className="flex items-center gap-2 mb-4 pb-4 border-b border-defense-border">
          <div className="w-3 h-3 bg-defense-border rounded-full"></div>
          <div className="w-3 h-3 bg-defense-border rounded-full"></div>
          <div className="w-3 h-3 bg-defense-border rounded-full"></div>
          <div className="text-xs font-mono text-defense-muted ml-auto">UI_PREVIEW_SYS_01</div>
        </div>
        <div className="flex-1 flex flex-col gap-3">
          <div className="h-6 w-1/3 bg-defense-700"></div>
          <div className="h-4 w-full bg-defense-700"></div>
          <div className="h-4 w-5/6 bg-defense-700"></div>
          <div className="h-4 w-4/6 bg-defense-700"></div>
          <div className="mt-auto h-24 w-full bg-defense-900 border border-defense-border flex items-end p-2 gap-1">
             <div className="w-2 bg-defense-accent h-1/4"></div>
             <div className="w-2 bg-defense-accent h-2/4"></div>
             <div className="w-2 bg-defense-accent h-1/2"></div>
             <div className="w-2 bg-defense-accent h-3/4"></div>
             <div className="w-2 bg-defense-accent h-full"></div>
             <div className="w-2 bg-defense-accent h-2/4"></div>
             <div className="w-2 bg-defense-accent h-1/4"></div>
          </div>
        </div>
      </div>
    </section>
  );
}