"use client";

import { useEffect, useRef, useState } from "react";

type Tour = {
  id: string;          // template id (matches MP4 filename minus _tour.mp4)
  city: string;
  title: string;       // "1 BR Jugendstil"
  area: number;        // m²
  rooms: number;
  ceiling: number;     // m
  stops: number;       // number of tour stops
  style: string;       // one-line descriptor
};

const TOURS: Tour[] = [
  {
    id: "eu_de_1bed_munich_schwabing",
    city: "Munich",
    title: "1 BR Jugendstil",
    area: 50, rooms: 5, ceiling: 3.1, stops: 5,
    style: "Schwabing Altbau, period mouldings, oak parquet, French windows",
  },
  {
    id: "eu_fr_1bed_paris_marais",
    city: "Paris",
    title: "1 BR Haussmann",
    area: 50, rooms: 5, ceiling: 2.9, stops: 5,
    style: "Marais Haussmann, herringbone parquet, French windows, refined neutrals",
  },
  {
    id: "in_2bhk_bangalore_vastu",
    city: "Bangalore",
    title: "2 BR Vastu",
    area: 92, rooms: 7, ceiling: 2.7, stops: 5,
    style: "Vastu-compliant layout, granite floor, contemporary Indian design",
  },
  {
    id: "gl_jp_1ldk_tokyo_mansion",
    city: "Tokyo",
    title: "1 BR Mansion",
    area: 32, rooms: 6, ceiling: 2.4, stops: 4,
    style: "Modern minimalist, light wood, low furniture, skyline view",
  },
  {
    id: "gl_ae_1bed_dubai_marina",
    city: "Dubai",
    title: "1 BR Marina",
    area: 75, rooms: 6, ceiling: 2.7, stops: 5,
    style: "Marina luxe, marble palette, floor-to-ceiling windows, harbour view",
  },
  {
    id: "gl_us_1bed_nyc_tenement_walkup",
    city: "New York",
    title: "1 BR Tenement",
    area: 32, rooms: 4, ceiling: 2.7, stops: 4,
    style: "Tenement walkup, hardwood, exposed brick, brownstone view",
  },
  {
    id: "gl_au_2bed_sydney_modern",
    city: "Sydney",
    title: "2 BR Modern",
    area: 90, rooms: 9, ceiling: 2.7, stops: 5,
    style: "Coastal modern, indoor-outdoor flow, polished concrete, sea view",
  },
];

const ACCENT = "#FFC83D";

export default function ToursPage() {
  const [activeIdx, setActiveIdx] = useState(0);
  const [autoplay, setAutoplay] = useState(true);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const active = TOURS[activeIdx];

  // When the active tour changes, reset the video to the beginning.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    if (autoplay) v.play().catch(() => undefined);
  }, [activeIdx, autoplay]);

  // Auto-advance to the next tour when current ends (if autoplay is on).
  function onEnded() {
    if (autoplay) {
      setActiveIdx((i) => (i + 1) % TOURS.length);
    }
  }

  return (
    <main
      className="min-h-screen text-white"
      style={{
        background:
          "radial-gradient(circle at 18% 8%, rgba(255,200,61,0.10), transparent 30%)," +
          "radial-gradient(circle at 84% 92%, rgba(60,180,255,0.10), transparent 32%)," +
          "linear-gradient(180deg, #0a0c10 0%, #131722 60%, #1a1d28 100%)",
        fontFamily: "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <header className="flex items-center justify-between mb-10">
          <div>
            <div className="text-2xl font-extrabold tracking-tight">
              BIM <span style={{ color: ACCENT }}>Coordinator</span>
            </div>
            <div className="text-sm uppercase tracking-[0.28em] text-white/50 mt-1">
              Animated tour gallery · 7 templates · BIM-conditioned panoramas
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-white/70 cursor-pointer">
            <input
              type="checkbox"
              checked={autoplay}
              onChange={(e) => setAutoplay(e.target.checked)}
              className="w-4 h-4 accent-yellow-400"
            />
            Auto-advance
          </label>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-[420px_minmax(0,1fr)] gap-8">
          {/* Player */}
          <section className="flex flex-col">
            <div
              className="relative w-full overflow-hidden rounded-3xl shadow-2xl"
              style={{ aspectRatio: "9 / 16", background: "#0a0c10" }}
            >
              <video
                ref={videoRef}
                key={active.id}
                src={`/${active.id}_tour.mp4`}
                poster={`/tour_thumbs/${active.id}.jpg`}
                onEnded={onEnded}
                controls
                playsInline
                className="absolute inset-0 w-full h-full object-cover"
              />
            </div>

            {/* Active tour metadata card */}
            <div
              className="mt-5 p-5 rounded-2xl"
              style={{
                background: "rgba(20, 24, 34, 0.85)",
                border: "1px solid rgba(255, 200, 61, 0.25)",
              }}
            >
              <div
                className="text-sm uppercase tracking-[0.28em]"
                style={{ color: "#67e8f9" }}
              >
                {active.city}
              </div>
              <div className="text-3xl font-extrabold leading-tight mt-1">
                {active.title.split(" ").slice(0, 2).join(" ")}{" "}
                <span style={{ color: ACCENT }}>
                  {active.title.split(" ").slice(2).join(" ")}
                </span>
              </div>
              <div className="text-white/60 text-sm mt-2">
                {active.area} m² · {active.rooms} rooms · {active.ceiling} m ceiling ·{" "}
                {active.stops} stops · IFC valid 35/35
              </div>
              <div className="text-white/80 text-sm mt-3 leading-relaxed">
                {active.style}
              </div>
            </div>
          </section>

          {/* Tour list */}
          <section>
            <div className="flex items-baseline justify-between mb-5">
              <h2 className="text-xl font-bold tracking-tight">
                {TOURS.length} BIM templates · same pipeline, different city
              </h2>
              <div className="text-xs text-white/50 uppercase tracking-widest">
                Now playing: {activeIdx + 1} / {TOURS.length}
              </div>
            </div>

            <ul className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {TOURS.map((t, i) => {
                const isActive = i === activeIdx;
                return (
                  <li key={t.id}>
                    <button
                      onClick={() => setActiveIdx(i)}
                      className="group block w-full text-left rounded-2xl overflow-hidden transition-transform"
                      style={{
                        background: "rgba(20, 24, 34, 0.9)",
                        border: isActive
                          ? `2px solid ${ACCENT}`
                          : "2px solid rgba(255,255,255,0.08)",
                        transform: isActive ? "translateY(-3px)" : undefined,
                        boxShadow: isActive
                          ? "0 18px 50px rgba(0,0,0,0.5), 0 0 0 4px rgba(255,200,61,0.15)"
                          : "0 8px 24px rgba(0,0,0,0.3)",
                      }}
                    >
                      <div
                        className="relative"
                        style={{ aspectRatio: "9 / 16", background: "#0a0c10" }}
                      >
                        <img
                          src={`/tour_thumbs/${t.id}.jpg`}
                          alt={`${t.city} ${t.title}`}
                          className="absolute inset-0 w-full h-full object-cover"
                          loading="lazy"
                        />
                        {isActive && (
                          <div
                            className="absolute top-2 right-2 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest"
                            style={{
                              background: ACCENT,
                              color: "#1a1612",
                              boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                            }}
                          >
                            Now playing
                          </div>
                        )}
                      </div>
                      <div className="px-3 py-3">
                        <div
                          className="text-[10px] uppercase tracking-widest"
                          style={{ color: "#67e8f9" }}
                        >
                          {t.city}
                        </div>
                        <div className="text-sm font-bold mt-0.5">{t.title}</div>
                        <div className="text-xs text-white/50 mt-1">
                          {t.area} m² · {t.rooms} rooms · {t.stops} stops
                        </div>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>

            {/* "How it works" footer */}
            <div
              className="mt-8 p-5 rounded-2xl text-sm leading-relaxed"
              style={{
                background: "rgba(20, 24, 34, 0.5)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div
                className="text-xs uppercase tracking-[0.28em] mb-2"
                style={{ color: ACCENT }}
              >
                How each tour was built
              </div>
              <div className="text-white/75">
                One BIM template (polygon coords + windows + ceiling) →{" "}
                <span className="text-white">5 panoramic SDXL renders</span> conditioned
                on each room's metadata →{" "}
                <span className="text-white">parameterised HyperFrames composition</span>{" "}
                that auto-lays out the floor plan from the polygons →{" "}
                <span className="text-white">47s vertical MP4</span>. Same script ran for
                all 7 cities; each took ~50 seconds end-to-end.
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
