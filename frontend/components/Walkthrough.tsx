"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Template, Room, Vec2 } from "@/lib/types";
import { walkthroughRoomUrl, dollhouseUrl, PrewarmStatus } from "@/lib/api";
import { translateRoomName } from "@/lib/i18n";
import { useLang } from "./LanguageContext";
import type { Walk3DHandle } from "./Walk3D";
import type { WalkCubemapHandle } from "./WalkCubemap";
import type { WalkSplatHandle } from "./WalkSplat";

// Dynamic imports — playcanvas is a large client-only module, and this
// also avoids any TSX parser ambiguity from `forwardRef<T, P>()`.
const Walk3D = dynamic(() => import("./Walk3D"), { ssr: false });
const WalkCubemap = dynamic(() => import("./WalkCubemap"), { ssr: false });
const WalkSplat = dynamic(() => import("./WalkSplat"), { ssr: false });

type Mode = "photo" | "walk3d" | "photoreal" | "splat";

// Room types we skip in the walkthrough (no useful interior view)
const SKIP_TYPES = new Set([
  "corridor", "stairs", "balcony", "loggia", "terrace",
  "store", "wardrobe", "abstellraum", "utility", "passage",
]);
const ALWAYS_INCLUDE = new Set([
  "kitchen", "kueche", "kochnische",
  "bathroom", "wc", "bad",
]);

// Logical tour order: living -> dining -> bedrooms -> kitchen -> office -> bath
const ORDER: Record<string, number> = {
  living: 0, dining: 1,
  master_bedroom: 2, bedroom: 3,
  kitchen: 4, kueche: 4, kochnische: 4,
  office: 5, study: 5,
  bathroom: 6, bad: 6, wc: 7,
  diele: 8, flur: 8, entry: 8,
};

interface Stop {
  kind: "dollhouse" | "room";
  id: string;          // 'dollhouse' or room.id
  name: string;
  area_sqm?: number;
  type?: string;
  url: string;
}

interface Props {
  template: Template;
  /** Background prewarm status for the photoreal cubemaps (passed in
   *  from DetailView). When `done`, the modal opens directly in the
   *  Photoreal-walk tab starting at the entry room. While still
   *  rendering, we show progress and let the user use Photo / 3D walk. */
  prewarm?: PrewarmStatus | null;
  onClose?: () => void;
}

function pickRooms(template: Template): Room[] {
  let rooms: Room[] = template.rooms || [];
  if (rooms.length === 0 && template.floors?.length) {
    rooms = template.floors.flatMap((fl) => fl.rooms || []);
  }
  const chosen = rooms.filter((r) => {
    const t = (r.type || "").toLowerCase();
    if (SKIP_TYPES.has(t)) return false;
    const area = r.area_sqm ?? 0;
    if (!ALWAYS_INCLUDE.has(t) && area < 4) return false;
    return true;
  });
  chosen.sort((a, b) => {
    const oa = ORDER[(a.type || "").toLowerCase()] ?? 9;
    const ob = ORDER[(b.type || "").toLowerCase()] ?? 9;
    if (oa !== ob) return oa - ob;
    return (b.area_sqm ?? 0) - (a.area_sqm ?? 0);
  });
  return chosen;
}

/**
 * Floor plan with the active room shaded yellow. Lightweight inline
 * SVG (no API call) so the highlight is instant on room change.
 */
function Minimap({ template, activeRoomId, onRoomClick }: {
  template: Template;
  activeRoomId: string | null;
  onRoomClick?: (roomId: string) => void;
}) {
  const polygon = template.boundary.polygon;
  const rooms = template.rooms ?? [];
  const xs = polygon.map((p) => p[0]);
  const ys = polygon.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = maxX - minX, h = maxY - minY;
  const PAD = 0.5;
  const vbW = w + 2 * PAD;
  const vbH = h + 2 * PAD;

  const tx = (p: Vec2) => `${p[0] - minX + PAD},${vbH - (p[1] - minY + PAD)}`;

  return (
    <svg
      viewBox={`0 0 ${vbW} ${vbH}`}
      style={{ width: "100%", height: "auto", display: "block",
                background: "white", borderRadius: 8 }}
    >
      <polygon
        points={polygon.map(tx).join(" ")}
        fill="#fafafa" stroke="#222" strokeWidth={0.08}
      />
      {rooms.map((r) => {
        const isActive = r.id === activeRoomId;
        const clickable = onRoomClick != null;
        return (
          <g key={r.id}>
            <polygon
              points={r.polygon.map(tx).join(" ")}
              fill={isActive ? "rgba(255,200,60,0.55)" : "#f0f0f0"}
              stroke={isActive ? "#FFC83D" : "#888"}
              strokeWidth={isActive ? 0.12 : 0.04}
              style={{ cursor: clickable ? "pointer" : "default",
                        transition: "fill 0.2s",
                        pointerEvents: clickable ? "auto" : "none" }}
              onClick={clickable ? () => onRoomClick!(r.id) : undefined}
            >
              {clickable && (
                <title>{r.name} — click to walk here</title>
              )}
            </polygon>
            {/* Room name label, only on hover via title; here render
                small label centred for orientation */}
            {(() => {
              const cx = r.polygon.reduce((s, p) => s + p[0], 0) / r.polygon.length;
              const cy = r.polygon.reduce((s, p) => s + p[1], 0) / r.polygon.length;
              return (
                <text
                  x={cx - minX + PAD}
                  y={vbH - (cy - minY + PAD)}
                  fontSize="0.32"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={isActive ? "#7a5a00" : "#888"}
                  fontWeight={isActive ? 700 : 500}
                  style={{ pointerEvents: "none",
                            fontFamily: "system-ui, sans-serif" }}
                >
                  {r.name.length > 12 ? r.name.slice(0, 11) + "…" : r.name}
                </text>
              );
            })()}
          </g>
        );
      })}
    </svg>
  );
}

export default function Walkthrough({ template, prewarm, onClose }: Props) {
  const { lang } = useLang();
  const [idx, setIdx] = useState(0);
  const [loaded, setLoaded] = useState<Set<string>>(new Set());
  const [showDollhouse, setShowDollhouse] = useState(false);
  // If the photoreal cubemaps are already done, open in that tab. Otherwise
  // start in Photo so the user has something to look at while it builds.
  const [mode, setMode] = useState<Mode>(
    prewarm?.done ? "photoreal" : "photo",
  );
  const [walk3dActiveRoom, setWalk3dActiveRoom] = useState<string | null>(null);
  const [cubemapActiveRoom, setCubemapActiveRoom] = useState<string | null>(
    prewarm?.entry_room_id ?? null,
  );
  const walk3dRef = useRef<Walk3DHandle | null>(null);
  const cubemapRef = useRef<WalkCubemapHandle | null>(null);
  const splatRef = useRef<WalkSplatHandle | null>(null);

  // If prewarm completes while modal is open, auto-flip to photoreal tab.
  // (Only first time it goes from non-done -> done; respect later user clicks.)
  const autoFlippedRef = useRef(false);
  useEffect(() => {
    if (prewarm?.done && !autoFlippedRef.current && mode === "photo") {
      setMode("photoreal");
      autoFlippedRef.current = true;
      if (prewarm.entry_room_id) setCubemapActiveRoom(prewarm.entry_room_id);
    }
  }, [prewarm?.done, prewarm?.entry_room_id, mode]);

  const stops = useMemo<Stop[]>(() => {
    const dollhouse: Stop = {
      kind: "dollhouse", id: "dollhouse",
      name: "Whole apartment",
      url: dollhouseUrl(template.id),
    };
    const rooms = pickRooms(template).map<Stop>((r) => ({
      kind: "room", id: r.id, name: r.name,
      area_sqm: r.area_sqm, type: r.type,
      url: walkthroughRoomUrl(template.id, r.id),
    }));
    return showDollhouse ? [dollhouse, ...rooms] : rooms;
  }, [template, showDollhouse]);

  // Reset to first stop when template changes
  useEffect(() => {
    setIdx(0);
    setLoaded(new Set());
  }, [template.id]);

  // After the current image is loaded, prefetch the next one in the
  // background by issuing a parallel fetch (with the same URL the next
  // <img> tag will use, so the browser cache picks it up). Sequential —
  // only one prefetch in flight at a time.
  useEffect(() => {
    const cur = stops[idx];
    if (!cur || !loaded.has(cur.id)) return;
    const next = stops[idx + 1];
    if (!next) return;
    const ctrl = new AbortController();
    fetch(next.url, { signal: ctrl.signal }).catch(() => {});
    return () => ctrl.abort();
  }, [loaded, idx, stops]);

  // Keyboard navigation — only in photo mode (3D mode owns its own keys).
  useEffect(() => {
    if (mode !== "photo") return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, i - 1));
      if (e.key === "ArrowRight") setIdx((i) => Math.min(stops.length - 1, i + 1));
      if (e.key === "Escape" && onClose) onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stops.length, onClose, mode]);

  // Esc to close also in 3D mode (but only when pointer is NOT locked,
  // so the user's first Esc just releases the look-cursor).
  useEffect(() => {
    if (mode !== "walk3d") return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && document.pointerLockElement == null && onClose) {
        onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, mode]);

  const cur = stops[idx];
  if (!cur) {
    return (
      <div style={{ padding: 32, color: "#aaa" }}>
        No habitable rooms found in this template.
      </div>
    );
  }

  const isLoaded = loaded.has(cur.id);

  return (
    <div className="walkthrough" style={{ display: "grid",
        gridTemplateColumns: "1fr 320px", gap: 24, padding: 24,
        background: "#0f1115", color: "#eee", borderRadius: 12,
        position: "relative", minHeight: 520 }}>

      {/* Mode tabs (top-left of the hero) */}
      <div style={{ position: "absolute", top: 36, left: 36, zIndex: 5,
                      background: "rgba(15,17,21,0.85)", borderRadius: 8,
                      padding: 4, display: "flex", gap: 2,
                      border: "1px solid #2a2d33" }}>
        <button
          onClick={() => setMode("photo")}
          style={{
            background: mode === "photo" ? "#FFC83D" : "transparent",
            color: mode === "photo" ? "#1a1a1a" : "#bbb",
            border: "none", padding: "6px 14px", borderRadius: 6,
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}
        >📷 Photo tour</button>
        {/* 🎮 3D walk hidden — photoreal cubemap walk supersedes it.
            To re-enable for engineering tests, append ?dev to the URL. */}
        {typeof window !== "undefined" && window.location.search.includes("dev") && (
          <button
            onClick={() => setMode("walk3d")}
            style={{
              background: mode === "walk3d" ? "#FFC83D" : "transparent",
              color: mode === "walk3d" ? "#1a1a1a" : "#bbb",
              border: "none", padding: "6px 14px", borderRadius: 6,
              fontSize: 13, fontWeight: 600, cursor: "pointer",
            }}
          >🎮 3D walk</button>
        )}
        <button
          onClick={() => setMode("photoreal")}
          style={{
            background: mode === "photoreal" ? "#FFC83D" : "transparent",
            color: mode === "photoreal" ? "#1a1a1a" : "#bbb",
            border: "none", padding: "6px 14px", borderRadius: 6,
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}
          title="Matterport-style 360° tour — photoreal cubemaps per room"
        >✨ Photoreal walk</button>
        <button
          onClick={() => setMode("splat")}
          style={{
            background: mode === "splat" ? "#FFC83D" : "transparent",
            color: mode === "splat" ? "#1a1a1a" : "#bbb",
            border: "none", padding: "6px 14px", borderRadius: 6,
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}
          title="Free-walk through a 3D Gaussian Splat trained on synthetic SDXL views"
        >🪐 Splat walk</button>
      </div>

      {/* Prewarm progress badge — shown while cubemaps are still rendering */}
      {prewarm && prewarm.started && !prewarm.done && (
        <div style={{
          position: "absolute", top: 36, right: 64, zIndex: 5,
          background: "rgba(15,17,21,0.85)", color: "#FFC83D",
          border: "1px solid #2a2d33", borderRadius: 8,
          padding: "6px 12px", fontSize: 12, fontWeight: 500,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{
            width: 14, height: 14, borderRadius: "50%",
            border: "2px solid #444", borderTopColor: "#FFC83D",
            animation: "spin 1s linear infinite", display: "inline-block",
          }} />
          Building photoreal tour…&nbsp;
          <strong>{prewarm.ready.length}/{prewarm.total}</strong>
          {prewarm.current_name && (
            <span style={{ color: "#bbb", fontWeight: 400 }}>
              · {prewarm.current_name}
            </span>
          )}
        </div>
      )}
      {prewarm && prewarm.done && mode !== "photoreal" && (
        <div style={{
          position: "absolute", top: 36, right: 64, zIndex: 5,
          background: "#16331f", color: "#7CE39A",
          border: "1px solid #1f4a2c", borderRadius: 8,
          padding: "6px 12px", fontSize: 12, fontWeight: 600,
          cursor: "pointer",
        }}
        onClick={() => setMode("photoreal")}
        title="Switch to the photoreal cubemap walk"
        >
          ✓ Photoreal walk ready · click to enter
        </div>
      )}
      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      {/* Hero — Photo tour OR 3D walk */}
      <div style={{ position: "relative", aspectRatio: "3/2",
                     background: "#1a1d23", borderRadius: 10, overflow: "hidden" }}>

        {mode === "walk3d" && (
          <Walk3D
            ref={walk3dRef}
            template={template}
            onActiveRoomChange={setWalk3dActiveRoom}
          />
        )}

        {mode === "photoreal" && (
          <WalkCubemap
            ref={cubemapRef}
            template={template}
            onActiveRoomChange={setCubemapActiveRoom}
            initialRoomId={prewarm?.entry_room_id ?? cubemapActiveRoom ?? undefined}
          />
        )}

        {mode === "splat" && (
          <WalkSplat
            ref={splatRef}
            template={template}
          />
        )}

        {mode === "photo" && (
        <div style={{ display: "contents" }}>
        <img
          key={cur.id}
          src={cur.url}
          alt={cur.name}
          onLoad={() => setLoaded((prev) => new Set([...prev, cur.id]))}
          onError={() => setLoaded((prev) => new Set([...prev, cur.id]))}
          style={{
            position: "absolute", inset: 0,
            width: "100%", height: "100%", objectFit: "cover",
            opacity: isLoaded ? 1 : 0,
            transition: "opacity 0.25s ease",
          }}
        />

        {/* Loading overlay */}
        {!isLoaded && (
          <div style={{ position: "absolute", inset: 0,
                          display: "flex", alignItems: "center",
                          justifyContent: "center", flexDirection: "column",
                          background: "#1a1d23", color: "#888" }}>
            <div className="spinner" style={{
              width: 36, height: 36, borderRadius: "50%",
              border: "3px solid #333", borderTopColor: "#FFC83D",
              animation: "spin 1s linear infinite", marginBottom: 12,
            }} />
            <div style={{ fontSize: 13 }}>
              Rendering photoreal interior...
            </div>
            <div style={{ fontSize: 11, color: "#666", marginTop: 6 }}>
              ~2s · SDXL-turbo + Depth ControlNet on Apple MPS
            </div>
          </div>
        )}

        {/* Caption */}
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0,
                        padding: "32px 20px 16px",
                        background: "linear-gradient(transparent, rgba(0,0,0,0.85))" }}>
          <div style={{ fontSize: 24, fontWeight: 600 }}>
            {cur.kind === "dollhouse"
              ? "Whole apartment — dollhouse cutaway"
              : translateRoomName(cur.name, lang)}
          </div>
          <div style={{ fontSize: 13, color: "#bbb", marginTop: 4 }}>
            {cur.kind === "room"
              ? `${cur.type ?? ""}${cur.area_sqm ? `  ·  ${cur.area_sqm} m²` : ""}  ·  ${idx + 1} / ${stops.length}`
              : `Walls cut away · all rooms visible · ${idx + 1} / ${stops.length}`}
          </div>
        </div>

        {/* Prev/Next chevrons */}
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          aria-label="Previous room"
          style={{
            position: "absolute", left: 16, top: "50%",
            transform: "translateY(-50%)",
            width: 48, height: 48, borderRadius: "50%",
            background: "rgba(0,0,0,0.55)", color: "white",
            border: "1px solid #444", fontSize: 24,
            cursor: idx === 0 ? "default" : "pointer",
            opacity: idx === 0 ? 0.3 : 1,
          }}
        >‹</button>
        <button
          onClick={() => setIdx((i) => Math.min(stops.length - 1, i + 1))}
          disabled={idx === stops.length - 1}
          aria-label="Next room"
          style={{
            position: "absolute", right: 16, top: "50%",
            transform: "translateY(-50%)",
            width: 48, height: 48, borderRadius: "50%",
            background: "rgba(0,0,0,0.55)", color: "white",
            border: "1px solid #444", fontSize: 24,
            cursor: idx === stops.length - 1 ? "default" : "pointer",
            opacity: idx === stops.length - 1 ? 0.3 : 1,
          }}
        >›</button>

        </div>)}

        {/* Close button — visible in BOTH modes */}
        {onClose && (
          <button
            onClick={onClose}
            aria-label="Close walkthrough"
            style={{
              position: "absolute", top: 12, right: 12,
              width: 32, height: 32, borderRadius: "50%",
              background: "rgba(0,0,0,0.55)", color: "white",
              border: "1px solid #444", fontSize: 18,
              cursor: "pointer", zIndex: 10,
            }}
          >×</button>
        )}
      </div>

      {/* Sidebar */}
      <aside style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Minimap */}
        <div style={{ background: "#1a1d23", borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 8,
                          display: "flex", justifyContent: "space-between",
                          alignItems: "center" }}>
            <span>Floor plan · click any room to walk</span>
            <span style={{ color: "#FFC83D", fontSize: 11, fontWeight: 600 }}>
              You are here
            </span>
          </div>
          <Minimap
            template={template}
            activeRoomId={
              mode === "walk3d"
                ? walk3dActiveRoom
                : mode === "photoreal"
                  ? cubemapActiveRoom
                  : (cur.kind === "room" ? cur.id : null)
            }
            onRoomClick={(roomId) => {
              if (mode === "photoreal") {
                cubemapRef.current?.goToRoom(roomId);
              } else if (mode === "walk3d") {
                walk3dRef.current?.teleportToRoom(roomId);
              } else if (mode === "photo") {
                // Find this room in the stops list and select it
                const target = stops.findIndex(
                  (s) => s.kind === "room" && s.id === roomId,
                );
                if (target >= 0) setIdx(target);
              }
            }}
          />
        </div>

        {/* Toggle dollhouse */}
        <label style={{ display: "flex", alignItems: "center", gap: 8,
                          fontSize: 13, color: "#bbb", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showDollhouse}
            onChange={(e) => {
              setShowDollhouse(e.target.checked);
              setIdx(0);
            }}
          />
          Include dollhouse cutaway
        </label>

        {/* Room list */}
        <div style={{ background: "#1a1d23", borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 8 }}>
            Tour ({stops.length} stops)
          </div>
          <ol style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {stops.map((s, i) => {
              const active = mode === "walk3d"
                ? walk3dActiveRoom === s.id
                : mode === "photoreal"
                  ? cubemapActiveRoom === s.id
                  : i === idx;
              return (
                <li
                  key={s.id}
                  onClick={() => {
                    setIdx(i);
                    if (mode === "walk3d" && s.kind === "room") {
                      walk3dRef.current?.teleportToRoom(s.id);
                    } else if (mode === "photoreal" && s.kind === "room") {
                      cubemapRef.current?.goToRoom(s.id);
                    }
                  }}
                  style={{
                    padding: "8px 12px", borderRadius: 6, cursor: "pointer",
                    color: active ? "#FFC83D" : "#bbb",
                    background: active ? "rgba(255,200,60,0.12)" : "transparent",
                    fontWeight: active ? 600 : 400, fontSize: 14,
                    display: "flex", justifyContent: "space-between",
                    transition: "background 0.15s",
                  }}
                >
                  <span>
                    {i + 1}. {s.kind === "dollhouse"
                      ? "Whole apartment"
                      : translateRoomName(s.name, lang)}
                  </span>
                  {s.area_sqm ? (
                    <span style={{ color: "#666", fontSize: 12 }}>{s.area_sqm} m²</span>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </div>

        <div style={{ fontSize: 11, color: "#666", lineHeight: 1.5 }}>
          ← → arrow keys · click rooms to jump · Esc to close.
          <br />Photoreal interior generated by SDXL-turbo + Depth ControlNet
          conditioned on the BIM 3D scene.
        </div>
      </aside>

      <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
