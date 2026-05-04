"use client";

import { useMemo, useRef, useState } from "react";
import type { Template, Vec2 } from "@/lib/types";
import { useLang } from "./LanguageContext";
import { translateRoomName } from "@/lib/i18n";

const COLLINEAR_TOL = 0.005;
const MIN_SEGMENT_LEN = 0.05;
const SNAP_DISTANCE_M = 0.6;

interface Edge {
  p1: Vec2;
  p2: Vec2;
}

interface DragState {
  kind: "door" | "window";
  index: number;
  validEdges: Edge[];
  snapPosition: Vec2 | null;
  currentMouse: Vec2;
}

// Architectural-plan palette: warm cream for living spaces, cool blue for outdoor,
// pale green for wet rooms, neutral for circulation. Saturated enough to read at
// thumbnail size, muted enough to feel like a real plan.
const ROOM_COLORS: Record<string, string> = {
  living: "#FFF7DC",
  kitchen: "#FFEFC7", kochnische: "#FFEFC7",
  dining: "#FFE8D1",
  bedroom: "#F5E8E0", master_bedroom: "#F0DDD0",
  bathroom: "#E0EDDA", wc: "#E0EDDA",
  balcony: "#D6E5EE",
  pooja: "#FFE5B4",
  utility: "#E8E5E0", abstellraum: "#E8E5E0",
  corridor: "#F5F0E8", diele: "#F5F0E8", entry: "#F5F0E8",
  study: "#E5EEF5",
  wardrobe: "#EFE5DD",
  stairs: "#FCD9A1",
};

const ROOM_ABBR: Record<string, string> = {
  living: "Liv.",
  kitchen: "Kit.", kochnische: "Kit.",
  dining: "Dr.",
  bedroom: "Br.", master_bedroom: "Mbr.",
  bathroom: "Bath", wc: "WC",
  balcony: "Balc.",
  pooja: "Pooja",
  utility: "Util.", abstellraum: "Util.",
  corridor: "Hall", diele: "Hall", entry: "Entry",
  study: "Study",
  wardrobe: "W.I.C.",
  stairs: "Stairs",
};

function polygonEdges(polygon: Vec2[]): Edge[] {
  const out: Edge[] = [];
  for (let i = 0; i < polygon.length; i++) {
    out.push({ p1: polygon[i], p2: polygon[(i + 1) % polygon.length] });
  }
  return out;
}

function polygonBBox(polygon: Vec2[]): { x: number; y: number; w: number; h: number } {
  const xs = polygon.map(p => p[0]);
  const ys = polygon.map(p => p[1]);
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    w: Math.max(...xs) - Math.min(...xs),
    h: Math.max(...ys) - Math.min(...ys),
  };
}

function pointOnEdge(p1: Vec2, p2: Vec2, point: Vec2, tol = 0.011): boolean {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
  const L = Math.hypot(dx, dy);
  if (L < tol) return false;
  const ux = dx / L, uy = dy / L;
  const qx = point[0] - p1[0], qy = point[1] - p1[1];
  const perp = Math.abs(qx * uy - qy * ux);
  if (perp > tol) return false;
  const t = qx * ux + qy * uy;
  return -tol <= t && t <= L + tol;
}

function findHostEdgeForPoint(
  point: Vec2,
  boundary: Vec2[],
  rooms: { polygon: Vec2[] }[],
): Edge | null {
  for (const e of polygonEdges(boundary)) {
    if (pointOnEdge(e.p1, e.p2, point)) return e;
  }
  for (const r of rooms) {
    for (const e of polygonEdges(r.polygon)) {
      if (pointOnEdge(e.p1, e.p2, point)) return e;
    }
  }
  return null;
}

function collinearOverlap(a: Edge, b: Edge, tol: number): [Vec2, Vec2] | null {
  const dx = a.p2[0] - a.p1[0];
  const dy = a.p2[1] - a.p1[1];
  const L = Math.hypot(dx, dy);
  if (L < tol) return null;
  const ux = dx / L;
  const uy = dy / L;
  const perp = (p: Vec2) => Math.abs((p[0] - a.p1[0]) * uy - (p[1] - a.p1[1]) * ux);
  if (perp(b.p1) > tol || perp(b.p2) > tol) return null;
  const proj = (p: Vec2) => (p[0] - a.p1[0]) * ux + (p[1] - a.p1[1]) * uy;
  const tb1 = proj(b.p1);
  const tb2 = proj(b.p2);
  const blo = Math.min(tb1, tb2);
  const bhi = Math.max(tb1, tb2);
  const lo = Math.max(0, blo);
  const hi = Math.min(L, bhi);
  if (hi - lo < MIN_SEGMENT_LEN) return null;
  return [
    [a.p1[0] + ux * lo, a.p1[1] + uy * lo],
    [a.p1[0] + ux * hi, a.p1[1] + uy * hi],
  ];
}

interface Props {
  template: Template;
  onMove?: (kind: "door" | "window", index: number, newPos: Vec2) => Promise<void>;
  width?: number;
  height?: number;
  busy?: boolean;
  activeFloor?: number;
  onActiveFloorChange?: (floor: number) => void;
}

export default function Plan2D({ template, onMove, width = 600, height = 600, busy,
                                  activeFloor: activeFloorProp, onActiveFloorChange }: Props) {
  const { lang } = useLang();
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeFloorLocal, setActiveFloorLocal] = useState(0);
  const activeFloor = activeFloorProp ?? activeFloorLocal;
  const setActiveFloor = onActiveFloorChange ?? setActiveFloorLocal;

  // Keep ref in sync with state for handlers that read latest value
  dragRef.current = drag;

  // If multi-floor, project the active floor into single-floor shape.
  // Clamp activeFloor to the valid range — guards against stale prop when
  // user navigates from a duplex (activeFloor=1) to a single-floor template.
  const isMultifloor = !!template.floors && template.floors.length > 0;
  const safeActiveFloor = isMultifloor
    ? Math.min(Math.max(0, activeFloor), template.floors!.length - 1)
    : 0;
  const activeFloorData = isMultifloor ? template.floors![safeActiveFloor] : null;
  const rooms = activeFloorData ? activeFloorData.rooms : template.rooms;
  const doors = activeFloorData ? activeFloorData.doors : template.doors;
  const windows = activeFloorData ? activeFloorData.windows : template.windows;
  const boundary = (activeFloorData?.boundary_polygon) ?? template.boundary.polygon;
  const bbox = useMemo(() => {
    const xs = boundary.map(p => p[0]);
    const ys = boundary.map(p => p[1]);
    return {
      x: Math.min(...xs),
      y: Math.min(...ys),
      w: Math.max(...xs) - Math.min(...xs),
      h: Math.max(...ys) - Math.min(...ys),
    };
  }, [boundary]);

  const pad = 24;
  const scale = Math.min((width - 2 * pad) / bbox.w, (height - 2 * pad) / bbox.h);
  const cw = bbox.w * scale;
  const ch = bbox.h * scale;
  const ox = (width - cw) / 2 - bbox.x * scale;
  const oy = height - (height - ch) / 2 + bbox.y * scale;

  function tx(p: Vec2): [number, number] {
    return [ox + p[0] * scale, oy - p[1] * scale];
  }
  function txInv(sx: number, sy: number): Vec2 {
    return [(sx - ox) / scale, (oy - sy) / scale];
  }

  function computeValidEdges(kind: "door" | "window", index: number): Edge[] {
    if (kind === "window") {
      const win = windows[index];
      const room = rooms.find(r => r.id === win.room);
      if (!room) return [];
      const result: Edge[] = [];
      for (const re of polygonEdges(room.polygon)) {
        for (const be of polygonEdges(boundary)) {
          const ov = collinearOverlap(re, be, COLLINEAR_TOL);
          if (ov) result.push({ p1: ov[0], p2: ov[1] });
        }
      }
      return result;
    }
    const door = doors[index];
    if (door.from === "outside" || door.to === "outside") {
      const otherId = door.from === "outside" ? door.to : door.from;
      const room = rooms.find(r => r.id === otherId);
      if (!room) return [];
      const result: Edge[] = [];
      for (const re of polygonEdges(room.polygon)) {
        for (const be of polygonEdges(boundary)) {
          const ov = collinearOverlap(re, be, COLLINEAR_TOL);
          if (ov) result.push({ p1: ov[0], p2: ov[1] });
        }
      }
      return result;
    }
    const fromRoom = rooms.find(r => r.id === door.from);
    const toRoom = rooms.find(r => r.id === door.to);
    if (!fromRoom || !toRoom) return [];
    const result: Edge[] = [];
    for (const fe of polygonEdges(fromRoom.polygon)) {
      for (const te of polygonEdges(toRoom.polygon)) {
        const ov = collinearOverlap(fe, te, COLLINEAR_TOL);
        if (ov) result.push({ p1: ov[0], p2: ov[1] });
      }
    }
    return result;
  }

  function snap(point: Vec2, edges: Edge[], halfWidth: number): Vec2 | null {
    let bestDist = Infinity;
    let bestPos: Vec2 | null = null;
    for (const e of edges) {
      const ux = e.p2[0] - e.p1[0];
      const uy = e.p2[1] - e.p1[1];
      const L = Math.hypot(ux, uy);
      if (L < MIN_SEGMENT_LEN) continue;
      const nx = ux / L;
      const ny = uy / L;
      const dx = point[0] - e.p1[0];
      const dy = point[1] - e.p1[1];
      const t = dx * nx + dy * ny;
      const tClamped = Math.max(halfWidth, Math.min(L - halfWidth, t));
      if (L - 2 * halfWidth < 0) continue;
      const sx = e.p1[0] + nx * tClamped;
      const sy = e.p1[1] + ny * tClamped;
      const dist = Math.hypot(point[0] - sx, point[1] - sy);
      if (dist < bestDist && dist < SNAP_DISTANCE_M) {
        bestDist = dist;
        bestPos = [sx, sy];
      }
    }
    return bestPos;
  }

  function getMouseWorld(e: React.MouseEvent | MouseEvent): Vec2 {
    // Translate CSS pixel coords to SVG viewBox coords by scaling against
    // the rendered size — the viewBox is fixed at width × height regardless
    // of how the SVG is sized in CSS.
    const rect = svgRef.current!.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * width;
    const sy = ((e.clientY - rect.top) / rect.height) * height;
    return txInv(sx, sy);
  }

  function onOpeningMouseDown(e: React.MouseEvent, kind: "door" | "window", index: number) {
    if (busy) return;
    e.stopPropagation();
    e.preventDefault();
    const validEdges = computeValidEdges(kind, index);
    const world = getMouseWorld(e);
    const next: DragState = { kind, index, validEdges, snapPosition: null, currentMouse: world };
    dragRef.current = next;
    setDrag(next);
    setError(null);
  }

  function onSvgMouseMove(e: React.MouseEvent) {
    const d = dragRef.current;
    if (!d) return;
    const world = getMouseWorld(e);
    const opening = d.kind === "door" ? doors[d.index] : windows[d.index];
    const halfWidth = opening.width_mm / 1000 / 2;
    const snapped = snap(world, d.validEdges, halfWidth);
    const next = { ...d, currentMouse: world, snapPosition: snapped };
    dragRef.current = next;
    setDrag(next);
  }

  async function onSvgMouseUp() {
    const captured = dragRef.current;
    if (!captured) return;
    dragRef.current = null;
    setDrag(null);
    if (captured.snapPosition && onMove) {
      try {
        await onMove(captured.kind, captured.index, captured.snapPosition);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } else if (!captured.snapPosition) {
      setError("Drop on a highlighted edge — that position is not valid.");
      setTimeout(() => setError(null), 3000);
    }
  }

  return (
    <div className="relative w-full h-full flex flex-col">
      {isMultifloor && (
        <div className="flex gap-1 px-2 py-1.5 bg-white border-b border-gray-200 rounded-t">
          {template.floors!.map((f, i) => (
            <button
              key={i}
              onClick={() => setActiveFloor(i)}
              className={`text-xs px-3 py-1 rounded font-medium ${
                safeActiveFloor === i
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {f.name}
            </button>
          ))}
        </div>
      )}
      <div className="relative flex-1">
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseMove={onSvgMouseMove}
        onMouseUp={onSvgMouseUp}
        onMouseLeave={onSvgMouseUp}
        className="bg-gray-50 rounded select-none"
        style={{ touchAction: "none" }}
      >
        <rect x={0} y={0} width={width} height={height} fill="#FAFAFA" />

        <polygon
          points={boundary.map(p => tx(p).join(",")).join(" ")}
          fill="white"
          stroke="#222"
          strokeWidth={3}
        />

        {/* Room fills */}
        {rooms.map(room => {
          const color = ROOM_COLORS[room.type] ?? "#EFEFEF";
          const points = room.polygon.map(p => tx(p).join(",")).join(" ");
          return (
            <polygon key={`fill-${room.id}`}
                     points={points} fill={color}
                     stroke="#1a1a1a" strokeWidth={2}
                     pointerEvents="none" />
          );
        })}

        {/* Furniture per room */}
        {rooms.map(room => {
          const bb = polygonBBox(room.polygon);
          const fragments: React.ReactElement[] = [];
          const ux = scale; // world meters → SVG units (1m * scale)

          // Helper to draw a thin-stroked rect in world coords
          const wRect = (
            wx: number, wy: number, wW: number, wH: number,
            opts: { fill?: string; stroke?: string; rx?: number; key: string }
          ) => {
            const [sx, sy] = tx([wx, wy + wH]);
            return (
              <rect key={opts.key}
                    x={sx} y={sy}
                    width={wW * ux} height={wH * ux}
                    fill={opts.fill ?? "white"}
                    stroke={opts.stroke ?? "#555"} strokeWidth={1}
                    rx={opts.rx} />
            );
          };
          const wEllipse = (cx: number, cy: number, rx: number, ry: number,
                              opts: { key: string }) => {
            const [sx, sy] = tx([cx, cy]);
            return <ellipse key={opts.key} cx={sx} cy={sy}
                              rx={rx * ux} ry={ry * ux}
                              fill="white" stroke="#555" strokeWidth={1} />;
          };
          const wLine = (x1: number, y1: number, x2: number, y2: number,
                          opts: { key: string; w?: number }) => {
            const [a, b] = [tx([x1, y1]), tx([x2, y2])];
            return <line key={opts.key} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
                          stroke="#555" strokeWidth={opts.w ?? 1} />;
          };

          // Bathroom: tub (if room large enough), toilet, sink
          if (room.type === "bathroom") {
            const cx = bb.x + bb.w / 2, cy = bb.y + bb.h / 2;
            const orient = bb.w > bb.h ? "horizontal" : "vertical";
            // Tub: along longest wall, 1.7m × 0.7m
            if (Math.min(bb.w, bb.h) > 1.5 && Math.max(bb.w, bb.h) > 1.8) {
              if (orient === "horizontal") {
                fragments.push(wRect(bb.x + 0.1, bb.y + 0.1, Math.min(1.7, bb.w - 0.2), 0.7,
                                      { key: `${room.id}-tub`, rx: 4 }));
              } else {
                fragments.push(wRect(bb.x + 0.1, bb.y + 0.1, 0.7, Math.min(1.7, bb.h - 0.2),
                                      { key: `${room.id}-tub`, rx: 4 }));
              }
            }
            // Toilet
            const toiletX = orient === "horizontal" ? bb.x + bb.w - 0.5 : bb.x + 0.1;
            const toiletY = orient === "horizontal" ? bb.y + 0.1 : bb.y + bb.h - 0.6;
            fragments.push(wEllipse(toiletX + 0.2, toiletY + 0.25, 0.18, 0.22,
                                     { key: `${room.id}-toilet` }));
            fragments.push(wRect(toiletX + 0.05, toiletY, 0.3, 0.15,
                                  { key: `${room.id}-toiletbox`, rx: 1 }));
            // Sink
            const sinkX = orient === "horizontal" ? bb.x + bb.w - 0.5 : bb.x + bb.w - 0.5;
            const sinkY = orient === "horizontal" ? bb.y + bb.h - 0.5 : bb.y + 0.1;
            fragments.push(wEllipse(sinkX + 0.2, sinkY + 0.2, 0.18, 0.13,
                                     { key: `${room.id}-sink` }));
          }
          // WC: just toilet
          if (room.type === "wc") {
            fragments.push(wEllipse(bb.x + bb.w / 2, bb.y + bb.h / 2, 0.18, 0.22,
                                     { key: `${room.id}-toilet` }));
          }
          // Kitchen / kitchenette: L-shape counter + sink + stove
          if (room.type === "kitchen" || room.type === "kochnische") {
            const counterDepth = 0.6;
            // Counter along top wall
            fragments.push(wRect(bb.x, bb.y + bb.h - counterDepth, bb.w, counterDepth,
                                  { key: `${room.id}-counter1`, fill: "#FAFAFA" }));
            // Counter along right wall (L-shape)
            if (bb.h > 2 && bb.w > 2.5) {
              fragments.push(wRect(bb.x + bb.w - counterDepth, bb.y, counterDepth, bb.h - counterDepth,
                                    { key: `${room.id}-counter2`, fill: "#FAFAFA" }));
            }
            // Sink (square in counter)
            fragments.push(wRect(bb.x + 0.4, bb.y + bb.h - counterDepth + 0.1, 0.5, 0.4,
                                  { key: `${room.id}-sink`, rx: 2 }));
            // Stove (4 burners)
            const stoveX = bb.x + 1.2, stoveY = bb.y + bb.h - counterDepth + 0.1;
            fragments.push(wRect(stoveX, stoveY, 0.5, 0.4, { key: `${room.id}-stove` }));
            for (let bx = 0; bx < 2; bx++) for (let by = 0; by < 2; by++) {
              fragments.push(wEllipse(stoveX + 0.12 + bx * 0.25, stoveY + 0.1 + by * 0.18,
                                       0.06, 0.06, { key: `${room.id}-bn${bx}${by}` }));
            }
            // Kitchen island if room is big
            if (bb.w > 4 && bb.h > 3.5) {
              const ix = bb.x + bb.w / 2 - 1, iy = bb.y + 0.8;
              fragments.push(wRect(ix, iy, 2, 0.8, { key: `${room.id}-island`, fill: "#F5F5F5", rx: 2 }));
            }
          }
          // Bedroom / Master bedroom: bed + bedside tables
          if (room.type === "bedroom" || room.type === "master_bedroom") {
            const isMaster = room.type === "master_bedroom";
            const bedW = isMaster ? 1.8 : 1.4;
            const bedH = 2.0;
            // Place bed against the short edge if possible, leaving walking room
            const horizontalRoom = bb.w > bb.h;
            let bx = horizontalRoom ? bb.x + 0.3 : bb.x + (bb.w - bedW) / 2;
            let by = horizontalRoom ? bb.y + (bb.h - bedH) / 2 : bb.y + 0.3;
            if (bedW + 0.6 < bb.w && bedH + 0.6 < bb.h) {
              fragments.push(wRect(bx, by, bedW, bedH, {
                key: `${room.id}-bed`, fill: "#FFFFFF", rx: 3,
              }));
              // Pillows (rounded squares at the head)
              fragments.push(wRect(bx + 0.1, by + bedH - 0.5, bedW * 0.4, 0.35,
                                    { key: `${room.id}-pil1`, fill: "#F0F0F0", rx: 4 }));
              if (isMaster) {
                fragments.push(wRect(bx + bedW * 0.5, by + bedH - 0.5, bedW * 0.4, 0.35,
                                      { key: `${room.id}-pil2`, fill: "#F0F0F0", rx: 4 }));
              }
              // Bedside tables
              if (bx - 0.55 > bb.x) {
                fragments.push(wRect(bx - 0.55, by + bedH - 0.5, 0.5, 0.5,
                                      { key: `${room.id}-bt1`, rx: 2 }));
              }
            }
          }
          // Living: sofa + coffee table
          if (room.type === "living") {
            const sofaW = Math.min(2.5, bb.w * 0.5);
            const sofaH = 0.85;
            const sx = bb.x + (bb.w - sofaW) / 2;
            const sy = bb.y + 0.3;
            fragments.push(wRect(sx, sy, sofaW, sofaH, { key: `${room.id}-sofa`, fill: "#F0E8D8", rx: 3 }));
            // Coffee table
            const ctW = Math.min(1.2, sofaW * 0.6);
            const ctH = 0.5;
            fragments.push(wRect(sx + (sofaW - ctW) / 2, sy + sofaH + 0.3, ctW, ctH,
                                  { key: `${room.id}-ct`, rx: 2 }));
          }
          // Dining: table + chairs (oval)
          if (room.type === "dining") {
            const cx = bb.x + bb.w / 2, cy = bb.y + bb.h / 2;
            const tableW = Math.min(1.6, bb.w * 0.55);
            const tableH = Math.min(0.9, bb.h * 0.45);
            fragments.push(wEllipse(cx, cy, tableW / 2, tableH / 2, { key: `${room.id}-table` }));
          }
          // Stairs (corridor with name containing 'stairs', or stairs type)
          if (room.type === "stairs" ||
              (room.type === "corridor" && /stair|landing/i.test(room.name))) {
            const treads = 8;
            const horiz = bb.w > bb.h;
            const stepW = horiz ? bb.w / treads : bb.w * 0.7;
            const stepH = horiz ? bb.h * 0.7 : bb.h / treads;
            const stx = horiz ? bb.x : bb.x + (bb.w - stepW) / 2;
            const sty = horiz ? bb.y + (bb.h - stepH) / 2 : bb.y;
            for (let i = 0; i < treads; i++) {
              fragments.push(horiz
                ? wLine(stx + i * stepW, sty, stx + i * stepW, sty + stepH,
                          { key: `${room.id}-st${i}`, w: 1 })
                : wLine(stx, sty + i * stepH, stx + stepW, sty + i * stepH,
                          { key: `${room.id}-st${i}`, w: 1 }));
            }
          }
          return <g key={`furn-${room.id}`} pointerEvents="none" opacity={0.85}>{fragments}</g>;
        })}

        {/* Room labels — bold abbreviations + dimensions */}
        {rooms.map(room => {
          const bb = polygonBBox(room.polygon);
          const cx = bb.x + bb.w / 2;
          const cy = bb.y + bb.h / 2;
          const [scx, scy] = tx([cx, cy]);
          const abbr = ROOM_ABBR[room.type] ?? translateRoomName(room.name, lang);
          const dims = `${bb.w.toFixed(1)}m × ${bb.h.toFixed(1)}m`;
          return (
            <g key={`lbl-${room.id}`} pointerEvents="none">
              <text x={scx} y={scy - 4}
                    fontSize={13} fontWeight={700}
                    textAnchor="middle" fill="#1a1a1a"
                    fontFamily='"Bookman Old Style", Georgia, serif'>
                {abbr}
              </text>
              <text x={scx} y={scy + 11}
                    fontSize={9} textAnchor="middle" fill="#444"
                    fontFamily='"Bookman Old Style", Georgia, serif'>
                {dims}
              </text>
            </g>
          );
        })}

        {drag && drag.validEdges.map((e, i) => {
          const [a, b] = [tx(e.p1), tx(e.p2)];
          return (
            <line
              key={`ve${i}`}
              x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
              stroke="#10b981"
              strokeWidth={6}
              strokeLinecap="round"
              opacity={0.5}
              pointerEvents="none"
            />
          );
        })}

        {drag && drag.snapPosition && (() => {
          const [gx, gy] = tx(drag.snapPosition);
          const isDoor = drag.kind === "door";
          return (
            <g pointerEvents="none">
              <circle cx={gx} cy={gy} r={14} fill="#10b981" opacity={0.25} />
              <rect
                x={gx - (isDoor ? 6 : 8)}
                y={gy - (isDoor ? 6 : 4)}
                width={isDoor ? 12 : 16}
                height={isDoor ? 12 : 8}
                fill={isDoor ? "#ef4444" : "#3b82f6"}
                stroke="#fff"
                strokeWidth={2}
                opacity={0.85}
              />
            </g>
          );
        })()}

        {/* Doors: architectural symbol — quarter-arc swing + door panel */}
        {doors.map((d, i) => {
          const isDragging = drag?.kind === "door" && drag.index === i;
          const pos = isDragging ? drag.currentMouse : (d.position as Vec2);
          const widthM = d.width_mm / 1000;
          // Find host edge to orient door properly
          const host = findHostEdgeForPoint(pos, boundary, rooms);
          const halfW = widthM / 2;
          // Direction unit vector along the wall
          let ux = 1, uy = 0;
          if (host) {
            const hL = Math.hypot(host.p2[0] - host.p1[0], host.p2[1] - host.p1[1]);
            ux = (host.p2[0] - host.p1[0]) / hL;
            uy = (host.p2[1] - host.p1[1]) / hL;
          }
          // Door panel endpoints in world
          const x1w = pos[0] - ux * halfW;
          const y1w = pos[1] - uy * halfW;
          const x2w = pos[0] + ux * halfW;
          const y2w = pos[1] + uy * halfW;
          // Perpendicular for the swing direction (default: into +Y from the wall)
          const px = -uy, py = ux;
          // Open-position endpoint (90° swing into room)
          const openX = x1w + px * widthM;
          const openY = y1w + py * widthM;
          // Convert to screen
          const [sx1, sy1] = tx([x1w, y1w]);
          const [sx2, sy2] = tx([x2w, y2w]);
          const [sox, soy] = tx([openX, openY]);
          // Wall-gap mask (matches wall thickness ~0.23m): bright stroke breaking the wall line
          const radiusPx = Math.hypot(sox - sx1, soy - sy1);
          // Path: M(sx1,sy1) → arc(rad) to (sox,soy) → L(sx1,sy1)
          const arcPath = `M ${sx1} ${sy1} A ${radiusPx} ${radiusPx} 0 0 1 ${sox} ${soy}`;
          return (
            <g key={`d${i}`}
               style={{ cursor: busy ? "not-allowed" : "grab" }}
               opacity={isDragging ? 0.4 : 1}
               onMouseDown={e => onOpeningMouseDown(e, "door", i)}>
              <title>{`Door: ${d.from} → ${d.to} (${d.width_mm}mm)`}</title>
              {/* White wall-gap to "cut" the wall stroke */}
              <line x1={sx1} y1={sy1} x2={sx2} y2={sy2}
                    stroke="white" strokeWidth={6} strokeLinecap="butt" />
              {/* Door panel */}
              <line x1={sx1} y1={sy1} x2={sox} y2={soy}
                    stroke="#222" strokeWidth={2} />
              {/* Swing arc */}
              <path d={arcPath} stroke="#666" strokeWidth={1}
                    fill="none" strokeDasharray="3 2" />
              {/* Pivot dot */}
              <circle cx={sx1} cy={sy1} r={2.5} fill="#222" />
              {/* Invisible drag-target (large hit area) */}
              <rect x={Math.min(sx1, sx2) - 6}
                    y={Math.min(sy1, sy2) - 6}
                    width={Math.abs(sx2 - sx1) + 12}
                    height={Math.abs(sy2 - sy1) + 12}
                    fill="transparent" />
            </g>
          );
        })}

        {/* Windows: architectural symbol — double parallel line on the wall */}
        {windows.map((w, i) => {
          const isDragging = drag?.kind === "window" && drag.index === i;
          const pos = isDragging ? drag.currentMouse : (w.position as Vec2);
          const widthM = w.width_mm / 1000;
          const host = findHostEdgeForPoint(pos, boundary, rooms);
          let ux = 1, uy = 0;
          if (host) {
            const hL = Math.hypot(host.p2[0] - host.p1[0], host.p2[1] - host.p1[1]);
            ux = (host.p2[0] - host.p1[0]) / hL;
            uy = (host.p2[1] - host.p1[1]) / hL;
          }
          const halfW = widthM / 2;
          const x1w = pos[0] - ux * halfW;
          const y1w = pos[1] - uy * halfW;
          const x2w = pos[0] + ux * halfW;
          const y2w = pos[1] + uy * halfW;
          const [sx1, sy1] = tx([x1w, y1w]);
          const [sx2, sy2] = tx([x2w, y2w]);
          // Offset perpendicular for the double-line glass effect
          const px = -uy, py = ux;
          const offset = 2;
          const [ox1, oy1] = [sx1 + px * 0, sy1 + py * 0];
          // Skip the offset point compute, just push lines
          return (
            <g key={`w${i}`}
               style={{ cursor: busy ? "not-allowed" : "grab" }}
               opacity={isDragging ? 0.4 : 1}
               onMouseDown={e => onOpeningMouseDown(e, "window", i)}>
              <title>{`Window: ${w.room} (${w.width_mm}mm)`}</title>
              {/* White wall-gap */}
              <line x1={sx1} y1={sy1} x2={sx2} y2={sy2}
                    stroke="white" strokeWidth={6} strokeLinecap="butt" />
              {/* Outer wall line */}
              <line x1={sx1} y1={sy1} x2={sx2} y2={sy2}
                    stroke="#222" strokeWidth={1} />
              {/* Inner glass line (offset) */}
              <line x1={sx1 + 0} y1={sy1 + 0}
                    x2={sx2 + 0} y2={sy2 + 0}
                    stroke="#3b82f6" strokeWidth={1.5} opacity={0.7} />
              {/* End caps */}
              <line x1={sx1} y1={sy1 - 3} x2={sx1} y2={sy1 + 3}
                    stroke="#222" strokeWidth={1} />
              <line x1={sx2} y1={sy2 - 3} x2={sx2} y2={sy2 + 3}
                    stroke="#222" strokeWidth={1} />
              <rect x={Math.min(sx1, sx2) - 6}
                    y={Math.min(sy1, sy2) - 6}
                    width={Math.abs(sx2 - sx1) + 12}
                    height={Math.abs(sy2 - sy1) + 12}
                    fill="transparent" />
            </g>
          );
        })}

        {!drag && (
          <text
            x={width / 2}
            y={height - 6}
            fontSize={10}
            textAnchor="middle"
            fill="#94a3b8"
            pointerEvents="none"
          >
            Drag doors (red) or windows (blue) — valid edges highlight in green
          </text>
        )}
      </svg>

      {error && (
        <div className="absolute top-2 left-2 right-2 bg-red-50 border border-red-200 text-red-700 text-xs rounded px-3 py-2 shadow-sm">
          {error}
        </div>
      )}
      {busy && (
        <div className="absolute inset-0 bg-white/40 flex items-center justify-center pointer-events-none">
          <div className="bg-white border border-gray-200 px-3 py-1.5 rounded text-xs text-gray-700 shadow">
            Updating…
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
