"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Template, Vec2 } from "@/lib/types";

const COLOR_BY_TYPE: Record<string, number> = {
  kitchen: 0xfff3cd, kochnische: 0xfff3cd,
  living: 0xd1ecf1,
  bedroom: 0xe2d5f0, master_bedroom: 0xd5c8e0,
  bathroom: 0xd4edda, wc: 0xd4edda,
  balcony: 0xf8f9fa,
  pooja: 0xffe5b4,
  utility: 0xe0e0e0, abstellraum: 0xe0e0e0,
  corridor: 0xf5f5f5, diele: 0xf5f5f5, entry: 0xf5f5f5,
  dining: 0xffe5e5,
  study: 0xe5f0ff,
};

const WALL_COLOR = 0xeeeeee;
const FLOOR_COLOR = 0xf3f4f6;
const DOOR_COLOR = 0xff6633;
const WINDOW_COLOR = 0x4488ff;

interface Props {
  template: Template;
  activeFloor?: number;
}

// Compute exterior wall segments (boundary edges minus opening gaps)
function computeExteriorSegments(
  boundary: Vec2[],
  doors: { position: Vec2; width_mm: number }[],
  windows: { position: Vec2; width_mm: number }[],
): [Vec2, Vec2][] {
  const ON_EDGE = 0.011;
  const openings = [...doors, ...windows];
  const segments: [Vec2, Vec2][] = [];
  for (let i = 0; i < boundary.length; i++) {
    const p1 = boundary[i];
    const p2 = boundary[(i + 1) % boundary.length];
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const L = Math.hypot(dx, dy);
    if (L < 1e-6) continue;
    const ux = dx / L;
    const uy = dy / L;

    // Find openings on this edge
    const ops: { t_start: number; t_end: number }[] = [];
    for (const op of openings) {
      const qx = op.position[0] - p1[0];
      const qy = op.position[1] - p1[1];
      const perp = Math.abs(qx * uy - qy * ux);
      if (perp > ON_EDGE) continue;
      const t = qx * ux + qy * uy;
      if (t < -ON_EDGE || t > L + ON_EDGE) continue;
      const halfW = op.width_mm / 1000 / 2;
      ops.push({
        t_start: Math.max(0, t - halfW),
        t_end: Math.min(L, t + halfW),
      });
    }
    ops.sort((a, b) => a.t_start - b.t_start);

    let cursor = 0;
    for (const op of ops) {
      if (op.t_start > cursor + 0.05) {
        segments.push([
          [p1[0] + ux * cursor, p1[1] + uy * cursor],
          [p1[0] + ux * op.t_start, p1[1] + uy * op.t_start],
        ]);
      }
      cursor = Math.max(cursor, op.t_end);
    }
    if (L - cursor > 0.05) {
      segments.push([
        [p1[0] + ux * cursor, p1[1] + uy * cursor],
        [p1[0] + ux * L, p1[1] + uy * L],
      ]);
    }
  }
  return segments;
}

// Find interior wall segments (shared room edges minus interior door gaps)
function computeInteriorSegments(
  rooms: { id: string; polygon: Vec2[] }[],
  doors: { from: string; to: string; position: Vec2; width_mm: number }[],
): [Vec2, Vec2][] {
  const ON_EDGE = 0.011;
  const COLLINEAR = 0.001;
  const all: { id: string; edge: [Vec2, Vec2] }[] = [];
  for (const r of rooms) {
    for (let i = 0; i < r.polygon.length; i++) {
      all.push({
        id: r.id,
        edge: [r.polygon[i], r.polygon[(i + 1) % r.polygon.length]],
      });
    }
  }

  const seen = new Set<string>();
  const shared: [Vec2, Vec2][] = [];
  for (let i = 0; i < all.length; i++) {
    for (let j = i + 1; j < all.length; j++) {
      if (all[i].id === all[j].id) continue;
      const overlap = collinearOverlap(all[i].edge, all[j].edge, COLLINEAR);
      if (!overlap) continue;
      const key = edgeKey(overlap);
      if (seen.has(key)) continue;
      seen.add(key);
      shared.push(overlap);
    }
  }

  const interiorDoors = doors.filter((d) => d.from !== "outside" && d.to !== "outside");
  const final: [Vec2, Vec2][] = [];
  for (const seg of shared) {
    const [p1, p2] = seg;
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const L = Math.hypot(dx, dy);
    const ux = dx / L;
    const uy = dy / L;
    const ops: { t_start: number; t_end: number }[] = [];
    for (const d of interiorDoors) {
      const qx = d.position[0] - p1[0];
      const qy = d.position[1] - p1[1];
      const perp = Math.abs(qx * uy - qy * ux);
      if (perp > ON_EDGE) continue;
      const t = qx * ux + qy * uy;
      if (t < -ON_EDGE || t > L + ON_EDGE) continue;
      const halfW = d.width_mm / 1000 / 2;
      ops.push({
        t_start: Math.max(0, t - halfW),
        t_end: Math.min(L, t + halfW),
      });
    }
    ops.sort((a, b) => a.t_start - b.t_start);

    let cursor = 0;
    for (const op of ops) {
      if (op.t_start > cursor + 0.05) {
        final.push([
          [p1[0] + ux * cursor, p1[1] + uy * cursor],
          [p1[0] + ux * op.t_start, p1[1] + uy * op.t_start],
        ]);
      }
      cursor = Math.max(cursor, op.t_end);
    }
    if (L - cursor > 0.05) {
      final.push([
        [p1[0] + ux * cursor, p1[1] + uy * cursor],
        [p1[0] + ux * L, p1[1] + uy * L],
      ]);
    }
  }
  return final;
}

function collinearOverlap(
  a: [Vec2, Vec2],
  b: [Vec2, Vec2],
  tol: number,
): [Vec2, Vec2] | null {
  const [a1, a2] = a;
  const dx = a2[0] - a1[0];
  const dy = a2[1] - a1[1];
  const L = Math.hypot(dx, dy);
  if (L < tol) return null;
  const ux = dx / L;
  const uy = dy / L;

  function perp(p: Vec2): number {
    return Math.abs((p[0] - a1[0]) * uy - (p[1] - a1[1]) * ux);
  }
  if (perp(b[0]) > tol || perp(b[1]) > tol) return null;

  function proj(p: Vec2): number {
    return (p[0] - a1[0]) * ux + (p[1] - a1[1]) * uy;
  }
  const tb1 = proj(b[0]);
  const tb2 = proj(b[1]);
  const blo = Math.min(tb1, tb2);
  const bhi = Math.max(tb1, tb2);
  const lo = Math.max(0, blo);
  const hi = Math.min(L, bhi);
  if (hi - lo < 0.05) return null;
  return [
    [a1[0] + ux * lo, a1[1] + uy * lo],
    [a1[0] + ux * hi, a1[1] + uy * hi],
  ];
}

function edgeKey(seg: [Vec2, Vec2]): string {
  const a = [Math.round(seg[0][0] * 1000), Math.round(seg[0][1] * 1000)] as const;
  const b = [Math.round(seg[1][0] * 1000), Math.round(seg[1][1] * 1000)] as const;
  const ord = a[0] < b[0] || (a[0] === b[0] && a[1] <= b[1]) ? [a, b] : [b, a];
  return `${ord[0][0]},${ord[0][1]}-${ord[1][0]},${ord[1][1]}`;
}

function buildFloor(
  scene: THREE.Scene,
  boundaryPolygon: Vec2[],
  rooms: Template["rooms"],
  doors: Template["doors"],
  windows: Template["windows"],
  thickness: number,
  height: number,
  elevation: number,
) {
  const boundaryShape = new THREE.Shape(
    boundaryPolygon.map(([x, y]) => new THREE.Vector2(x, y)),
  );
  const slab = new THREE.Mesh(
    new THREE.ExtrudeGeometry(boundaryShape, { depth: 0.15, bevelEnabled: false }),
    new THREE.MeshStandardMaterial({ color: FLOOR_COLOR, roughness: 0.9 }),
  );
  slab.rotation.x = -Math.PI / 2;
  slab.position.y = elevation - 0.15;
  scene.add(slab);

  for (const room of rooms) {
    const shape = new THREE.Shape(
      room.polygon.map(([x, y]) => new THREE.Vector2(x, y)),
    );
    const color = COLOR_BY_TYPE[room.type] ?? 0xefefef;
    const floor = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, { depth: 0.02, bevelEnabled: false }),
      new THREE.MeshStandardMaterial({ color, roughness: 0.7, transparent: true, opacity: 0.85 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = elevation;
    scene.add(floor);
  }

  function addWallSegment(p1: Vec2, p2: Vec2) {
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const L = Math.hypot(dx, dy);
    if (L < 0.05) return;
    const angle = Math.atan2(dy, dx);
    const cx = (p1[0] + p2[0]) / 2;
    const cy = (p1[1] + p2[1]) / 2;
    const wall = new THREE.Mesh(
      new THREE.BoxGeometry(L, height, thickness),
      new THREE.MeshStandardMaterial({ color: WALL_COLOR, roughness: 0.6 }),
    );
    wall.position.set(cx, elevation + height / 2, -cy);
    wall.rotation.y = -angle;
    scene.add(wall);
  }

  for (const seg of computeExteriorSegments(boundaryPolygon, doors, windows)) {
    addWallSegment(seg[0], seg[1]);
  }
  for (const seg of computeInteriorSegments(rooms, doors)) {
    addWallSegment(seg[0], seg[1]);
  }

  for (const d of doors) {
    const dh = (d.height_mm ?? 2100) / 1000;
    const dw = d.width_mm / 1000;
    const door = new THREE.Mesh(
      new THREE.BoxGeometry(dw, dh, 0.08),
      new THREE.MeshStandardMaterial({ color: DOOR_COLOR, roughness: 0.5 }),
    );
    door.position.set(d.position[0], elevation + dh / 2, -d.position[1]);
    scene.add(door);
  }

  for (const w of windows) {
    const wh = (w.height_mm ?? 1200) / 1000;
    const ww = w.width_mm / 1000;
    const sill = (w.sill_mm ?? 900) / 1000;
    const win = new THREE.Mesh(
      new THREE.BoxGeometry(ww, wh, 0.08),
      new THREE.MeshStandardMaterial({ color: WINDOW_COLOR, roughness: 0.4, opacity: 0.6, transparent: true }),
    );
    win.position.set(w.position[0], elevation + sill + wh / 2, -w.position[1]);
    scene.add(win);
  }
}

function buildScene(template: Template, scene: THREE.Scene, activeFloor?: number) {
  const t = template.boundary.wall_thickness_mm / 1000;
  const defaultH = template.boundary.ceiling_height_mm / 1000;

  if (template.floors && template.floors.length > 0) {
    // If activeFloor is provided, render only that one (matches the 2D selector).
    // Otherwise render all floors stacked.
    const floorsToRender = activeFloor !== undefined
      ? [{ fl: template.floors[activeFloor], idx: activeFloor }]
      : template.floors.map((fl, idx) => ({ fl, idx }));
    for (const { fl, idx } of floorsToRender) {
      const flH = (fl.ceiling_height_mm ?? template.boundary.ceiling_height_mm) / 1000;
      // Compute elevation from accumulated heights of floors below
      let flElev = fl.elevation_mm !== undefined ? fl.elevation_mm / 1000 : 0;
      if (fl.elevation_mm === undefined) {
        for (let i = 0; i < idx; i++) {
          flElev += (template.floors[i].ceiling_height_mm
            ?? template.boundary.ceiling_height_mm) / 1000;
        }
      }
      const poly = fl.boundary_polygon ?? template.boundary.polygon;
      buildFloor(scene, poly, fl.rooms, fl.doors, fl.windows, t, flH, flElev);
    }
  } else {
    buildFloor(scene, template.boundary.polygon, template.rooms,
                template.doors, template.windows, t, defaultH, 0);
  }
}

export default function Viewer3D({ template, activeFloor }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f7f8);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(20, 30, 10);
    scene.add(dir);

    // Build the scene
    buildScene(template, scene, activeFloor);

    // Frame the model
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim / Math.tan((Math.PI / 4) / 2) * 0.7;
    camera.position.set(center.x + dist * 0.7, dist * 0.8, center.z + dist * 0.7);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.update();

    let raf = 0;
    function animate() {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    animate();

    function onResize() {
      const w = container.clientWidth;
      const hh = container.clientHeight;
      camera.aspect = w / hh;
      camera.updateProjectionMatrix();
      renderer.setSize(w, hh);
    }
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [template, activeFloor]);

  return <div ref={containerRef} className="w-full h-full bg-gray-50 rounded-lg overflow-hidden" />;
}
