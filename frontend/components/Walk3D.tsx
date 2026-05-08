"use client";

/**
 * Real-time 3D walkthrough of the BIM template using PlayCanvas.
 *
 * Architecture:
 *   1. Fetch a GLB binary from `/api/templates/{id}/gltf` (server builds it
 *      from the same trimesh meshes the depth ControlNet sees).
 *   2. Fetch the template JSON for room polygons + door positions —
 *      these drive the spawn point and a simple AABB-based wall collision
 *      check so the player can't walk through walls.
 *   3. Start a PlayCanvas Application, mount the GLB as the world,
 *      add a sun + ambient + fill light, and a first-person camera with
 *      WASD movement + mouse-look (PointerLock on click).
 *   4. Expose a `teleportTo(roomId)` hook on a ref so the parent's
 *      room-list sidebar can fly the camera into a specific room.
 */

import {
  forwardRef, useEffect, useImperativeHandle,
  useMemo, useRef, useState,
} from "react";
import type { Template, Room, Vec2 } from "@/lib/types";
import { gltfUrl } from "@/lib/api";

interface Props {
  template: Template;
  /** Optional callback fired when the camera enters a different room. */
  onActiveRoomChange?: (roomId: string | null) => void;
}

export interface Walk3DHandle {
  teleportToRoom: (roomId: string) => void;
}

/* ------------------------------------------------------------------------ */
/* Helpers                                                                  */
/* ------------------------------------------------------------------------ */

function roomCentroid(r: Room): { x: number; y: number } {
  const xs = r.polygon.map((p: Vec2) => p[0]);
  const ys = r.polygon.map((p: Vec2) => p[1]);
  return {
    x: xs.reduce((a, b) => a + b, 0) / xs.length,
    y: ys.reduce((a, b) => a + b, 0) / ys.length,
  };
}

/** Axis-aligned bounding box of a room with optional padding (negative = inset). */
function roomAabb(r: Room, pad = -0.15) {
  const xs = r.polygon.map((p: Vec2) => p[0]);
  const ys = r.polygon.map((p: Vec2) => p[1]);
  return {
    x0: Math.min(...xs) - pad, y0: Math.min(...ys) - pad,
    x1: Math.max(...xs) + pad, y1: Math.max(...ys) + pad,
  };
}

function pointInsideAnyRoom(rooms: Room[], x: number, y: number): boolean {
  for (const r of rooms) {
    const b = roomAabb(r, 0.15); // 15cm inset from wall
    if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) return true;
  }
  return false;
}

function findRoomAt(rooms: Room[], x: number, y: number): Room | null {
  for (const r of rooms) {
    const b = roomAabb(r, 0);
    if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) return r;
  }
  return null;
}

/** Choose a sensible spawn: just inside the main entry door, looking inward. */
function pickSpawn(template: Template): { x: number; y: number; yaw: number } {
  const rooms = (template.rooms?.length
    ? template.rooms
    : (template.floors?.[0]?.rooms || [])) as Room[];
  const doors = template.doors?.length
    ? template.doors
    : (template.floors?.[0]?.doors || []);

  if (rooms.length === 0) return { x: 0, y: 0, yaw: 0 };

  const main = doors.find((d) => d.is_main_entry) ?? doors[0];
  if (main && main.position) {
    const targetRoom =
      rooms.find((r) => r.id === main.to) ??
      rooms.find((r) => r.id === main.from) ??
      rooms[0];
    const c = roomCentroid(targetRoom);
    const [dx, dy] = main.position;
    const yaw = Math.atan2(c.y - dy, c.x - dx);
    return {
      x: dx + Math.cos(yaw) * 0.8,
      y: dy + Math.sin(yaw) * 0.8,
      yaw,
    };
  }
  const biggest = [...rooms].sort((a, b) => (b.area_sqm ?? 0) - (a.area_sqm ?? 0))[0];
  const c = roomCentroid(biggest);
  return { x: c.x, y: c.y, yaw: 0 };
}

/* ------------------------------------------------------------------------ */
/* Component                                                                */
/* ------------------------------------------------------------------------ */

const Walk3D = forwardRef<Walk3DHandle, Props>(function Walk3D(
  { template, onActiveRoomChange },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Track the player's logical room each frame so the parent can highlight it.
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null);

  // Stable list of rooms (flatten multi-floor for now)
  const rooms: Room[] = useMemo(() => {
    if (template.rooms?.length) return template.rooms;
    return (template.floors?.[0]?.rooms ?? []) as Room[];
  }, [template]);

  // Stable spawn
  const spawn = useMemo(() => pickSpawn(template), [template]);

  // Imperative teleport handle for the parent
  const teleportRef = useRef<((roomId: string) => void) | null>(null);
  useImperativeHandle(ref, () => ({
    teleportToRoom: (roomId: string) => teleportRef.current?.(roomId),
  }), []);

  // Notify parent when active room changes
  useEffect(() => {
    onActiveRoomChange?.(activeRoomId);
  }, [activeRoomId, onActiveRoomChange]);

  // PlayCanvas lifecycle
  useEffect(() => {
    if (!canvasRef.current) return;
    let stopped = false;
    let app: any = null;
    let cleanup: (() => void) | null = null;

    (async () => {
      const pc = await import("playcanvas");
      if (stopped || !canvasRef.current) return;

      const canvas = canvasRef.current;

      app = new pc.Application(canvas, {
        mouse: new pc.Mouse(canvas),
        keyboard: new pc.Keyboard(window),
        graphicsDeviceOptions: { preferWebGl2: true, alpha: false },
      });
      app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
      app.setCanvasResolution(pc.RESOLUTION_AUTO);
      // Generous ambient + matching gamma so PBR materials read at indoor
      // light levels. The previous 0.32 was way too dim for a small room
      // with windows you can't see (since we don't model exterior light).
      app.scene.ambientLight = new pc.Color(0.55, 0.55, 0.58);
      app.scene.exposure = 1.2;
      // gammaCorrection — defaults vary by version; force it on
      try { (app.scene as any).gammaCorrection = pc.GAMMA_SRGB; } catch {}
      try { (app.scene as any).toneMapping = pc.TONEMAP_LINEAR; } catch {}

      // Sun (directional light) — strong, angled down INTO the apartment
      const sun = new pc.Entity("sun");
      sun.addComponent("light", {
        type: "directional",
        color: new pc.Color(1.0, 0.97, 0.92),
        intensity: 2.6,
        castShadows: true,
        shadowDistance: 50,
        shadowResolution: 1024,
        shadowBias: 0.08,
        normalOffsetBias: 0.06,
      });
      sun.setEulerAngles(55, 25, 0);
      app.root.addChild(sun);

      // Soft fill from the opposite side
      const fill = new pc.Entity("fill");
      fill.addComponent("light", {
        type: "directional",
        color: new pc.Color(0.85, 0.90, 1.0),
        intensity: 0.9,
        castShadows: false,
      });
      fill.setEulerAngles(40, -150, 0);
      app.root.addChild(fill);

      // A bare-bulb omni at the apartment centroid so the back rooms
      // aren't pitch-dark (single sun + ambient isn't enough for an
      // interior with no real window light).
      const apartmentCx = rooms.reduce((s, r) => s + roomCentroid(r).x, 0) / rooms.length;
      const apartmentCy = rooms.reduce((s, r) => s + roomCentroid(r).y, 0) / rooms.length;
      const omni = new pc.Entity("omni");
      omni.addComponent("light", {
        type: "omni",
        color: new pc.Color(1.0, 0.96, 0.88),
        intensity: 1.5,
        range: 20,
      });
      omni.setPosition(apartmentCx, 2.4, -apartmentCy);
      app.root.addChild(omni);

      // Camera
      const camera = new pc.Entity("camera");
      camera.addComponent("camera", {
        clearColor: new pc.Color(0.62, 0.74, 0.88), // soft blue sky
        farClip: 200,
        nearClip: 0.05,
        fov: 75,
      });
      // World coords: BIM (X, Y) -> world (X, Z) with Z = -Y, eye height
      // on world Y. Use lookAt() to set the initial direction — bullet-
      // proof against euler/yaw conversion bugs.
      const camPos = new pc.Vec3(spawn.x, 1.65, -spawn.y);
      const lookAhead = 2.0;
      const targetX = spawn.x + Math.cos(spawn.yaw) * lookAhead;
      const targetY = spawn.y + Math.sin(spawn.yaw) * lookAhead;
      const camTgt = new pc.Vec3(targetX, 1.5, -targetY);
      camera.setPosition(camPos);
      camera.lookAt(camTgt);
      app.root.addChild(camera);

      // Load GLB
      const url = gltfUrl(template.id);
      app.assets.loadFromUrl(url, "container", (err: any, asset: any) => {
        if (stopped) return;
        if (err) {
          console.error("[Walk3D] GLB load error", err);
          setLoadError(`Failed to load 3D model: ${err}`);
          setLoading(false);
          return;
        }
        const entity = asset.resource.instantiateRenderEntity();
        app.root.addChild(entity);
        // Compute model bounds for debug
        try {
          const aabb = (entity as any).render?.meshInstances?.[0]?.aabb;
          // eslint-disable-next-line no-console
          console.log("[Walk3D] model loaded; first AABB:", aabb);
        } catch (_) { /* ignore */ }
        // eslint-disable-next-line no-console
        console.log(`[Walk3D] camera at (${camPos.x.toFixed(2)},${camPos.y.toFixed(2)},${camPos.z.toFixed(2)})  looking toward (${camTgt.x.toFixed(2)},${camTgt.y.toFixed(2)},${camTgt.z.toFixed(2)})`);
        setLoading(false);
      });

      // First-person controls — derive starting yaw/pitch from the
      // camera's current orientation (set by lookAt above).
      const initialEuler = camera.getEulerAngles();
      let yawAngle = initialEuler.y;
      let pitchAngle = initialEuler.x;
      const moveSpeed = 3.0;
      const sprintMul = 1.8;

      const keys = {
        forward: false, back: false, left: false, right: false,
        sprint: false, jump: false,
      };

      const onKeyDown = (e: any) => {
        if (e.key === pc.KEY_W || e.key === pc.KEY_UP) keys.forward = true;
        if (e.key === pc.KEY_S || e.key === pc.KEY_DOWN) keys.back = true;
        if (e.key === pc.KEY_A || e.key === pc.KEY_LEFT) keys.left = true;
        if (e.key === pc.KEY_D || e.key === pc.KEY_RIGHT) keys.right = true;
        if (e.key === pc.KEY_SHIFT) keys.sprint = true;
      };
      const onKeyUp = (e: any) => {
        if (e.key === pc.KEY_W || e.key === pc.KEY_UP) keys.forward = false;
        if (e.key === pc.KEY_S || e.key === pc.KEY_DOWN) keys.back = false;
        if (e.key === pc.KEY_A || e.key === pc.KEY_LEFT) keys.left = false;
        if (e.key === pc.KEY_D || e.key === pc.KEY_RIGHT) keys.right = false;
        if (e.key === pc.KEY_SHIFT) keys.sprint = false;
      };
      app.keyboard.on(pc.EVENT_KEYDOWN, onKeyDown);
      app.keyboard.on(pc.EVENT_KEYUP, onKeyUp);

      // PointerLock on canvas click
      const onCanvasClick = () => {
        if (document.pointerLockElement !== canvas) {
          canvas.requestPointerLock?.();
        }
      };
      canvas.addEventListener("click", onCanvasClick);

      const onMouseMove = (e: any) => {
        if (document.pointerLockElement !== canvas) return;
        yawAngle -= e.dx * 0.18;
        pitchAngle = Math.max(-85, Math.min(85, pitchAngle - e.dy * 0.18));
      };
      app.mouse.on(pc.EVENT_MOUSEMOVE, onMouseMove);

      // Per-frame update: integrate WASD + collide against room AABBs
      const updateHandler = (dt: number) => {
        // Apply look angles to camera
        camera.setLocalEulerAngles(pitchAngle, yawAngle, 0);

        // Use PlayCanvas's camera.forward / camera.right vectors —
        // bullet-proof against yaw-convention mistakes.
        const fwdVec = camera.forward;
        const rightVec = camera.right;
        const fwd = { x: fwdVec.x, z: fwdVec.z };
        const right = { x: rightVec.x, z: rightVec.z };

        let dx = 0, dz = 0;
        if (keys.forward) { dx += fwd.x; dz += fwd.z; }
        if (keys.back)    { dx -= fwd.x; dz -= fwd.z; }
        if (keys.right)   { dx += right.x; dz += right.z; }
        if (keys.left)    { dx -= right.x; dz -= right.z; }
        const len = Math.hypot(dx, dz);
        if (len > 0.001) {
          const speed = moveSpeed * (keys.sprint ? sprintMul : 1.0) * dt;
          dx = (dx / len) * speed;
          dz = (dz / len) * speed;
          const pos = camera.getPosition();
          // World Z is BIM-Y (negated). BIM-X stays X.
          // Try X then Z separately so we slide along walls instead of stopping.
          const tryX = pos.x + dx;
          const tryZ = pos.z + dz;
          const bimX = tryX;
          const bimY = -pos.z;
          const bimXOnly = tryX;
          const bimYZOnly = -tryZ;
          let newX = pos.x;
          let newZ = pos.z;
          if (pointInsideAnyRoom(rooms, bimXOnly, -pos.z)) newX = tryX;
          if (pointInsideAnyRoom(rooms, newX, -tryZ))      newZ = tryZ;
          camera.setPosition(newX, pos.y, newZ);

          // Track active room for the sidebar
          const cur = findRoomAt(rooms, newX, -newZ);
          const newId = cur?.id ?? null;
          if (newId !== activeRoomId) setActiveRoomId(newId);
        }
      };
      app.on("update", updateHandler);

      // Teleport hook for parent
      teleportRef.current = (roomId: string) => {
        const r = rooms.find((rr) => rr.id === roomId);
        if (!r) return;
        const c = roomCentroid(r);
        camera.setPosition(c.x, 1.65, -c.y);
      };

      app.start();

      cleanup = () => {
        canvas.removeEventListener("click", onCanvasClick);
        try { document.exitPointerLock?.(); } catch (e) { /* ignore */ }
        app.off("update", updateHandler);
        app.destroy();
      };
    })().catch((e) => {
      console.error("[Walk3D] init failed", e);
      setLoadError(`Init failed: ${e}`);
      setLoading(false);
    });

    return () => {
      stopped = true;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [template.id]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%",
                    background: "#0a0c10", borderRadius: 10, overflow: "hidden" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block",
                  cursor: "crosshair" }}
      />
      {loading && !loadError && (
        <div style={{ position: "absolute", inset: 0, display: "flex",
                        flexDirection: "column", alignItems: "center",
                        justifyContent: "center", color: "#aaa",
                        background: "rgba(10,12,16,0.85)" }}>
          <div className="spinner3d" style={{
            width: 36, height: 36, borderRadius: "50%",
            border: "3px solid #333", borderTopColor: "#FFC83D",
            animation: "spin 1s linear infinite", marginBottom: 12,
          }} />
          <div style={{ fontSize: 13 }}>Building 3D scene from BIM…</div>
        </div>
      )}
      {loadError && (
        <div style={{ position: "absolute", inset: 0, display: "flex",
                        alignItems: "center", justifyContent: "center",
                        color: "#f88", padding: 20, textAlign: "center" }}>
          {loadError}
        </div>
      )}
      {!loading && !loadError && (
        <div style={{ position: "absolute", left: 12, bottom: 12,
                        background: "rgba(0,0,0,0.65)", color: "white",
                        padding: "8px 12px", borderRadius: 6,
                        fontSize: 12, fontFamily: "monospace",
                        pointerEvents: "none" }}>
          <strong>Click</strong> to look around · <strong>WASD</strong> to walk · <strong>Shift</strong> to run · <strong>Esc</strong> to release
        </div>
      )}
      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
});

export default Walk3D;
