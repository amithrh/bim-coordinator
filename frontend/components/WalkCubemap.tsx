"use client";

/**
 * Photoreal Matterport-style walkthrough.
 *
 * For each room, the backend renders 6 photoreal cubemap faces using
 * SDXL-turbo + Depth ControlNet conditioned on the BIM 3D scene from
 * that room's centroid. PlayCanvas loads them as a cubemap skybox; the
 * user can mouse-look freely around the room and click any room in the
 * sidebar to teleport (the new room's cubemap loads with a quick
 * crossfade).
 *
 * Latency: ~12s cold render per room (then cached on disk on the
 * server forever). Instant on second visit.
 */

import {
  forwardRef, useEffect, useImperativeHandle, useRef, useState,
} from "react";
import type { Template, Room } from "@/lib/types";
import { fetchCubemapManifest, CubemapManifest } from "@/lib/api";

interface Props {
  template: Template;
  /** Notify parent when the user "moves into" a different room. */
  onActiveRoomChange?: (roomId: string | null) => void;
  /** Initial room id; if omitted, picks the largest non-circulation. */
  initialRoomId?: string;
}

export interface WalkCubemapHandle {
  goToRoom: (roomId: string) => void;
}

const FACE_ORDER = ["posx", "negx", "posy", "negy", "posz", "negz"] as const;
type FaceLabel = typeof FACE_ORDER[number];

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

function pickInitialRoomId(template: Template): string | undefined {
  let rooms: Room[] = template.rooms || [];
  if (rooms.length === 0 && template.floors?.length) {
    rooms = template.floors.flatMap((f) => f.rooms || []);
  }
  // Avoid corridors / balconies (boring tour stops)
  const skip = new Set(["corridor", "stairs", "balcony", "loggia",
                          "terrace", "wardrobe", "store"]);
  const eligible = rooms.filter((r) => !skip.has((r.type || "").toLowerCase()));
  if (eligible.length === 0) return rooms[0]?.id;
  // Prefer 'living' first
  const living = eligible.find((r) => (r.type || "").toLowerCase() === "living");
  if (living) return living.id;
  // Else largest by area
  return eligible.sort((a, b) => (b.area_sqm ?? 0) - (a.area_sqm ?? 0))[0].id;
}

async function loadImageAsBitmap(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

const WalkCubemap = forwardRef<WalkCubemapHandle, Props>(function WalkCubemap(
  { template, onActiveRoomChange, initialRoomId },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeRoomId, setActiveRoomId] = useState<string | null>(
    initialRoomId ?? pickInitialRoomId(template) ?? null,
  );
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  // Fade overlay opacity 0..1 (1 = fully black, mid-crossfade)
  const [fade, setFade] = useState(0);

  // When parent updates initialRoomId after mount (e.g. prewarm finished
  // and reported the entry room), pick it up.
  useEffect(() => {
    if (initialRoomId && initialRoomId !== activeRoomId) {
      setActiveRoomId(initialRoomId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialRoomId]);

  // Imperative goToRoom for the parent
  const goToRoomRef = useRef<((roomId: string) => void) | null>(null);
  useImperativeHandle(ref, () => ({
    goToRoom: (roomId: string) => goToRoomRef.current?.(roomId),
  }), []);

  // App-level refs we keep across effect runs
  const appRef = useRef<any>(null);
  const skyboxRef = useRef<any>(null);

  useEffect(() => {
    onActiveRoomChange?.(activeRoomId);
  }, [activeRoomId, onActiveRoomChange]);

  /* -------- Mount PlayCanvas application once per template -------- */
  useEffect(() => {
    if (!canvasRef.current) return;
    let stopped = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      const pc = await import("playcanvas");
      if (stopped || !canvasRef.current) return;

      const canvas = canvasRef.current;
      const app = new pc.Application(canvas, {
        mouse: new pc.Mouse(canvas),
        graphicsDeviceOptions: { preferWebGl2: true, alpha: false },
      });
      app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
      app.setCanvasResolution(pc.RESOLUTION_AUTO);
      // Skybox-only rendering — no extra lighting needed since the
      // cubemap IS the lighting for the scene.
      app.scene.ambientLight = new pc.Color(0, 0, 0);
      try { (app.scene as any).gammaCorrection = pc.GAMMA_SRGB; } catch {}
      try { (app.scene as any).toneMapping = pc.TONEMAP_LINEAR; } catch {}
      // Skybox starts transparent / dim until we load a cubemap
      app.scene.skyboxIntensity = 1.0;

      // Camera — natural human FOV (~60°). Wider FOVs distort the
      // cubemap stitching seams; 60° is what real-estate photos use.
      const camera = new pc.Entity("camera");
      camera.addComponent("camera", {
        clearColor: new pc.Color(0.05, 0.06, 0.08),
        farClip: 100,
        nearClip: 0.05,
        fov: 60,
      });
      camera.setPosition(0, 0, 0);
      app.root.addChild(camera);

      // ----- Drag-to-look (Matterport-style) -----
      //
      // The cubemap is 6 independently-generated SDXL faces that don't
      // perfectly seam — looking too far up/down exposes the inconsistent
      // posy/negy faces. So we restrict pitch tightly: ±18° gives a
      // natural "looking around at eye height" range, hides the seams,
      // and matches what you'd actually do touring a property.

      // TARGET angles — set by mouse drag
      let targetYaw = 180;   // start facing -Z (room "north")
      let targetPitch = 0;   // start exactly at horizon
      // ACTUAL angles — exponentially smoothed toward the target
      let yaw = 180;
      let pitch = 0;

      // Tour-friendly tuning — slow + heavy easing so it feels gliding,
      // not game-y. Nothing here exceeds the speed a human would expect
      // when grabbing and dragging a real photo.
      const sensitivityX = 0.08;  // deg per pixel horizontally
      const sensitivityY = 0.05;  // even slower vertically — discourage tilt
      const smoothing = 0.12;     // lower = silkier ease
      const PITCH_MIN = -18;
      const PITCH_MAX = +18;

      // Reset hook — called when room changes so each new room starts
      // looking forward, not whatever angle the previous room ended on.
      const resetView = (newYaw = 180) => {
        targetYaw = newYaw;
        targetPitch = 0;
        yaw = newYaw;
        pitch = 0;
      };
      (canvas as any).__walkResetView = resetView;

      let dragging = false;
      let lastX = 0, lastY = 0;

      const onMouseDown = (e: MouseEvent) => {
        dragging = true;
        lastX = e.clientX; lastY = e.clientY;
        canvas.style.cursor = "grabbing";
      };
      const onMouseUp = () => {
        dragging = false;
        canvas.style.cursor = "grab";
      };
      const onMouseLeave = () => {
        dragging = false;
        canvas.style.cursor = "grab";
      };
      const onMouseMove = (e: MouseEvent) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        targetYaw -= dx * sensitivityX;
        targetPitch = Math.max(PITCH_MIN,
                                  Math.min(PITCH_MAX,
                                            targetPitch - dy * sensitivityY));
      };
      // Touch — slightly more sensitive on small screens
      const onTouchStart = (e: TouchEvent) => {
        if (e.touches.length === 0) return;
        dragging = true;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
      };
      const onTouchMove = (e: TouchEvent) => {
        if (!dragging || e.touches.length === 0) return;
        const t = e.touches[0];
        const dx = t.clientX - lastX;
        const dy = t.clientY - lastY;
        lastX = t.clientX; lastY = t.clientY;
        targetYaw -= dx * sensitivityX * 1.4;
        targetPitch = Math.max(PITCH_MIN,
                                  Math.min(PITCH_MAX,
                                            targetPitch - dy * sensitivityY * 1.4));
      };
      const onTouchEnd = () => { dragging = false; };

      canvas.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mouseup", onMouseUp);
      canvas.addEventListener("mouseleave", onMouseLeave);
      canvas.addEventListener("mousemove", onMouseMove);
      canvas.addEventListener("touchstart", onTouchStart, { passive: true });
      canvas.addEventListener("touchmove", onTouchMove, { passive: true });
      canvas.addEventListener("touchend", onTouchEnd);
      canvas.style.cursor = "grab";

      const onUpdate = () => {
        // Pure ease-toward-target — no inertia, no overshoot. This is
        // what virtual-tour viewers do (Matterport, Zillow 3D). Keeps
        // the camera obviously reactive to your drag without ever
        // continuing on its own.
        yaw += (targetYaw - yaw) * smoothing;
        pitch += (targetPitch - pitch) * smoothing;
        camera.setLocalEulerAngles(pitch, yaw, 0);
      };
      app.on("update", onUpdate);

      app.start();
      appRef.current = app;

      cleanup = () => {
        canvas.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mouseup", onMouseUp);
        canvas.removeEventListener("mouseleave", onMouseLeave);
        canvas.removeEventListener("mousemove", onMouseMove);
        canvas.removeEventListener("touchstart", onTouchStart as any);
        canvas.removeEventListener("touchmove", onTouchMove as any);
        canvas.removeEventListener("touchend", onTouchEnd);
        app.off("update", onUpdate);
        app.destroy();
        appRef.current = null;
        skyboxRef.current = null;
      };
    })().catch((e) => {
      console.error("[WalkCubemap] init failed", e);
      setStatus(`Init failed: ${e}`);
    });

    return () => {
      stopped = true;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [template.id]);

  /* -------- Load + apply cubemap whenever activeRoomId changes -------- */
  useEffect(() => {
    if (!activeRoomId) return;
    let cancelled = false;

    (async () => {
      const pc = await import("playcanvas");
      if (cancelled) return;
      const app = appRef.current;
      if (!app) return;

      setLoading(true);
      setStatus(`Rendering ${activeRoomId}…`);
      setLatencyMs(null);
      setFade(1); // fade to black for the transition
      const t0 = performance.now();

      try {
        const manifest: CubemapManifest =
          await fetchCubemapManifest(template.id, activeRoomId);
        if (cancelled) return;
        setStatus(manifest.cached ? "Loading cached cubemap…" : "Loading freshly rendered cubemap…");

        // Load the 6 face images in parallel — the server already serialised
        // the SDXL renders, so the client can fetch them all at once.
        const imgs = await Promise.all(
          FACE_ORDER.map((f) => loadImageAsBitmap(manifest.faces[f])),
        );
        if (cancelled) return;

        // Build a PlayCanvas cubemap texture
        const tex = new pc.Texture(app.graphicsDevice, {
          cubemap: true,
          width: imgs[0].width,
          height: imgs[0].height,
          format: pc.PIXELFORMAT_R8_G8_B8_A8,
          minFilter: pc.FILTER_LINEAR_MIPMAP_LINEAR,
          magFilter: pc.FILTER_LINEAR,
          addressU: pc.ADDRESS_CLAMP_TO_EDGE,
          addressV: pc.ADDRESS_CLAMP_TO_EDGE,
          mipmaps: true,
        });
        // PlayCanvas expects 6 sources in posx,negx,posy,negy,posz,negz order
        tex.setSource(imgs);

        // Apply as skybox
        app.scene.skybox = tex;
        app.scene.skyboxIntensity = 1.0;
        app.scene.skyboxMip = 0;

        skyboxRef.current = tex;
        setLatencyMs(Math.round(performance.now() - t0));
        setStatus("");

        // Reset the view so each new room starts at horizon — no
        // carry-over from the previous room's tilt/yaw.
        if (canvasRef.current) {
          const reset = (canvasRef.current as any).__walkResetView;
          if (typeof reset === "function") reset(180);
        }

        // brief tick so the skybox is rendered at least once before fade-in
        requestAnimationFrame(() => setFade(0));
      } catch (e: any) {
        console.error("[WalkCubemap] load failed", e);
        setStatus(`Failed to load cubemap: ${e?.message || e}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [activeRoomId, template.id]);

  // goToRoom hook for parent
  goToRoomRef.current = (roomId: string) => setActiveRoomId(roomId);

  /* -------- Keyboard navigation through rooms -------- */
  useEffect(() => {
    const rooms = (template.rooms?.length
      ? template.rooms
      : (template.floors?.[0]?.rooms || [])) as Room[];
    const skip = new Set(["corridor", "stairs"]);
    const order = rooms.filter((r) => !skip.has((r.type || "").toLowerCase()));
    if (order.length < 2) return;

    function onKey(e: KeyboardEvent) {
      const idx = order.findIndex((r) => r.id === activeRoomId);
      if (idx < 0) return;
      if (e.key === "[" || e.key === "ArrowLeft") {
        const prev = order[Math.max(0, idx - 1)];
        if (prev && prev.id !== activeRoomId) setActiveRoomId(prev.id);
      } else if (e.key === "]" || e.key === "ArrowRight") {
        const nxt = order[Math.min(order.length - 1, idx + 1)];
        if (nxt && nxt.id !== activeRoomId) setActiveRoomId(nxt.id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [template, activeRoomId]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%",
                    background: "#080a0e", borderRadius: 10, overflow: "hidden" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block",
                  cursor: "grab" }}
      />

      {/* Fade overlay — covers canvas to black during room transitions */}
      <div
        aria-hidden
        style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          background: "black",
          opacity: fade, transition: "opacity 0.45s ease",
        }}
      />

      {loading && (
        <div style={{ position: "absolute", inset: 0, display: "flex",
                        flexDirection: "column", alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(8,10,14,0.85)", color: "#ddd",
                        pointerEvents: "none" }}>
          <div style={{
            width: 40, height: 40, borderRadius: "50%",
            border: "3px solid #333", borderTopColor: "#FFC83D",
            animation: "spin 1s linear infinite", marginBottom: 14,
          }} />
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
            {status || "Loading…"}
          </div>
          <div style={{ fontSize: 11, color: "#888" }}>
            First view of a room: ~12s · then cached forever
          </div>
        </div>
      )}

      {!loading && status && (
        <div style={{ position: "absolute", inset: 0, display: "flex",
                        alignItems: "center", justifyContent: "center",
                        color: "#f88", padding: 20, textAlign: "center",
                        background: "rgba(8,10,14,0.85)" }}>
          {status}
        </div>
      )}

      {!loading && !status && (
        <div style={{ position: "absolute", left: 12, bottom: 12,
                        background: "rgba(0,0,0,0.65)", color: "white",
                        padding: "8px 12px", borderRadius: 6,
                        fontSize: 12,
                        pointerEvents: "none" }}>
          <strong>Drag</strong> to look around &nbsp;·&nbsp; <strong>click rooms on the floor plan</strong> to walk there
        </div>
      )}

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
});

export default WalkCubemap;
