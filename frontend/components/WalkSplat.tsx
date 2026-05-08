"use client";

/**
 * Truly walkable photoreal 3D scene rendered as a Gaussian Splat
 * trained from synthetic SDXL+ControlNet views of the BIM apartment.
 *
 * Loaded via PlayCanvas's GSplatComponent (engine 2.x). Free 6DoF
 * navigation: WASD walks, mouse drag looks, Q/E for up/down.
 *
 * The .ply is served by /api/templates/{id}/splat and is ~25 MB for a
 * 5-room apartment (~100k Gaussians). Browser-cached after first load.
 */

import {
  forwardRef, useEffect, useImperativeHandle,
  useRef, useState,
} from "react";
import type { Template } from "@/lib/types";
import { splatUrl, getSplatStatus } from "@/lib/api";

interface Props {
  template: Template;
  onClose?: () => void;
}

export interface WalkSplatHandle {
  resetCamera: () => void;
}

const WalkSplat = forwardRef<WalkSplatHandle, Props>(function WalkSplat(
  { template, onClose },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [phase, setPhase] = useState<"checking" | "absent" | "loading" | "ready" | "error">("checking");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const resetCamRef = useRef<(() => void) | null>(null);

  useImperativeHandle(ref, () => ({
    resetCamera: () => resetCamRef.current?.(),
  }), []);

  useEffect(() => {
    if (!canvasRef.current) return;
    let stopped = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      // Check availability first so the UI can show a clear "not trained" state
      const status = await getSplatStatus(template.id).catch(() => ({ available: false } as const));
      if (stopped) return;
      if (!status.available) {
        setPhase("absent");
        return;
      }

      const pc = await import("playcanvas");
      if (stopped || !canvasRef.current) return;

      const canvas = canvasRef.current;
      setPhase("loading");

      // Pick a graphics device based on what's available — WebGL2 is
      // fine for splats, no need to force WebGPU.
      const app = new pc.Application(canvas, {
        mouse: new pc.Mouse(canvas),
        keyboard: new pc.Keyboard(window),
        graphicsDeviceOptions: { preferWebGl2: true, alpha: false },
      });
      app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
      app.setCanvasResolution(pc.RESOLUTION_AUTO);
      app.scene.ambientLight = new pc.Color(0.5, 0.5, 0.5);

      // Camera
      const camera = new pc.Entity("camera");
      camera.addComponent("camera", {
        clearColor: new pc.Color(0.06, 0.07, 0.1),
        farClip: 100,
        nearClip: 0.05,
        fov: 70,
      });
      camera.setPosition(0, 1.65, 4);
      camera.lookAt(0, 1.5, 0);
      app.root.addChild(camera);

      const initialPos = camera.getPosition().clone();
      const initialTarget = new pc.Vec3(0, 1.5, 0);

      resetCamRef.current = () => {
        camera.setPosition(initialPos);
        camera.lookAt(initialTarget);
        const e = camera.getEulerAngles();
        yaw = e.y; pitch = e.x;
      };

      // ----- Drag-to-look + WASD walk -----
      let yaw = camera.getEulerAngles().y;
      let pitch = camera.getEulerAngles().x;
      let dragging = false;
      let lastX = 0, lastY = 0;

      const onDown = (e: MouseEvent) => {
        dragging = true;
        lastX = e.clientX; lastY = e.clientY;
        canvas.style.cursor = "grabbing";
      };
      const onUp = () => { dragging = false; canvas.style.cursor = "grab"; };
      const onMove = (e: MouseEvent) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        yaw -= dx * 0.18;
        pitch = Math.max(-80, Math.min(80, pitch - dy * 0.18));
      };
      canvas.addEventListener("mousedown", onDown);
      window.addEventListener("mouseup", onUp);
      canvas.addEventListener("mousemove", onMove);
      canvas.style.cursor = "grab";

      const keys = { fwd: false, back: false, left: false, right: false,
                       up: false, down: false, sprint: false };
      const onKD = (e: any) => {
        if (e.key === pc.KEY_W || e.key === pc.KEY_UP)    keys.fwd = true;
        if (e.key === pc.KEY_S || e.key === pc.KEY_DOWN)  keys.back = true;
        if (e.key === pc.KEY_A || e.key === pc.KEY_LEFT)  keys.left = true;
        if (e.key === pc.KEY_D || e.key === pc.KEY_RIGHT) keys.right = true;
        if (e.key === pc.KEY_Q || e.key === pc.KEY_SPACE) keys.up = true;
        if (e.key === pc.KEY_E)                            keys.down = true;
        if (e.key === pc.KEY_SHIFT)                        keys.sprint = true;
      };
      const onKU = (e: any) => {
        if (e.key === pc.KEY_W || e.key === pc.KEY_UP)    keys.fwd = false;
        if (e.key === pc.KEY_S || e.key === pc.KEY_DOWN)  keys.back = false;
        if (e.key === pc.KEY_A || e.key === pc.KEY_LEFT)  keys.left = false;
        if (e.key === pc.KEY_D || e.key === pc.KEY_RIGHT) keys.right = false;
        if (e.key === pc.KEY_Q || e.key === pc.KEY_SPACE) keys.up = false;
        if (e.key === pc.KEY_E)                            keys.down = false;
        if (e.key === pc.KEY_SHIFT)                        keys.sprint = false;
      };
      app.keyboard.on(pc.EVENT_KEYDOWN, onKD);
      app.keyboard.on(pc.EVENT_KEYUP, onKU);

      const onUpdate = (dt: number) => {
        camera.setLocalEulerAngles(pitch, yaw, 0);
        const fwd = camera.forward;
        const right = camera.right;
        const speed = (keys.sprint ? 4.5 : 2.0) * dt;
        const dx = (keys.fwd ? fwd.x : 0) - (keys.back ? fwd.x : 0)
                 + (keys.right ? right.x : 0) - (keys.left ? right.x : 0);
        const dy = (keys.up ? 1 : 0) - (keys.down ? 1 : 0);
        const dz = (keys.fwd ? fwd.z : 0) - (keys.back ? fwd.z : 0)
                 + (keys.right ? right.z : 0) - (keys.left ? right.z : 0);
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1e-3) {
          const p = camera.getPosition();
          camera.setPosition(p.x + dx * speed, p.y + dy * speed, p.z + dz * speed);
        }
      };
      app.on("update", onUpdate);
      app.start();

      // ----- Load the trained splat .ply -----
      const url = splatUrl(template.id);
      app.assets.loadFromUrl(url, "gsplat", (err: any, asset: any) => {
        if (stopped) return;
        if (err) {
          console.error("[WalkSplat] PLY load failed", err);
          setErrorMsg(`Failed to load splat: ${err}`);
          setPhase("error");
          return;
        }
        const splat = new pc.Entity("splat");
        splat.addComponent("gsplat", { asset });
        app.root.addChild(splat);
        // Try to extract the splat count for the badge
        try {
          const count = (asset as any)?.resource?.numSplats
                          ?? (asset as any)?.resource?.splatBuffer?.numSplats
                          ?? null;
          if (typeof count === "number") setSplatCount(count);
        } catch {}
        setPhase("ready");
      });

      cleanup = () => {
        canvas.removeEventListener("mousedown", onDown);
        window.removeEventListener("mouseup", onUp);
        canvas.removeEventListener("mousemove", onMove);
        app.off("update", onUpdate);
        app.destroy();
      };
    })().catch((e) => {
      console.error("[WalkSplat] init failed", e);
      setErrorMsg(`Init failed: ${e}`);
      setPhase("error");
    });

    return () => {
      stopped = true;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [template.id]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%",
                    background: "#080a0e", borderRadius: 10, overflow: "hidden" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block",
                  cursor: phase === "ready" ? "grab" : "default" }}
      />

      {phase === "checking" && (
        <div style={overlayStyle()}>
          <Spinner /> Checking for trained splat…
        </div>
      )}

      {phase === "loading" && (
        <div style={overlayStyle()}>
          <Spinner />
          <div style={{ fontSize: 14, fontWeight: 600, marginTop: 14 }}>
            Loading 3D Gaussian Splat…
          </div>
          <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
            ~25 MB · cached after first load
          </div>
        </div>
      )}

      {phase === "absent" && (
        <div style={overlayStyle()}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>
            Splat not trained for this template yet
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 8,
                          maxWidth: 460, textAlign: "center", lineHeight: 1.5 }}>
            Path B (3D Gaussian Splat) is pre-baked offline:
            <br />
            <code>python scripts/render_splat_dataset.py {template.id} && python scripts/train_splat.py {template.id}</code>
          </div>
        </div>
      )}

      {phase === "error" && (
        <div style={{ ...overlayStyle(), color: "#f88" }}>
          {errorMsg || "Unknown error"}
        </div>
      )}

      {phase === "ready" && (
        <>
          <div style={{ position: "absolute", left: 12, bottom: 12,
                          background: "rgba(0,0,0,0.7)", color: "white",
                          padding: "8px 12px", borderRadius: 6,
                          fontSize: 12, pointerEvents: "none" }}>
            <strong>Drag</strong> to look · <strong>WASD</strong> to walk · <strong>Q/E</strong> up/down · <strong>Shift</strong> sprint
            {splatCount && (
              <span style={{ marginLeft: 12, color: "#FFC83D" }}>
                · {splatCount.toLocaleString()} splats
              </span>
            )}
          </div>
        </>
      )}

      {onClose && (
        <button
          onClick={onClose}
          aria-label="Close"
          style={{ position: "absolute", top: 12, right: 12,
                    width: 32, height: 32, borderRadius: "50%",
                    background: "rgba(0,0,0,0.7)", color: "white",
                    border: "1px solid #444", fontSize: 18, cursor: "pointer" }}
        >×</button>
      )}

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
});

export default WalkSplat;

/* helpers */
function overlayStyle(): React.CSSProperties {
  return {
    position: "absolute", inset: 0, display: "flex",
    flexDirection: "column", alignItems: "center", justifyContent: "center",
    background: "rgba(8,10,14,0.9)", color: "#ddd", padding: 24,
  };
}

function Spinner() {
  return (
    <div style={{
      width: 36, height: 36, borderRadius: "50%",
      border: "3px solid #333", borderTopColor: "#FFC83D",
      animation: "spin 1s linear infinite",
    }} />
  );
}
