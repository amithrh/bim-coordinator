"use client";

import { useEffect, useState } from "react";
import type { Template, Vec2 } from "@/lib/types";
import { postModify, postMoveOpening } from "@/lib/api";
import Viewer3D from "./Viewer3D";
import Plan2D from "./Plan2D";
import AdjustmentPanel from "./AdjustmentPanel";
import ActionPanel from "./ActionPanel";
import { useLang } from "./LanguageContext";
import { translateText } from "@/lib/i18n";

function bedroomLabel(n: number): string {
  if (n === 0) return "Studio";
  if (n === 1) return "1 BR";
  return `${n} BR`;
}

interface Props {
  template: Template;
  onBack: () => void;
}

export default function DetailView({ template: baseTemplate, onBack }: Props) {
  const { lang } = useLang();
  const [current, setCurrent] = useState<Template>(baseTemplate);
  const [modifiedId, setModifiedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeFloor, setActiveFloor] = useState(0);

  useEffect(() => {
    setCurrent(baseTemplate);
    setModifiedId(null);
    setError(null);
    setActiveFloor(0);
  }, [baseTemplate]);

  async function applyMods(mods: { area_scale?: number; ceiling_height_mm?: number; rotation_deg?: number }) {
    setBusy(true);
    setError(null);
    try {
      // Always apply against the original template — sliders are absolute, not relative
      const res = await postModify(baseTemplate.id, mods);
      if (!res.ok) {
        setError(res.message || res.errors?.[0] || "Modification failed");
      } else if (res.template && res.modified_id) {
        setCurrent(res.template);
        setModifiedId(res.modified_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function moveOpening(kind: "door" | "window", index: number, newPos: Vec2) {
    setBusy(true);
    setError(null);
    try {
      // Move against the CURRENT template (chains with prior mods)
      const res = await postMoveOpening(current.id, kind, index, newPos);
      if (!res.ok) {
        const msg = res.message || res.errors?.[0] || "Move failed";
        setError(msg);
        throw new Error(msg);
      }
      if (res.template && res.modified_id) {
        setCurrent(res.template);
        setModifiedId(res.modified_id);
      }
    } finally {
      setBusy(false);
    }
  }

  const isModified = modifiedId !== null;
  const exportId = isModified ? modifiedId! : baseTemplate.id;

  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="text-sm text-blue-600 hover:underline flex items-center gap-1"
      >
        ← Back to results
      </button>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-xl font-bold">
            {bedroomLabel(baseTemplate.metadata.bedrooms)} — {baseTemplate.metadata.city_inspiration}
            {baseTemplate.floors && baseTemplate.floors.length > 1 && (
              <span className="ml-2 text-xs font-medium text-blue-700 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded align-middle">
                {baseTemplate.floors.length}-storey
              </span>
            )}
          </h2>
          {isModified && (
            <span className="text-xs px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
              modified
            </span>
          )}
        </div>
        <p className="text-sm text-gray-600">{translateText(baseTemplate.metadata.description, lang)}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: 2D — interactive */}
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">2D Floor Plan</div>
          <div className="aspect-square bg-gray-50 rounded overflow-hidden">
            <Plan2D template={current} onMove={moveOpening} busy={busy}
                     activeFloor={activeFloor} onActiveFloorChange={setActiveFloor} />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            {current.metadata.total_area_sqm} m² · {current.boundary.ceiling_height_mm} mm ceiling
          </div>
        </div>

        {/* Middle: 3D */}
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">3D Preview</div>
          <div className="aspect-square">
            <Viewer3D template={current} activeFloor={current.floors ? activeFloor : undefined} />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Drag to rotate · scroll to zoom
          </div>
        </div>

        {/* Right: actions + adjustments */}
        <div className="space-y-4">
          <AdjustmentPanel
            baseAreaSqm={baseTemplate.metadata.total_area_sqm}
            baseCeilingMm={baseTemplate.boundary.ceiling_height_mm}
            onApply={applyMods}
            busy={busy}
          />
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
              {error}
            </div>
          )}
          <ActionPanel templateId={exportId} isModified={isModified} />
        </div>
      </div>
    </div>
  );
}
