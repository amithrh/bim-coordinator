"use client";

import { useState } from "react";

interface Props {
  baseAreaSqm: number;
  baseCeilingMm: number;
  onApply: (mods: { area_scale?: number; ceiling_height_mm?: number; rotation_deg?: number }) => void;
  busy?: boolean;
}

export default function AdjustmentPanel({ baseAreaSqm, baseCeilingMm, onApply, busy }: Props) {
  const [areaScale, setAreaScale] = useState(1.0);
  const [ceilingMm, setCeilingMm] = useState(baseCeilingMm);
  const [rotationDeg, setRotationDeg] = useState(0);

  const targetArea = (baseAreaSqm * areaScale).toFixed(1);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <h3 className="font-semibold text-sm">Adjust this plan</h3>

      <div>
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>Total area</span>
          <span className="font-mono">{targetArea} m² ({Math.round(areaScale * 100)}%)</span>
        </div>
        <input
          type="range"
          min={0.7}
          max={1.4}
          step={0.05}
          value={areaScale}
          onChange={(e) => setAreaScale(parseFloat(e.target.value))}
          className="w-full"
          disabled={busy}
        />
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>Ceiling height</span>
          <span className="font-mono">{ceilingMm} mm</span>
        </div>
        <input
          type="range"
          min={2400}
          max={3500}
          step={100}
          value={ceilingMm}
          onChange={(e) => setCeilingMm(parseInt(e.target.value, 10))}
          className="w-full"
          disabled={busy}
        />
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>Rotation</span>
          <span className="font-mono">{rotationDeg}°</span>
        </div>
        <div className="flex gap-2">
          {[0, 90, 180, 270].map((d) => (
            <button
              key={d}
              onClick={() => setRotationDeg(d)}
              disabled={busy}
              className={`flex-1 text-xs py-1.5 rounded border ${
                rotationDeg === d
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white border-gray-300 hover:border-blue-500"
              }`}
            >
              {d}°
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={() => onApply({
          area_scale: areaScale === 1 ? undefined : areaScale,
          ceiling_height_mm: ceilingMm === baseCeilingMm ? undefined : ceilingMm,
          rotation_deg: rotationDeg === 0 ? undefined : rotationDeg,
        })}
        disabled={busy || (areaScale === 1 && ceilingMm === baseCeilingMm && rotationDeg === 0)}
        className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:bg-gray-300"
      >
        {busy ? "Applying…" : "Apply changes"}
      </button>
    </div>
  );
}
