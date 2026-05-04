"use client";

import type { Brief } from "@/lib/types";

interface Props {
  brief: Brief;
}

export default function BriefSummary({ brief }: Props) {
  const chips: Array<[string, string | number | null]> = [
    ["region", brief.region],
    ["city", brief.city],
    ["bedrooms", brief.bedrooms],
    ["area", brief.total_area_sqm ? `${brief.total_area_sqm} m²` : null],
    ["vastu", brief.vastu_compliant ? "yes" : null],
    ["size", brief.size_label],
  ].filter(([, v]) => v !== null && v !== undefined);

  return (
    <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 mb-6">
      <div className="text-xs text-gray-500 mb-2">Brief understood as:</div>
      <div className="flex flex-wrap gap-2">
        {chips.map(([k, v]) => (
          <span
            key={k}
            className="text-xs px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-100"
          >
            <span className="text-blue-400">{k}:</span> <span className="font-medium">{v}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
