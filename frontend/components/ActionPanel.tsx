"use client";

import { ifcUrl } from "@/lib/api";

interface Props {
  templateId: string;
  isModified: boolean;
}

export default function ActionPanel({ templateId, isModified }: Props) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-2">
      <h3 className="font-semibold text-sm mb-2">Export</h3>
      <a
        href={ifcUrl(templateId, isModified)}
        download
        className="block text-center py-2.5 rounded-lg bg-gray-900 text-white font-medium hover:bg-gray-800"
      >
        ⬇ Download IFC
      </a>
      <button
        disabled
        className="w-full py-2.5 rounded-lg bg-gray-100 text-gray-400 font-medium cursor-not-allowed"
        title="Hackathon Day 3 wires this to a real Bimplus tenant"
      >
        Open in Allplan (stub)
      </button>
      <button
        disabled
        className="w-full py-2.5 rounded-lg bg-gray-100 text-gray-400 font-medium cursor-not-allowed"
        title="Hackathon Day 2 wires this to ifcclash + ifctester"
      >
        Run BIM Coordinator validation (stub)
      </button>
    </div>
  );
}
