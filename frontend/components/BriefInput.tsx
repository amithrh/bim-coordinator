"use client";

import { useState } from "react";

const EXAMPLES = [
  "2BHK in Bangalore for my parents, Vastu, around 90 sqm",
  "Berlin Altbau Zweizimmer with high ceilings, 65 sqm",
  "5-zimmer family house in Stuttgart",
  "Modern Hamburg apartment with harbour balcony",
  "Cologne Neubau 2-zimmer with balcony",
  "Munich studio for a student, 35 sqm",
  "Dusseldorf Dachgeschoss with roof terrace",
  "Premium 3BHK in Delhi with pooja room",
  "1BHK compact in Mumbai with balcony",
  "Frankfurt 4-zimmer family flat",
];

interface Props {
  onSubmit: (text: string) => void;
  busy?: boolean;
}

export default function BriefInput({ onSubmit, busy }: Props) {
  const [text, setText] = useState("");

  function submit(t: string) {
    if (!t.trim() || busy) return;
    setText(t);
    onSubmit(t);
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">BIM Coordinator</h1>
      <p className="text-sm text-gray-500 mb-6">
        Describe the home you want. The system will surface 4 matching floor plans.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g., 2BHK in Bangalore for my parents, Vastu, around 90 sqm"
          className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:border-blue-500 bg-white"
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !text.trim()}
          className="px-6 py-3 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:bg-gray-300"
        >
          {busy ? "Searching…" : "Find plans"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2 mt-4">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            onClick={() => submit(e)}
            disabled={busy}
            className="text-xs px-3 py-1.5 rounded-full bg-white border border-gray-300 hover:border-blue-500 hover:bg-blue-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {e}
          </button>
        ))}
      </div>
    </div>
  );
}
