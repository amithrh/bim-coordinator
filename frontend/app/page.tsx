"use client";

import { useState } from "react";
import type { Brief, Card } from "@/lib/types";
import { postBrief, postRetrieve } from "@/lib/api";
import BriefInput from "@/components/BriefInput";
import BriefSummary from "@/components/BriefSummary";
import CardGallery from "@/components/CardGallery";
import DetailView from "@/components/DetailView";
import LanguageToggle from "@/components/LanguageToggle";

export default function Home() {
  const [brief, setBrief] = useState<Brief | null>(null);
  const [cards, setCards] = useState<Card[]>([]);
  const [selected, setSelected] = useState<Card | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(text: string) {
    setBusy(true);
    setError(null);
    setSelected(null);
    setCards([]);
    try {
      const b = await postBrief(text);
      setBrief(b);
      const { cards: c } = await postRetrieve(b);
      setCards(c);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-8 sm:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-end mb-4">
          <LanguageToggle />
        </div>
        {!selected && (
          <BriefInput onSubmit={handleSubmit} busy={busy} />
        )}

        {brief && !selected && (
          <div className="mt-8">
            <BriefSummary brief={brief} />
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 mb-6">
                {error}
              </div>
            )}
            <CardGallery cards={cards} onSelect={setSelected} />
          </div>
        )}

        {selected && (
          <div className="mt-2">
            <DetailView template={selected.template} onBack={() => setSelected(null)} />
          </div>
        )}
      </div>
    </main>
  );
}
