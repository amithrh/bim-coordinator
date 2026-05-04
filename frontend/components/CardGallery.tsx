"use client";

import type { Card } from "@/lib/types";
import TemplateCard from "./TemplateCard";

interface Props {
  cards: Card[];
  onSelect: (card: Card) => void;
}

export default function CardGallery({ cards, onSelect }: Props) {
  if (cards.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No matching templates. Try a different brief.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => (
        <TemplateCard key={c.template.id} card={c} onClick={() => onSelect(c)} />
      ))}
    </div>
  );
}
