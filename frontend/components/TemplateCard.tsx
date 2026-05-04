"use client";

import type { Card } from "@/lib/types";
import { svgUrl } from "@/lib/api";
import { useLang } from "./LanguageContext";
import { translateText } from "@/lib/i18n";

function bedroomLabel(n: number): string {
  if (n === 0) return "Studio";
  if (n === 1) return "1 BR";
  return `${n} BR`;
}

interface Props {
  card: Card;
  onClick: () => void;
}

export default function TemplateCard({ card, onClick }: Props) {
  const { lang } = useLang();
  const t = card.template;
  return (
    <button
      onClick={onClick}
      className="text-left bg-white border border-gray-200 rounded-lg overflow-hidden hover:border-blue-500 hover:shadow-md transition-all"
    >
      <div className="aspect-square bg-gray-50 border-b border-gray-100">
        <img
          src={svgUrl(t.id)}
          alt={t.id}
          className="w-full h-full object-contain"
        />
      </div>
      <div className="p-3">
        <h3 className="font-semibold text-sm mb-1">
          {bedroomLabel(t.metadata.bedrooms)} — {t.metadata.city_inspiration}
        </h3>
        <div className="text-xs text-gray-500 mb-2">
          {t.metadata.total_area_sqm} m² · {t.metadata.bathrooms} bath · {translateText(t.metadata.style, lang)}
        </div>
        <ul className="text-xs text-gray-600 space-y-0.5">
          {card.reasoning.slice(0, 2).map((r, i) => (
            <li key={i}>• {r}</li>
          ))}
        </ul>
      </div>
    </button>
  );
}
