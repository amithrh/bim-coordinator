"use client";

import { useLang } from "./LanguageContext";

export default function LanguageToggle() {
  const { lang, setLang } = useLang();
  return (
    <div className="inline-flex rounded-md border border-gray-300 bg-white text-xs overflow-hidden">
      <button
        onClick={() => setLang("en")}
        className={`px-3 py-1.5 font-medium transition-colors ${
          lang === "en" ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-50"
        }`}
        title="Show all text in English"
      >
        EN
      </button>
      <button
        onClick={() => setLang("de")}
        className={`px-3 py-1.5 font-medium transition-colors border-l border-gray-300 ${
          lang === "de" ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-50"
        }`}
        title="German labels (architect view)"
      >
        DE
      </button>
    </div>
  );
}
