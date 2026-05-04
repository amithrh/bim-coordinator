"use client";

import { createContext, useContext, useState, ReactNode } from "react";
import type { Lang } from "@/lib/i18n";

interface LangCtxValue {
  lang: Lang;
  setLang: (l: Lang) => void;
}

const LangCtx = createContext<LangCtxValue>({
  lang: "en",
  setLang: () => {},
});

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");
  return <LangCtx.Provider value={{ lang, setLang }}>{children}</LangCtx.Provider>;
}

export function useLang() {
  return useContext(LangCtx);
}
