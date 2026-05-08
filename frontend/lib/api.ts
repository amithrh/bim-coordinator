import type { Brief, Card, Template } from "./types";

const BASE = "/api";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function postBrief(text: string): Promise<Brief> {
  return jsonFetch<Brief>(`${BASE}/brief`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export async function postRetrieve(brief: Brief, topN = 4): Promise<{ cards: Card[] }> {
  return jsonFetch(`${BASE}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ brief, top_n: topN }),
  });
}

export interface ModifyResult {
  ok: boolean;
  template?: Template;
  modified_id?: string;
  errors?: string[];
  message?: string;
}

export async function postModify(
  template_id: string,
  mods: Record<string, number>,
): Promise<ModifyResult> {
  const res = await fetch(`${BASE}/modify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template_id, mods }),
  });
  return res.json();
}

export async function postMoveOpening(
  template_id: string,
  kind: "door" | "window",
  index: number,
  new_position: [number, number],
): Promise<ModifyResult> {
  const res = await fetch(`${BASE}/move_opening`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template_id, kind, index, new_position }),
  });
  return res.json();
}

export function svgUrl(template_id: string, isModified = false): string {
  return isModified ? `${BASE}/modified/${template_id}/svg` : `${BASE}/templates/${template_id}/svg`;
}

export function ifcUrl(template_id: string, isModified = false): string {
  return isModified ? `${BASE}/modified/${template_id}/ifc` : `${BASE}/templates/${template_id}/ifc`;
}

/**
 * URL for the per-room photoreal interior render. Pass through
 * focus_room_id to walk the customer through the apartment.
 */
export function walkthroughRoomUrl(
  template_id: string,
  room_id: string,
  width = 768,
  height = 512,
): string {
  const qs = new URLSearchParams({
    room_id,
    view: "interior",
    width: String(width),
    height: String(height),
  });
  return `${BASE}/render/walkthrough/${template_id}?${qs.toString()}`;
}

/** Whole-flat dollhouse cutaway. */
export function dollhouseUrl(
  template_id: string,
  width = 768,
  height = 512,
): string {
  const qs = new URLSearchParams({
    view: "dollhouse",
    width: String(width),
    height: String(height),
  });
  return `${BASE}/render/walkthrough/${template_id}?${qs.toString()}`;
}

/** Pre-warm both pipelines (call once on app load). */
export async function warmupRender(includeControlNet = true): Promise<unknown> {
  const qs = includeControlNet ? "?controlnet=true" : "";
  const res = await fetch(`${BASE}/render/warmup${qs}`);
  return res.json();
}

/** glTF/GLB binary URL for the PlayCanvas 3D walkthrough. */
export function gltfUrl(template_id: string): string {
  return `${BASE}/templates/${template_id}/gltf`;
}
