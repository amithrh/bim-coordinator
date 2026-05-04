export type Vec2 = [number, number];

export interface Room {
  id: string;
  name: string;
  type: string;
  polygon: Vec2[];
  area_sqm?: number;
  notes?: string;
}

export interface Door {
  from: string;
  to: string;
  position: Vec2;
  width_mm: number;
  height_mm?: number;
  swing?: "in" | "out" | "slide";
  is_main_entry?: boolean;
}

export interface WindowDef {
  room: string;
  position: Vec2;
  width_mm: number;
  height_mm?: number;
  sill_mm?: number;
}

export interface TemplateMetadata {
  region: "india" | "europe";
  country: string;
  city_inspiration?: string;
  size_label: string;
  total_area_sqm: number;
  bedrooms: number;
  bathrooms: number;
  style: string;
  description: string;
  suitable_for: string[];
  tags: string[];
  vastu_compliant?: boolean;
  kitchen_facing?: string;
}

export interface Floor {
  name: string;
  elevation_mm?: number;
  ceiling_height_mm?: number;
  boundary_polygon?: Vec2[];
  rooms: Room[];
  doors: Door[];
  windows: WindowDef[];
}

export interface Template {
  id: string;
  metadata: TemplateMetadata;
  boundary: {
    polygon: Vec2[];
    wall_thickness_mm: number;
    ceiling_height_mm: number;
    orientation_north_deg?: number;
  };
  rooms: Room[];
  doors: Door[];
  windows: WindowDef[];
  floors?: Floor[];
}

export interface Brief {
  raw: string;
  region: "india" | "europe" | null;
  country: string | null;
  city: string | null;
  size_label: string | null;
  bedrooms: number | null;
  total_area_sqm: number | null;
  vastu_compliant: boolean | null;
}

export interface Card {
  template: Template;
  score: number;
  raw_cosine: number;
  reasoning: string[];
}
