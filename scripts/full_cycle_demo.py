"""End-to-end full-cycle demo for the BIM Coordinator.

Runs the complete pipeline for a single brief and writes a composed
poster image showing every stage:

    natural-language brief
      |
      v   POST /api/reason       (MiniLM retrieval + Llama reasoning)
    ranked candidate templates
      |
      v   pick top match
    chosen template (metadata + IFC validation)
      |
      v   render_template_svg
    floor plan SVG  (architectural, walls + fixtures + door swings)
      |
      +---> GET /api/render/depth/{id}?view=interior   -> depth map
      |     POST /api/render mode=faithful view=interior -> photoreal interior
      |
      +---> GET /api/render/depth/{id}?view=dollhouse   -> depth map
            POST /api/render mode=faithful view=dollhouse -> photoreal dollhouse

Output: /tmp/full_cycle_<id>.png   (large composed poster)
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import cairosvg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.floorplan_renderer import render_template_svg

API = "http://127.0.0.1:8009"

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

LABEL_BG = (28, 28, 28)
HEADER_BG = (16, 18, 22)
PAPER = (22, 22, 22)
TXT_HI = (245, 245, 245)
TXT_MD = (180, 180, 180)
TXT_LO = (140, 140, 140)
ACCENT = (0, 200, 160)


def _font(size, bold=False, family="Helvetica"):
    paths = [
        f"/System/Library/Fonts/{family}.ttc",
        f"/System/Library/Fonts/Supplemental/{family}.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=1 if bold else 0)
        except Exception:
            continue
    return ImageFont.load_default()


def label(text, w, h, fontsize=14, bg=LABEL_BG, fg=TXT_HI,
            align="center", bold=False, pad_x=10):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    f = _font(fontsize, bold=bold)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if align == "center":
        x = (w - tw) // 2
    elif align == "right":
        x = w - tw - pad_x
    else:
        x = pad_x
    y = (h - th) // 2 - 2
    d.text((x, y), text, fill=fg, font=f)
    return img


def fit_height(im, target_h, pad_to_w=None):
    new_w = int(im.width * target_h / im.height)
    out = im.resize((new_w, target_h), Image.Resampling.LANCZOS)
    if pad_to_w and pad_to_w > out.width:
        canvas = Image.new("RGB", (pad_to_w, target_h), PAPER)
        canvas.paste(out, ((pad_to_w - out.width) // 2, 0))
        return canvas
    return out


def card(title, subtitle, body_img, w, body_h, accent=ACCENT):
    """A titled card containing one image + 2 lines of caption."""
    HEAD_H = 56
    BODY_H = body_h
    CAP_H  = 22
    h = HEAD_H + BODY_H + CAP_H
    canvas = Image.new("RGB", (w, h), PAPER)
    # Header strip
    canvas.paste(label(title, w, HEAD_H, fontsize=18, bg=HEADER_BG,
                        fg=TXT_HI, align="left", bold=True, pad_x=18), (0, 0))
    # Accent stripe under title
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, HEAD_H - 3, w, HEAD_H), fill=accent)
    # Body image
    body = fit_height(body_img, BODY_H, pad_to_w=w)
    canvas.paste(body, (0, HEAD_H))
    # Caption strip
    canvas.paste(label(subtitle, w, CAP_H, fontsize=11, bg=PAPER,
                        fg=TXT_MD, align="center"), (0, HEAD_H + BODY_H))
    return canvas


def big_text_card(title, lines, w, h, accent=ACCENT, body_font=14, line_pad=6,
                    body_align="left"):
    """A card containing only text — used for the brief, the LLM reasoning,
    and the IFC validation summary."""
    HEAD_H = 56
    canvas = Image.new("RGB", (w, h), PAPER)
    canvas.paste(label(title, w, HEAD_H, fontsize=18, bg=HEADER_BG,
                        fg=TXT_HI, align="left", bold=True, pad_x=18), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, HEAD_H - 3, w, HEAD_H), fill=accent)
    f = _font(body_font)
    f_bold = _font(body_font, bold=True)
    y = HEAD_H + 12
    for line in lines:
        if isinstance(line, tuple):
            txt, color, bold = line
            font = f_bold if bold else f
        else:
            txt, color, font = line, TXT_HI, f
        # word-wrap
        words = txt.split()
        line_w = []
        cur = ""
        for w_ in words:
            test = (cur + " " + w_).strip()
            if d.textlength(test, font=font) <= w - 36:
                cur = test
            else:
                line_w.append(cur)
                cur = w_
        if cur:
            line_w.append(cur)
        for ln in line_w:
            d.text((18, y), ln, fill=color, font=font)
            y += font.size + line_pad
        y += 2
    return canvas


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_reason(brief: str) -> dict:
    print(f"\n[1/6] /api/reason  '{brief}'")
    t0 = time.time()
    r = requests.post(f"{API}/api/reason", json={"brief_text": brief})
    r.raise_for_status()
    out = r.json()
    print(f"      backend={out.get('llm_backend')} "
          f"model={out.get('llm_model')} "
          f"latency={out.get('llm_latency_s', 0):.2f}s "
          f"http={time.time()-t0:.2f}s "
          f"({len(out.get('candidates', []))} candidates)")
    return out


def stage_template_meta(tid: str) -> dict:
    print(f"\n[2/6] /api/templates/{tid}/json")
    t0 = time.time()
    r = requests.get(f"{API}/api/templates/{tid}/json")
    r.raise_for_status()
    out = r.json()
    print(f"      {out.get('id')} · {out['metadata'].get('country','?')} · "
          f"{out['metadata'].get('total_area_sqm', '?')} m² "
          f"({time.time()-t0:.2f}s)")
    return out


def stage_validate_ifc(tid: str, template_path: Path) -> dict:
    print(f"\n[3/6] /api/templates/{tid}/ifc + schema validation")
    t0 = time.time()
    r = requests.get(f"{API}/api/templates/{tid}/ifc")
    if r.status_code != 200:
        return {"ok": False, "error": f"http {r.status_code}", "size": 0,
                "passed": 0, "total": 0}
    ifc_bytes = r.content
    fetch_s = time.time() - t0

    # Save IFC to a temp file and run verify_ifc
    ifc_path = Path(f"/tmp/full_cycle_{tid}.ifc")
    ifc_path.write_bytes(ifc_bytes)

    from scripts.verify_ifc import verify
    t1 = time.time()
    try:
        passed, total, _lines = verify(template_path, ifc_path)
    except Exception as e:
        print(f"      verify failed: {e!r}")
        return {"ok": False, "passed": 0, "total": 0,
                "ifc_size_kb": len(ifc_bytes)/1024,
                "error": repr(e)}
    val_s = time.time() - t1
    print(f"      fetched {len(ifc_bytes)/1024:.1f} KB in {fetch_s:.2f}s; "
          f"validated {passed}/{total} checks in {val_s:.2f}s")
    return {"ok": passed == total, "passed": passed, "total": total,
            "ifc_size_kb": len(ifc_bytes)/1024}


def stage_floor_plan(template: dict, out_path: str) -> tuple[Image.Image, float]:
    print(f"\n[4/6] render_template_svg → architectural floor plan")
    t0 = time.time()
    svg_path = out_path.replace(".png", ".svg")
    render_template_svg(template, svg_path, size=1024)
    cairosvg.svg2png(url=svg_path, write_to=out_path,
                      output_width=1400, background_color="white")
    elapsed = time.time() - t0
    print(f"      {out_path} in {elapsed:.2f}s")
    return Image.open(out_path).convert("RGB"), elapsed


def stage_render(tid: str, view: str) -> tuple[Image.Image, dict, float]:
    print(f"\n[5-6/6] /api/render mode=faithful view={view}")
    t0 = time.time()
    r = requests.post(f"{API}/api/render", json={
        "template_id": tid, "mode": "faithful", "view": view,
    })
    r.raise_for_status()
    elapsed = time.time() - t0
    # requests' Response.headers is case-insensitive — normalise to lowercase
    headers = {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-render")}
    print(f"      {view}: {headers.get('x-render-latency-s', '?')}s render, "
          f"{elapsed:.2f}s round-trip; "
          f"prompt: {headers.get('x-render-prompt', '')[:90]}...")
    return Image.open(io.BytesIO(r.content)).convert("RGB"), headers, elapsed


def stage_depth(tid: str, view: str) -> Image.Image:
    r = requests.get(f"{API}/api/render/depth/{tid}", params={"view": view})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Compose poster
# ---------------------------------------------------------------------------

def main(brief: str, force_template: str | None = None) -> None:
    t_total = time.time()

    reason = stage_reason(brief)
    candidates = reason.get("candidates", [])
    if not candidates:
        sys.exit("no candidates")

    if force_template:
        tid = force_template
    else:
        # Just take the top retrieval result for simplicity (LLM-rerank text
        # is also in reason["llm_response"], but we want a deterministic pick).
        tid = candidates[0]["template"]["id"]

    template = stage_template_meta(tid)
    md = template.get("metadata", {})

    template_path = ROOT / "data" / "templates"
    # find the JSON file by id (search across regions)
    json_path = next(template_path.glob(f"*/{tid}.json"))
    val = stage_validate_ifc(tid, json_path)

    fp_img, fp_t = stage_floor_plan(template, f"/tmp/full_cycle_{tid}_floorplan.png")

    int_img, int_meta, int_t = stage_render(tid, "interior")
    dh_img,  dh_meta,  dh_t  = stage_render(tid, "dollhouse")
    int_depth = stage_depth(tid, "interior")
    dh_depth  = stage_depth(tid, "dollhouse")

    total_t = time.time() - t_total
    print(f"\n[done] full cycle in {total_t:.2f}s\n")

    # ---- Compose poster ----
    PAGE_W = 1980
    GUTTER = 18
    BG = (10, 10, 10)

    # Top header
    HEADER_H = 110

    # 4 cards: brief+reasoning(text), template+IFC(text+small), depth(2 panels)
    # +2 cards for floor plan, photoreal interior, photoreal dollhouse
    # Layout:
    #   Row 1: header (full width)
    #   Row 2: [brief+reasoning] [template+IFC] [floor plan]   (3 columns)
    #   Row 3: [depth interior | depth dollhouse]              (2 columns)
    #   Row 4: [photoreal interior | photoreal dollhouse]      (2 columns)

    col_w_3 = (PAGE_W - 4 * GUTTER) // 3
    col_w_2 = (PAGE_W - 3 * GUTTER) // 2

    # ---- Cards ----
    # 2.1 — brief + reasoning (text card)
    reasoning_text = (reason.get("llm_response") or "(no reasoning)").strip()
    # truncate for display
    reasoning_short = reasoning_text[:1200] + ("..." if len(reasoning_text) > 1200 else "")
    brief_lines = [
        ("Brief:", TXT_MD, True),
        (brief, TXT_HI, False),
        (" ", TXT_HI, False),
        ("Stage 1 — MiniLM retrieval:", TXT_MD, True),
        (f"{len(candidates)} candidates · top: {candidates[0]['template']['id']}", TXT_HI, False),
        (" ", TXT_HI, False),
        ("Stage 2 — Fine-tuned Llama 3.2 reasoning:", TXT_MD, True),
        (f"backend={reason.get('llm_backend','?')} · "
         f"model={reason.get('llm_model','?')[:40]} · "
         f"{reason.get('llm_latency_s',0):.2f}s", TXT_LO, False),
        (" ", TXT_HI, False),
        (reasoning_short, TXT_HI, False),
    ]
    BRIEF_CARD_H = 760
    brief_card = big_text_card("1.  Brief → reasoning", brief_lines,
                                col_w_3, BRIEF_CARD_H, body_font=13)

    # 2.2 — template + IFC validation card
    rooms = template.get("rooms", [])
    room_summary = ", ".join([f"{r['name']} {r.get('area_sqm','?')}m²"
                                for r in rooms[:6]])
    val_passed = val.get("passed", 0)
    val_total = val.get("total", 0)
    val_color = ACCENT if val.get("ok") else (255, 120, 120)
    template_lines = [
        ("Picked template:", TXT_MD, True),
        (tid, TXT_HI, False),
        (" ", TXT_HI, False),
        ("Country / city / size:", TXT_MD, True),
        (f"{md.get('country','?')} · {md.get('city_inspiration','?')} · "
         f"{md.get('total_area_sqm','?')} m² · {md.get('bedrooms','?')} BR · "
         f"ceiling {template.get('boundary',{}).get('ceiling_height_mm',2700)/1000:.1f} m",
         TXT_HI, False),
        (" ", TXT_HI, False),
        ("Style:", TXT_MD, True),
        (md.get("style","")[:140], TXT_HI, False),
        (" ", TXT_HI, False),
        ("Rooms:", TXT_MD, True),
        (room_summary, TXT_HI, False),
        (" ", TXT_HI, False),
        ("IFC validation:", TXT_MD, True),
        (f"{val_passed}/{val_total} schema checks passed  "
         f"·  {val.get('ifc_size_kb',0):.1f} KB",
         val_color, True),
    ]
    template_card = big_text_card("2.  Template + IFC", template_lines,
                                    col_w_3, BRIEF_CARD_H, body_font=13,
                                    accent=val_color)

    # 2.3 — floor plan card
    fp_card = card(
        "3.  Floor plan (architectural)",
        f"walls + door swings + window glyphs + fixtures  ·  {fp_t:.2f}s",
        fp_img, col_w_3, BRIEF_CARD_H - 78,
    )

    # 3 — depth maps row
    DEPTH_H = 320
    depth_int_card = card(
        "4a.  BIM depth — interior",
        f"camera at {int_meta.get('x-render-focus-room','?')} window-wall  ·  "
        f"ceiling {float(int_meta.get('x-render-ceiling-m', 0)):.1f}m",
        int_depth, col_w_2, DEPTH_H,
    )
    depth_dh_card = card(
        "4b.  BIM depth — dollhouse",
        f"{int_meta.get('x-render-focus-room','')} included  ·  "
        f"{dh_meta.get('x-render-rooms-count','?')} rooms  ·  "
        f"{dh_meta.get('x-render-boundary-w','?')}×"
        f"{dh_meta.get('x-render-boundary-d','?')}m",
        dh_depth, col_w_2, DEPTH_H,
    )

    # 4 — photoreal row
    REAL_H = 560
    real_int_card = card(
        "5a.  Faithful interior render",
        f"SDXL-turbo + Depth ControlNet  ·  "
        f"{int_meta.get('x-render-latency-s','?')}s  ·  "
        f"camera aimed at BIM window-wall",
        int_img, col_w_2, REAL_H,
    )
    real_dh_card = card(
        "5b.  Faithful dollhouse render",
        f"SDXL-turbo + Depth ControlNet  ·  "
        f"{dh_meta.get('x-render-latency-s','?')}s  ·  "
        f"full-flat cutaway from above",
        dh_img, col_w_2, REAL_H,
    )

    # Compose
    page_h = (HEADER_H + GUTTER
                + BRIEF_CARD_H + GUTTER
                + DEPTH_H + 78 + GUTTER
                + REAL_H + 78 + GUTTER)
    page = Image.new("RGB", (PAGE_W, page_h), BG)

    # Header
    header = Image.new("RGB", (PAGE_W, HEADER_H), HEADER_BG)
    d = ImageDraw.Draw(header)
    fb = _font(28, bold=True)
    fs = _font(14)
    d.text((24, 18), "BIM Coordinator — full pipeline cycle", fill=TXT_HI, font=fb)
    d.text((24, 56), f"Brief → retrieval → reasoning → IFC → "
                      f"floor plan → BIM depth → photorealistic render",
            fill=TXT_MD, font=fs)
    d.text((24, 78), f"Total wall-clock: {total_t:.2f}s   ·   "
                      f"server: {API}",
            fill=TXT_LO, font=fs)
    page.paste(header, (0, 0))

    y = HEADER_H + GUTTER
    # Row 2 (3 cols)
    page.paste(brief_card,    (GUTTER,                          y))
    page.paste(template_card, (GUTTER * 2 + col_w_3,            y))
    page.paste(fp_card,       (GUTTER * 3 + col_w_3 * 2,        y))
    y += BRIEF_CARD_H + GUTTER

    # Row 3 (2 cols, depth)
    page.paste(depth_int_card, (GUTTER,                  y))
    page.paste(depth_dh_card,  (GUTTER * 2 + col_w_2,    y))
    y += DEPTH_H + 78 + GUTTER

    # Row 4 (2 cols, photoreal)
    page.paste(real_int_card, (GUTTER,                  y))
    page.paste(real_dh_card,  (GUTTER * 2 + col_w_2,    y))

    out = f"/tmp/full_cycle_{tid}.png"
    page.save(out)
    print(f"poster -> {out}  ({page.width}×{page.height})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        brief = " ".join(sys.argv[1:])
    else:
        brief = ("1 BHK apartment in Munich Schwabing, classic Jugendstil "
                 "style with high ceilings, around 50 sqm")
    main(brief, force_template="eu_de_1bed_munich_schwabing")
