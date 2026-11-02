"""Prompts and strict JSON schemas for the vision model."""

from __future__ import annotations

REQ_TYPES = ["size", "location", "depth", "radius", "angle", "chamfer", "gdt", "surface_finish",
             "thread", "note", "basic", "reference", "other"]
KINDS = ["dimension", "hole_callout", "gdt_frame", "note", "surface_finish", "radius", "angle", "chamfer",
         "thread", "basic_dimension", "reference_dimension"]


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


S = {"type": "string"}
BOX = {"type": "array", "items": {"type": "integer"}}

REQUIREMENT = _obj({
    "type": {"type": "string", "enum": REQ_TYPES},
    "label": S,
    "text": S,
    "nominal": S,
    "lower": S,
    "upper": S,
    "gdt_symbol": S,
    "datums": {"type": "array", "items": S},
    "modifiers": {"type": "array", "items": S},
})

ITEM = _obj({
    "kind": {"type": "string", "enum": KINDS},
    "text": S,
    "multiplier": {"type": "integer"},
    "box": BOX,
    "confidence": {"type": "string", "enum": ["high", "low"]},
    "requirements": {"type": "array", "items": REQUIREMENT},
})

READ_SCHEMA = _obj({"items": {"type": "array", "items": ITEM}})

AUDIT_ITEM = _obj({
    "text": S,
    "box": BOX,
    "what": {"type": "string", "enum": ["requirement", "not_requirement", "erased_fragment"]},
    "reason": S,
    "kind": {"type": "string", "enum": KINDS},
    "multiplier": {"type": "integer"},
    "requirements": {"type": "array", "items": REQUIREMENT},
})
AUDIT_SCHEMA = _obj({"items": {"type": "array", "items": AUDIT_ITEM}})

OVERVIEW_SCHEMA = _obj({
    "border_box": BOX,
    "title_block_box": BOX,
    "revision_block_box": BOX,
    "general_notes_box": BOX,
    "zone_columns": {"type": "array", "items": S},
    "zone_rows": {"type": "array", "items": S},
})

TITLE_SCHEMA = _obj({
    "part_number": S,
    "part_name": S,
    "drawing_number": S,
    "revision": S,
    "sheet": S,
    "units": {"type": "string", "enum": ["in", "mm", "unknown"]},
    "material": S,
    "finish": S,
    "general_tolerances": {"type": "array", "items": _obj({"applies_to": S, "plus_minus": S})},
})

_REQ_RULES = (
    "For each item also split it into its inspectable requirements (one per Form 3 line). Examples: "
    "a hole callout 'Ø.182-.192 ↧.250±.010' is two requirements (hole diameter .182 to .192, depth .240 to .260); "
    "a counterbore line adds counterbore diameter and depth; a feature control frame row is one gdt requirement "
    "(gdt_symbol such as position, flatness, profile of a surface, perpendicularity, angularity; nominal = the tolerance "
    "value; lower empty, upper = the tolerance value; datums in order; modifiers such as MMC, LMC, diameter); "
    "a note that imposes a requirement is one note requirement with the full note text. "
    "For number requirements give nominal, lower and upper exactly as numbers on the drawing would give them "
    "(compute lower and upper from ± or limits; leave them empty when the drawing prints no tolerance). "
    "Basic dimensions (in a box) are type basic; dimensions marked REF or in parentheses are type reference. "
    "Copy every number exactly as printed; never invent a value that is not on the drawing."
)

READ_PROMPT = (
    "You are an object grounding expert reading one tile of a mechanical engineering drawing for an AS9102 first "
    "article inspection. Find EVERY inspectable requirement printed in this image. Do not miss any, including small, "
    "rotated or vertical text, stacked tolerances, text touching lines, and notes. Include: every dimension value "
    "(also basic and reference ones), every hole, counterbore, countersink, thread and depth callout, every GD&T "
    "feature control frame (each frame row is a separate item), radii, angles, chamfers, surface finish symbols, "
    "and every note or general note that imposes a requirement (including UNLESS OTHERWISE SPECIFIED tolerance blocks, "
    "material and finish notes, BREAK SHARP EDGES and similar). "
    "Do NOT include: view or section labels (SECTION A-A, DETAIL B), datum feature letters on their own, border zone "
    "letters or numbers, revision table rows, title block administrative fields (drawn by, dates, scale, sheet), "
    "part lists or hardware part numbers, or text that is cut off at the image edge (it will be read from the next tile). "
    "Give text verbatim with its multiplier from 'nX' or '(TYP n PLACES)' (1 if none), and a tight box "
    "[x_min, y_min, x_max, y_max] around the printed text (and its frame, if boxed) on a 0-1000 grid of this image. "
    + _REQ_RULES
)

AUDIT_PROMPT = (
    "You are an object grounding expert checking an engineering drawing tile for an AS9102 first article inspection. "
    "Every requirement that was already found has been erased from this image (painted white). Your job is to find "
    "anything that was missed: any dimension value, tolerance, hole or thread callout, GD&T feature control frame, "
    "surface finish symbol, radius, angle, chamfer, or note that imposes a requirement and is still visible. "
    "Look carefully at small, rotated, faint or crowded text and at text touching lines. "
    "Report every piece of text or symbol you can still see with a tight box on a 0-1000 grid of this image and say "
    "what it is: 'requirement' (a complete missed requirement; also fill kind, multiplier and requirements), "
    "'not_requirement' (view or section labels, datum letters, zone markers, title block fields, revision table, "
    "part lists, hardware part numbers and quantities such as MS27039-1-23(4) or AN960D10L(4), find numbers, "
    "applicability notes such as (ON-5), scale, sheet numbers, empty frames, leaders and arrows) with a short reason, "
    "or 'erased_fragment' (any incomplete piece of text: a stray digit, a partial word or number, a symbol edge, "
    "anything that looks cut off or sits next to a white patch). Only call something a requirement when it is complete "
    "and readable on its own. If nothing is left, return an empty list. "
    + _REQ_RULES
)

OVERVIEW_PROMPT = (
    "This is a whole engineering drawing sheet, shrunk. Return boxes on a 0-1000 grid of this image for: the drawing "
    "border (inner frame), the title block, the revision block (empty box [0,0,0,0] if none), and the general notes "
    "area (empty box if none). Also list the border zone labels: zone_columns left to right as printed along the top "
    "or bottom edge (for example ['8','7',...,'1']), and zone_rows top to bottom as printed along the side (for example "
    "['D','C','B','A']). Empty lists if the border has no zone labels."
)

TITLE_PROMPT = (
    "These are crops of an engineering drawing's title block and general notes. Read: part number, part name, drawing "
    "number, revision letter, sheet, units (inches or millimeters; decide from notes such as 'DIMENSIONS ARE IN "
    "INCHES' or from the values), material, finish, and the general tolerances that apply when a dimension has no "
    "tolerance printed (for example applies_to '.XX' plus_minus '.01', applies_to '.XXX' plus_minus '.005', "
    "applies_to 'ANGLES' plus_minus '0.5', applies_to 'FRACTIONS' plus_minus '1/32'). Empty strings or lists when "
    "not shown. Copy values exactly."
)

CONFIRM_SCHEMA = _obj({
    "verdict": {"type": "string", "enum": ["separate_requirement", "same_as_listed", "not_requirement"]},
    "same_as": S,
    "full_text": S,
    "reason": S,
    "kind": {"type": "string", "enum": KINDS},
    "multiplier": {"type": "integer"},
    "requirements": {"type": "array", "items": REQUIREMENT},
})


def confirm_prompt(listed: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"  [{n}] {t}" for n, t in listed) or "  (none nearby)"
    return (
        "This is a crop of the original engineering drawing. A search of the erased page flagged the text in the RED "
        "box as an inspectable requirement that the first read missed. Requirements already listed in this area are "
        "outlined in BLUE and numbered:\n" + lines + "\n"
        "Decide: 'separate_requirement' if the red-box text is a complete inspectable requirement (dimension, "
        "tolerance, hole or thread callout, GD&T frame, surface finish, or a note that imposes a requirement) that is "
        "NOT already covered by a blue item; 'same_as_listed' if it is the same text as, or part of, a blue item (give "
        "its number in same_as); 'not_requirement' for labels, hardware part numbers and quantities, applicability "
        "notes, title block fields, find numbers or fragments. For a separate requirement give its full text exactly "
        "as printed (read the whole callout, not just the red box), kind, multiplier and requirements. "
        + _REQ_RULES
    )
