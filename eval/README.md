# Whiteout answer keys

Ground truth for measuring how well Whiteout finds every inspectable requirement on a drawing: dimensions,
tolerances, hole and thread callouts, each feature control frame row, radii, angles, chamfers, surface finish
symbols, and notes that impose a requirement. One key per sheet in `eval/keys/<id>__p<page>.json`. Every drawing
file a key uses is listed in `data/MANIFEST.json` with its source URL and licence. The drawings themselves live
in `data/raw/` (not in git); the manifest says where to download each one.

## Key format

```
{"id": "...", "page": 1, "width": W, "height": H, "dpi": D, "file": "data/raw/...",
 "source": "official" | "hand", "doubts": ["..."],
 "items": [{"key": "k1", "text": "exact printed text", "kind": "...", "multiplier": 1,
            "box": [x0, y0, x1, y1] | null}]}
```

- `width`, `height` are the size of `whiteout.ingest.load(file)[page - 1].image` (PDFs at 200 dpi grayscale,
  longest side at most 6400 px, scans cleaned and deskewed). Boxes are pixels in that image.
- `kind` is one of dimension, hole_callout, gdt_frame, note, surface_finish, radius, angle, chamfer, thread,
  basic_dimension, reference_dimension.
- `multiplier` is the printed feature count ("4X", "(3)", "2 PLACES"); 1 when none is printed.
- Optional item fields: `table` (a cell of a dimension table), `official_id` (NIST element ID such as D9-1, T12),
  `title_block` (a title-block value).
- Frame rows are written cell by cell with `|`, e.g. `⌖|⌀.005Ⓜ|A|B|C`. Stacked limits are written upper/lower,
  e.g. `1.565/1.555`.
- `doubts` lists every choice made while writing the key, with key ids. Read them before disputing a
  result.

Included: every printed instance of a requirement (a value repeated in two views is two items), numbered notes
that impose something (material, finish, marking, cleaning, deburr, flatness), the checked lines of the Grumman
"manufacturing operations / supplementary data" block as one note, and title-block general tolerances.
Excluded: view and section labels, datum letters, zone labels, title block admin fields, revision tables, parts
lists, hardware part numbers alone, notes that impose nothing (units, applicable standards, cross references).

## Sources, licences, counts

Counts (90 keys, 2086 items, 65 drawing files; every item has a box):

| Source | Key source | Files | Keyed sheets | Items |
|---|---|---|---|---|
| Apollo LM drawings | hand | 19 | 19 | 750 |
| Cadsy synthetic drawings | official labels | 25 | 25 | 371 |
| NIST FTC 6 to 11 | official element list | 6 | 22 | 305 |
| NIST CTC 1 to 5 | official element list | 5 | 8 | 99 |
| NIST D2MI 903 to 908 | hand | 6 | 12 | 226 |
| US military standard sheets | hand | 4 | 4 | 335 |

Items per kind: dimension 997, gdt_frame 339, note 205, basic_dimension 119, hole_callout 156, radius 99,
chamfer 61, surface_finish 38, reference_dimension 26, thread 24, angle 22. Tables: the MS sheets are mostly
tabulated dimensions (308 of their 335 items carry `table: true`), and apollo_b503_n0006 has a 31-row table.

| Source | Licence | Notes |
|---|---|---|
| Apollo LM drawings (Grumman, 1960s), NARA aperture cards scanned by the Virtual AGC project, archive.org items `apertureCardBox502NARASW_images` and `apertureCardBox503NARASW_images` | Public Domain Mark 1.0 on the archive.org items | Contractor drawings, so 17 USC 105 does not apply directly; the public domain status rests on the archive.org marking and NARA holding them. Credit "NARA RG 255, scanned by the Virtual AGC project". |
| Cadsy CAD technical drawings, huggingface.co/datasets/cadsy/cad-technical-drawings | Apache License 2.0 (copy in data/raw/cadsy/LICENSE); models from ADSKAILab/Zero-To-CAD-100k, Apache 2.0 | Synthetic, clean, one ASME and one ISO profile per part. |
| NIST MBE PMI test cases, FTC 6 to 11 and CTC 1 to 5 (NIST_MBE_PMI_FTC_Definitions.zip, NIST_MBE_PMI_CTC_Definitions.zip) | US Government work, public domain (17 USC 105) | NIST asks for acknowledgement; the NIST logo may not be used. |
| NIST D2MI mechanical reference dataset 827-9999-903 to 908 (NIST-D2MI-Models.zip) | US Government work, public domain (17 USC 105) | Vector PDFs from NX. |
| US military standard sheets (MS16562C, MS21042M, MS21919E, MS35338L), everyspec.com copies | US Government works | Each copy carries a "Downloaded from everyspec.com" line; official copies are on ASSIST (quicksearch.dla.mil). MS16562C, MS21042M and MS35338L print Distribution Statement A; MS21919E sheet 1 prints none. |

"NOT FOR PHOTO REPRODUCTION" stamp: of the keyed Apollo sheets only apollo_b502_n0040 carries it. It is a 1960s
shop marking, not a licence term, but avoid showing that sheet in public material.

## How the keys were made

- Cadsy (official): `eval/make_cadsy_keys.py` converts the Cadsy label files. Kept: `dimension`, `callout`,
  `feature_control_frame` and the ISO title-block "General tolerance" value. A labelled frame with two rows is split
  into two items with the box cut into equal row heights. Kinds are inferred from the callout text (R radius,
  C chamfer, ⌀ with THRU, depth, counterbore, countersink or PCD hole callout, otherwise dimension). A frame that
  sits directly under an "nX" hole callout takes that count as its multiplier. Label boxes are normalised, so they
  scale straight onto the loaded PNG (2480 x 1754, unchanged by the loader).
- NIST FTC and CTC (official element list, text read from the drawing): NIST publishes an `elem_ids` copy of
  each drawing page that prints an element ID next to every annotation, and for FTC a Feature and Specification
  Index (FSI) that lists every element. The elem_ids pages are rasters on a different page size, so each was
  registered onto the plain page render (ORB features, similarity transform, median error about 1 px). Keys have
  one item per dimension element (D, with sub IDs such as D9-1 kept apart), one per frame row of each tolerance
  element (T), and the qualifier strings (STR) and notes that change a requirement. Datum feature symbols, datum
  targets, represented line elements and coordinate systems are left out. Text is copied as printed on the
  drawing, boxes are measured on the plain page render, and each item keeps its `official_id`. The CTC
  spreadsheet only lists the highlighted test specifications, so for CTC the elem_ids IDs are the checklist.
- Hand keys (Apollo, MS, D2MI): read sheet by sheet from the loaded image in crops of about 900 px (MS sheets
  enlarged x2), boxes measured on a coordinate grid, then every box drawn back on the sheet and checked. No
  Whiteout run or model call was used to write any key.

## Spot check of earlier hand keys

Three earlier hand keys were checked by drawing every box on the sheet and re-reading the sheet:
apollo_b503_n0172, nist_d2mi_906 page 2 and milstd_ms21919e. Findings and fixes:

- apollo_b503_n0172 (26 items): every box sits on its text and the readings match the sheet, including the
  checked lines of the Grumman block. No change.
- nist_d2mi_906 page 2 (43 items): boxes and texts match, nothing missing. No change. One cross-sheet
  inconsistency noted, not changed: `2X ⌀.370 - .380` is keyed hole_callout here and dimension on 904 page 2
  (the 904 doubts read it as the boss outside diameter).
- milstd_ms21919e (7 items): k2 (`.199/.209 Dia 2 holes in line`) clipped the leading decimal points; box
  widened to [697, 659, 918, 699]. k4 (joggle flushness note) clipped its last digit; box widened to
  [626, 1129, 873, 1169]. Added a doubt that the lettered dimensions (W, D, R, T, G, E, X, S, B) are placeholders
  whose values are tabulated on sheet 2, and that the flag note triangles point to sheet 3.

## Known doubts and limits

- Apollo scans: a few words are blacked out on the cards, some values are overwritten by hand
  (apollo_b503_n0172 k24 .005 may be revised to .002), and faint check marks in the Grumman block were read
  one at a time. Old style "THIS SURFACE TO BE FLAT WITHIN .002" leader notes are keyed as notes, not frames.
  "TYP" without a count gets multiplier 1. Limits printed with tight hyphens are written with spaced hyphens.
- MS sheets: values live in tables by dash number, the figure dimensions are letters only and are not keyed.
  The MS21042M scan dropped most decimal points; cells are keyed as printed ("167" means .167). Strength and
  torque cells are keyed as table notes; part number, dash number, nominal size and recommended hole size
  columns are left out. A few blurred cells are listed in each key's doubts.
- NIST FTC and CTC: most annotations are drawn rotated (isometric views), so their axis-aligned boxes are larger
  than the text and often overlap. Values are keyed as printed where the drawing and NIST's spreadsheet or index
  disagree (CTC 2 T2 prints ⌀1Ⓛ, spreadsheet ⌀1.2Ⓛ; CTC 5 T6 prints datum A, spreadsheet B; FTC 9 D10 and D11
  print 3X, index lists four holes). Two-row frames get one item per row; where the ID page labels the frame
  once, the rows carry made-up sub IDs (T2-1, T2-2). The general profile note is tagged with its index element
  (T23, T37) on FTC 6 and 10 and as "note 4" on FTC 7 and 8. All-around circles on leaders are not written into
  the text. Frames and depth lines stacked under an "nX" callout take that count as multiplier, as in the D2MI
  keys. Datum targets with an area size are left out.
- Cadsy: kinds come from the callout text, not from Cadsy (Cadsy only says dimension, callout or frame). The
  per-row frame boxes are an even split of the labelled frame box. The ASME profile sheets have no general
  tolerance cell, so only the 12 ISO sheets carry a general tolerance note. Some labelled boxes for small rotated
  dimensions are only a few pixels wide.
- Inclusion of notes is the main source of disagreement in any key: notes that only point to other drawings,
  units, standards and CAD model notes are left out; each key lists the notes it left out.
