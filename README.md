# Whiteout

**Balloon every requirement on an engineering drawing, fill the AS9102 first article forms, then erase everything that
was found so you can see what is left.**

Live app: https://whiteout-955819688070.us-central1.run.app (three sample drawings open instantly; uploads run live)

Backup link: https://whiteout-backup-955819688070.us-east1.run.app (same app; the first visit can take about 12 seconds to start)

Built for the Meta Global AI Developer Hackathon (Oct 27 to Nov 4 2026), track "Push the Frontier" (manufacturing).
Everything in this repository was written during the hackathon.

## The problem

Before an aerospace or defense shop ships the first part from a new drawing or a new revision, it proves every
requirement on the drawing was measured: the AS9102 first article inspection. A quality technician numbers every
dimension, tolerance, GD&T frame and note on the drawing with a circle ("balloon"), types each one into Form 3, and
measures the part against each line. A drawing can carry 50 to 500 of them. Vendors put the manual ballooning at 2 to 8
hours per drawing ([Net-Inspect](https://www.net-inspect.com/blog/how-automated-ballooning-accelerates-first-article-inspection/),
[GroundControl](https://www.gndctl.com/insights-library/automated-ballooning-software-fai-buyers-guide-2026)), and a
requirement with no balloon has no Form 3 line, so the report is incomplete.

Auto-ballooning tools exist. The inspector still has to trust that the list is complete, which is why some still
balloon by hand. Whiteout makes completeness something you can look at.

## What it does

1. **Read.** The sheet (vector PDF, scan, or photo) is cut into tiles small enough for Muse Spark to read at full
   resolution (Spark shrinks anything above about 3.2 megapixels; we measured it). Every requirement comes back with
   its printed text, its place on the tile, and its tolerance split into Form 3 lines (a hole callout with a depth is
   two lines).
2. **Check in code.** Model boxes drift, so each one is snapped onto the real ink: drawing lines are removed, letters
   are grouped into text blocks, and boxes are matched to blocks across the whole sheet at once, with Tesseract as a
   second reader that confirms the characters. Tolerance limits are recomputed from the printed text; a number the
   model gives that is not on the drawing is flagged.
3. **Whiteout.** Everything found is painted white. Muse Spark searches what is left on tiles shifted half a step, and
   each candidate miss is checked again on the original drawing next to the requirements already listed. Whatever
   the system missed is still printed on an otherwise blank page, where a person sees it in seconds.
4. **Report.** Balloons numbered in reading order with drawing zones, AS9102 Forms 1 to 3 as a spreadsheet, a
   ballooned PDF. Results are typed, imported from a CMM file, or spoken ("balloon fourteen, point two five one two")
   through Muse Voice Transcribe. Code, not the model, decides pass or fail.
5. **Revision.** When revision B arrives, the two sheets are aligned in code and compared pixel by pixel. Requirements
   whose ink did not change keep their measured values; only changed, added and newly listed ones go on the
   measurement list.
6. **Open weights.** The same pipeline runs on Muse Glimmer 30B (Q4 GGUF with its vision projector and the DFlash
   draft model) served by llama.cpp on a rented GPU, for drawings that should not go to a model API.

## From the ballooned drawing to the signed report

The ballooned drawing is where first article inspection starts, not where it ends. Whiteout carries on through the
rest of the job:

7. **Plan.** Every Form 3 line gets an instrument and a method, chosen in code from the requirement type and the width
   of its tolerance band (the instrument should resolve a tenth of the band): caliper or micrometer, GO and NO-GO pins
   with the pin sizes, thread gauges, height gauge, profilometer, CMM. Muse Spark sorts the notes into certificate,
   visual, functional or not inspectable. Lines are grouped into setups so the part is picked up as few times as
   possible.
8. **CMM program.** A starting program in DMIS for the lines that go to the CMM: datum alignment, holes with their
   size tolerances, position frames with their material condition, each tied to the hole callout it sits under. With
   the part's STEP model, hole centres and axes come from the model; without it the holes are probed by hand. A CMM
   programmer reviews it before it runs.
9. **Guided measuring, hands-free.** Each line is read aloud (the browser's voice; Meta's API has no text to speech),
   the inspector says the value and Muse Voice Transcribe hears it, and code decides pass or fail. Or the inspector
   photographs the caliper: Muse Spark finds the display and reads each digit's lit segments, a seven-segment
   decoder in code reads the same display, and the value is written only when both agree. Otherwise the person picks.
10. **Ask the drawing.** "Which lines refer to datum B?", "what is the worst-case wall under counterbore 17 if the part
    is .500 thick?" Muse Spark answers from the ballooned sheets and the Form 3 lines and cites the balloons. For a
    stack-up it only picks the chain and the signs; code adds up worst case and root sum square.
11. **Check a supplier's report.** The buyer's side: a supplier's AS9102 report, as a spreadsheet in any layout (read
    in code) or a scanned PDF (read by Muse Spark), is checked row by row against the drawing: requirements with no
    row, limits copied wrong, results outside the limits marked accepted, missing results, repeated numbers, rows that
    are not on the drawing.
12. **Drawing against the 3D model.** In model-based definition the STEP model carries its own dimensions and
    tolerances (AP242 PMI), and drawing and model drift apart. A STEP reader in plain Python pulls the PMI and the
    holes out of the model, and every item is compared with what was read on the drawing.
13. **Redraw old drawings.** A 1960s scan becomes a clean vector drawing (SVG and PDF): line work traced in code
    (thinning, line and arc fitting, nothing invented), every text read by Muse Spark and set in type where it was
    printed. Then Muse Spark proofreads the redraw against the scan, region by region, and every place where the text
    differs is marked in red for a person to check. The redraw is a reading aid, not a released drawing.

## Meta models and what each one does

| Model | Job |
|---|---|
| Muse Spark 1.3 (Meta Model API) | Reads every tile with strict JSON output; reads the title block and general tolerances; searches the erased page; checks each candidate miss on the original; sorts notes into how they are verified; reads gauge displays segment by segment; answers questions about the drawing and picks stack-up chains; reads scanned supplier reports |
| Muse Voice Transcribe 1.0 | Push-to-talk and hands-free measured values with keyword biasing; a strict parser only accepts values for a real Form 3 line |
| Muse Glimmer 30B (open weights) | The whole read and search pipeline on a rented L4 GPU through llama.cpp, with Meta's DFlash draft model for speed |

## Measured results

All numbers come from `eval/` and can be rerun. Model calls are cached, so a rerun of the same code costs nothing.

**Answer keys on 67 public drawings** (`eval/results/keys_main/SUMMARY.json`). 1,679 requirements, in keys made
before any run and independent of Whiteout: NIST's official element lists for the FTC and CTC test cases, Cadsy's
official labels, and keys we wrote for the NIST D2MI, Apollo and military standard sheets.

| | Requirements found | 95% interval |
|---|---|---|
| Whiteout (Muse Spark 1.3) | **96.9%** (1,627 of 1,679) | 96.0 to 97.6 |
| Plain OCR (Tesseract + a dimension pattern) | 16.6% | |

| Source | Sheets | Requirements | Whiteout | Plain OCR |
|---|---|---|---|---|
| NIST D2MI (CAD, GD&T) | 12 | 226 | 99.1% | 27.0% |
| Apollo Lunar Module scans, 1960s | 15 | 589 | 97.5% | 24.3% |
| NIST CTC test cases | 5 | 69 | 97.1% | 1.5% |
| NIST FTC test cases | 6 | 89 | 96.6% | 2.2% |
| US military standard sheets | 4 | 335 | 96.4% | 4.2% |
| Cadsy synthetic drawings | 25 | 371 | 95.2% | 15.4% |

- Of the requirements found, the text was exactly right 89% of the time and the multiplier (4X and so on) 95%.
- 74% of what Whiteout listed matched a key item. The others include pieces of larger key items, table values, notes
  the keys left out on purpose ("GRAIN DIRECTION OPTIONAL") and a few real requirements the keys missed
  ("TO BE FLAT .005 TIR").
  The inspector removes an unwanted line with one click.
- The search of the erased page added 3 items over the 67 sheets, 1 of them in a key. The first read already finds
  almost everything; the erased page is mainly for the person checking.

**Planted requirements** (`eval/results/seeded_main2/SUMMARY.json`). 144 requirements drawn into 12 public drawings (10 real, 2 synthetic) in
six hard conditions (faint, tiny, small, rotated 90 degrees, sitting on a drawing line, plain). The first read found
139 (96.5%). Of the 5 it missed, 4 were still fully visible on the erased page; the fifth was planted touching another
label and was erased with it.

**Revision delta** (`samples/nist-904-revB.pdf`, our own edit). Whiteout reported exactly the 3 changed values, the 1
removed dimension and the 1 added note, plus 3 items the earlier report had not listed; 7 lines to measure instead of
78, and 42 measured results carried forward.

**Open weights** (`eval/results/keys_glimmer/SUMMARY.json`). Muse Glimmer 30B (4-bit, with its vision projector and
the DFlash draft model) on one NVIDIA L4 read 6 of the answer-key sheets: 3 Apollo scans and 3 NIST D2MI CAD sheets,
240 requirements. It found 220 (91.7%); Muse Spark found 234 (97.5%) on the same sheets. Most of the gap is one faint
Apollo scan (45 of 57, Spark 54). Glimmer also listed more lines that match no key item (46 against 25), so the person
checking has more to delete. A sheet took 22 to 70 minutes (median 52), so on one L4 this is an overnight batch, not
an interactive tool. We planned 10 sheets; the GPU's 6 hour stop came after 6.

**Redraw** (`eval/results/redraw/SUMMARY.json`, `PROOF_SUMMARY.json`). Six Apollo scans with answer keys. On average
98.5% of each scan's line ink is covered by a traced line, and 99.98% of traced line pixels sit on scan ink (within 3
pixels). Plain Tesseract reads 67.9% of the key words on the redraw against 57.1% on the scan. 80% of key items are
set with exactly the key's text; the misses are mostly long tables where a value lands one row off. The proofreading
pass marked 55 of the 60 key items the redraw got wrong, and also marked 51 of the 230 it got right.

**Fine-tuned Glimmer** (`eval/results/keys_glimmer_ft/COMPARISON.json`, scripts in `deploy/finetune/`). A LoRA
adapter (rank 16) trained on Muse Spark's own tile reads of 63 answer-key sheets, cut into 2,108 crops of 1344 x 672
pixels. The 6 test sheets above and every other page of their drawings were left out, and no answer key was used in
training. 160 steps (640 crops) took 118 minutes on one L4 with Unsloth's 4-bit checkpoint; llama.cpp serves the stock
4-bit model with the adapter (`--lora`). On the same 6 sheets it found 232 of 240 (96.7%, 95% CI 93.6 to 98.3), against
220 for stock Glimmer and 234 for Muse Spark, at a median 20 minutes a sheet instead of 52 (9 to 43). It reads the
smaller tiles it was trained on, so the gain is the adapter and the tile size together, not the adapter alone. It also
listed more lines that match no key item (57 against 46; 31 of them on one faint Apollo scan). The VM ran Tesseract 5,
the Mac 5.5. The whole job runs on the VM from one startup script (`deploy/finetune/vm_driver.sh`) and the VM deletes
itself at the end.

**Supplier report check** (`eval/results/fair_check/SUMMARY.json`). 18 supplier reports we generated for the NIST and
Apollo samples, in two layouts unlike ours (merged group headers, Min/Max or nominal and plus/minus columns, the
supplier's own numbering on some), 4 of them as PDFs read by Muse Spark. 120 errors planted, 20 of each kind: all 120
reported, and no findings on the 8 clean reports. Ten more random draws of the spreadsheets: 957 of 958 planted errors
reported, none on 60 clean reports. The one miss is a true ambiguity (two balloons with the same printed text). These
reports are synthetic; real ones will be messier.

**Drawing against the 3D model** (`eval/results/model_check/SUMMARY.json`). NIST's 11 CTC and FTC test models: the
reader recovered 325 of the 326 dimension and tolerance elements NIST lists (293 with the same value and tolerance).
57 edits planted in 12 models (tolerances doubled, nominals moved, GD&T values doubled, tolerances and frames deleted):
56 reported. It also turned up real disagreements in NIST's own files, for example FTC 09 stores its 30 GD&T values as
inch numbers with a millimetre unit, and FTC 10's model lacks two radius callouts that are on the drawing.

**Ask the drawing** (`eval/results/ask/SUMMARY.json`). 27 questions on the three samples whose answers were worked out
in code from the Form 3 lines: limits of a line 9 of 9, stack-up chains (lines and signs) 6 of 6, list questions
("which balloons refer to datum A") 91 of 91 balloons cited, with 6 extra citations, each a related line such as the
note that describes datum A.

**Gauge photos** (`eval/results/gauge/SUMMARY.json`). 36 synthetic phone photos of caliper and micrometer displays at
an angle, with blur, noise and, on 6, glare over a digit. Muse Spark alone read 22 of the 30 clear ones right. With
the second reader in code, 9 of 30 were written without asking, none of them wrong; the rest, and all 6 glare photos,
went to the person to pick. Real instrument photos have not been measured yet.

**Cost and time.** About $0.30 per CAD sheet and $0.40 to $0.80 per large scan with Muse Spark 1.3; 40 to 70 seconds
of processing per sheet.

## Try it

Open the live app and pick a sample, or upload a PDF, PNG, JPEG or TIFF (up to 6 sheets). The public demo has a daily
model budget; if it is used up, the samples still work.

## Run it yourself

```bash
uv sync
export MODEL_API_KEY=...            # Meta Model API key
uv run whiteout run drawing.pdf --out out/          # ballooned PNGs, erased pages, AS9102_forms.xlsx, result.json
uv run whiteout serve --port 8080                    # the web app (front end: cd web && npm i && npm run build)
```

Open weights: see `deploy/glimmer_vm_startup.sh` (llama.cpp + Glimmer on one L4), then
`WHITEOUT_OFFLINE_URL=http://HOST:8000/v1 uv run whiteout run drawing.pdf --engine offline`.

Tests: `uv run pytest`. Measurements: `eval/run_keys.py` (answer keys), `eval/run_seeded.py` (planted requirements).

## Data and licences

The code in this repository is open source under the MIT License (see `LICENSE`). The drawings and other data keep
their own licences, listed below.

Test drawings are public domain or Apache 2.0: NIST MBE PMI test cases and D2MI drawings (17 USC 105), Apollo Lunar
Module drawings from the US National Archives (Internet Archive, Public Domain Mark), US military standard sheets
(Distribution Statement A), and the Cadsy synthetic drawings (Apache 2.0). Sources, URLs and licences per file are in
`data/MANIFEST.json`; how the answer keys were made is in `eval/README.md`. Revision B of NIST 827-9999-904 in
`samples/` is our own edit, made to test the revision feature; its changes are listed in
`samples/nist-904-revB.changes.json`.

## Limits

- A person checks and signs the report. Whiteout prepares it; it does not accept parts.
- Boxes on rotated or very faint text are the weakest step; such items are flagged for a check.
- Glimmer on one L4 is slow (22 to 70 minutes per sheet) and finds less than Spark (91.7% against 97.5% on 6 sheets).
- The open-weights mode keeps drawings off model APIs. It is a technical control, not an export-compliance
  certification.
- The revision comparison assumes the new revision is the same sheet layout, moved or scaled at most.

## Team

Nhat Nam Tran.
