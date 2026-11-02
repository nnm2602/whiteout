"""Core records passed between pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in sheet pixels

ReqType = Literal[
    "size", "location", "depth", "radius", "angle", "chamfer", "gdt", "surface_finish",
    "thread", "note", "basic", "reference", "other",
]


class Requirement(BaseModel):
    """One inspectable line on Form 3."""

    type: ReqType = "other"
    label: str = ""  # e.g. "hole diameter", "counterbore depth"
    text: str = ""  # requirement as printed
    nominal: float | None = None
    lower: float | None = None  # lowest acceptable value
    upper: float | None = None  # highest acceptable value
    unit: str = ""
    gdt_symbol: str = ""
    datums: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    attribute: bool = False  # pass/fail by inspection, no number
    tolerance_source: Literal["printed", "general", "none", "model"] = "none"
    flags: list[str] = Field(default_factory=list)  # e.g. "model limits differ from printed text"


class Characteristic(BaseModel):
    id: str  # stable id within a job
    number: str = ""  # balloon number, e.g. "7" or "7.2" for 4X instances
    kind: str = "dimension"
    text: str  # verbatim as read
    multiplier: int = 1
    box: Box  # tight box in sheet pixels after snapping
    raw_box: Box | None = None  # box as returned by the model
    sheet: int = 0
    zone: str = ""
    confidence: Literal["high", "low"] = "high"
    found_in: str = "read"  # "read", "audit-1", "audit-2", "text-layer", "manual"
    requirements: list[Requirement] = Field(default_factory=list)
    balloon_xy: tuple[int, int] | None = None
    flags: list[str] = Field(default_factory=list)
    ignored: bool = False  # flagged by the audit as not a requirement


class SheetInfo(BaseModel):
    index: int
    width: int
    height: int
    dpi: float = 200.0
    part_number: str = ""
    part_name: str = ""
    drawing_number: str = ""
    revision: str = ""
    units: str = "in"
    material: str = ""
    finish: str = ""
    general_tolerances: list[dict] = Field(default_factory=list)  # {"decimals":3,"plus_minus":0.005} / {"kind":"angle",...}
    zone_columns: list[str] = Field(default_factory=list)  # left to right
    zone_rows: list[str] = Field(default_factory=list)  # top to bottom
    border_box: Box | None = None
    title_block_box: Box | None = None
    has_text_layer: bool = False


class AuditFind(BaseModel):
    pass_no: int
    text: str
    box: Box
    verdict: Literal["new", "duplicate", "not_requirement", "fragment"]
    reason: str = ""
    char_id: str | None = None


class AuditPass(BaseModel):
    pass_no: int
    finds: list[AuditFind] = Field(default_factory=list)
    new_count: int = 0
    residue_png: str = ""  # relative path of the residue image used for this pass
