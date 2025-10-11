from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .parser import ParsedMeddraData


def _entry(code: Optional[str], term: Optional[str], level: str) -> Optional[Dict[str, str]]:
    if not code:
        return None
    return {"code": str(code), "term": term or "", "level": level}


@dataclass
class HierarchyItem:
    llt: Optional[Dict[str, str]]
    pt: Optional[Dict[str, str]]
    hlt: Optional[Dict[str, str]]
    hlgt: Optional[Dict[str, str]]
    soc: Optional[Dict[str, str]]

    def to_dict(self) -> Dict[str, Optional[Dict[str, str]]]:
        return {
            "LLT": self.llt,
            "PT": self.pt,
            "HLT": self.hlt,
            "HLGT": self.hlgt,
            "SOC": self.soc,
        }


class MeddraHierarchy:
    """Utility for resolving MedDRA hierarchical relationships."""

    def __init__(self, parsed: ParsedMeddraData):
        self.parsed = parsed
        self.code_to_level = parsed.code_to_level
        self.llt_to_pt = parsed.llt_to_pt
        self.llt_terms = dict(zip(parsed.llt["llt_code"], parsed.llt["llt_name"]))
        self.pt_terms = dict(zip(parsed.pt["pt_code"], parsed.pt["pt_name"]))
        self.hlt_terms = dict(zip(parsed.hlt["hlt_code"], parsed.hlt["hlt_name"]))
        self.hlgt_terms = dict(zip(parsed.hlgt["hlgt_code"], parsed.hlgt["hlgt_name"]))
        self.soc_terms = dict(zip(parsed.soc["soc_code"], parsed.soc["soc_name"]))
        self.pt_context = self._build_pt_context(parsed.mdhier, parsed.llt)

    def _build_pt_context(self, mdhier: pd.DataFrame, llt_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        pt_context: Dict[str, Dict[str, str]] = {}
        llt_to_pt = dict(zip(llt_df["llt_code"], llt_df["pt_code"]))

        for _, row in mdhier.iterrows():
            llt_code = str(row.get("llt_code", ""))
            pt_code = llt_to_pt.get(llt_code)
            if not pt_code:
                continue
            is_primary = str(row.get("primary_soc_flag", "")).upper() == "Y"
            current = pt_context.get(pt_code)
            if current and not (is_primary and not current.get("primary", False)):
                # Keep the existing context unless the new row is primary and existing is not.
                if current.get("primary", False) and not is_primary:
                    continue
            pt_context[pt_code] = {
                "primary": is_primary,
                "hlt_code": str(row.get("hlt_code", "")),
                "hlgt_code": str(row.get("hlgt_code", "")),
                "soc_code": str(row.get("soc_code", "")),
                "hlt_name": row.get("hlt_name", ""),
                "hlgt_name": row.get("hlgt_name", ""),
                "soc_name": row.get("soc_name", ""),
            }
        return pt_context

    def get_level(self, code: str) -> Optional[str]:
        return self.code_to_level.get(str(code))

    def resolve(self, code: str, level: Optional[str] = None) -> HierarchyItem:
        code = str(code)
        level = level or self.get_level(code)

        llt_info = None
        pt_info = None
        hlt_info = None
        hlgt_info = None
        soc_info = None

        if level == "LLT":
            llt_info = _entry(code, self.llt_terms.get(code), "LLT")
            pt_code = self.llt_to_pt.get(code)
            if pt_code:
                pt_info, hlt_info, hlgt_info, soc_info = self._resolve_pt_chain(pt_code)
        elif level == "PT":
            pt_info, hlt_info, hlgt_info, soc_info = self._resolve_pt_chain(code)
        elif level == "HLT":
            hlt_info = _entry(code, self.hlt_terms.get(code), "HLT")
            hlgt_code = self._find_parent_code(level="HLT", code=code)
            if hlgt_code:
                hlgt_info = _entry(hlgt_code, self.hlgt_terms.get(hlgt_code), "HLGT")
                soc_code = self._find_parent_code(level="HLGT", code=hlgt_code)
                if soc_code:
                    soc_info = _entry(soc_code, self.soc_terms.get(soc_code), "SOC")
        elif level == "HLGT":
            hlgt_info = _entry(code, self.hlgt_terms.get(code), "HLGT")
            soc_code = self._find_parent_code(level="HLGT", code=code)
            if soc_code:
                soc_info = _entry(soc_code, self.soc_terms.get(soc_code), "SOC")
        elif level == "SOC":
            soc_info = _entry(code, self.soc_terms.get(code), "SOC")

        return HierarchyItem(llt=llt_info, pt=pt_info, hlt=hlt_info, hlgt=hlgt_info, soc=soc_info)

    def _resolve_pt_chain(
        self, pt_code: str
    ) -> tuple[Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        pt_code = str(pt_code)
        pt_info = _entry(pt_code, self.pt_terms.get(pt_code), "PT")
        context = self.pt_context.get(pt_code, {})

        hlt_code = context.get("hlt_code", "")
        hlgt_code = context.get("hlgt_code", "")
        soc_code = context.get("soc_code", "")

        hlt_info = _entry(hlt_code, self.hlt_terms.get(hlt_code), "HLT") if hlt_code else None
        hlgt_info = _entry(hlgt_code, self.hlgt_terms.get(hlgt_code), "HLGT") if hlgt_code else None
        soc_info = _entry(soc_code, self.soc_terms.get(soc_code), "SOC") if soc_code else None

        return pt_info, hlt_info, hlgt_info, soc_info

    def _find_parent_code(self, *, level: str, code: str) -> Optional[str]:
        code = str(code)
        if level == "HLT":
            for context in self.pt_context.values():
                if context.get("hlt_code") == code:
                    return context.get("hlgt_code")
        if level == "HLGT":
            for context in self.pt_context.values():
                if context.get("hlgt_code") == code:
                    return context.get("soc_code")
        return None
