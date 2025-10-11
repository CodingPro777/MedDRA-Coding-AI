from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import clean_field, ensure_required_files, parse_version_folder


@dataclass
class ParsedMeddraData:
    """Container for parsed MedDRA resources."""

    language: str
    version: str
    terms: pd.DataFrame
    documents: pd.DataFrame
    llt: pd.DataFrame
    pt: pd.DataFrame
    hlt: pd.DataFrame
    hlgt: pd.DataFrame
    soc: pd.DataFrame
    mdhier: pd.DataFrame
    llt_to_pt: Dict[str, str]
    code_to_level: Dict[str, str]


class MeddraParser:
    """Parse MedDRA ASCII resources into structured data frames."""

    REQUIRED_ASC_FILES: Iterable[str] = (
        "llt.asc",
        "pt.asc",
        "hlt.asc",
        "hlgt.asc",
        "soc.asc",
        "mdhier.asc",
    )

    def __init__(self, version_dir: Path):
        self.version_dir = Path(version_dir)
        ensure_required_files(self.version_dir, self.REQUIRED_ASC_FILES)
        meta = parse_version_folder(self.version_dir.name)
        self.language = meta["language"]
        self.version = meta["version"]

    # ------------------------------
    # Public API
    # ------------------------------
    def parse(self) -> ParsedMeddraData:
        """Parse all MedDRA assets for the given version."""
        llt_df = self._load_llt()
        pt_df = self._load_pt()
        hlt_df = self._load_level_file("hlt.asc", ("hlt_code", "hlt_name"))
        hlgt_df = self._load_level_file("hlgt.asc", ("hlgt_code", "hlgt_name"))
        soc_df = self._load_level_file("soc.asc", ("soc_code", "soc_name", "soc_abbrev"))
        mdhier_df = self._load_mdhier()

        terms_df, llt_hierarchy_df = self._build_terms_dataframe(
            llt_df=llt_df,
            pt_df=pt_df,
            hlt_df=hlt_df,
            hlgt_df=hlgt_df,
            soc_df=soc_df,
            mdhier_df=mdhier_df,
        )

        llt_to_pt = dict(zip(llt_df["llt_code"], llt_df["pt_code"]))

        code_to_level: Dict[str, str] = {}
        code_to_level.update({code: "LLT" for code in llt_df["llt_code"]})
        code_to_level.update({code: "PT" for code in pt_df["pt_code"]})
        code_to_level.update({code: "HLT" for code in hlt_df["hlt_code"]})
        code_to_level.update({code: "HLGT" for code in hlgt_df["hlgt_code"]})
        code_to_level.update({code: "SOC" for code in soc_df["soc_code"]})

        return ParsedMeddraData(
            language=self.language,
            version=self.version,
            terms=terms_df,
            documents=self._build_documents_dataframe(
                llt_df=llt_df,
                pt_df=pt_df,
                hlt_df=hlt_df,
                hlgt_df=hlgt_df,
                soc_df=soc_df,
                llt_paths=llt_hierarchy_df,
                language=self.language,
                version=self.version,
            ),
            llt=llt_df,
            pt=pt_df,
            hlt=hlt_df,
            hlgt=hlgt_df,
            soc=soc_df,
            mdhier=mdhier_df,
            llt_to_pt=llt_to_pt,
            code_to_level=code_to_level,
        )

    # ------------------------------
    # Loading helpers
    # ------------------------------
    def _load_llt(self) -> pd.DataFrame:
        df = self._read_asc("llt.asc", ("llt_code", "llt_name", "pt_code"))
        df = df.drop_duplicates(subset=["llt_code"]).reset_index(drop=True)
        return df

    def _load_pt(self) -> pd.DataFrame:
        df = self._read_asc("pt.asc", ("pt_code", "pt_name", "null_1", "hlt_code"))
        df = df.drop(columns=[col for col in df.columns if col.startswith("null_")], errors="ignore")
        df = df.drop_duplicates(subset=["pt_code"]).reset_index(drop=True)
        return df

    def _load_level_file(self, filename: str, columns: Iterable[str]) -> pd.DataFrame:
        df = self._read_asc(filename, columns)
        primary_col = next(iter(columns))
        df = df.drop_duplicates(subset=[primary_col]).reset_index(drop=True)
        return df

    def _load_mdhier(self) -> pd.DataFrame:
        columns = (
            "llt_code",
            "hlt_code",
            "hlgt_code",
            "soc_code",
            "llt_name",
            "hlt_name",
            "hlgt_name",
            "soc_name",
            "soc_abbrev",
            "soc_sorting",
            "soc_code_dup",
            "primary_soc_flag",
            "unused",
        )
        df = self._read_asc("mdhier.asc", columns)
        df = df.drop(columns=["soc_sorting", "soc_code_dup", "unused"], errors="ignore")
        df = df.drop_duplicates(subset=["llt_code"]).reset_index(drop=True)
        return df

    def _read_asc(self, filename: str, columns: Iterable[str]) -> pd.DataFrame:
        """Read a pipe-delimited MedDRA ASCII file with the expected columns."""
        path = self.version_dir / filename
        requested_columns: List[str] = list(columns)
        records: List[Dict[str, str]] = []

        with path.open("r", encoding="latin1") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("$")
                if not parts or not parts[0]:
                    continue
                record: Dict[str, str] = {}
                for idx, column in enumerate(requested_columns):
                    value = parts[idx] if idx < len(parts) else ""
                    record[column] = clean_field(value)
                records.append(record)

        df = pd.DataFrame(records, columns=requested_columns)
        for column in requested_columns:
            df[column] = df[column].astype(str).fillna("").map(clean_field)
        return df

    # ------------------------------
    # Data assembly
    # ------------------------------
    def _build_terms_dataframe(
        self,
        *,
        llt_df: pd.DataFrame,
        pt_df: pd.DataFrame,
        hlt_df: pd.DataFrame,
        hlgt_df: pd.DataFrame,
        soc_df: pd.DataFrame,
        mdhier_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge parsed ASCII files into a unified table."""
        llt_terms = llt_df.rename(
            columns={"llt_code": "code", "llt_name": "term", "pt_code": "parent_code"}
        )
        llt_terms = llt_terms.assign(level="LLT", parent_level="PT")

        # Derive PT → HLT mappings using mdhier (primary SOC rows preferred)
        primary_mask = mdhier_df["primary_soc_flag"].str.upper() == "Y"
        primary_rows = mdhier_df.loc[primary_mask]
        primary_hierarchy = (
            primary_rows[["llt_code", "hlt_code", "hlgt_code", "soc_code"]]
            .merge(llt_df[["llt_code", "pt_code"]], on="llt_code", how="left")
            .dropna(subset=["pt_code"])
        )
        if primary_hierarchy.empty:
            primary_hierarchy = (
                mdhier_df[["llt_code", "hlt_code", "hlgt_code", "soc_code"]]
                .merge(llt_df[["llt_code", "pt_code"]], on="llt_code", how="left")
                .dropna(subset=["pt_code"])
            )
        pt_to_hlt = primary_hierarchy.drop_duplicates(subset=["pt_code"])

        pt_terms = pt_df.merge(pt_to_hlt[["pt_code", "hlt_code"]], how="left", on="pt_code")
        pt_terms = pt_terms.rename(
            columns={"pt_code": "code", "pt_name": "term", "hlt_code": "parent_code"}
        )
        pt_terms = pt_terms.assign(level="PT", parent_level="HLT")

        hlt_terms = hlt_df.merge(
            primary_hierarchy[["hlt_code", "hlgt_code"]].drop_duplicates(), how="left", on="hlt_code"
        )
        hlt_terms = hlt_terms.rename(
            columns={"hlt_code": "code", "hlt_name": "term", "hlgt_code": "parent_code"}
        )
        hlt_terms = hlt_terms.assign(level="HLT", parent_level="HLGT")

        hlgt_terms = hlgt_df.merge(
            primary_hierarchy[["hlgt_code", "soc_code"]].drop_duplicates(), how="left", on="hlgt_code"
        )
        hlgt_terms = hlgt_terms.rename(
            columns={"hlgt_code": "code", "hlgt_name": "term", "soc_code": "parent_code"}
        )
        hlgt_terms = hlgt_terms.assign(level="HLGT", parent_level="SOC")

        soc_terms = soc_df.rename(
            columns={"soc_code": "code", "soc_name": "term", "soc_abbrev": "soc_abbrev"}
        )
        soc_terms = soc_terms.assign(level="SOC", parent_code="", parent_level="")

        combined_df = pd.concat(
            [llt_terms, pt_terms, hlt_terms, hlgt_terms, soc_terms], ignore_index=True, axis=0
        )
        combined_df["code"] = combined_df["code"].astype(str)
        combined_df["term"] = combined_df["term"].astype(str)
        combined_df["level"] = combined_df["level"].astype(str)
        combined_df["parent_code"] = combined_df["parent_code"].fillna("").astype(str)
        combined_df["parent_level"] = combined_df["parent_level"].fillna("").astype(str)
        if "soc_abbrev" in combined_df.columns:
            combined_df["soc_abbrev"] = combined_df["soc_abbrev"].fillna("").astype(str)

        pt_lookup = dict(zip(pt_df["pt_code"].astype(str), pt_df["pt_name"]))
        hlt_lookup = dict(zip(hlt_df["hlt_code"].astype(str), hlt_df["hlt_name"]))
        hlgt_lookup = dict(zip(hlgt_df["hlgt_code"].astype(str), hlgt_df["hlgt_name"]))
        soc_lookup = dict(zip(soc_df["soc_code"].astype(str), soc_df["soc_name"]))

        llt_path_map: Dict[str, Dict[str, str]] = {}
        pt_path_map: Dict[str, Dict[str, str]] = {}
        hlt_path_map: Dict[str, Dict[str, str]] = {}
        hlgt_path_map: Dict[str, Dict[str, str]] = {}
        for _, row in primary_hierarchy.iterrows():
            llt_code = str(row.get("llt_code", ""))
            pt_code = str(row.get("pt_code", ""))
            hlt_code = str(row.get("hlt_code", ""))
            hlgt_code = str(row.get("hlgt_code", ""))
            soc_code = str(row.get("soc_code", ""))

            if llt_code and llt_code not in llt_path_map:
                llt_path_map[llt_code] = {
                    "pt_code": pt_code,
                    "hlt_code": hlt_code,
                    "hlgt_code": hlgt_code,
                    "soc_code": soc_code,
                }
            if pt_code and pt_code not in pt_path_map:
                pt_path_map[pt_code] = {
                    "hlt_code": hlt_code,
                    "hlgt_code": hlgt_code,
                    "soc_code": soc_code,
                }
            if hlt_code and hlt_code not in hlt_path_map:
                hlt_path_map[hlt_code] = {
                    "hlgt_code": hlgt_code,
                    "soc_code": soc_code,
                }
            if hlgt_code and hlgt_code not in hlgt_path_map:
                hlgt_path_map[hlgt_code] = {"soc_code": soc_code}

        def build_embedding_text(row: pd.Series) -> str:
            level = row.get("level", "")
            term = row.get("term", "")
            code = row.get("code", "")
            pieces = [f"{level}: {term}"] if term else []

            def append_level(label: str, code_value: str, lookup: Dict[str, str]) -> None:
                if not code_value:
                    return
                name = lookup.get(code_value)
                if name:
                    pieces.append(f"{label}: {name}")

            if level == "LLT":
                path = llt_path_map.get(code, {})
                pt_code = path.get("pt_code") or row.get("parent_code", "")
                append_level("PT", pt_code, pt_lookup)
                hlt_code = path.get("hlt_code") or pt_path_map.get(pt_code, {}).get("hlt_code", "")
                append_level("HLT", hlt_code, hlt_lookup)
                hlgt_code = path.get("hlgt_code") or hlt_path_map.get(hlt_code, {}).get("hlgt_code", "")
                append_level("HLGT", hlgt_code, hlgt_lookup)
                soc_code = path.get("soc_code") or hlgt_path_map.get(hlgt_code, {}).get("soc_code", "")
                append_level("SOC", soc_code, soc_lookup)
            elif level == "PT":
                path = pt_path_map.get(code, {})
                append_level("HLT", path.get("hlt_code", ""), hlt_lookup)
                append_level("HLGT", path.get("hlgt_code", ""), hlgt_lookup)
                append_level("SOC", path.get("soc_code", ""), soc_lookup)
            elif level == "HLT":
                path = hlt_path_map.get(code, {})
                append_level("HLGT", path.get("hlgt_code", ""), hlgt_lookup)
                append_level("SOC", path.get("soc_code", ""), soc_lookup)
            elif level == "HLGT":
                path = hlgt_path_map.get(code, {})
                append_level("SOC", path.get("soc_code", ""), soc_lookup)

            return " | ".join(pieces) if pieces else term

        combined_df["embedding_text"] = combined_df.apply(build_embedding_text, axis=1)
        combined_df["display_text"] = combined_df["embedding_text"]

        combined_df = combined_df.drop_duplicates(subset=["code", "level"]).reset_index(drop=True)
        llt_hierarchy = primary_hierarchy.drop_duplicates(subset=["llt_code"])
        return combined_df, llt_hierarchy

    def _build_documents_dataframe(
        self,
        *,
        llt_df: pd.DataFrame,
        pt_df: pd.DataFrame,
        hlt_df: pd.DataFrame,
        hlgt_df: pd.DataFrame,
        soc_df: pd.DataFrame,
        llt_paths: pd.DataFrame,
        language: str,
        version: str,
    ) -> pd.DataFrame:
        llt_to_terms = {str(code): str(name) for code, name in zip(llt_df["llt_code"], llt_df["llt_name"])}
        pt_lookup = {str(code): str(name) for code, name in zip(pt_df["pt_code"], pt_df["pt_name"])}
        hlt_lookup = {str(code): str(name) for code, name in zip(hlt_df["hlt_code"], hlt_df["hlt_name"])}
        hlgt_lookup = {str(code): str(name) for code, name in zip(hlgt_df["hlgt_code"], hlgt_df["hlgt_name"])}
        soc_lookup = {str(code): str(name) for code, name in zip(soc_df["soc_code"], soc_df["soc_name"])}

        records: List[Dict[str, str]] = []
        seen_llt: set[str] = set()

        for _, row in llt_paths.iterrows():
            llt_code = str(row.get("llt_code", ""))
            pt_code = str(row.get("pt_code", ""))
            hlt_code = str(row.get("hlt_code", ""))
            hlgt_code = str(row.get("hlgt_code", ""))
            soc_code = str(row.get("soc_code", ""))

            if not llt_code or llt_code in seen_llt:
                continue
            seen_llt.add(llt_code)

            llt_name = llt_to_terms.get(llt_code, "")
            pt_name = pt_lookup.get(pt_code, "")
            hlt_name = hlt_lookup.get(hlt_code, "")
            hlgt_name = hlgt_lookup.get(hlgt_code, "")
            soc_name = soc_lookup.get(soc_code, "")

            content_lines = [
                f"LLT: {llt_name} ({llt_code})",
                f"PT: {pt_name} ({pt_code})",
                f"HLT: {hlt_name} ({hlt_code})",
                f"HLGT: {hlgt_name} ({hlgt_code})",
                f"SOC: {soc_name} ({soc_code})",
            ]

            records.append(
                {
                    "doc_id": llt_code,
                    "llt_code": llt_code,
                    "llt_term": llt_name,
                    "pt_code": pt_code,
                    "pt_term": pt_name,
                    "hlt_code": hlt_code,
                    "hlt_term": hlt_name,
                    "hlgt_code": hlgt_code,
                    "hlgt_term": hlgt_name,
                    "soc_code": soc_code,
                    "soc_term": soc_name,
                    "level": "LLT",
                    "term": llt_name,
                    "document_text": "\n".join(content_lines),
                    "display_text": " | ".join(content_lines),
                    "language": language,
                    "version": version,
                }
            )

        documents_df = pd.DataFrame(records)
        if documents_df.empty:
            return documents_df

        documents_df = documents_df.fillna("")
        return documents_df
