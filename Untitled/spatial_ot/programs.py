from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Program:
    name: str
    self_genes: tuple[str, ...]
    neighborhood_genes: tuple[str, ...]
    source_genes: tuple[str, ...] = ()
    receiver_genes: tuple[str, ...] = ()
    target_genes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramLibrary:
    programs: tuple[Program, ...]
    self_mask: np.ndarray
    neighborhood_mask: np.ndarray
    source_mask: np.ndarray
    receiver_target_mask: np.ndarray
    program_names: tuple[str, ...]

    @property
    def n_programs(self) -> int:
        return len(self.programs)

    @property
    def communication_program_indices(self) -> list[int]:
        indices = []
        for idx, program in enumerate(self.programs):
            if program.source_genes and (program.receiver_genes or program.target_genes):
                indices.append(idx)
        return indices


def _validate_gene_list(value, field_name: str, program_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"Program '{program_name}' field '{field_name}' must be a list of gene names.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for gene in value:
        gene_str = str(gene).strip()
        if not gene_str or gene_str in seen:
            continue
        cleaned.append(gene_str)
        seen.add(gene_str)
    return tuple(cleaned)


def load_programs(path: str | Path | None) -> tuple[Program, ...]:
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("Program JSON must contain a top-level object.")
    unknown_top = sorted(set(payload) - {"programs"})
    if unknown_top:
        raise KeyError(f"Unknown top-level keys in program JSON: {', '.join(unknown_top)}")
    items = payload.get("programs", [])
    if not isinstance(items, list):
        raise TypeError("'programs' must be a list of program definitions.")
    programs = []
    seen_names: set[str] = set()
    allowed_fields = {"name", "self_genes", "neighborhood_genes", "source_genes", "receiver_genes", "target_genes"}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Program entry at index {idx} must be an object.")
        unknown_fields = sorted(set(item) - allowed_fields)
        if unknown_fields:
            raise KeyError(f"Program entry {idx} has unknown fields: {', '.join(unknown_fields)}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Program entry {idx} is missing a non-empty 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate program name '{name}' in program JSON.")
        seen_names.add(name)
        self_genes = _validate_gene_list(item.get("self_genes", []), "self_genes", name)
        neighborhood_genes = _validate_gene_list(item.get("neighborhood_genes", []), "neighborhood_genes", name)
        source_genes = _validate_gene_list(item.get("source_genes", []), "source_genes", name)
        receiver_genes = _validate_gene_list(item.get("receiver_genes", []), "receiver_genes", name)
        target_genes = _validate_gene_list(item.get("target_genes", []), "target_genes", name)
        if not any([self_genes, neighborhood_genes, source_genes, receiver_genes, target_genes]):
            raise ValueError(f"Program '{name}' has no genes in any component.")
        programs.append(
            Program(
                name=name,
                self_genes=self_genes,
                neighborhood_genes=neighborhood_genes,
                source_genes=source_genes,
                receiver_genes=receiver_genes,
                target_genes=target_genes,
            )
        )
    return tuple(programs)


def collect_program_genes(programs: tuple[Program, ...]) -> set[str]:
    genes: set[str] = set()
    for program in programs:
        genes.update(program.self_genes)
        genes.update(program.neighborhood_genes)
        genes.update(program.source_genes)
        genes.update(program.receiver_genes)
        genes.update(program.target_genes)
    return genes


def build_program_library(programs: tuple[Program, ...], gene_names: list[str]) -> ProgramLibrary:
    gene_index = {gene: idx for idx, gene in enumerate(gene_names)}
    n_programs = len(programs)
    n_genes = len(gene_names)
    self_mask = np.zeros((n_programs, n_genes), dtype=np.float32)
    neighborhood_mask = np.zeros((n_programs, n_genes), dtype=np.float32)
    source_mask = np.zeros((n_programs, n_genes), dtype=np.float32)
    receiver_target_mask = np.zeros((n_programs, n_genes), dtype=np.float32)

    for i, program in enumerate(programs):
        for gene in program.self_genes:
            if gene in gene_index:
                self_mask[i, gene_index[gene]] = 1.0
        for gene in program.neighborhood_genes:
            if gene in gene_index:
                neighborhood_mask[i, gene_index[gene]] = 1.0
        for gene in program.source_genes:
            if gene in gene_index:
                source_mask[i, gene_index[gene]] = 1.0
        for gene in (*program.receiver_genes, *program.target_genes):
            if gene in gene_index:
                receiver_target_mask[i, gene_index[gene]] = 1.0

    return ProgramLibrary(
        programs=programs,
        self_mask=self_mask,
        neighborhood_mask=neighborhood_mask,
        source_mask=source_mask,
        receiver_target_mask=receiver_target_mask,
        program_names=tuple(program.name for program in programs),
    )


def score_programs(expression: np.ndarray, gene_names: list[str], library: ProgramLibrary) -> np.ndarray:
    if library.n_programs == 0:
        return np.zeros((expression.shape[0], 0), dtype=np.float32)
    scores = np.zeros((expression.shape[0], library.n_programs), dtype=np.float32)
    standardized = expression - expression.mean(axis=0, keepdims=True)
    denom = expression.std(axis=0, keepdims=True) + 1e-6
    standardized = standardized / denom
    for idx in range(library.n_programs):
        mask = (library.self_mask[idx] + library.neighborhood_mask[idx]) > 0
        if mask.any():
            scores[:, idx] = standardized[:, mask].mean(axis=1)
    return scores


def summarize_program_coverage(programs: tuple[Program, ...], gene_names: list[str]) -> dict:
    gene_set = set(gene_names)
    per_program = []
    total_requested = 0
    total_kept = 0
    total_missing = 0
    for program in programs:
        requested = sorted(
            set(program.self_genes)
            | set(program.neighborhood_genes)
            | set(program.source_genes)
            | set(program.receiver_genes)
            | set(program.target_genes)
        )
        kept = [gene for gene in requested if gene in gene_set]
        missing = [gene for gene in requested if gene not in gene_set]
        total_requested += len(requested)
        total_kept += len(kept)
        total_missing += len(missing)
        per_program.append(
            {
                "name": program.name,
                "requested_gene_count": len(requested),
                "kept_gene_count": len(kept),
                "missing_gene_count": len(missing),
                "kept_genes": kept,
                "missing_genes": missing,
            }
        )
    return {
        "n_programs": len(programs),
        "requested_gene_count_total": total_requested,
        "kept_gene_count_total": total_kept,
        "missing_gene_count_total": total_missing,
        "programs": per_program,
    }
