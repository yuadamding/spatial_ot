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


def load_programs(path: str | Path | None) -> tuple[Program, ...]:
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text())
    programs = []
    for item in payload.get("programs", []):
        programs.append(
            Program(
                name=item["name"],
                self_genes=tuple(item.get("self_genes", [])),
                neighborhood_genes=tuple(item.get("neighborhood_genes", [])),
                source_genes=tuple(item.get("source_genes", [])),
                receiver_genes=tuple(item.get("receiver_genes", [])),
                target_genes=tuple(item.get("target_genes", [])),
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
