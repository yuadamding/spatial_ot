from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "pipelines" / "visium_hd_p2_crc"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_spaceranger_count = _load_module(
    "run_spaceranger_count",
    PIPELINE_DIR / "07_run_spaceranger_count.py",
)
generate_cohort_inputs = _load_module(
    "generate_cohort_inputs",
    PIPELINE_DIR / "28_generate_cohort_p2_like_inputs.py",
)


def test_detect_existing_run_status_marks_completed_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        run_dir = runs_dir / "example_run"
        (run_dir / "outs").mkdir(parents=True)
        (run_dir / "_finalstate").write_text("complete\n", encoding="utf-8")
        (run_dir / "_timestamp").write_text(
            "start: 2026-04-16 15:06:17\nend: 2026-04-16 19:55:48\n", encoding="utf-8"
        )

        status = run_spaceranger_count.detect_existing_run_status(
            runs_dir, "example_run"
        )

        assert status["completed"] is True
        assert status["finalstate_exists"] is True
        assert status["outs_dir_exists"] is True
        assert status["timestamp_preview"] == [
            "start: 2026-04-16 15:06:17",
            "end: 2026-04-16 19:55:48",
        ]


def test_run_spaceranger_count_payload_marks_existing_completed_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        runs_dir = root_dir / "runs"
        run_id = "example_run"
        run_dir = runs_dir / run_id
        (run_dir / "outs").mkdir(parents=True)
        (run_dir / "_finalstate").write_text("complete\n", encoding="utf-8")
        (run_dir / "_timestamp").write_text(
            "start: 2026-04-16 15:06:17\nend: 2026-04-16 19:55:48\n", encoding="utf-8"
        )

        fastqs_dir = root_dir / "fastqs"
        fastqs_dir.mkdir()
        fastq_name = "DemoSample_S1_L001_R1_001.fastq.gz"
        (fastqs_dir / fastq_name).write_text("", encoding="utf-8")
        (fastqs_dir / ".extraction_complete.json").write_text(
            '{"expected_fastq_count": 1, "expected_fastq_names": ["DemoSample_S1_L001_R1_001.fastq.gz"]}\n',
            encoding="utf-8",
        )
        transcriptome = root_dir / "ref"
        transcriptome.mkdir()
        probe_set = root_dir / "probe.csv"
        slidefile = root_dir / "slide.vlf"
        cytaimage = root_dir / "cyta.tif"
        image = root_dir / "image.btf"
        for path in [probe_set, slidefile, cytaimage, image]:
            path.write_text("", encoding="utf-8")

        generated_shell = root_dir / "generated" / "run.sh"
        output_json = root_dir / "reports" / "spaceranger.json"
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_DIR / "07_run_spaceranger_count.py"),
                "--spaceranger-bin",
                "/bin/true",
                "--transcriptome",
                str(transcriptome),
                "--probe-set",
                str(probe_set),
                "--fastqs-dir",
                str(fastqs_dir),
                "--sample-prefix",
                "DemoSample",
                "--slide",
                "H1-DEMO",
                "--area",
                "A1",
                "--slidefile",
                str(slidefile),
                "--cytaimage",
                str(cytaimage),
                "--image",
                str(image),
                "--run-id",
                run_id,
                "--runs-dir",
                str(runs_dir),
                "--generated-shell",
                str(generated_shell),
                "--output-json",
                str(output_json),
                "--localcores",
                "4",
                "--localmem",
                "8",
            ],
            check=True,
        )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["status"] == "already_complete"
        assert payload["execute_performed"] is False
        assert payload["existing_run"]["completed"] is True


def test_detect_existing_run_status_marks_stale_lock_when_unheld() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        run_dir = runs_dir / "example_run"
        run_dir.mkdir(parents=True)
        (run_dir / "_lock").write_text("2026-04-20 13:19:35\n", encoding="utf-8")

        original_lock_holders = run_spaceranger_count.find_lock_holders
        original_run_processes = run_spaceranger_count.find_run_processes
        try:
            run_spaceranger_count.find_lock_holders = lambda path: []
            run_spaceranger_count.find_run_processes = lambda run_id: []
            status = run_spaceranger_count.detect_existing_run_status(
                runs_dir, "example_run"
            )
        finally:
            run_spaceranger_count.find_lock_holders = original_lock_holders
            run_spaceranger_count.find_run_processes = original_run_processes

        assert status["lock_exists"] is True
        assert status["running"] is False
        assert status["stale_lock"] is True


def test_run_spaceranger_count_execute_removes_stale_lock_before_resume() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        runs_dir = root_dir / "runs"
        run_id = "example_run"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "_lock").write_text("2026-04-20 13:19:35\n", encoding="utf-8")

        fastqs_dir = root_dir / "fastqs"
        fastqs_dir.mkdir()
        fastq_name = "DemoSample_S1_L001_R1_001.fastq.gz"
        (fastqs_dir / fastq_name).write_text("", encoding="utf-8")
        (fastqs_dir / ".extraction_complete.json").write_text(
            '{"expected_fastq_count": 1, "expected_fastq_names": ["DemoSample_S1_L001_R1_001.fastq.gz"]}\n',
            encoding="utf-8",
        )
        transcriptome = root_dir / "ref"
        transcriptome.mkdir()
        probe_set = root_dir / "probe.csv"
        slidefile = root_dir / "slide.vlf"
        cytaimage = root_dir / "cyta.tif"
        image = root_dir / "image.btf"
        for path in [probe_set, slidefile, cytaimage, image]:
            path.write_text("", encoding="utf-8")

        generated_shell = root_dir / "generated" / "run.sh"
        output_json = root_dir / "reports" / "spaceranger.json"
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_DIR / "07_run_spaceranger_count.py"),
                "--spaceranger-bin",
                "/bin/true",
                "--transcriptome",
                str(transcriptome),
                "--probe-set",
                str(probe_set),
                "--fastqs-dir",
                str(fastqs_dir),
                "--sample-prefix",
                "DemoSample",
                "--slide",
                "H1-DEMO",
                "--area",
                "A1",
                "--slidefile",
                str(slidefile),
                "--cytaimage",
                str(cytaimage),
                "--image",
                str(image),
                "--run-id",
                run_id,
                "--runs-dir",
                str(runs_dir),
                "--generated-shell",
                str(generated_shell),
                "--output-json",
                str(output_json),
                "--localcores",
                "4",
                "--localmem",
                "8",
                "--execute",
            ],
            check=True,
        )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["status"] == "executed"
        assert payload["stale_lock_removed"] is True
        assert (run_dir / "_lock").exists() is False


def test_generate_methoddev_if_possible_allows_cell_marker_without_azimuth() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        sample_dir = root_dir / "spatial" / "visium_hd" / "sample_demo_crc"
        pipeline_dir = root_dir / "pipelines" / "visium_hd_p2_crc"
        pipeline_dir.mkdir(parents=True)
        work_dir = root_dir / "work" / "visium_hd_demo_crc"
        run_id = "run123"
        segmented = work_dir / "runs" / run_id / "outs" / "segmented_outputs"
        segmented.mkdir(parents=True)
        (segmented / "filtered_feature_cell_matrix.h5").write_text("", encoding="utf-8")
        (segmented / "cell_segmentations.geojson").write_text("{}", encoding="utf-8")

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: Path | None = None):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        original_run = generate_cohort_inputs._run
        try:
            generate_cohort_inputs._run = fake_run
            result = generate_cohort_inputs._generate_methoddev_if_possible(
                pipeline_dir=pipeline_dir,
                sample_dir=sample_dir,
                sample_prefix="DemoSample",
                slide="H1-DEMO",
                area="A1",
                run_id=run_id,
                force_outputs=False,
                root_dir=root_dir,
            )
        finally:
            generate_cohort_inputs._run = original_run

        assert result["cell_marker_inputs_available"] is True
        assert result["methoddev_inputs_available"] is False
        assert "cell_marker_h5ad" in result
        assert "methoddev_h5ad" not in result
        assert len(calls) == 1
        assert calls[0][1].endswith("24_make_cell_polygon_marker_gene_umap3d.py")


def test_sample_summary_treats_already_running_as_in_progress() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        sample_dir = root_dir / "spatial" / "visium_hd" / "sample_demo_crc"
        (sample_dir / "input").mkdir(parents=True)
        (sample_dir / "input" / "Demo_alignment_file.json").write_text(
            json.dumps({"serialNumber": "H1-DEMO", "area": "A1"}),
            encoding="utf-8",
        )

        original_prepare_workspace = generate_cohort_inputs._prepare_workspace
        original_extract_fastqs = generate_cohort_inputs._extract_fastqs
        original_generate_spaceranger_command = (
            generate_cohort_inputs._generate_spaceranger_command
        )
        original_generate_methoddev = (
            generate_cohort_inputs._generate_methoddev_if_possible
        )
        original_sample_prefix = generate_cohort_inputs._sample_prefix
        try:
            generate_cohort_inputs._prepare_workspace = lambda *args, **kwargs: {
                "work_dir": str(root_dir / "work"),
                "report_dir": str(root_dir / "reports"),
                "generated_dir": str(root_dir / "generated"),
                "log_dir": str(root_dir / "logs"),
                "fastq_extract_dir": str(root_dir / "fastqs"),
                "public_baseline_dir": str(root_dir / "public"),
                "runs_dir": str(root_dir / "runs"),
                "exports_dir": str(root_dir / "exports"),
            }
            generate_cohort_inputs._extract_fastqs = lambda *args, **kwargs: {
                "status": "ok"
            }
            generate_cohort_inputs._generate_spaceranger_command = (
                lambda *args, **kwargs: (
                    "run123",
                    {"status": "already_running"},
                )
            )
            generate_cohort_inputs._generate_methoddev_if_possible = (
                lambda *args, **kwargs: {}
            )
            generate_cohort_inputs._sample_prefix = lambda sample_input_dir: "Demo"

            result = generate_cohort_inputs._sample_summary(
                root_dir=root_dir,
                pipeline_dir=root_dir / "pipelines" / "visium_hd_p2_crc",
                sample_dir=sample_dir,
                execute_spaceranger=True,
                force_outputs=False,
                localcores=4,
                localmem=8,
            )
        finally:
            generate_cohort_inputs._prepare_workspace = original_prepare_workspace
            generate_cohort_inputs._extract_fastqs = original_extract_fastqs
            generate_cohort_inputs._generate_spaceranger_command = (
                original_generate_spaceranger_command
            )
            generate_cohort_inputs._generate_methoddev_if_possible = (
                original_generate_methoddev
            )
            generate_cohort_inputs._sample_prefix = original_sample_prefix

        assert result["status"] == "spaceranger_running_or_ready"


def test_move_spatial_ot_inputs_run_once_copies_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        work_export_dir = root_dir / "work" / "visium_hd_demo_crc" / "exports"
        work_export_dir.mkdir(parents=True)
        src = work_export_dir / "demo_crc_cells_marker_genes_umap3d.h5ad"
        src.write_bytes(b"demo-payload")
        dest_dir = root_dir / "staged"
        script = PIPELINE_DIR / "29_move_spatial_ot_inputs.sh"
        env = os.environ.copy()
        env["RUN_ONCE"] = "1"
        env["ROOT"] = str(root_dir)
        env["SLEEP_SECONDS"] = "1"

        subprocess.run(
            ["bash", str(script), str(dest_dir)],
            check=True,
            cwd=root_dir,
            env=env,
        )

        dest = dest_dir / src.name
        assert dest.read_bytes() == b"demo-payload"


def test_step25_wrapper_defaults_to_non_rgb_h5ad() -> None:
    wrapper = (PIPELINE_DIR / "25_run_cell_spatial_niche_analysis.sh").read_text(
        encoding="utf-8"
    )
    assert "p2_crc_cells_marker_genes_umap3d.h5ad" in wrapper
    assert "p2_crc_cells_marker_genes_umap3d_rgb.h5ad" not in wrapper
