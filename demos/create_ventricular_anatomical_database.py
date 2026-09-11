"""Create a SimCardEMS LV population with cellular and anatomical variability.

Each population member receives independently sampled cellular parameters and
its own idealized left-ventricular geometry. Geometry generation uses the
``cardiac-geometries`` command-line program.

Examples
--------
Create geometries and simulation configurations::

    python demos/create_ventricular_anatomical_database.py --samples 25

Generate the files and then run all simulations sequentially::

    python demos/create_ventricular_anatomical_database.py --samples 25 --run

Use ``--skip-geometries`` only to inspect the sampled design without invoking
the mesher; the resulting simulation configs cannot be run until their
geometries have been generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


SUPPORTED_ANATOMY = {
    "r_short_endo",
    "r_short_epi",
    "r_long_endo",
    "r_long_epi",
    "fiber_angle_endo",
    "fiber_angle_epi",
}


def _read_ranges(section: object, name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if not isinstance(section, dict) or not section:
        raise ValueError(f"{name!r} must be a non-empty JSON object")

    names: List[str] = []
    lower: List[float] = []
    upper: List[float] = []
    for parameter, limits in section.items():
        if (
            not isinstance(parameter, str)
            or not isinstance(limits, list)
            or len(limits) != 2
            or not all(isinstance(value, (int, float)) for value in limits)
        ):
            raise ValueError(f"{name}.{parameter} must have the form [minimum, maximum]")
        lo, hi = map(float, limits)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"{name}.{parameter} requires finite minimum < maximum")
        names.append(parameter)
        lower.append(lo)
        upper.append(hi)
    return names, np.asarray(lower), np.asarray(upper)


def load_design_config(
    path: Path,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Dict[str, object], Dict[str, str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("The design config must be a JSON object")

    cellular_names, cellular_lo, cellular_hi = _read_ranges(data.get("cellular"), "cellular")
    anatomy_names, anatomy_lo, anatomy_hi = _read_ranges(data.get("anatomy"), "anatomy")
    unknown = set(anatomy_names) - SUPPORTED_ANATOMY
    if unknown:
        raise ValueError(f"Unsupported anatomical parameters: {', '.join(sorted(unknown))}")

    geometry = data.get("geometry", {})
    units = data.get("units", {})
    if not isinstance(geometry, dict) or not isinstance(units, dict):
        raise ValueError("'geometry' and 'units' must be JSON objects")

    names = cellular_names + anatomy_names
    lower = np.concatenate((cellular_lo, anatomy_lo))
    upper = np.concatenate((cellular_hi, anatomy_hi))
    return cellular_names, anatomy_names, lower, upper, dict(geometry), dict(units)


def latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    rng = np.random.default_rng(seed)
    design = np.empty((samples, dimensions))
    for column in range(dimensions):
        design[:, column] = (rng.permutation(samples) + rng.random(samples)) / samples
    return design


def validate_anatomy(anatomy: Mapping[str, float]) -> None:
    for inner, outer in (("r_short_endo", "r_short_epi"), ("r_long_endo", "r_long_epi")):
        if inner in anatomy and outer in anatomy and anatomy[outer] <= anatomy[inner]:
            raise ValueError(f"Invalid anatomy: {outer} must exceed {inner}")
    if "r_short_endo" in anatomy and "r_long_endo" in anatomy:
        if anatomy["r_long_endo"] <= anatomy["r_short_endo"]:
            raise ValueError("Invalid anatomy: r_long_endo must exceed r_short_endo")
    if "r_short_epi" in anatomy and "r_long_epi" in anatomy:
        if anatomy["r_long_epi"] <= anatomy["r_short_epi"]:
            raise ValueError("Invalid anatomy: r_long_epi must exceed r_short_epi")


def geometry_command(
    geometry_dir: Path, anatomy: Mapping[str, float], fixed: Mapping[str, object]
) -> List[str]:
    command = ["cardiac-geometries", "create-lv-ellipsoid", str(geometry_dir.resolve())]
    for name, value in anatomy.items():
        command.extend((f"--{name.replace('_', '-')}", str(value)))
    command.extend(("--psize-ref", str(float(fixed.get("psize_ref", 7.0)))))
    command.extend(("--fiber-space", str(fixed.get("fiber_space", "Quadrature_3"))))
    command.append("--create-fibers")
    return command


def generate_geometry(
    geometry_dir: Path, anatomy: Mapping[str, float], fixed: Mapping[str, object]
) -> Tuple[Path, Path]:
    executable = shutil.which("cardiac-geometries")
    if executable is None:
        raise RuntimeError(
            "Could not find 'cardiac-geometries'. Run this script in the SimCardEMS/FEniCS "
            "environment where cardiac-geometries is installed."
        )
    command = geometry_command(geometry_dir, anatomy, fixed)
    command[0] = executable
    subprocess.run(command, check=True)
    geometry_path = geometry_dir / "lv_ellipsoid.h5"
    schema_path = geometry_dir / "lv_ellipsoid.json"
    if not geometry_path.is_file() or not schema_path.is_file():
        raise RuntimeError(f"Geometry generator did not create expected files in {geometry_dir}")
    return geometry_path, schema_path


def ventricular_config(
    member_dir: Path, factors_file: Path, geometry_path: Path, schema_path: Path
) -> Dict[str, object]:
    return {
        "outdir": str(member_dir.resolve()),
        "outfilename": "results.h5",
        "geometry_path": str(geometry_path.resolve()),
        "geometry_schema_path": str(schema_path.resolve()),
        "T": 600.0,
        "dt": 0.05,
        "save_freq": 1,
        "PCL": 600.0,
        "coupling_type": "explicit_ORdmm_Land",
        "mechanics_solve_strategy": "fixed",
        "dt_mech": 1.0,
        "spring": 0.01,
        "popu_factors_file": str(factors_file.resolve()),
        "show_progress_bar": True,
    }


def create_database(
    config_path: Path, outdir: Path, samples: int, seed: int, skip_geometries: bool = False
) -> List[Path]:
    cellular_names, anatomy_names, lower, upper, fixed, units = load_design_config(config_path)
    names = cellular_names + anatomy_names
    values = lower + latin_hypercube(samples, len(names), seed) * (upper - lower)
    outdir.mkdir(parents=True, exist_ok=True)
    config_paths: List[Path] = []

    with (outdir / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *names])
        for index, row in enumerate(values):
            model_name = f"m{index:04d}"
            member_dir = outdir / model_name
            geometry_dir = member_dir / "geometry"
            member_dir.mkdir(exist_ok=True)
            sampled = dict(zip(names, map(float, row)))
            cellular = {name: sampled[name] for name in cellular_names}
            anatomy = {name: sampled[name] for name in anatomy_names}
            validate_anatomy(anatomy)

            factors_path = member_dir / "population_factors.json"
            factors_path.write_text(json.dumps(cellular, indent=2) + "\n")
            (member_dir / "anatomy.json").write_text(json.dumps(anatomy, indent=2) + "\n")

            geometry_path = geometry_dir / "lv_ellipsoid.h5"
            schema_path = geometry_dir / "lv_ellipsoid.json"
            if not skip_geometries:
                geometry_path, schema_path = generate_geometry(geometry_dir, anatomy, fixed)

            simulation_path = member_dir / "config.json"
            simulation_path.write_text(
                json.dumps(
                    ventricular_config(member_dir, factors_path, geometry_path, schema_path), indent=2
                )
                + "\n"
            )
            config_paths.append(simulation_path)
            writer.writerow([model_name, *(sampled[name] for name in names)])

    metadata = {
        "method": "random Latin hypercube",
        "samples": samples,
        "seed": seed,
        "design_config": str(config_path.resolve()),
        "cellular_parameters": cellular_names,
        "anatomical_parameters": anatomy_names,
        "anatomical_units": units,
        "fixed_geometry": fixed,
        "geometries_generated": not skip_geometries,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return config_paths


def run_database(config_paths: Sequence[Path]) -> None:
    import simcardems

    for index, path in enumerate(config_paths, start=1):
        print(f"Running {path.parent.name} ({index}/{len(config_paths)})")
        data = json.loads(path.read_text())
        geometry_path = Path(data["geometry_path"])
        if not geometry_path.is_file():
            raise FileNotFoundError(f"Missing geometry for {path.parent.name}: {geometry_path}")
        config = simcardems.Config(**data)
        runner = simcardems.Runner(config)
        runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=config.show_progress_bar)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("ventricular_anatomical_ranges.json"),
        help="cellular/anatomical design JSON",
    )
    parser.add_argument("--samples", "-n", type=int, default=600, help="number of models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir", type=Path, default=Path("ventricular_anatomical_database")
    )
    parser.add_argument("--skip-geometries", action="store_true")
    parser.add_argument("--run", action="store_true", help="run simulations after generation")
    args = parser.parse_args()
    if args.run and args.skip_geometries:
        parser.error("--run cannot be combined with --skip-geometries")
    return args


def main() -> None:
    args = parse_args()
    paths = create_database(
        args.config, args.outdir, args.samples, args.seed, skip_geometries=args.skip_geometries
    )
    print(f"Created {len(paths)} models in {args.outdir.resolve()}")
    if args.run:
        run_database(paths)


if __name__ == "__main__":
    main()
