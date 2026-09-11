"""Create a Latin-hypercube population for the ventricular example.

The script only creates the database files by default. Pass ``--run`` to also
execute every simulation (which can take a long time).

Example
-------
python demos/create_ventricular_database.py \
    demos/ventricular_parameter_ranges.json --samples 20 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_ranges(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Read and validate ``{"parameter": [minimum, maximum]}`` JSON."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("The ranges file must be a non-empty JSON object")

    names: List[str] = []
    lower: List[float] = []
    upper: List[float] = []
    for name, limits in data.items():
        if (
            not isinstance(name, str)
            or not isinstance(limits, list)
            or len(limits) != 2
            or not all(isinstance(value, (int, float)) for value in limits)
        ):
            raise ValueError(f"{name!r} must have the form [minimum, maximum]")
        lo, hi = map(float, limits)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"{name!r} requires finite values with minimum < maximum")
        names.append(name)
        lower.append(lo)
        upper.append(hi)

    return names, np.asarray(lower), np.asarray(upper)


def latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    """Return a reproducible random Latin-hypercube design in [0, 1)."""
    if samples < 1:
        raise ValueError("--samples must be at least 1")

    rng = np.random.default_rng(seed)
    design = np.empty((samples, dimensions))
    for column in range(dimensions):
        # One random point in every equally sized interval, independently
        # shuffled in each dimension.
        design[:, column] = (rng.permutation(samples) + rng.random(samples)) / samples
    return design


def ventricular_config(root: Path, member_dir: Path, factors_file: Path) -> Dict[str, object]:
    """Build a complete Config dictionary for one LV simulation."""
    geometry_dir = root / "demos" / "geometries"
    return {
        "outdir": str(member_dir.resolve()),
        "outfilename": "results.h5",
        "geometry_path": str((geometry_dir / "lv_ellipsoid.h5").resolve()),
        "geometry_schema_path": str((geometry_dir / "lv_ellipsoid.json").resolve()),
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


def create_database(ranges_path: Path, outdir: Path, samples: int, seed: int) -> List[Path]:
    """Generate population-factor and simulation-config files."""
    names, lower, upper = load_ranges(ranges_path)
    unit_design = latin_hypercube(samples, len(names), seed)
    values = lower + unit_design * (upper - lower)

    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    config_paths: List[Path] = []

    with (outdir / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *names])

        for index, row in enumerate(values):
            model_name = f"m{index:04d}"
            member_dir = outdir / model_name
            member_dir.mkdir(exist_ok=True)

            factors = {name: float(value) for name, value in zip(names, row)}
            factors_path = member_dir / "population_factors.json"
            factors_path.write_text(json.dumps(factors, indent=2) + "\n")

            config_path = member_dir / "config.json"
            config = ventricular_config(root, member_dir, factors_path)
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            config_paths.append(config_path)
            writer.writerow([model_name, *(factors[name] for name in names)])

    metadata = {
        "method": "random Latin hypercube",
        "samples": samples,
        "seed": seed,
        "ranges_file": str(ranges_path.resolve()),
        "parameters": names,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return config_paths


def run_database(config_paths: List[Path]) -> None:
    """Run all generated simulations sequentially."""
    import simcardems

    for index, config_path in enumerate(config_paths, start=1):
        print(f"Running {config_path.parent.name} ({index}/{len(config_paths)})")
        data = json.loads(config_path.read_text())
        config = simcardems.Config(**data)
        runner = simcardems.Runner(config)
        runner.solve(
            T=config.T,
            save_freq=config.save_freq,
            show_progress_bar=config.show_progress_bar,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, default=Path("simcardems/demos/ventricular_parameter_ranges.json"), help="JSON file containing parameter min/max ranges")
    parser.add_argument("--samples", "-n", type=int, default=25, help="number of models (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("ventricular_database"),
        help="database directory (default: ventricular_database)",
    )
    parser.add_argument("--run", action="store_false", help="run simulations after generating files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = create_database(args.ranges, args.outdir, args.samples, args.seed)
    print(f"Created {len(config_paths)} models in {args.outdir.resolve()}")
    if args.run:
        run_database(config_paths)


if __name__ == "__main__":
    main()
