# Checks for the code in this folder, to run after any change:
#
#   python code/check_code.py            consistency checks only
#   python code/check_code.py save       the same, then, if everything passed, save the smoke runs' results as the
#                                        baseline (code/cache/check_code_baseline.npz)
#   python code/check_code.py compare    the same, then compare the smoke runs' results with that baseline, bit for
#                                        bit: run save before a change that should leave every result alone, and
#                                        compare after it
#
# 1. Static checks: every module imports; nothing defined goes unused (top-level definitions, NamedTuple fields,
#    config constants); no line is longer than 120 characters; no module that runs during a solve reads a lattice or
#    mesh size from config.
# 2. Smoke runs (smoke_runs): a copy of this folder, with the problem shrunk in the copy's config.py (this folder's
#    config, cache and outputs are never touched), runs both entry points, MAIN_FP.py first for the cross-checks, and
#    every check they print must PASS, none FAIL or SKIPPED.
#
# Exits with status 1 if anything fails. Takes about five minutes. The checks test consistency, not results: a change
# that alters the numbers still passes if every solver still agrees with the others. compare is what catches that.

import ast
import importlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple
import numpy as np

code_folder = pathlib.Path(__file__).resolve().parent
baseline_path = code_folder / "cache" / "check_code_baseline.npz"
entry_points = ["MAIN_FP", "MAIN_Newton"]
# Small enough to run in seconds. The geometry and model switches are set too, so a local study configuration cannot
# change what the smoke runs cover: the square leaves 2 whole matrix partitions on each side, 3 elements per partition.
smoke_run_overrides = {"partition_number_per_side": "9", "element_number_per_side": "27",
                       "strain_increment_count": "12", "timing_repeat_count": "1", "inclusion_shape": '"square"',
                       "inclusion_side_length": "(5/9) * 1e-3", "include_tfa_model": "True",
                       "include_non_partitioned_ls": "True", "matched_stiffness_control": "False"}

class SmokeRun(NamedTuple):
    # The config.py values this run changes on top of smoke_run_overrides; whether every solver must complete every
    # step (abandoning the load path prints no FAIL, so it is caught separately); and, for a configuration that must be
    # refused, the text its error must contain, which then replaces every other requirement.
    overrides: dict
    require_every_step: bool = False
    expected_error: str = None

smoke_runs = {
    "monotonic": SmokeRun({}),
    # The only test of elastic unloading, reverse yielding and passing through zero imposed strain with partitions
    # still yielding, which a monotonic path never reaches. Its hardening is fixed positive, so a local softening
    # study cannot end the path legitimately. 4 increments per quarter still yields at both zero crossings.
    "cyclic": SmokeRun({"load_path_shape": '"cyclic"', "strain_increment_count": "4",
                        "matrix_hardening_modulus": "10e6"}, require_every_step=True),
    # The LS regression gate, where LS must reproduce TFA.
    "matched stiffness": SmokeRun({"matched_stiffness_control": "True"}),
    "LS only": SmokeRun({"include_tfa_model": "False"}),
    # The exact solution every solver must reproduce (laminate.py): a layer 3 partitions thick, loaded in tension and
    # shear so that every interface-normal component, ε11, γ12 and γ13, takes part.
    "laminate": SmokeRun({"inclusion_shape": '"laminate"', "inclusion_side_length": "(3/9) * 1e-3",
                          "max_macro_strain": "np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.01])"}),
    # A circle resolved on the elements, with one partition per element so that no partition is mixed. The three LS
    # lattices are then one lattice, so the fine and coarse LS solvers must agree with partitioned LS.
    "circle": SmokeRun({"inclusion_shape": '"circle"', "inclusion_radius": "0.35e-3", "element_number_per_side": "9"}),
    # The same circle on coarser partitions cuts through some of them, which must stop the run.
    "mixed partitions": SmokeRun({"inclusion_shape": '"circle"', "inclusion_radius": "0.35e-3"},
                                 expected_error="partitions contain both phases"),
}
maximum_line_length = 120
# The modules a solve runs, which must take every size from the arrays they are given, never from config, so that
# solvers on different lattices (partitioned, fine and coarse LS) can run side by side.
online_modules = ["convergence.py", "laminate.py", "lattice_fft.py", "load_path.py", "ls_solvers.py", "material.py",
                  "online_timing.py", "tfa_solvers.py"]

## ------- Static Checks ------- ##

def get_sources():
    return {path.name: path.read_text() for path in sorted(code_folder.glob("*.py"))}

def count_word(sources, word):
    return sum(len(re.findall(rf"\b{re.escape(word)}\b", source)) for source in sources.values())

def check_modules_import():
    sys.path.insert(0, str(code_folder))
    problems = []
    for path in sorted(code_folder.glob("*.py")):
        if path.stem != "check_code":
            try:
                importlib.import_module(path.stem)
            except Exception as error:
                problems.append(f"{path.name} does not import: {type(error).__name__}: {error}")
    return problems

def check_nothing_unused(sources):
    problems = []
    for file_name, source in sources.items():
        if file_name == "check_code.py":
            continue
        for node in ast.parse(source).body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                if file_name == "config.py":
                    # Read elsewhere as config.<name>, or used by config.py's own derived values and checks.
                    used = (any(f"config.{name}" in text for text in sources.values())
                            or count_word({file_name: source}, name) > 1)
                else:
                    used = count_word(sources, name) > 1 or name in ("main", "run_name")
                if not used:
                    problems.append(f"{file_name}: {name} is defined but never used")
            is_named_tuple = isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "NamedTuple" for base in node.bases)
            if is_named_tuple:
                for field in node.body:
                    if isinstance(field, ast.AnnAssign):
                        field_name = field.target.id
                        # Fields are read as .field, or by name in strings, as report.get_fastest_repeat_per_step does.
                        reads = sum(len(re.findall(rf"\.{field_name}\b|[\"']{field_name}[\"']|\b{field_name}=", text))
                                    for text in sources.values())
                        if reads == 0:
                            problems.append(f"{file_name}: field {node.name}.{field_name} is never read")
    return problems

def check_formatting(sources):
    problems = []
    for file_name, source in sources.items():
        for line_number, line in enumerate(source.split("\n"), start=1):
            if len(line) > maximum_line_length:
                problems.append(f"{file_name}:{line_number}: {len(line)} characters")
            if line != line.rstrip():
                problems.append(f"{file_name}:{line_number}: trailing whitespace")
    return problems

def check_no_global_lattice_size(sources):
    return [f"{file_name}:{line_number}: an online module reads a lattice or mesh size from config"
            for file_name in online_modules
            for line_number, line in enumerate(sources[file_name].split("\n"), start=1)
            if re.search(r"config\.(partition_|element_|dof_count)", line)]

## ------- Smoke Runs ------- ##

def run_smoke_test(run_name, smoke_run):
    # Returns the problems found and every array the entry points saved, keyed "<run>/<entry point>/<array>".
    problems = []
    saved_arrays = {}
    with tempfile.TemporaryDirectory() as temporary_folder:
        copy_folder = pathlib.Path(temporary_folder)
        for path in code_folder.glob("*.py"):
            shutil.copy(path, copy_folder / path.name)
        config_path = copy_folder / "config.py"
        config_source = config_path.read_text()
        for name, value in {**smoke_run_overrides, **smoke_run.overrides}.items():
            config_source, replaced = re.subn(rf"^{name}(\s*)= .*$", rf"{name}\g<1>= {value}", config_source, count=1,
                                              flags=re.M)
            if replaced != 1:
                return [f"{run_name} smoke run: config.py has no line setting {name}"], {}
        config_path.write_text(config_source)

        for entry_point in entry_points:
            run = subprocess.run([sys.executable, f"{entry_point}.py"], cwd=copy_folder, capture_output=True, text=True)
            if smoke_run.expected_error is not None:
                refused = run.returncode != 0 and smoke_run.expected_error in run.stderr
                if not refused:
                    problems.append(f"{run_name} {entry_point}.py should have stopped with an error containing "
                                    f"{smoke_run.expected_error!r}")
                print(f"  {entry_point}.py: {'stopped with the expected error' if refused else 'was not refused'}")
                continue
            passed = run.stdout.count("[PASS]")
            not_passed = [line.strip() for line in run.stdout.splitlines() if "[FAIL]" in line or "[SKIPPED]" in line
                          or (smoke_run.require_every_step and "abandoned the load path" in line)]
            if run.returncode != 0:
                problems.append(f"{run_name} {entry_point}.py exited with status {run.returncode}:\n"
                                f"{run.stderr[-1500:]}")
            problems += [f"{run_name} {entry_point}.py: {line}" for line in not_passed]
            print(f"  {entry_point}.py: exit status {run.returncode}, {passed} checks passed, "
                  f"{len(not_passed)} failed, skipped or abandoned")
            # Each entry point's run name is its file name without MAIN_, and it saves only when every check passed.
            results_path = copy_folder / "output" / f"{entry_point.removeprefix('MAIN_')}_results.npz"
            if results_path.exists():
                with np.load(results_path) as saved:
                    saved_arrays.update({f"{run_name}/{entry_point}/{name}": saved[name] for name in saved.files})
    return problems, saved_arrays

## ------- Result Baseline ------- ##

def compare_with_baseline(saved_arrays):
    # Every array the smoke runs saved must equal the baseline's bit for bit. A changed problem/ entry means config.py's
    # inputs changed since the baseline was saved, so the results are not comparable; save a new baseline first.
    if not baseline_path.exists():
        return [f"no baseline at {baseline_path}: run python code/check_code.py save first"]
    with np.load(baseline_path) as baseline:
        baseline_arrays = {name: baseline[name] for name in baseline.files}
    differences = [f"only in the baseline: {name}" for name in sorted(set(baseline_arrays) - set(saved_arrays))]
    differences += [f"not in the baseline: {name}" for name in sorted(set(saved_arrays) - set(baseline_arrays))]
    for name in sorted(set(baseline_arrays) & set(saved_arrays)):
        value, baseline_value = saved_arrays[name], baseline_arrays[name]
        if not np.array_equal(value, baseline_value):
            detail = (f"largest difference {np.abs(value - baseline_value).max():.3e}"
                      if value.dtype.kind == "f" and value.shape == baseline_value.shape else "differs")
            differences.append(f"{name}: {detail}")
    return differences

## ------- Main ------- ##

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode not in ("check", "save", "compare"):
        sys.exit(f"unknown mode {mode!r}: use no argument, save or compare")

    sources = get_sources()
    problems = []
    for description, check in [("modules import", check_modules_import),
                               ("nothing unused", lambda: check_nothing_unused(sources)),
                               ("formatting", lambda: check_formatting(sources)),
                               ("no global lattice size online", lambda: check_no_global_lattice_size(sources))]:
        found = check()
        print(f"[{'PASS' if not found else 'FAIL'}] {description}")
        problems += found
    saved_arrays = {}
    for run_name, smoke_run in smoke_runs.items():
        print(f"{run_name} smoke run of every entry point on a small copy of the problem:")
        found, run_arrays = run_smoke_test(run_name, smoke_run)
        print(f"[{'PASS' if not found else 'FAIL'}] {run_name} smoke run")
        problems += found
        saved_arrays.update(run_arrays)

    if mode == "save":
        if problems:
            print("\nbaseline not saved, since not every check passed")
        else:
            baseline_path.parent.mkdir(exist_ok=True)
            np.savez(baseline_path, **saved_arrays)
            print(f"\nbaseline of {len(saved_arrays)} saved arrays written to {baseline_path}")
    if mode == "compare":
        differences = compare_with_baseline(saved_arrays)
        print(f"[{'PASS' if not differences else 'FAIL'}] results identical to the baseline "
              f"({len(saved_arrays)} saved arrays)")
        problems += differences

    if problems:
        print("\nproblems:\n" + "\n".join(f"  {problem}" for problem in problems))
        sys.exit(1)
    print("\nall checks passed")

if __name__ == "__main__":
    main()
