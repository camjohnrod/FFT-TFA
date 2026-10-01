# Consistency checks for the code in this folder, to run after any change:
#
#   python code/check_code.py
#
# 1. Static checks: every module imports; nothing defined goes unused (top-level definitions, NamedTuple fields,
#    config constants); no line is longer than 120 characters; none of the names retired in the 2026-09-30
#    restructuring is back.
# 2. Smoke run: a copy of this folder, with the grid, step count and repeats shrunk in the copy's config.py (this
#    folder's config, cache and outputs are never touched), runs all four entry points in the order their
#    cross-checks need, and every check they print must PASS, none FAIL or SKIPPED.
#
# Exits with status 1 if anything fails. Takes about a minute. It checks consistency, not results: a change that
# alters the numbers still passes if every solver still agrees with the others.

import ast
import importlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

code_folder = pathlib.Path(__file__).resolve().parent
entry_points = ["MAIN_TFA_FP", "MAIN_LS_FP", "MAIN_TFA_Newton", "MAIN_LS_Newton"]
retired_names = ["MAIN_Testing", "plots_testing", "output_testing", "cache_testing", "post_process",
                 "reference_solver_method", "fixed_point_tolerance", "divergence_residual_limit",
                 "fixed_point_max_iterations", "operator_applications"]
# Small enough to run in seconds, and 9 partitions per side suits both inclusion shapes at their default sizes.
smoke_run_overrides = {"partition_number_per_side": "9", "element_number_per_side": "27",
                       "strain_increment_count": "12", "timing_repeat_count": "1"}
maximum_line_length = 120

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

def check_no_retired_names(sources):
    return [f"{file_name}: retired name {name!r} is back" for file_name, source in sources.items()
            if file_name != "check_code.py" for name in retired_names if name in source]

def run_smoke_test():
    problems = []
    with tempfile.TemporaryDirectory() as temporary_folder:
        copy_folder = pathlib.Path(temporary_folder)
        for path in code_folder.glob("*.py"):
            shutil.copy(path, copy_folder / path.name)
        config_path = copy_folder / "config.py"
        config_source = config_path.read_text()
        for name, value in smoke_run_overrides.items():
            config_source, replaced = re.subn(rf"^{name}(\s*)= .*$", rf"{name}\g<1>= {value}", config_source, count=1,
                                              flags=re.M)
            if replaced != 1:
                return [f"smoke run: config.py has no line setting {name}"]
        config_path.write_text(config_source)

        for entry_point in entry_points:
            run = subprocess.run([sys.executable, f"{entry_point}.py"], cwd=copy_folder, capture_output=True, text=True)
            passed = run.stdout.count("[PASS]")
            not_passed = [line.strip() for line in run.stdout.splitlines() if "[FAIL]" in line or "[SKIPPED]" in line]
            if run.returncode != 0:
                problems.append(f"{entry_point}.py exited with status {run.returncode}:\n{run.stderr[-1500:]}")
            problems += [f"{entry_point}.py: {line}" for line in not_passed]
            print(f"  {entry_point}.py: exit status {run.returncode}, {passed} checks passed, "
                  f"{len(not_passed)} failed or skipped")
    return problems

def main():
    sources = get_sources()
    problems = []
    for description, check in [("modules import", check_modules_import),
                               ("nothing unused", lambda: check_nothing_unused(sources)),
                               ("formatting", lambda: check_formatting(sources)),
                               ("no retired names", lambda: check_no_retired_names(sources))]:
        found = check()
        print(f"[{'PASS' if not found else 'FAIL'}] {description}")
        problems += found
    print("smoke run of every entry point on a small copy of the problem:")
    found = run_smoke_test()
    print(f"[{'PASS' if not found else 'FAIL'}] smoke run")
    problems += found

    if problems:
        print("\nproblems:\n" + "\n".join(f"  {problem}" for problem in problems))
        sys.exit(1)
    print("\nall checks passed")

if __name__ == "__main__":
    main()
