# GitPython security update validation

GitPython **3.1.43 → 3.1.62** addresses reviewed security advisories including
[CVE-2026-87817 / GHSA-239g-whfq-7xj9](https://github.com/advisories/GHSA-239g-whfq-7xj9).
Only this project dependency changes. Tracked files must not impersonate the
real `.git` directory/configuration. No exploitation of TRACE-GFN is claimed.

In a new Python 3.10 virtual environment:

```sh
python -m pip install -r tests/requirements-gitpython.txt
python tests/test_gitpython_compatibility.py
python -m pip check
```

Validated on macOS arm64 / Python 3.10.19. The real WandB Git client is used:
metadata, branch, remote URL, dirty/untracked state, diffs, local clone,
bare repository and linked worktree checks pass before and after.
The project leaves WandB unpinned: this comparison uses **WandB 0.18.7**
and holds the other **19** test dependencies identical, matching project
pins where specified. This does not prove compatibility with every possible
resolution of the project's unpinned dependencies.

The inert tracked-file fixture reproduces wrong Git-directory discovery on
3.1.43 and verifies the real directory/configuration wins with 3.1.62.
No executable hook or external config include is created. All three updated
checks and dependency checks pass. OSV reports no advisories for 3.1.62 on
2026-10-02. For a baseline comparison, replace only the GitPython pin with
3.1.43 in a second environment and use `GITPYTHON_BASELINE=1` to skip the
expected failing security test.

This is a scoped Git/WandB test environment. Full `uv sync`/Linux/CUDA
solving, graph extensions, models, GPU training/generation, Python 3.11 and
WandB server communication are unverified. Other dependency advisories
remain; the frozen test requirements are not a general secure lockfile.
