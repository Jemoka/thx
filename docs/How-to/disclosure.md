# Disclosing Theseus code

The public Theseus repository is [Jemoka/thx](https://github.com/Jemoka/thx).
The Python package and CLI are still named `theseus`.

Files are private by default. An empty `.thx_disclose` file releases all tracked
files in its folder and subfolders, including binary assets. `docs/` uses this
convention. The control file itself is not exported. Explicit file regions
still restrict a file inside a disclosed folder. Use folder disclosure only
where all future files are intended to be public.

Release a text file by putting `!!THX_DISCLOSE`
on its first line, wrapped in the language's comment syntax: for example
`# !!THX_DISCLOSE`, `// !!THX_DISCLOSE`, or `<!-- !!THX_DISCLOSE -->`.
All in-file markers and the `.thx_disclose` filename are case-insensitive.

To release only part of a file, put `<<<THX_DISCLOSE` on a standalone comment
line before each public region and `>>>THX_DISCLOSE` on a standalone comment
line after it. Only those regions are exported, in order. Marker lines are
removed. Do not nest regions or combine region markers with a whole-file
marker. A misplaced whole-file marker or unmatched region fails CI.

Keep complete Python statements inside public regions. For a package
initializer, disclose the imports and `__all__` entries for released modules.
Keep private imports and private names outside those regions. The registry
already loads initializers lazily and registers the included jobs; no separate
public registry implementation is needed.

New scientific experiments, datasets, evaluations, and model implementations
remain private until explicitly released. The initial release boundary matches
the previous `thx` tree. New documentation under `docs/` is disclosed automatically and should
refer to `https://github.com/Jemoka/thx`, including in the private checkout.

Binary assets, strict data formats, and executable scripts whose shebang must
remain first use exact paths in `tools/disclosure/raw_files.json` in the source
repository. This is an explicit whole-file release list, not a directory glob.

## Preview and validation

In the source checkout, run `python3 tools/disclosure/export.py /tmp/thx-preview`
with a fresh output path. Only tracked files are considered, so stage new files
before previewing them. The exporter checks Python syntax. CI formats the released Python with Ruff
0.12.2 to clean up whitespace left by removed regions, then builds and
installs the exported wheel, imports every public module, checks registry keys
against `tools/disclosure/baseline.json`, and exercises the CLI. An intentional
scientific release requires reviewing and updating that baseline as well as
adding disclosure markers.

Google Copybara syncs the validated snapshot on pushes to source `master`.
Pull requests run validation and a disposable local Git mirror test without
publishing. Outdated queued runs are skipped so they cannot overwrite a newer
public snapshot. Private source history and commit messages are never inputs to
Copybara. Unchanged public snapshots produce no commit; removing a release
marker removes the corresponding file on the next sync. Previously published
contents remain in public Git history.

Mirror commits use `thxbara <theseus@jemoka.com>` and the subject
`theseus sync commit`. The first sync credits the project's co-authors.
A content digest identifies the exported snapshot without exposing private
source revisions. The source repository's `THX_DEPLOY_KEY` secret must contain
a write-enabled deploy key scoped to `Jemoka/thx`.
