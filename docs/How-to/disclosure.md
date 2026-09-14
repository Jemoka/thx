# Disclosing Theseus code

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

The public Theseus repository is [Jemoka/thx](https://github.com/Jemoka/thx).
The Python package and CLI are still named `theseus`.

## Release or hide content

Files are private by default. All marker names are case-insensitive, and in-file
markers use the language's comment syntax.

| Scope | Disclose | Keep private |
|---|---|---|
| Folder and descendants | `.thx_disclose` file | `.thx_private` file |
| Whole file | `!!THX_DISCLOSE` line anywhere | `!!THX_PRIVATE` line anywhere |
| Region | `<<<THX_DISCLOSE` … `>>>THX_DISCLOSE` | `<<<THX_PRIVATE` … `>>>THX_PRIVATE` |

For example, use `# !!THX_DISCLOSE` in Python or
`<!-- !!THX_PRIVATE -->` in Markdown. Region delimiters must each occupy a
standalone comment line. Folder control files may be empty and are not exported.

**Smaller scopes take precedence:** the nearest folder marker overrides an
ancestor's marker; a file marker overrides its folder; a region overrides the
file policy; and an inner region overrides its enclosing region. Closing a region
restores its enclosing policy. Closing markers must match the open region's kind.
If both policies occur at the same scope, private wins, regardless of marker
order. This applies to both folder marker files and whole-file directives.

Content outside regions follows the file or inherited folder policy. In an
otherwise private file, disclose regions select what is released. In an
otherwise public file, private regions select what is hidden. Whole-file markers
and nested regions can be combined.

`docs/` is disclosed recursively. A `!!THX_PRIVATE` comment keeps an
individual guide private. A nested folder containing `.thx_private` keeps its contents
private unless a smaller scope explicitly discloses them. Binary assets follow
folder policies. Whole-file directives may appear anywhere on a standalone line,
so scripts can keep their shebang first. They apply to the entire file, including
text before the directive; region policies still take precedence. Conflicting
whole-file directives default to private. Inline mentions in prose or strings are not
directives. Formats that cannot contain a directive can use a folder marker.

## Keep the public package usable

Keep complete Python statements inside public regions. For a package initializer,
disclose the imports and `__all__` entries for released modules. Keep private
imports and private names outside those regions. The registry already loads
initializers lazily and registers the included jobs.

New scientific experiments, datasets, evaluations, and model implementations
remain private until explicitly released. The checked baseline preserves the
existing public scientific boundary. Documentation repository links should point
to `https://github.com/Jemoka/thx`, including in the private checkout.

## Preview and validation

Run `python3 tools/disclosure/export.py /tmp/thx-preview` in the source checkout,
using a fresh output path. Only tracked files are considered, so stage new files
before previewing them. The exporter rejects malformed markers and invalid Python.
CI formats the released Python with Ruff 0.12.2, with its cache disabled, then
builds and installs the public wheel, imports every public module, checks registry
keys against `tools/disclosure/baseline.json`, and exercises the CLI. An intentional
scientific release requires reviewing and updating that baseline too.

Google Copybara syncs the validated snapshot on pushes to source `master`.
Pull requests validate and test a disposable local Git mirror without publishing.
Outdated queued runs are skipped. Private source history and commit messages
never reach Copybara. Cache and build artifacts are rejected.

Unchanged public snapshots produce no commit. Removing disclosure removes the
corresponding file on the next sync, but previously published contents remain in
Git history.

Mirror commits use `thxbara <theseus@jemoka.com>` and the subject
`theseus sync commit`. The initial commit credits the project's co-authors.
A public-content digest identifies subsequent snapshots without exposing private
source revisions. The source repository's `THX_DEPLOY_KEY` secret contains a
write-enabled deploy key scoped to `Jemoka/thx`.
