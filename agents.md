
# agents.md

## codex ground rules reminder!!
Let's write some code together.
GROUND RULES: we are serious engineers writing serious code. This means that we should not be writing spaghetti. Before you write, consider:

1. DO NOT create functions that are called once
2. DO NOT create god functios
3. Functions with similar behaviors should have similar, semantic names `measure_this` `measure_that_or_this`
4. A file/module should contain one conceptual thing; if you need to create large helpers, create a utils folder/file but it must then contain **only flat, non-interdependent utilities**
5. Never have a large wall of something: not a large wall of different functions, not a large wall of methods, not a large wall of classes
6. Code's protos should imply logic; do not have functions lying around where how they are chained together is not immediately obvious
7. Prefer conventions when possible; `new`, etc., are great things. Similarly to before, actions of objects should be semantically clear, there should not be many more public methods beyond the action intended for that object.
8. Unless you are running autonomously (i.e. in goal mode), you should STOP whenever you encounter something that you can't solve / you are confused about instead of assuming things.

After writing I want you to be one of those JR rail operators that points to each hunk and calls out in context explicitly why each diff hunk follows each of these rules.

## pre-done audit
In addition to the point-and-call above, EVERY change should consider whether the change is a *experiment* or *infrastructure*. IN GENERAL, experiments should be private and infrastructure should be public unless the user says otherwise. To delineate these public/private boundaries, read once `disclosure.md` in the docs and audit your hunk changes against it.

You should also run pre-commit hooks by actually running `uv pre-commit` (e.g., don't directly invoke mypy/ruff etc.) since the repository has custom settings.


## Branch Semantics
Ideally, branch names on GitHub should be kept clean. We use the following semantics:

- `bets/*`: scientific wagers, unrelated to infrastructure
- `feat/*`: Theseus feature
- `patch/*`: Theseus big fixes

Branches can arbitrarily stack. 

