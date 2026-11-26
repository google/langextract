# Change the skill, not the adapter

Point directly at the directory containing `SKILL.md`. No registration or planner callback is required:

```python
from langextract_skills.runtime import extract_with_skill

result = extract_with_skill(
    'orders.txt', skill='./sample_skills/confirmed-orders',
    prompt_description='Extract each order ID verbatim.', examples=examples,
    model_id=model_id,  # caller-configured LangExtract provider
)
print(result.skill_trace)  # name, manifest_sha256, executed_tools (in order)
```

The supplied model ID is used for both planning and extraction. Pass an initialized `model=` to share one configured instance. Optional `allow_tools=` narrows the built-in read-only capabilities; an empty grant fails closed. An advanced custom `executor=` owns its own tool policy and cannot be combined with `allow_tools`. Trace metadata excludes prompts, file contents, credentials and tool arguments. Custom executors that do not report execution return `executed_tools=None` rather than a fabricated trace.

The bridge takes a caller-selected local Agent Skills directory. There is no skill-name table, manifest digest allowlist or new Python module to install for each skill. The deterministic registered adapter API remains available separately.

**Data flow:** `SKILL.md` body + user task + granted tool descriptions → planner → validated JSON plan. Only then does trusted code read the selected input. Its text goes to ordinary LangExtract inference; a sidecar projects aligned character intervals onto any available slide spans. The planner does not invent source text or coordinates. Plain-text and filtered results have document identity and character offsets in *prepared text*, not a guessed slide or byte offset in the original. Filtering intentionally discards slide spans rather than reusing stale offsets.

## Try two skills

Install the example package as in [README](README.md), obtain the pinned local official PPTX checkout there, and work in `examples/skill_preprocessing`. Configure an existing LangExtract provider and set `MODEL_ID` to its model identifier. These commands explicitly authorize sending the skill instructions and task to that model for planning, then prepared document text and examples for extraction. The CLI prints prepared text and results: use synthetic or authorized inputs only.

```sh
python run_skill.py "$LX_TEST_SKILL" tests/regions.pptx \
  --allow markitdown_pptx --model-id "$MODEL_ID" \
  --task 'Read presentation content. Extract each regional revenue statement verbatim in order.' \
  --example-class revenue --example-text 'West: $3M'

python run_skill.py sample_skills/confirmed-orders tests/orders.txt \
  --allow read_text --allow keep_lines --model-id "$MODEL_ID" \
  --task 'Extract each order ID verbatim.' --example-class order_id --example-text PO-123
```

For a private local run on Apple Silicon, optionally install `mlx-lm==0.32.2` and replace `--model-id "$MODEL_ID"` with `--mlx-local /path/to/installed/instruct-model`. MLX is **not** an integration dependency. Neither path auto-installs skill dependencies or downloads skill code. In application code `PlanningExecutor(planner_callable, allowed_tools=...)` accepts any callable returning a JSON string, so the planner need not use the extraction provider. A different trusted executor can implement `SkillExecutor.prepare` without modifying the bridge.

## What exactly is supported?

The reference executor is deliberately small: one planning turn, at most eight validated calls, one local reader (UTF-8 text or bounded MarkItDown PPTX conversion), followed by zero or more literal line filters. It handles previously unseen instructions expressible using these capabilities. To try an unseen skill, create a directory with a matching `name`, a `description` and Markdown instructions; select it instead of the example directory. No custom DSL or tool names are required in its instructions.

The planner receives the manifest body but not the source contents, local paths, resources or environment. `allowed-tools` in frontmatter does not confer privileges. The entire plan must validate before any tool runs: fixed operations, no user-supplied paths, no arbitrary regular expressions, shell, network calls, script execution or synthesized document text. A remote planner/provider is a separate, explicit network choice. Provider timeouts and credentials remain the caller's responsibility. A legitimate but incorrect allowed plan is possible; validation checks permissions and structure, **not** semantic correctness. Use the pinned baseline where reproducibility matters more than flexibility.

This is not a universal agent runtime: it cannot follow a bundled script or reference file, plan interactively over tool output, modify files, or run every task described by a complex skill. The official PPTX reading workflow is tested for the specific revision in the README, not its editing, creation or rendering instructions. An unsupported workflow should produce an empty plan and fail. No new tools are granted to rescue a failed plan.

## Real-model acceptance test

```sh
# Both are local directories. Without either, this test skips rather than download.
export LX_LOCAL_MODEL=/path/to/local/MLX-model
export LX_TEST_SKILL=/path/to/pinned/skills/pptx
python -m pytest tests -q
```

Verified with `mlx-community/Qwen2.5-7B-Instruct-4bit` snapshot `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`, on Apple Silicon. **32 passed**, including the existing deterministic tests. The real-model test uses the **public convenience API, same local model and default tool grants** across three invocations:

| Input / selected skill | Observed extraction |
|---|---|
| Official PPTX skill, synthetic deck | `North: $12M` on slide 1; `South: $12M` on slide 2 |
| Order ledger, independently authored Markdown instructions | Only `PO-240`, in `CONFIRMED: PO-240` |
| Same ledger, same skill name and task; change only `CONFIRMED:` to `PENDING:` in the body | Only `PO-108`, in `PENDING: PO-108` |

The last pair distinguishes interpretation of new prose from dispatch by filename, skill name or format. Results are evidence of capability on these inputs, not a cross-model success-rate claim. No paid provider calls were made.
