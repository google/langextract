# langextract-usage — Agent Skill

This directory is an [Agent Skill](https://agentskills.io) source that teaches
AI coding assistants how to use LangExtract correctly. It follows the open
Agent Skills format (`SKILL.md` with YAML frontmatter, optional
`references/`, `examples/`).

This directory is **not** auto-discovered by any tool from the `skills/`
path it lives in here. To use it, copy or symlink the directory into
whichever path your Agent-Skills-compatible tool reads from.

## Activate

Pick your tool, then install at either project scope (this repo only) or
user scope (all your projects). Paths below are **canonical examples** per
each tool's 2026 docs — some tools (Copilot, Codex) scan several paths,
so any of the documented locations works. Check the linked docs for the
full list if your tool has moved.

Project-scope paths are relative to your workspace/repo root. Run the copy
or symlink command from there.

| Tool | Project-scope path (example) | User-scope path (example) |
|---|---|---|
| [Google Antigravity](https://antigravity.google/docs/skills) | `.agents/skills/langextract-usage/` | `~/.gemini/antigravity/skills/langextract-usage/` |
| [Anthropic Claude Code](https://code.claude.com/docs/en/skills) | `.claude/skills/langextract-usage/` | `~/.claude/skills/langextract-usage/` |
| [OpenAI Codex](https://developers.openai.com/codex/skills) | `.agents/skills/langextract-usage/` | `~/.agents/skills/langextract-usage/` |
| [GitHub Copilot](https://docs.github.com/en/copilot) | `.github/skills/langextract-usage/` | `~/.copilot/skills/langextract-usage/` |

Notes:
- **Antigravity** also recognizes legacy `.agent/skills/` (singular) for
  backward compatibility; new installs should use `.agents/skills/`
  (plural).
- **Copilot** also accepts `.claude/skills/` or `.agents/skills/` at project
  scope, and `~/.claude/skills/` or `~/.agents/skills/` at user scope.
- **Codex** scans `.agents/skills/` from the current directory up through
  the repo root, so a project-scope install works from any subdirectory.
- A project-scope `.agents/skills/langextract-usage/` therefore activates
  the skill in Antigravity, Codex, and Copilot simultaneously.

### Copy (static install)

```bash
cp -R skills/langextract-usage <tool-skill-path>
```

### Symlink (tracks this repo's updates)

```bash
ln -s "$(pwd)/skills/langextract-usage" <tool-skill-path>
```

### Other tools

- **Agent-Skills-compatible tools not listed above** — point the tool at
  this directory as a skill source, following its own docs.
- **Tools that read their own instruction format** (e.g. Cursor
  `.cursor/rules/`, Windsurf `.windsurf/rules/`, Aider `CONVENTIONS.md`) —
  copy the relevant sections of `SKILL.md` / `references/*.md` into that
  tool's convention. The content is portable; only the activation
  mechanism differs.

After activating, restart your agent session if the tool does not
auto-detect changes.

## Contents

- `SKILL.md` — entry point; install, basic usage, key parameters, working with
  results, common issues
- `references/providers.md` — Gemini, OpenAI, Ollama, `ModelConfig`, custom
  provider plugins
- `references/resolver-params.md` — fuzzy alignment tuning
- `references/prompt-validation.md` — `PromptValidationLevel` details
- `examples/` — runnable scripts (`basic_extraction.py`,
  `relationship_extraction.py`, `multiple_documents.py`)

## Maintenance

When changing user-facing LangExtract APIs (imports, `lx.extract()` kwargs,
resolver or validation behavior), please update the affected files in this
subtree in the same PR. If you spot drift, open an issue or PR.
