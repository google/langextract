## Unreleased

- Fix: Avoid importing provider modules during explicit provider resolution
  and ensure builtins/plugins are loaded before pattern matching. This
  prevents optional provider imports (which may require API keys) from
  happening during model resolution for local models. (fixes #113)
