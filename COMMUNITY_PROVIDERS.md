# Community Provider Plugins

Community-developed provider plugins that extend LangExtract with additional model backends.

**Supporting the Community:** Star plugin repositories you find useful and add 👍 reactions to their tracking issues to support maintainers' efforts.

**⚠️ Important:** These are community-maintained packages. Please review the [safety guidelines](#safety-disclaimer) before use.

## Plugin Registry

| Plugin Name | PyPI Package | Maintainer | GitHub Repo | Description | Issue Link |
|-------------|--------------|------------|-------------|-------------|------------|
| AWS Bedrock | `langextract-bedrock` | [@andyxhadji](https://github.com/andyxhadji) | [andyxhadji/langextract-bedrock](https://github.com/andyxhadji/langextract-bedrock) | AWS Bedrock provider for LangExtract, supports all models & inference profiles | [#148](https://github.com/google/langextract/issues/148) |
| LiteLLM | `langextract-litellm` | [@JustStas](https://github.com/JustStas) | [JustStas/langextract-litellm](https://github.com/JustStas/langextract-litellm) | LiteLLM provider for LangExtract, supports all models covered in LiteLLM, including OpenAI, Azure, Anthropic, etc., See [LiteLLM's supported models](https://docs.litellm.ai/docs/providers) | [#187](https://github.com/google/langextract/issues/187) |
| Llama.cpp | `langextract-llamacpp` | [@fgarnadi](https://github.com/fgarnadi) | [fgarnadi/langextract-llamacpp](https://github.com/fgarnadi/langextract-llamacpp) | Llama.cpp provider for LangExtract, supports GGUF models from HuggingFace and local files | [#199](https://github.com/google/langextract/issues/199) |
| Outlines | `langextract-outlines` | [@RobinPicard](https://github.com/RobinPicard) | [dottxt-ai/langextract-outlines](https://github.com/dottxt-ai/langextract-outlines) | Outlines provider for LangExtract, supports structured generation for various local and API-based models | [#101](https://github.com/google/langextract/issues/101) |
| vLLM | `langextract-vllm` | [@wuli666](https://github.com/wuli666) | [wuli666/langextract-vllm](https://github.com/wuli666/langextract-vllm) | vLLM provider for LangExtract, supports local and distributed model serving | [#236](https://github.com/google/langextract/issues/236) |
<!-- ADD NEW PLUGINS ABOVE THIS LINE -->

## How to Add Your Plugin (PR Checklist)

Copy this row template, replace placeholders, and insert **above** the marker line:

```markdown
| Your Plugin | `langextract-provider-yourname` | [@yourhandle](https://github.com/yourhandle) | [yourorg/yourrepo](https://github.com/yourorg/yourrepo) | Brief description (min 10 chars) | [#456](https://github.com/google/langextract/issues/456) |
```

**Before submitting your PR:**
- [ ] PyPI package name starts with `langextract-` (recommended: `langextract-provider-<name>`)
- [ ] PyPI package is published (or will be soon) and listed in backticks
- [ ] Maintainer(s) listed as GitHub profile links (comma-separated if multiple)
- [ ] Repository link points to public GitHub repo
- [ ] Description clearly explains what your provider does
- [ ] Issue Link points to a tracking issue in the LangExtract repository for integration and usage feedback (plugin-specific features and discussions can optionally happen in the plugin's repository)
- [ ] Entries are sorted alphabetically by Plugin Name

## Documentation

For detailed plugin development instructions, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/README.md).

## Safety Disclaimer

Community plugins are independently developed and maintained. While we encourage community contributions, the LangExtract team cannot guarantee the safety, security, or functionality of third-party packages.

**Before installing any plugin, we recommend:**

- **Review the code** - Examine the source code and dependencies on GitHub
- **Check community feedback** - Read issues and discussions for user experiences
- **Verify the maintainer** - Look for active maintenance and responsive support
- **Test safely** - Try plugins in isolated environments before production use
- **Assess security needs** - Consider your specific security requirements

Community plugins are used at your own discretion. When in doubt, reach out to the community through the plugin's issue tracker or the main LangExtract discussions.
