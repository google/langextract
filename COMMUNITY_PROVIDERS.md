# Community Provider Plugins

Community-developed provider plugins that extend LangExtract with additional model backends.

**Supporting the Community:** Star plugin repositories you find useful and add 👍 reactions to their tracking issues to support maintainers' efforts.

**⚠️ Important:** These are community-maintained packages. Please review the [safety guidelines](#safety-disclaimer) before use.

## Plugin Registry

| Plugin Name | PyPI Package | Maintainer | GitHub Repo | Description | Issue Link |
|-------------|--------------|------------|-------------|-------------|------------|
| Example Provider | `langextract-provider-example` | [@google](https://github.com/google) | [google/langextract](https://github.com/google/langextract) | Reference implementation for creating custom providers | [#123](https://github.com/google/langextract/issues/123) |
| AWS Bedrock | `langextract-bedrock` | [@andyxhadji](https://github.com/andyxhadji) | [andyxhadji/langextract-bedrock](https://github.com/andyxhadji/langextract-bedrock) | AWS Bedrock provider for LangExtract, supports all models & inference profiles | [#148](https://github.com/google/langextract/issues/148) |
| LiteLLM | `langextract-litellm` | [@JustStas](https://github.com/JustStas) | [JustStas/langextract-litellm](https://github.com/JustStas/langextract-litellm) | LiteLLM provider for LangExtract, supports all models covered in LiteLLM, including OpenAI, Azure, Anthropic, etc., See [LiteLLM's supported models](https://docs.litellm.ai/docs/providers) | [#187](https://github.com/google/langextract/issues/187) |

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
