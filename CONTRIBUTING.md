# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows HAI-DEF's
[Community guidelines](https://developers.google.com/health-ai-developer-foundations/community-guidelines)

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub.
We have templates to help guide you:

- **[Bug Report](.github/ISSUE_TEMPLATE/1-bug.md)**: For reporting bugs or unexpected behavior
- **[Feature Request](.github/ISSUE_TEMPLATE/2-feature-request.md)**: For suggesting new features or improvements

When creating an issue, GitHub will prompt you to choose the appropriate template.
Please provide as much detail as possible to help us understand and address your concern.

## Contribution Process

### 1. Development Setup

To get started, clone the repository and install the necessary dependencies for development and testing. Detailed instructions can be found in the [Installation from Source](https://github.com/google/langextract#from-source) section of the `README.md`.

**Windows Users**: The formatting scripts use bash. Please use one of:
- Git Bash (comes with Git for Windows)
- WSL (Windows Subsystem for Linux)
- PowerShell with bash-compatible commands

### 2. Code Style and Formatting

This project uses automated tools to maintain a consistent code style. Before submitting a pull request, please format your code:

```bash
# Run the auto-formatter
./autoformat.sh
```

This script uses:
- `isort` to organize imports with Google style (single-line imports)
- `pyink` (Google's fork of Black) to format code according to Google's Python Style Guide

You can also run the formatters manually:
```bash
isort langextract tests
pyink langextract tests --config pyproject.toml
```

Note: The formatters target only `langextract` and `tests` directories by default to avoid
formatting virtual environments or other non-source directories.

### 3. Pre-commit Hooks (Recommended)

For automatic formatting checks before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### 4. Linting and Testing

All contributions must pass linting checks and unit tests. Please run these locally before submitting your changes:

```bash
# Run linting with Pylint 3.x
pylint --rcfile=.pylintrc langextract tests

# Run tests
pytest tests
```

**Note on Pylint Configuration**: We use a modern, minimal configuration that:
- Only disables truly noisy checks (not entire categories)
- Keeps critical error detection enabled
- Uses plugins for enhanced docstring and type checking
- Aligns with our pyink formatter (80-char lines, 2-space indents)

For full testing across Python versions:
```bash
tox  # runs pylint + pytest on Python 3.10 and 3.11
```

### 5. Submit Your Pull Request

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

When you create a pull request, GitHub will automatically populate it with our
[pull request template](.github/PULL_REQUEST_TEMPLATE/pull_request_template.md).
Please fill out all sections of the template to help reviewers understand your changes.

#### Pull Request Guidelines

- **Keep PRs focused and small**: Each PR should address a single, specific change. This makes review easier and faster.
- **Reference related issues**: Use "Fixes #123" or "Addresses #123" in your PR description to link to relevant issues.
- **Single-change commits**: A PR should typically comprise a single git commit. Squash multiple commits before submitting.
- **Clear description**: Explain what your change does and why it's needed.
- **Ensure all tests pass**: Check that both formatting and tests are green before requesting review.
- **Respond to feedback promptly**: Address reviewer comments in a timely manner.

If your change is large or complex, consider:
- Opening an issue first to discuss the approach
- Breaking it into multiple smaller PRs
- Clearly explaining in the PR description why a larger change is necessary

For more details, read HAI-DEF's
[Contributing guidelines](https://developers.google.com/health-ai-developer-foundations/community-guidelines#contributing)
