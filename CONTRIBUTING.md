# Contributing to AgriGuardAI

Thank you for your interest in contributing to **AgriGuardAI**! We welcome contributions from the community to help make this AI-powered sustainable crop disease detection system even better.

## How to Contribute

### 1. Setup the Development Environment
Please refer to the [Installation Instructions](README.md#%EF%B8%8F-installation-instructions) in the `README.md` to get your local environment running. We recommend using Docker for a seamless experience.

### 2. Branching Strategy
We use a standard branching strategy:
- `main`: The stable production branch.
- `dev`: The active development branch.
- Feature branches: `feature/your-feature-name`
- Bugfix branches: `bugfix/issue-description`

Before you start working, please create a new branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

### 3. Coding Standards & Commit Message Style
- **PEP8 Guidelines**: All Python code should follow standard PEP8 guidelines. Please run `black` and `isort` on your code before committing.
- **Type Hints**: Ensure FastAPI schemas and internal services are properly typed using Python type hints.
- **Commit Messages**: We follow [Conventional Commits](https://www.conventionalcommits.org/).
  - `feat: add AI fallback system`
  - `fix: resolve database connection timeout`
  - `docs: update API endpoints in README`

### 4. Pull Requests
1. Push your branch to your forked repository.
2. Open a Pull Request against the `main` branch.
3. Use the provided Pull Request template and fill in all the details.
4. Ensure all GitHub Actions / automated tests pass (if configured).
5. A maintainer will review your code and may request changes before merging.

## Reporting Bugs
If you find a bug, please use the "Bug Report" issue template and provide as much detail as possible, including steps to reproduce the issue.

Thank you for contributing!
