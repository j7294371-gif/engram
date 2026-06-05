# Contributing to Engram

Thank you for your interest in contributing to Engram!

## Development Setup

```bash
git clone https://github.com/j7294371-gif/memore.git
cd memore
pip install -e ".[dev]"
```

## Code Quality

```bash
# Run tests
pytest -v

# Lint check
ruff check src/

# Type check
mypy src/memore/
```

## Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Make your changes.
4. Run tests and ensure they pass.
5. Commit with conventional commit messages.
6. Push and open a PR.

## Conventional Commits

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — test additions/changes
- `refactor:` — code restructuring
- `chore:` — maintenance

## Code of Conduct

Be respectful, inclusive, and constructive.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
