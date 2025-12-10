# Contributing to SIH-25035

Thank you for your interest in contributing! Please follow these guidelines.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Error messages/logs

### Suggesting Features

1. Check if already proposed
2. Provide clear use case
3. Explain expected behavior
4. Suggest implementation if possible

### Submitting Code

1. **Fork and branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Follow code style**
   - Use PEP 8
   - Add docstrings
   - Write type hints
   - Keep functions small and focused

3. **Write tests**
   - Test new functionality
   - Ensure existing tests pass
   - Aim for >80% coverage

4. **Update documentation**
   - Add docstrings
   - Update README if needed
   - Include usage examples

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: clear description of changes"
   git push origin feature/your-feature
   ```

6. **Open a Pull Request**
   - Clear title and description
   - Link related issues
   - Ensure CI/CD passes

## Development Setup

```bash
# Clone and setup
git clone https://github.com/vijeth06/SIH-25035.git
cd SIH-25035
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```

## Commit Message Convention

```
type: brief description

Longer explanation if needed.

Fixes #issue-number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`

## Pull Request Process

1. Update version numbers if needed
2. Update CHANGELOG.md
3. Ensure all tests pass
4. Get approval from maintainers
5. Squash commits if requested
6. Merge and delete branch

## Questions?

Open an issue or discussion in the repository.

Thank you for contributing!
