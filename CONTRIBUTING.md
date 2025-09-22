# Contributing to ATC Hallucination Detection

Thank you for your interest in contributing to the ATC Hallucination Detection project! This document provides guidelines for contributing to the codebase.

## 🚀 Getting Started

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ATC_Hallucination.git
   cd ATC_Hallucination
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the guidelines below

3. Test your changes:
   ```bash
   python -m pytest tests/
   python verify_targeted_shifts.py  # Run verification tests
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

5. Push and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black src/ tests/ *.py
  ```
- Use [flake8](https://flake8.pycqa.org/) for linting:
  ```bash
  flake8 src/ tests/ *.py
  ```

### Documentation

- Add comprehensive docstrings to all functions and classes
- Use Google-style docstrings:
  ```python
  def function_name(param1: str, param2: int) -> bool:
      """Brief description of the function.

      Args:
          param1: Description of param1.
          param2: Description of param2.

      Returns:
          Description of return value.

      Raises:
          ValueError: Description of when this is raised.
      """
      pass
  ```

### Testing

- Write unit tests for new functionality
- Place tests in the `tests/` directory
- Use descriptive test names:
  ```python
  def test_targeted_shift_creates_single_agent_modification():
      """Test that targeted shifts only modify one agent."""
      pass
  ```

### Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature or functionality`
- `Fix: bug fixes`
- `Update: changes to existing functionality`
- `Docs: documentation changes`
- `Test: adding or updating tests`
- `Refactor: code restructuring without functional changes`

## 🧪 Adding New Features

### New Testing Scenarios

1. Add scenario JSON file to `scenarios/` directory
2. Update documentation in README.md
3. Add validation tests
4. Example scenario structure:
   ```json
   {
     "scenario_name": "your_scenario",
     "seed": 42,
     "center": {"lat": 52.0, "lon": 4.0, "alt_ft": 10000.0},
     "agents": [...]
   }
   ```

### New Shift Types

1. Update `create_conflict_inducing_shifts()` in `targeted_shift_tester.py`
2. Add handling in `create_targeted_shift()`
3. Update environment to support new shift types
4. Add comprehensive tests

### New Analysis Metrics

1. Update `HallucinationDetector` class
2. Add metric computation in analysis functions
3. Update result CSV schemas
4. Document new metrics in README

## 🐛 Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, package versions)
- Relevant log files or error messages

Use the bug report template:

```markdown
**Bug Description**
Brief description of the issue.

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.10.0]
- Package versions: [run `pip freeze`]

**Additional Context**
Any other relevant information.
```

## 🔬 Research Contributions

### New Algorithms

- Implement in `src/training/` directory
- Follow existing pattern for PPO/SAC integration
- Add comprehensive evaluation comparing to existing methods
- Document theoretical background and implementation details

### Evaluation Metrics

- Add to `src/analysis/` directory
- Ensure compatibility with existing result formats
- Provide benchmarks against standard metrics
- Include statistical significance testing

### Experimental Studies

- Use existing framework for reproducible experiments
- Document experimental setup thoroughly
- Provide analysis scripts and visualization code
- Include statistical analysis and confidence intervals

## 📊 Performance Considerations

- Profile performance-critical code
- Avoid loading large models/data in loops
- Use appropriate data structures (numpy arrays vs lists)
- Consider memory usage for large-scale experiments
- Test scalability with different episode counts

## 🔐 Security Guidelines

- Never commit sensitive data (API keys, credentials)
- Validate user inputs in CLI tools
- Use secure defaults for configuration
- Review dependencies for security vulnerabilities

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New functionality has tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No large files or sensitive data included
- [ ] Performance impact is considered
- [ ] Backwards compatibility is maintained

## 🤝 Code Review Process

1. **Automated Checks**: PRs run automated tests and style checks
2. **Peer Review**: At least one maintainer reviews code
3. **Testing**: New features are tested in isolation and integration
4. **Documentation**: Changes are reflected in documentation
5. **Merge**: Approved PRs are merged using squash commits

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Documentation**: Check README and inline documentation first

## 🏆 Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Academic publications (for research contributions)

Thank you for contributing to safer air traffic control systems! 🛩️