# Contributing to Python-Intan

Thank you for your interest in contributing to the python-intan package! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Questions](#questions)

---

## Code of Conduct

This project follows a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful, constructive, and professional in all interactions.

**Key principles:**
- Be welcoming and inclusive
- Be respectful of different viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## How Can I Contribute?

There are many ways to contribute to python-intan:

### 🐛 Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title** - Brief description of the problem
2. **Environment details**:
   - Python version
   - Package version (`import intan; print(intan.__version__)`)
   - Operating system
   - Relevant hardware (if applicable)
3. **Minimal reproducible example** - Simplest code that demonstrates the bug
4. **Expected behavior** - What you expected to happen
5. **Actual behavior** - What actually happened
6. **Error messages** - Full traceback if applicable
7. **Screenshots** - If relevant (GUI issues, plots, etc.)

**Example:**

```markdown
**Bug**: IntanRHXDevice fails to connect on Ubuntu 22.04

**Environment**:
- Python 3.10.12
- python-intan 0.0.3
- Ubuntu 22.04

**Code**:
\```python
from intan.interface import IntanRHXDevice
device = IntanRHXDevice()  # Fails here
\```

**Error**:
\```
ConnectionRefusedError: [Errno 111] Connection refused
\```

**Expected**: Should connect to RHX software running on localhost
**Actual**: Connection refused

**Notes**: RHX software is running with TCP servers enabled
```

### ✨ Suggesting Features

We welcome feature suggestions! Please create an issue with:

1. **Clear use case** - Why is this feature needed?
2. **Proposed solution** - How should it work?
3. **Alternatives considered** - Other approaches you've thought about
4. **Examples** - Code examples of how you'd use it

### 📝 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or clarify explanations
- Add examples
- Improve API documentation
- Create tutorials
- Translate documentation

### 🧪 Adding Examples

New examples help users learn the package:

- Real-world use cases
- Integration with other tools
- Novel applications
- Benchmark comparisons

Place examples in the appropriate `examples/` subdirectory and add documentation in `docs/source/examples/`.

### 🔧 Code Contributions

See [Contribution Workflow](#contribution-workflow) below.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/python-intan.git
cd python-intan
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n intan-dev python=3.10
conda activate intan-dev

# Or using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
# Install package in editable mode with development dependencies
pip install -e .

# Install development tools (optional)
pip install pytest black flake8 mypy
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## Contribution Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 2. Make Changes

- Write clear, readable code
- Follow style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run tests (if available)
pytest tests/

# Check code style
black intan/
flake8 intan/

# Type checking (optional)
mypy intan/
```

### 4. Commit Changes

Write clear commit messages:

```bash
git add .
git commit -m "Add feature: real-time RMS calculation

- Implement window_rms function in processing module
- Add unit tests for window_rms
- Update documentation with usage examples
- Closes #123"
```

**Commit message format:**
- **First line**: Brief summary (50 chars or less)
- **Body**: Detailed explanation (wrap at 72 chars)
- **Footer**: Reference issues (e.g., "Closes #123", "Fixes #456")

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

1. **Clear title** - What does this PR do?
2. **Description**:
   - What problem does it solve?
   - How does it solve it?
   - Any breaking changes?
   - Related issues?
3. **Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented if unavoidable)

### 6. Code Review

- Respond to reviewer feedback
- Make requested changes
- Push updates to your branch

### 7. Merge

Once approved, maintainers will merge your PR. Thank you for contributing!

---

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

**Line length**: 100 characters (not 79)

**Formatting**: Use `black` for automatic formatting

```bash
black intan/
```

**Imports**: Organize imports as:
1. Standard library
2. Third-party packages
3. Local imports

```python
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from intan.io import load_rhd_file
from intan.processing import filter_emg
```

**Naming conventions:**
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- `_leading_underscore` for private/internal functions

**Docstrings**: Use NumPy-style docstrings

```python
def filter_emg(data, filter_type, sample_rate, lowcut=None, highcut=None):
    """
    Apply filtering to EMG data.

    Parameters
    ----------
    data : np.ndarray
        EMG data array (channels, samples).
    filter_type : str
        Type of filter ('lowpass', 'highpass', 'bandpass').
    sample_rate : float
        Sampling rate in Hz.
    lowcut : float, optional
        Low cutoff frequency for bandpass/highpass.
    highcut : float, optional
        High cutoff frequency for bandpass/lowpass.

    Returns
    -------
    np.ndarray
        Filtered EMG data.

    Examples
    --------
    >>> emg_filtered = filter_emg(emg_data, 'bandpass', 4000,
    ...                            lowcut=20, highcut=500)
    """
    # Implementation
```

### Documentation Style

- Use **reStructuredText** (.rst) for Sphinx documentation
- Include code examples
- Link to relevant API references
- Keep explanations clear and concise
- Use proper heading hierarchy

### Example Code

- Examples should be self-contained when possible
- Include comments explaining non-obvious steps
- Handle errors gracefully
- Show expected output

---

## Testing

We use `pytest` for testing (when tests exist):

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_io.py

# Run with coverage
pytest --cov=intan tests/
```

**Writing tests:**

```python
import pytest
import numpy as np
from intan.processing import window_rms

def test_window_rms_basic():
    """Test basic RMS calculation."""
    # Create test data
    data = np.array([[1, 2, 3, 4, 5]])

    # Calculate RMS
    result = window_rms(data, window_size=3)

    # Assert expected behavior
    assert result.shape == data.shape
    assert np.all(result >= 0)

def test_window_rms_zeros():
    """Test RMS of zeros."""
    data = np.zeros((2, 100))
    result = window_rms(data, window_size=10)
    assert np.all(result == 0)
```

---

## Documentation

### Building Documentation

```bash
cd docs
pip install sphinx sphinx-autodoc-typehints myst-parser
make html
```

View at `docs/build/html/index.html`

### Adding Documentation

1. **API docs**: Docstrings are auto-generated
2. **Examples**: Add .rst files to `docs/source/examples/`
3. **Guides**: Add .rst files to `docs/source/info/`

### Documentation Checklist

- [ ] All public functions have docstrings
- [ ] Parameters and returns documented
- [ ] Examples included where helpful
- [ ] Cross-references to related functions
- [ ] Added to appropriate toctree

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (run `black` and `flake8`)
- [ ] All tests pass (if tests exist)
- [ ] New functionality has tests
- [ ] Documentation updated
- [ ] Examples added/updated if applicable
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description is complete

---

## Questions?

If you have questions about contributing:

1. Check the [FAQ](https://neuro-mechatronics-interfaces.github.io/python-intan/info/faqs.html)
2. Search existing [GitHub Issues](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues)
3. Open a new issue with your question
4. Email: jshulgac@andrew.cmu.edu

---

## Recognition

All contributors will be acknowledged in:
- GitHub contributors page
- Release notes
- Documentation (where appropriate)

We appreciate your time and effort in making python-intan better!

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to python-intan! 🎉**
