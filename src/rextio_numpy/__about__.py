"""Package version, isolated from ``__init__`` side effects.

setuptools reads ``[tool.setuptools.dynamic] version`` by parsing this module's
AST, so it stays a single literal assignment.
"""

__version__ = "0.1.2"
