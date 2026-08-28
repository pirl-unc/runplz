"""Legacy module path for :mod:`runplz.cli`.

The supported entry points are the ``runplz`` console script and
``python -m runplz.cli``. This shim keeps ``python -m runplz._cli`` working
for anyone who wired it up while the module was underscore-named.
"""

from runplz.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
