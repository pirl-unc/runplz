"""Legacy module path for :mod:`runplz.cli`.

The supported entry points are the ``runplz`` console script and
``python -m runplz.cli``. This shim keeps ``python -m runplz._cli`` working
for anyone who wired it up while the module was underscore-named.

Attributes are forwarded lazily for the same reason as
:mod:`runplz._bootstrap`: a by-value import would make patching
``runplz.cli.*`` invisible here.
"""


def __getattr__(name):
    from runplz import cli

    return getattr(cli, name)


def __dir__():
    from runplz import cli

    return sorted(set(globals()) | set(dir(cli)))


if __name__ == "__main__":
    from runplz.cli import main

    raise SystemExit(main())
