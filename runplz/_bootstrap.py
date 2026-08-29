"""Legacy module path for :mod:`runplz.bootstrap`.

Backends emit ``python -m runplz._bootstrap``, and a container running an
older runplz only has this path, so it stays for as long as that is true —
see the wire-contract note in :mod:`runplz.bootstrap`.

Attributes are forwarded lazily rather than bound at import, so patching
``runplz.bootstrap.main`` is visible through this module too. Binding by
value would silently decouple the two.
"""


def __getattr__(name):
    from runplz import bootstrap

    return getattr(bootstrap, name)


def __dir__():
    from runplz import bootstrap

    return sorted(set(globals()) | set(dir(bootstrap)))


if __name__ == "__main__":
    from runplz.bootstrap import main

    raise SystemExit(main())
