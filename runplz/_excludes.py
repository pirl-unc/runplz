"""Default file exclusion patterns applied to every host → remote transfer.

These patterns exist to stop common secret-bearing files from being
silently shipped to a Brev box or Modal image. Every backend that
transfers a local directory (brev's rsync_up, modal's add_local_dir)
should use this list as its baseline.

The list is intentionally conservative — it targets files that are
widely agreed-upon secrets and rarely intentionally deployed. Users
who *do* want these files in the remote environment should ship them
via their CI's proper secret-injection mechanism (Modal Secrets, env
vars injected via `App.function(env=...)`, etc.) rather than by
relaxing this list.

Patterns are glob-style, matched against path basenames (rsync's
default behavior for `--exclude=<pat>` without a leading slash, and
Modal's `add_local_dir(ignore=[...])` contract). Directory patterns
without trailing slashes still match the directory itself.
"""

DEFAULT_TRANSFER_EXCLUDES = (
    # dotenv + its common prod/dev variants. `.env.example` is left in.
    ".env",
    ".env.local",
    ".env.*.local",
    ".env.production",
    ".env.development",
    # private keys by extension — covers AWS access keys, TLS private
    # keys, SSH keys saved with custom names, service-account JSON
    # copies that were renamed *.key, etc.
    "*.pem",
    "*.key",
    # SSH keypairs by convention name. `.pub` is intentionally NOT
    # excluded (it's public).
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    # cloud-provider credential blobs
    "credentials.json",
    ".aws",
    ".ssh",
    ".netrc",
    ".git-credentials",
)
