"""Check that PREREG-v2.md and every file it froze still match their approved SHA-256 hashes.

Standard library only; no GPU or data needed. Same rule as hpc-package/run_v2.py: PREREG-v2.md must
match the hash approved in run_v2.py, and run_v2.py is hashed with that approval line blanked.
Usage: python3 verify_frozen.py
"""
import hashlib, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(rel):
    data = open(os.path.join(HERE, rel), "rb").read()
    if rel.endswith("run_v2.py"):
        data = re.sub(rb"^PREREG_SHA256 = .*$", b"PREREG_SHA256 = <approved>", data, flags=re.M)
    return hashlib.sha256(data).hexdigest()


approved = re.search(r'^PREREG_SHA256 = "([0-9a-f]{64})"', open(os.path.join(HERE, "hpc-package", "run_v2.py")).read(), re.M).group(1)
ok = sha256("PREREG-v2.md") == approved
print(f"{'OK ' if ok else 'BAD'} PREREG-v2.md")
# Later tables (amendment 1) supersede earlier rows for the same file, as dict() does in run_v2.py.
frozen = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", open(os.path.join(HERE, "PREREG-v2.md")).read(), re.M))
for rel, h in frozen.items():
    good = os.path.exists(os.path.join(HERE, rel)) and sha256(rel) == h
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {rel}")
sys.exit(0 if ok else 1)
