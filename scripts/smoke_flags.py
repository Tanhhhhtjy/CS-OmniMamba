"""Quick smoke test for new CLI flags on the server - 1 epoch each."""
import subprocess, sys

configs = [
    (["--loss", "mse"],    "E_smoke_mse"),
    (["--loss", "facl"],   "E_smoke_facl"),
    (["--optimizer", "adamw"], "E_smoke_adamw"),
    (["--scheduler", "cosine"], "E_smoke_cosine"),
    (["--loss", "facl", "--optimizer", "adamw", "--scheduler", "cosine"], "E_smoke_full"),
]

for extra, name in configs:
    cmd = [
        sys.executable, "-m", "src.train",
        "--run-name", name,
        "--epochs", "1",
        "--batch-size", "4",
        "--workers", "2",
    ] + extra
    print(f"\n{'='*60}\nRunning: {' '.join(extra)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(1)
    else:
        print("OK")

print("\nAll smoke tests passed.")
