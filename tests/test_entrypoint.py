import subprocess
import sys


def test_train_help():
    result = subprocess.run([sys.executable, "train.py", "--help"], capture_output=True)
    assert result.returncode == 0


def test_train_requires_confirm_flag():
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
    assert result.returncode != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--confirm-train" in combined
