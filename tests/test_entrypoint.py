import subprocess
import sys


def test_train_help():
    result = subprocess.run([sys.executable, "train.py", "--help"], capture_output=True)
    assert result.returncode == 0
