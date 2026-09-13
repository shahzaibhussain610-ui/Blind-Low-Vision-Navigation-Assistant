import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
test_paths = ["tests"]
print(f"Running: {' '.join(test_paths)}", flush=True)
try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *test_paths, "--tb=short", "-v"],
        capture_output=True, text=True, timeout=20
    )
    print("STDOUT:", result.stdout[-3000:], flush=True)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:], flush=True)
    print(f"Return code: {result.returncode}", flush=True)
except subprocess.TimeoutExpired:
    print(f"TIMEOUT after 20 seconds for {' '.join(test_paths)}!", flush=True)