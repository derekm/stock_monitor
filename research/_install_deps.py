import subprocess
import sys

# Install required packages
packages = ["polars", "tsfm-public", "torch", "torchvision", "torchaudio"]
for pkg in packages:
    print(f"Installing {pkg}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"  -> exit code: {result.returncode}")

# Verify
print("\nVerifying imports...")
try:
    import polars
    print(f"polars: {polars.__version__}")
except ImportError as e:
    print(f"polars: FAILED - {e}")

try:
    import tsfm_public
    print(f"tsfm_public: OK")
except ImportError as e:
    print(f"tsfm_public: FAILED - {e}")

try:
    import torch
    print(f"torch: {torch.__version__} CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"torch: FAILED - {e}")