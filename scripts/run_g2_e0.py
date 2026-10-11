"""TV2 E0 entrypoint; works from any current directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.e0 import main


if __name__ == "__main__":
    raise SystemExit(main())
