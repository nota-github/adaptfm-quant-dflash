"""Entrypoint: ``PYTHONPATH=src python -m serving`` (from the repo root).

Reads optional env vars and hands off to uvicorn:

  EQC_HOST          default 0.0.0.0
  EQC_PORT          default 8080  (competition contract)
  EQC_CONFIG_PATH   YAML config; consumed by app at startup
  EQC_USE_STUB=1    force the in-process stub engine (tests)
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("EQC_HOST", "0.0.0.0")
    port = int(os.environ.get("EQC_PORT", "8080"))
    uvicorn.run("serving.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
