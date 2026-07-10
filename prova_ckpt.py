#!/usr/bin/env python3
import os
from pathlib import Path

PATH = "~/projects/dataset/STABLE-DIFFUSION/sd-v1-1.safetensors"  # <-- metti qui il tuo path

def main() -> int:
    p = Path(PATH).expanduser()
    exists = p.exists()
    is_file = p.is_file() if exists else False
    readable = os.access(p, os.R_OK) if exists else False
    size = p.stat().st_size if (exists and is_file) else None

    print(f"path:     {p}")
    print(f"exists:   {exists}")
    print(f"is_file:  {is_file}")
    print(f"readable: {readable}")
    if size is not None:
        print(f"size:     {size} bytes")

    if not exists:
        print("RISULTATO: NON reperibile (non esiste)")
        return 1
    if not is_file:
        print("RISULTATO: NON reperibile (non è un file)")
        return 2
    if not readable:
        print("RISULTATO: NON reperibile (permessi di lettura mancanti)")
        return 3

    print("RISULTATO: reperibile ✅")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
