"""Download de AI4I 2020 Predictive Maintenance dataset.

python -m src.download_data
"""

import ssl
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DOEL = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"


def main() -> None:
    if DOEL.exists():
        print(f"Dataset bestaat al: {DOEL}")
        return

    DOEL.parent.mkdir(parents=True, exist_ok=True)

    # Sommige systemen hebben een verouderd SSL-certificaat
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"Downloaden van {URL} ...")
    req = urllib.request.Request(URL)
    with urllib.request.urlopen(req, context=ctx) as resp:
        DOEL.write_bytes(resp.read())
    print(f"Opgeslagen als {DOEL}")


if __name__ == "__main__":
    main()
