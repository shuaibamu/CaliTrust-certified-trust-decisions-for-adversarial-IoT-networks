"""Download the SNAP Bitcoin-OTC and Bitcoin-Alpha networks into data/ (or $CALITRUST_DATA)."""
import os, urllib.request

DATA = os.environ.get("CALITRUST_DATA", "data")
FILES = {"soc-sign-bitcoinotc.csv.gz": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
         "soc-sign-bitcoinalpha.csv.gz": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"}

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(DATA, name)
        if os.path.exists(dst):
            print("exists", dst); continue
        print("downloading", url); urllib.request.urlretrieve(url, dst)
        print("  ->", dst, os.path.getsize(dst), "bytes")
