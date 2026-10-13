"""Fetch only the probe images needed by compute_setting_a_stability.py.

Source: the public Hugging Face mirror of NIH ChestX-ray14
(alkzar90/NIH-Chest-X-ray-dataset), which stores the images in 12 zip archives.
The archives are read with HTTP range requests, so only the central directory
and the requested PNG members are transferred (about 0.4 MB per image), not the
40 GB of archives. Files that already exist are skipped.

    python paper/fetch_probe_images.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "ec2_final_run/demo_sample/images"
LIST = Path(__file__).resolve().parent / "probe_images_needed.txt"
URL = ("https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/"
       "data/images/images_{:03d}.zip")


class HttpFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, with a block cache."""

    BLOCK = 1 << 20

    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url                                  # resolved CDN location
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.cache = {}
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a = i * self.BLOCK
            b = min(a + self.BLOCK, self.size) - 1
            for attempt in range(4):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
            self.cache[i] = r.content
            self.fetched += len(r.content)
            if len(self.cache) > 48:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    need = sorted({ln.strip() for ln in open(LIST) if ln.strip() and not ln.startswith("#")})
    todo = set(n for n in need if not (DEST / n).exists())
    print(f"{len(need)} images needed, {len(need) - len(todo)} already present")
    total = 0
    for k in range(1, 13):
        if not todo:
            break
        f = HttpFile(URL.format(k))
        zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
        members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".png")}
        hit = sorted(todo & members.keys())
        for name in hit:
            data = zf.read(members[name])                 # zipfile verifies the CRC
            (DEST / name).write_bytes(data)
            todo.discard(name)
        total += f.fetched
        print(f"images_{k:03d}.zip: {len(members)} members, fetched {len(hit)} "
              f"({f.fetched / 1e6:.1f} MB transferred)", flush=True)
    print(f"done: {total / 1e6:.1f} MB transferred, {len(todo)} still missing")
    if todo:
        sys.exit("missing: " + ", ".join(sorted(todo)[:10]))


if __name__ == "__main__":
    main()
