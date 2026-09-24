"""Build the report, then make its table of contents agree with itself.

Hand-typed page numbers in a contents table go stale the moment a paragraph is
edited. This renders the document, reads back the page each heading actually
lands on, rewrites toc_pages.json, and rebuilds — repeating until the numbers
stop moving (adding a page to the TOC can push the body down by one).
"""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCX = HERE / "BCSE497J_ProjectI_Report.docx"
WORK = HERE / "toc_work"
JSONF = HERE / "toc_pages.json"

HEADS = [
    "1. INTRODUCTION", "1.1 Background", "1.2 Motivation", "1.3 Scope of the Project",
    "2. PROJECT DESCRIPTION AND GOALS", "2.1 Literature Review", "2.2 Research Gap",
    "2.3 Objectives", "2.4 Problem Statement", "2.5 Project Plan",
    "3. TECHNICAL SPECIFICATION", "3.1 Requirements", "3.1.1 Functional",
    "3.1.2 Non-Functional", "3.2 Feasibility Study", "3.2.1 Technical Feasibility",
    "3.2.2 Economic Feasibility", "3.2.3 Social Feasibility",
    "3.3 System Specification", "3.3.1 Hardware Specification",
    "3.3.2 Software Specification", "4. DESIGN APPROACH AND DETAILS",
    "4.1 System Architecture", "4.2 Design", "4.2.1 Data Flow Diagram",
    "4.2.2 Use Case Diagram", "4.2.3 Class Diagram", "4.2.4 Sequence Diagram",
    "5. REFERENCES",
]


def build():
    subprocess.run(["node", str(HERE / "build_report.js")], check=True,
                   capture_output=True)


def page_map(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                    "--outdir", str(run_dir), str(DOCX)],
                   check=True, capture_output=True)
    pdf = run_dir / (DOCX.stem + ".pdf")
    n = int(subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
            .stdout.split("Pages:")[1].split()[0])
    norm = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"],
                           capture_output=True, text=True).stdout
        norm.append(re.sub(r"\s+", " ", t).lower())
    body = next(i for i, t in enumerate(norm)
                if "1. introduction" in t and "table of contents" not in t)
    out = {}
    for h in HEADS:
        p = next((i for i in range(body, n) if h.lower() in norm[i]), None)
        out[h] = (p - body + 1) if p is not None else None
    return out, n


if __name__ == "__main__":
    if not JSONF.exists():
        JSONF.write_text(json.dumps({h: 1 for h in HEADS}))
    prev = None
    for attempt in range(1, 6):
        build()
        mapping, pages = page_map(WORK / f"pass{attempt}")
        print(f"pass {attempt}: {pages} pages", flush=True)
        if mapping == prev:
            print("table of contents is stable")
            break
        JSONF.write_text(json.dumps(mapping, indent=1))
        prev = mapping
    else:
        print("WARNING: page numbers did not settle")
    build()
    missing = [h for h, v in json.loads(JSONF.read_text()).items() if v is None]
    print("headings not located:", missing or "none")
