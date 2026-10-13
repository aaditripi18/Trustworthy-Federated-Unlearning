"""Attention-Fidelity-Preserving Federated Unlearning with Vision Transformers
for Chest X-Rays — technical archive & live model workspace.

A numbered concept path (narrative -> equation -> worked example with REAL
project numbers -> takeaway), two live workspaces running the actual trained
checkpoints, the measured results with an honest verdict, and the full training
configuration read from the run's own records.

Run from the repository root:
    .venv\\Scripts\\python.exe -m streamlit run demo\\streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.nih import NIH_LABELS

_MEAN, _STD = 0.5, 0.25

EXP_ROOT = REPO_ROOT / "ec2_final_run" / "tfu" / "experiments"
FULL_EXP = EXP_ROOT / "full_experiment"
SWEEP_EXP = EXP_ROOT / "exp_a1"
DEFAULT_IMAGES = REPO_ROOT / "ec2_final_run" / "demo_sample" / "images"
DEFAULT_CSV = REPO_ROOT / "ec2_final_run" / "demo_sample" / "Data_Entry_2017_v2020.csv"
ASSETS = Path(__file__).resolve().parent / "assets"
_FAVICON = ASSETS / "favicon.png"
# Charts plotted from the run CSVs by demo/make_figs.py. They are kept here
# rather than in experiments/*/figures so the training run's own output stays
# exactly as the job wrote it.
PRES_FIGS = ASSETS / "figures"
# Derived results behind the paper: setting-A stability, membership audit,
# gradient-ascent baseline, and the block diagram.
PAPER = REPO_ROOT / "paper"

PROJECT_TITLE = ("Attention-Fidelity-Preserving Federated Unlearning with "
                 "Vision Transformers for Chest X-Rays")

st.set_page_config(page_title=PROJECT_TITLE,
                   page_icon=str(_FAVICON) if _FAVICON.exists() else "🩺",
                   layout="wide", initial_sidebar_state="expanded")

# Brand mark, inline so it needs no network request: a federated hub with three
# retained clients and one faded — the client being forgotten.
BRAND_SVG = (
    '<svg viewBox="0 0 64 64" width="30" height="30" aria-hidden="true">'
    '<g stroke="#ffffff" stroke-width="3.4" stroke-linecap="round">'
    '<line x1="32" y1="32" x2="32" y2="15"/><line x1="32" y1="32" x2="49" y2="32"/>'
    '<line x1="32" y1="32" x2="32" y2="49"/></g>'
    '<line x1="32" y1="32" x2="15" y2="32" stroke="#ffffff" stroke-width="3.4" '
    'stroke-linecap="round" opacity=".3"/>'
    '<circle cx="32" cy="13" r="6" fill="#ffffff"/>'
    '<circle cx="51" cy="32" r="6" fill="#ffffff"/>'
    '<circle cx="32" cy="51" r="6" fill="#ffffff"/>'
    '<circle cx="13" cy="32" r="6" fill="#ffffff" opacity=".3"/>'
    '<circle cx="32" cy="32" r="8.4" fill="#ffffff"/>'
    '<circle cx="32" cy="32" r="3.6" fill="#2563eb"/></svg>')

# ---------------------------------------------------------------------------
# NOTE ON STREAMLIT INTERNALS (maintenance)
# The stylesheet targets a handful of Streamlit's private DOM hooks to restyle
# widgets it gives no theming API for. These are NOT public API and can change
# between releases. Verified against Streamlit 1.60.
#
#   [data-testid="stSidebar"]          sidebar container      (nav layout)
#   [data-testid="stVerticalBlock"]    element gap            (nav row spacing)
#   [data-testid="stElementContainer"] per-widget margins
#   [data-testid="stAppViewContainer"] main canvas            (blueprint grid)
#   [data-testid="stHeader"]           floating toolbar       (clearance)
#   [data-testid="stWidgetLabel"]      control labels
#   [data-testid="stSelectbox"] / [stFileUploader]            (control spacing)
#   [data-testid="stIconMaterial"]     nav icon glyph
#   [data-testid="stDecoration"]       top gradient bar       (hidden)
#   button[kind="primary"] + [data-testid="stBaseButton-primary"]
#                                      active nav item (both spellings targeted
#                                      on purpose: older and newer Streamlit)
#   .st-key-warmup                     keyed-widget class     (warm-up button)
#
# If a Streamlit upgrade breaks the look, check these first. Failure mode is
# cosmetic only — every page still renders and every control still works.
# ---------------------------------------------------------------------------
CSS = """
<style>
/* Webfonts are a progressive enhancement: if there is no internet the stacks
   below fall back to fonts already on the machine and the layout is unchanged.
   No icon font is imported — all icons are inline SVG. */
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@100..900&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* Palette. Only the four tokens below differ between light and dark; every
   other colour is mixed from them, so the whole archive inverts consistently.
   An injected :root block (see _active_theme) re-bases them at runtime. */
:root{
  --primary:#2563eb; --on-primary:#ffffff;
  --surface:#fdf8fd; --surface-low:#f3f0f4; --on-surface:#1a1b20;

  --on-surface-variant: color-mix(in srgb, var(--on-surface) 82%, var(--surface));
  --outline:            color-mix(in srgb, var(--on-surface) 66%, var(--surface));
  --outline-variant:    color-mix(in srgb, var(--on-surface) 30%, var(--surface));
  --surface-high:       color-mix(in srgb, var(--on-surface) 9%,  var(--surface));
  --card:               color-mix(in srgb, var(--on-surface) 3%,  var(--surface));
  --primary-container:  color-mix(in srgb, var(--primary) 14%, var(--surface));
  --on-primary-container: color-mix(in srgb, var(--primary) 82%, var(--on-surface));
  --error:#ba1a1a;
}
/* Fallback for environments where the theme cannot be detected in Python. */
@media (prefers-color-scheme: dark){
  :root{ --surface:#0e1117; --on-surface:#e8eaf0; --surface-low:#1a1d24;
         --primary:#7ea6ff; }
}

/* Body font. Deliberately NOT applied to bare span/div: that would fight
   KaTeX, which needs its own maths glyphs to render formulas correctly. */
html, body, .stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown td, .stMarkdown th{
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,ui-monospace,monospace;
}
h1,h2,h3,h4,h5{ font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif !important;
  letter-spacing:-.015em; }

/* zero-radius everywhere — the defining trait of the design */
*, *::before, *::after{ border-radius:0 !important; }

/* ---------- mathematics (KaTeX) ---------- */
/* No font-family is declared for .katex on purpose: KaTeX ships its own maths
   fonts and must be left alone, or the glyphs garble. */
.katex{ font-size:1.06em; line-height:1.5; color:var(--on-surface); }
.katex-display{
  margin:22px 0 !important; padding:18px 20px;
  border:1px solid var(--outline-variant); border-left:3px solid var(--primary);
  background:var(--card); overflow-x:auto; overflow-y:hidden;
}
.katex-display > .katex{ text-align:center; white-space:nowrap; }
.stMarkdown p .katex{ padding:0 1px; }
.fcap{ font-size:12.5px; color:var(--on-surface-variant); text-align:center;
  margin:-14px 0 24px 0; line-height:1.5; }

/* ---------- canvas ---------- */
[data-testid="stAppViewContainer"] > .main{
  background-color:var(--surface);
  background-image:
    linear-gradient(to right, color-mix(in srgb, var(--on-surface) 7%, transparent) 1px, transparent 1px),
    linear-gradient(to bottom, color-mix(in srgb, var(--on-surface) 7%, transparent) 1px, transparent 1px);
  background-size:16px 16px;
}
/* Streamlit floats a transparent toolbar (Deploy / ⋮) over the top of the page.
   Without this clearance the breadcrumb slides under it and looks clipped. */
.block-container{ padding-top:3.6rem; padding-bottom:3rem; max-width:1280px; }
[data-testid="stHeader"]{ background:transparent; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ width:320px !important;
  background:var(--surface); border-right:1px solid var(--outline-variant); }
[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }

.nav-brand{ display:flex; gap:14px; align-items:center; padding:0 8px 30px 8px; }
.nav-brand .mk{
  /* brand blue is fixed in both themes — a logo shouldn't change colour, and
     the white mark needs the deep blue behind it to stay legible */
  width:48px; height:48px; background:#2563eb; color:#ffffff;
  display:flex; align-items:center; justify-content:center;
  font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif;
  font-weight:700; font-size:17px; flex:0 0 48px;
}
.nav-brand .bname{ font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif;
  font-size:16px; font-weight:700; margin:0; line-height:1.25; color:var(--on-surface); }
.nav-brand p{ font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
  font-size:12px; color:var(--on-surface-variant); margin:4px 0 0 0; }

.grp{ font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
  font-size:12px; font-weight:600; letter-spacing:.18em; text-transform:uppercase;
  color:var(--outline); padding:26px 0 10px 20px; margin:0; }

/* nav rows: 8px apart, identical geometry whether active or not */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:8px !important; }
[data-testid="stSidebar"] [data-testid="stElementContainer"]{
  margin:0 !important; padding:0 !important; }
[data-testid="stSidebar"] .stButton{ margin:0 !important; padding:0 !important; }
[data-testid="stSidebar"] .stMarkdown{ margin:0 !important; }
[data-testid="stSidebar"] .stButton>button{
  width:100%; display:flex; align-items:center;
  justify-content:flex-start !important; gap:14px; text-align:left;
  padding:11px 20px !important; margin:0 !important;
  min-height:0 !important; height:auto;
  background:transparent; border:none !important;
  border-left:2px solid transparent !important;
  color:var(--on-surface-variant);
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
  font-size:13.5px; font-weight:400; line-height:1.4; transition:all .2s ease;
}
[data-testid="stSidebar"] .stButton>button p{
  font-size:13.5px !important; margin:0 !important; line-height:1.4;
  text-align:left !important; width:100%; }
/* Streamlit centres a button's inner content; force it left so the icons form
   one column and the 01/02/03/04 numbering lines up vertically. */
[data-testid="stSidebar"] .stButton>button > div,
[data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"]{
  width:100%; display:flex; justify-content:flex-start; text-align:left; }
[data-testid="stSidebar"] .stButton>button:hover{
  background:color-mix(in srgb, var(--primary) 8%, transparent);
  color:var(--on-surface); }
[data-testid="stSidebar"] .stButton>button:hover [data-testid="stIconMaterial"]{
  color:var(--primary); }
[data-testid="stSidebar"] .stButton>button:focus{ box-shadow:none !important; }

/* ACTIVE item = Streamlit "primary" button, restyled to the left-border +
   tinted-background + bold-primary-text treatment. */
[data-testid="stSidebar"] .stButton>button[kind="primary"],
[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{
  border-left:2px solid var(--primary) !important;
  background:color-mix(in srgb, var(--primary) 8%, transparent) !important;
  color:var(--primary) !important; font-weight:700 !important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"] p,
[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"] p{
  font-weight:700 !important; color:var(--primary) !important; }
[data-testid="stSidebar"] .stButton>button[kind="primary"] [data-testid="stIconMaterial"]{
  color:var(--primary) !important; }
[data-testid="stSidebar"] [data-testid="stIconMaterial"]{ font-size:20px; }

/* the warm-up action is NOT navigation — bordered, centred, so it never reads
   as a page link */
[data-testid="stSidebar"] .st-key-warmup button{
  justify-content:center !important; margin-top:10px !important;
  border:1px solid var(--outline-variant) !important;
  background:var(--card) !important; color:var(--on-surface-variant) !important;
  font-weight:600 !important; }
[data-testid="stSidebar"] .st-key-warmup button:hover{
  border-color:var(--primary) !important; color:var(--primary) !important; }
[data-testid="stSidebar"] .st-key-warmup button > div,
[data-testid="stSidebar"] .st-key-warmup button p{
  justify-content:center !important; text-align:center !important; }

/* widget labels in the design system's voice, so controls are self-explaining */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace !important;
  font-size:12px !important; font-weight:600; letter-spacing:.12em;
  text-transform:uppercase; color:var(--on-surface-variant);
  margin:0 0 5px 0 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"]{
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace; font-size:13px; }
[data-testid="stSidebar"] [data-testid="stSelectbox"],
[data-testid="stSidebar"] [data-testid="stFileUploader"]{ margin-bottom:14px !important; }

/* ---------- top bar ---------- */
.topbar{
  display:flex; align-items:center; gap:8px; line-height:1.7;
  border-bottom:1px solid var(--outline-variant);
  padding:6px 0 12px 0; margin-bottom:26px;
  font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif;
  font-size:16px; color:var(--on-surface-variant);
}
.topbar .sep{ color:var(--outline-variant); }
.topbar .cur{ color:var(--primary); font-weight:700; }

/* ---------- typography ---------- */
.kicker{
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace; font-size:12px;
  font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:var(--primary); border:1px solid var(--outline-variant);
  display:inline-block; padding:4px 10px; margin-bottom:14px; background:var(--card);
}
/* fluid headline: never overflows a narrow canvas, never shrinks on a wide one.
   Selector is h1.display so it outranks Streamlit's own heading rules. */
h1.display, .display{ font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif;
  font-size:clamp(27px, 3.3vw, 42px) !important; line-height:1.12;
  letter-spacing:-.02em; font-weight:700; color:var(--on-surface);
  margin:0 0 14px 0 !important; padding:0 !important; }
.lede{ font-size:16px; opacity:.92; max-width:62rem; line-height:1.65;
  color:var(--on-surface-variant); margin-bottom:26px; }
/* the formal submitted title, kept distinct from the plain-English headline */
.proj-title{ font-size:15px; font-weight:600; line-height:1.5; max-width:62rem;
  color:var(--on-surface); border-left:3px solid var(--primary);
  padding:2px 0 2px 12px; margin:-6px 0 20px 0; }
.stMarkdown p, .stMarkdown li{ font-size:14.5px; line-height:1.65;
  color:var(--on-surface-variant); }
.stMarkdown strong{ color:var(--on-surface); }

.badge{ display:inline-block; padding:4px 12px; margin:0 6px 8px 0;
  border-radius:0; font-size:12.5px; font-weight:600;
  border:1px solid var(--outline-variant); background:var(--card);
  color:var(--on-surface-variant); }

/* ---------- containers ---------- */
.panel{ border:1px solid var(--outline-variant); background:var(--surface-low);
  margin-bottom:26px; }
.panel-head{ border-bottom:1px solid var(--outline-variant); background:var(--card);
  padding:12px 16px; display:flex; gap:26px; justify-content:center; flex-wrap:wrap; }
.legend{ display:flex; align-items:center; gap:8px; font-size:13px;
  color:var(--on-surface); }
.dot{ width:9px; height:9px; display:inline-block; }
.panel-body{ padding:22px 20px; }
.panel-foot{ border-top:1px solid var(--outline-variant); background:var(--card);
  padding:18px 20px; }
.panel-foot .panel-title{ font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif;
  font-size:15px; font-weight:600; margin:0 0 12px 0; color:var(--on-surface);
  display:flex; align-items:center; gap:10px; }

.steps{ list-style:none; padding:0; margin:0; }
.steps li{ display:flex; gap:14px; margin-bottom:12px; }
/* one accent colour site-wide: step numbers use the primary, not an error red */
.steps .n{ color:var(--primary); font-weight:700; min-width:22px; }
.steps p{ margin:0; font-size:14px; line-height:1.6; color:var(--on-surface-variant); }

/* one grid = every card in a row is the same height, whatever its text length */
.kpi-row{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:14px; margin:0 0 18px 0; align-items:stretch; }
.kpi{ border:1px solid var(--outline-variant); background:var(--card);
  padding:14px 16px; height:100%; display:flex; flex-direction:column; }
.kpi .l{ font-size:12px; opacity:.78; margin-bottom:.15rem;
  text-transform:uppercase; letter-spacing:.08em; }
.kpi .v{ font-size:26px; font-weight:750; font-family:'Geist','Segoe UI',system-ui,Arial,sans-serif; }
.kpi .s{ font-size:12.5px; opacity:.72; margin-top:auto; padding-top:6px; line-height:1.45; }

.take{ border:1px solid var(--outline-variant); border-left:3px solid var(--primary);
  background:var(--primary-container); padding:14px 18px; margin:22px 0;
  font-size:14px; line-height:1.6; color:var(--on-primary-container); }
.take b{ color:var(--on-primary-container); }

.gt-chip{ display:inline-block; padding:3px 10px; margin:2px 4px 2px 0;
  font-size:12.5px; font-weight:600; background:var(--primary-container);
  border:1px solid var(--primary); color:var(--on-primary-container); }

.nextbox{ border:1px solid var(--outline-variant); background:var(--card);
  padding:12px 16px; margin-top:28px; font-size:13.5px;
  color:var(--on-surface-variant); }

/* ---------- contents rail ---------- */
.toc-wrap{ position:sticky; top:24px; }
.toc-h{ font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
  font-size:12px; font-weight:600; letter-spacing:.2em; text-transform:uppercase;
  color:var(--outline); margin-bottom:18px; }
.toc{ display:flex; flex-direction:column; }
.toc a{ display:block; border-left:1px solid var(--outline-variant);
  padding:6px 0 6px 16px; margin-bottom:12px;
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
  font-size:13px; line-height:1.5; color:var(--on-surface-variant);
  text-decoration:none; transition:all .2s ease; }
/* No permanent "active" styling on the first entry: nothing tracks scroll
   position, so highlighting one link would falsely claim "you are here". */
.toc a:hover{ color:var(--primary); border-left-color:var(--primary); }

/* ---------- tables ---------- */
.stMarkdown table{ border-collapse:collapse; width:100%; font-size:13.5px; }
.stMarkdown th{ background:var(--surface-high); border:1px solid var(--outline-variant);
  padding:8px 10px; text-align:left; color:var(--on-surface); font-weight:600; }
.stMarkdown td{ border:1px solid var(--outline-variant); padding:8px 10px;
  color:var(--on-surface-variant); vertical-align:top; }

/* ---------- diagram primitives ----------
   Every node identical in size and every connector identical in width, so
   diagrams stay aligned no matter how much text a box holds. */
.node{ border:1px solid var(--outline); background:var(--surface-high);
  padding:12px 10px; text-align:center; color:var(--on-surface);
  min-height:78px; box-sizing:border-box;
  display:flex; flex-direction:column; align-items:center; justify-content:center; }
.node .t{ color:var(--on-surface-variant); font-size:12px; font-weight:600;
  text-transform:uppercase; letter-spacing:.09em; margin-bottom:6px;
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace; }
.node .b{ font-size:13px; line-height:1.45;
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace; }
.node.hi{ border-color:var(--primary); background:var(--card); }
.node.hi .t{ color:var(--primary); }
.flow{ display:flex; align-items:stretch; justify-content:center; width:100%; }
.flow > .node{ flex:1 1 0; min-width:0; }
.arrow{ flex:0 0 30px; display:flex; align-items:center; justify-content:center;
  color:var(--outline); font-size:16px; line-height:1; user-select:none; }
/* parallel alternatives (not a sequence) — auto-fit collapses on narrow screens */
.grid3{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:14px; width:100%; }

.ico{ vertical-align:middle; flex:0 0 auto; display:inline-block; }
.topbar .ico{ color:var(--outline); }
.panel-foot .panel-title .ico{ color:var(--primary); width:18px; height:18px; }

.site-foot{ border-top:1px solid var(--outline-variant); margin-top:40px;
  padding-top:14px; font-size:12px; color:var(--outline);
  font-family:'JetBrains Mono','Cascadia Mono',Consolas,monospace; }
footer, [data-testid="stDecoration"]{ display:none !important; }

/* ---------- responsive: projectors, 125% scaling, small laptops ---------- */
@media (max-width:1250px){
  /* below this the contents rail is too narrow to be legible */
  .toc-wrap{ display:none; }
}
@media (max-width:1100px){
  [data-testid="stSidebar"]{ width:260px !important; }
  /* sequential diagrams reflow into a wrapping grid; arrows are dropped
     because the stage/step labels already carry the order */
  .flow{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:10px; }
  .flow > .node{ flex:none; }
  .arrow{ display:none; }
  .panel-body{ padding:18px 14px; }
  .panel-head{ gap:16px; }
}
@media (max-width:820px){
  .block-container{ padding-left:.7rem; padding-right:.7rem; }
  .node{ min-height:70px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _active_theme():
    """The theme actually in effect — 'light', 'dark', or None if it cannot be
    determined. Streamlit resolves 'System' to a concrete value for us, so this
    is correct for all three menu options. Returns None on older Streamlit, in
    which case the prefers-color-scheme fallback in CSS takes over."""
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else None
    except Exception:
        return None


_PALETTE = {
    "light": ("#fdf8fd", "#1a1b20", "#f3f0f4", "#2563eb"),
    "dark":  ("#0e1117", "#e8eaf0", "#1a1d24", "#7ea6ff"),
}
_theme = _active_theme()
if _theme:
    _s, _on, _low, _pri = _PALETTE[_theme]
    st.markdown(
        f"<style>:root{{--surface:{_s};--on-surface:{_on};"
        f"--surface-low:{_low};--primary:{_pri};}}</style>",
        unsafe_allow_html=True)


# ===========================================================================
# Cached loaders & inference helpers
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    import torch
    from src.models.vit import build_model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    # pretrained=False: the checkpoint's state_dict fully overwrites the weights,
    # so downloading ImageNet weights is pointless — and this keeps the demo
    # working with NO internet connection (important for the presentation room).
    model = build_model(dict(cfg["model"], pretrained=False),
                        cfg["data"]["image_size"], cfg["data"]["num_labels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt.get("seed")


@st.cache_data(show_spinner=False)
def load_summary(exp_dir: str):
    p = Path(exp_dir) / "final_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_repro(exp_dir: str):
    p = Path(exp_dir) / "reproducibility.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def per_config_stats(exp_dir: str):
    """mean ± std of every numeric metric, per config, computed straight from
    final_results.csv — so metrics the summary JSON omits (the *_to_original
    family, Pearson, IoU, MSE) are still available and never hand-typed."""
    import csv as _csv
    import statistics as _st
    from collections import defaultdict
    p = Path(exp_dir) / "final_results.csv"
    if not p.exists():
        return {}
    rows = list(_csv.DictReader(p.open()))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["config"]].append(r)
    out = {}
    for cfg, rs in grouped.items():
        d = {}
        for k in rs[0]:
            vals = []
            for r in rs:
                try:
                    vals.append(float(r[k]))
                except (TypeError, ValueError):
                    pass
            if vals and len(vals) == len(rs):
                d[k] = (_st.mean(vals), _st.stdev(vals) if len(vals) > 1 else 0.0)
        out[cfg] = d
    return out


@st.cache_data(show_spinner=False)
def paired_gain(exp_dir: str, metric: str, base_cfg: str = "finetune_retain"):
    """Per-seed paired difference against a baseline config. Pairing by seed is
    what makes a small mean difference interpretable."""
    import csv as _csv
    import statistics as _st
    from collections import defaultdict
    p = Path(exp_dir) / "final_results.csv"
    if not p.exists():
        return {}
    rows = list(_csv.DictReader(p.open()))
    base = {r["seed"]: float(r[metric]) for r in rows
            if r["config"] == base_cfg and r.get(metric) not in ("", None, "nan")}
    grouped = defaultdict(list)
    for r in rows:
        if r["config"] == base_cfg or r["seed"] not in base:
            continue
        try:
            grouped[r["config"]].append(float(r[metric]) - base[r["seed"]])
        except (TypeError, ValueError):
            pass
    return {cfg: (_st.mean(d), _st.stdev(d) if len(d) > 1 else 0.0,
                  all(x > 0 for x in d), d)
            for cfg, d in grouped.items() if d}


@st.cache_data(show_spinner=False)
def paper_table(name: str, keys: tuple = ("config",)):
    """mean and std over seeds of every numeric column of a derived result file in
    paper/ (setting-A stability, membership audit, gradient-ascent baseline),
    grouped by `keys` — read from the CSV, never hand-typed."""
    import pandas as pd
    fp = PAPER / name
    if not fp.exists():
        return {}
    df = pd.read_csv(fp)
    num = [c for c in df.columns
           if c not in keys and c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    out = {}
    for k, g in df.groupby(list(keys)):
        k = k if isinstance(k, tuple) else (k,)
        out[k if len(keys) > 1 else k[0]] = {
            c: (float(g[c].mean()), float(g[c].std(ddof=1)) if len(g) > 1 else 0.0)
            for c in num}
    return out


def pm(v, d: int = 3) -> str:
    return f"{v[0]:.{d}f} ± {v[1]:.{d}f}"


@st.cache_data(show_spinner=False)
def cooccurrence(csv_path: str):
    """How often each pair of findings appears together in the label file.
    Used to show that the model's extra flags are radiologically plausible
    confusions rather than arbitrary ones — with the counts, not just a claim."""
    from collections import Counter, defaultdict
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}, {}
    valid = set(NIH_LABELS)
    total, joint = Counter(), defaultdict(Counter)
    for lab in df["Finding Labels"]:
        s = {x.strip() for x in str(lab).split("|")} & valid
        for a in s:
            total[a] += 1
            for b in s:
                if a != b:
                    joint[a][b] += 1
    return {k: dict(v) for k, v in joint.items()}, dict(total)


@st.cache_data(show_spinner=False)
def load_groundtruth(csv_path: str):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        return {str(r["Image Index"]): str(r["Finding Labels"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def preprocess(pil_img, image_size: int):
    from PIL import Image
    im = pil_img.convert("L").resize((image_size, image_size), Image.BILINEAR)
    disp = np.asarray(im, dtype=np.uint8)
    x = (disp.astype(np.float32) / 255.0 - _MEAN) / _STD
    import torch
    return disp, torch.from_numpy(x[None, None]).float()


def _grid_size(model, image_size: int):
    gs = getattr(getattr(model, "patch_embed", None), "grid_size", None)
    if isinstance(gs, (tuple, list)) and len(gs) == 2:
        return int(gs[0]), int(gs[1])
    return image_size // 16, image_size // 16


@st.cache_data(show_spinner=False)
def run_inference(ckpt_path: str, img_key: str, _pil):
    """Cached forward pass. Keyed on (checkpoint, image) — the PIL object is
    prefixed with '_' so it is not hashed. Without this, every rerun (changing
    theme, navigating, moving a slider) recomputed the whole forward pass."""
    import torch
    from src.explain.torch_rollout import forward_with_maps
    model, cfg, _ = load_model(ckpt_path)
    size = cfg["data"]["image_size"]
    disp, tensor = preprocess(_pil, size)
    with torch.no_grad():
        logits, maps = forward_with_maps(model, tensor, _grid_size(model, size))
        return disp, torch.sigmoid(logits)[0].cpu().numpy(), maps[0].cpu().numpy()


def _colormap(name="inferno"):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except Exception:
        from matplotlib import cm
        return cm.get_cmap(name)


def overlay_heatmap(disp_uint8, rollout_map, alpha=0.45):
    from PIL import Image
    h, w = disp_uint8.shape
    m = Image.fromarray((rollout_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    m = np.asarray(m, dtype=np.float32) / 255.0
    heat = (_colormap()(m)[:, :, :3] * 255).astype(np.float32)
    base = np.stack([disp_uint8] * 3, axis=-1).astype(np.float32)
    return ((1 - alpha) * base + alpha * heat).clip(0, 255).astype(np.uint8)


def mean_of(cell) -> float:
    try:
        return float(str(cell).split("±")[0])
    except Exception:
        return float("nan")


def epu_options(seed_dir: Path) -> dict:
    """Discover epu_lambda*.pt in a seed dir -> {'EPU (λ=10)': path, ...}."""
    def lam_of(q: Path) -> str:
        return re.findall(r"epu_lambda([\d.]+)", q.stem)[0]   # stem: no '.pt'
    out = {}
    for p in sorted(seed_dir.glob("epu_lambda*.pt"), key=lambda q: float(lam_of(q))):
        out[f"EPU (λ={lam_of(p)})"] = p
    return out


# ===========================================================================
# UI primitives
# ===========================================================================
# Inline SVG icons — no downloaded font, so they render identically with no
# internet connection (a downloaded icon font would fall back to literal text).
_SVG_PATHS = {
    "menu": '<path d="M3 6h18M3 12h18M3 18h18"/>',
    # a numbered-list mark: this footer holds a step list, not a video. A
    # play button here would imply a clickable animation that does not exist.
    "steps": ('<path d="M9 6h12M9 12h12M9 18h12"/>'
              '<path d="M4 5.5h1.6M4.8 5.5V9M4 9h1.8"/>'
              '<path d="M4 11.5h1.7L4 14.5h1.8"/>'
              '<path d="M4 17h1.7v3.5H4"/>'),
}


def icon(name: str, cls: str = "") -> str:
    return (f'<svg class="ico {cls}" viewBox="0 0 24 24" width="18" height="18" '
            'fill="none" stroke="currentColor" stroke-width="1.6" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_SVG_PATHS.get(name, "")}</svg>')


def topbar(section_name: str, current: str):
    st.markdown(f'<div class="topbar">{icon("menu")}<span>{section_name}</span>'
                f'<span class="sep">/</span><span class="cur">{current}</span></div>',
                unsafe_allow_html=True)


def kicker(text: str):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def display(text: str):
    """The page title. A real <h1> (not a styled div) so the document outline
    and screen-reader navigation are correct."""
    st.markdown(f'<h1 class="display">{text}</h1>', unsafe_allow_html=True)


def lede(text: str):
    st.markdown(f'<div class="lede">{text}</div>', unsafe_allow_html=True)


def _kpi_html(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(_kpi_html(label, value, sub), unsafe_allow_html=True)


def kpi_row(cards: list[tuple[str, str, str]]):
    """A row of KPI cards rendered as ONE css grid, so every card is exactly the
    same height regardless of how many lines its label or sub-text wraps to."""
    st.markdown('<div class="kpi-row">' + "".join(_kpi_html(*c) for c in cards)
                + "</div>", unsafe_allow_html=True)


def _node(title: str, body: str, hi: bool = False) -> str:
    """One diagram box. Always the same structure, so always the same size."""
    return (f'<div class="node{" hi" if hi else ""}">'
            f'<div class="t">{title}</div><div class="b">{body}</div></div>')


def flow(nodes: list[tuple[str, str, bool]]) -> str:
    """Sequential diagram: equal-width boxes joined by identical arrows."""
    parts = []
    for i, (title, body, hi) in enumerate(nodes):
        if i:
            parts.append('<div class="arrow">&rarr;</div>')
        parts.append(_node(title, body, hi))
    return '<div class="flow">' + "".join(parts) + "</div>"


def grid3(nodes: list[tuple[str, str, bool]]) -> str:
    """Parallel alternatives: equal boxes, no arrows implied."""
    return '<div class="grid3">' + "".join(_node(*n) for n in nodes) + "</div>"


def formula(tex: str, caption: str = ""):
    """A display equation followed by a small explanatory caption."""
    st.latex(tex)
    if caption:
        st.markdown(f'<div class="fcap">{caption}</div>', unsafe_allow_html=True)


def panel(legend: list[tuple[str, str]], body_html: str,
          foot_title: str = "", steps: list[str] | None = None):
    """Bordered technical-diagram container: legend header, body, numbered footer."""
    leg = "".join(f'<div class="legend"><span class="dot" style="background:{c}"></span>{t}</div>'
                  for c, t in legend)
    # role/aria-label so the diagram is announced as one labelled figure rather
    # than a stream of loose text nodes
    label = foot_title or "technical diagram"
    html = (f'<div class="panel" role="figure" aria-label="{label}">'
            f'<div class="panel-head">{leg}</div>'
            f'<div class="panel-body">{body_html}</div>')
    if steps:
        items = "".join(f'<li><span class="n">{i+1}.</span><p>{s}</p></li>'
                        for i, s in enumerate(steps))
        # a <div>, not an <h4>: Streamlit attaches a clickable anchor link to every
        # heading it renders, and this caption is not a navigable section.
        html += (f'<div class="panel-foot">'
                 f'<div class="panel-title">{icon("steps")}{foot_title}</div>'
                 f'<ol class="steps">{items}</ol></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def takeaway(text: str):
    st.markdown(f'<div class="take"><b>Takeaway.</b> {text}</div>',
                unsafe_allow_html=True)


def describe_platform(rep: dict):
    """Turn the recorded platform string into readable machine details.

    Python's platform.platform() records the kernel and glibc, never the
    distribution name — so the OS is *derived* here (glibc release + the '-aws'
    kernel flavour Canonical ships), and the raw string is still displayed so
    nothing is asserted beyond the run record."""
    plat = str(rep.get("platform", ""))
    glibc = (re.search(r"glibc([\d.]+)", plat) or [None, None])[1] \
        if re.search(r"glibc([\d.]+)", plat) else None
    kernel = (re.search(r"Linux-([\w.\-]+?)-x86_64", plat).group(1)
              if re.search(r"Linux-([\w.\-]+?)-x86_64", plat) else "—")
    ubuntu = {"2.31": "20.04 LTS", "2.35": "22.04 LTS",
              "2.39": "24.04 LTS", "2.41": "25.04"}.get(glibc or "")
    on_aws = "-aws" in plat
    if ubuntu:
        os_name = f"Ubuntu {ubuntu}" + (" (AWS EC2)" if on_aws else "")
    else:
        os_name = "Linux" + (" (AWS EC2)" if on_aws else "")
    gpu = str(rep.get("cuda", {}).get("device_name", ""))
    cpus = rep.get("cpu_count")
    instance = "g4dn.xlarge" if ("T4" in gpu and cpus == 4) else None
    return os_name, kernel, glibc, instance, cpus


def slugify(text: str) -> str:
    """ASCII-only anchor id. Unicode fragments (e.g. 'λ') are unreliable in URL
    fragments, so symbols are transliterated before stripping."""
    t = text.lower().replace("λ", "lambda").replace("×", "x")
    t = t.replace("→", "-").replace("↔", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def section(title: str):
    """Section heading whose anchor is derived by the same slugify() the TOC
    uses — so a heading and its TOC link can never drift apart."""
    st.subheader(title, anchor=slugify(title))


def toc(items: list[str]):
    links = "".join(f'<a href="#{slugify(t)}">{t}</a>' for t in items)
    st.markdown('<div class="toc-wrap"><div class="toc-h">Table of Contents</div>'
                f'<div class="toc">{links}</div></div>', unsafe_allow_html=True)


def nextbox(label: str | None = None):
    """Pointer to the following page. With no argument the target is derived
    from the sidebar order, so the label can never drift out of sync with NAV."""
    if label is None:
        order = [lbl for _, items in NAV for lbl, _ in items]
        try:
            i = order.index(st.session_state.page)
        except ValueError:
            return
        if i + 1 >= len(order):
            return                      # last page: nothing to point at
        label = order[i + 1]
    st.markdown(f'<div class="nextbox">Next → <b>{label}</b> — select it in the '
                'sidebar</div>', unsafe_allow_html=True)


# ===========================================================================
# Navigation — grouped, trimmed to what matters
# ===========================================================================
NAV = [
    ("", [("Home", "home")]),
    ("Learn — the concept path", [
        ("01 · Federated Learning", "hub"),
        ("02 · Unlearning & Forgetting", "delete_sweep"),
        ("03 · EPU — proposed method", "shield"),
        ("04 · Attention Rollout", "visibility"),
    ]),
    ("Live — run the models", [
        ("Diagnosis", "monitor_heart"),
        ("Unlearning", "compare_arrows"),
    ]),
    ("Evidence", [
        ("Results & Verdict", "query_stats"),
        ("Training Setup", "tune"),
    ]),
    ("Context", [
        ("Related Work & Gap", "menu_book"),
        ("Scope & Design Decisions", "balance"),
        ("References", "format_quote"),
    ]),
]

if "page" not in st.session_state:
    st.session_state.page = "Home"


def nav_button(label: str, ic: str, active: bool) -> bool:
    """One nav row: icon + label. Active rows use Streamlit's 'primary' kind,
    which the CSS restyles — so active and inactive share identical geometry."""
    kw = dict(key=f"nav_{label}", width="stretch",
              type="primary" if active else "secondary")
    try:
        return st.button(label, icon=f":material/{ic}:", **kw)
    except TypeError:      # older Streamlit without icon support
        return st.button(label, **kw)


with st.sidebar:
    st.markdown(
        f'<div class="nav-brand"><div class="mk">{BRAND_SVG}</div>'
        # a div, not an <h1>: the page title is the only h1, so the document
        # outline stays correct and Streamlit adds no anchor here
        '<div><div class="bname">Trustworthy FU</div>'
        '<p>Technical Archive</p></div></div>',
        unsafe_allow_html=True)

    for group, items in NAV:
        if group:
            st.markdown(f'<div class="grp">{group}</div>', unsafe_allow_html=True)
        for label, ic in items:
            if nav_button(label, ic, st.session_state.page == label):
                st.session_state.page = label
                st.rerun()

    # A control is RENDERED only on the pages that actually consume it, so the
    # sidebar never shows a setting that does nothing. Values live in plain
    # session_state keys, so a selection survives navigating away and back.
    _p = st.session_state.page
    uses_exp = _p in {"Unlearning", "Results & Verdict", "Training Setup"}
    uses_seed = _p in {"Diagnosis", "Unlearning"}
    uses_img = _p in {"Diagnosis", "Unlearning"}

    EXP_OPTS = ["Main (λ = 0.5, 1.0)", "λ-sweep (λ = 1…100)"]
    SEED_OPTS = ["seed_42", "seed_43", "seed_44"]
    img_files = sorted((p.name for p in DEFAULT_IMAGES.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                       key=str) if DEFAULT_IMAGES.is_dir() else []

    st.session_state.setdefault("v_exp", EXP_OPTS[0])
    st.session_state.setdefault("v_seed", SEED_OPTS[0])
    if img_files and st.session_state.get("v_img") not in img_files:
        st.session_state.v_img = img_files[0]

    if uses_exp or uses_seed or uses_img:
        st.markdown('<div class="grp">Session</div>', unsafe_allow_html=True)
    if uses_exp:
        st.session_state.v_exp = st.selectbox(
            "Experiment run", EXP_OPTS, index=EXP_OPTS.index(st.session_state.v_exp),
            help="Which experiment campaign the metrics, figures and EPU "
                 "checkpoints are read from.")
    if uses_seed:
        st.session_state.v_seed = st.selectbox(
            "Random seed", SEED_OPTS, index=SEED_OPTS.index(st.session_state.v_seed),
            help="Everything was trained three times. This picks which trained copy "
                 "the live pages load; reported metrics always average all three.")
    uploaded = None
    if uses_img and img_files:
        st.session_state.v_img = st.selectbox(
            "Sample X-ray", img_files,
            index=img_files.index(st.session_state.v_img),
            help="Chest X-ray scored by this page.")
        uploaded = st.file_uploader("…or upload your own",
                                    type=["png", "jpg", "jpeg"])

    exp_choice = st.session_state.v_exp
    seed_choice = st.session_state.v_seed
    picked = st.session_state.get("v_img")
    sel_exp = FULL_EXP if exp_choice.startswith("Main") else SWEEP_EXP

page = st.session_state.page
gt_map = load_groundtruth(str(DEFAULT_CSV))
sum_main = load_summary(str(FULL_EXP))
sum_sweep = load_summary(str(SWEEP_EXP))
stab_a = paper_table("setting_a_stability.csv")
audit = paper_table("forgetting_attacks.csv", ("setting", "config"))
ga_res = paper_table("ga_results.csv")
HAVE_PAPER = bool(stab_a and audit and ga_res)
sel_summary = load_summary(str(sel_exp))
sel_seed_dir = sel_exp / seed_choice
orig_ckpt = FULL_EXP / seed_choice / "original.pt"     # canonical home of orig/oracle
oracle_ckpt = FULL_EXP / seed_choice / "oracle.pt"


def get_image():
    from PIL import Image
    if uploaded is not None:
        return Image.open(uploaded), uploaded.name
    if picked:
        return Image.open(DEFAULT_IMAGES / picked), picked
    return None, None


# Warm-up: caches the checkpoints ahead of time so the live pages respond
# instantly during a presentation instead of pausing on first click.
st.sidebar.markdown('<div class="grp">Tools</div>', unsafe_allow_html=True)
if st.sidebar.button("Warm up models", key="warmup", width="stretch",
                     help="Loads this seed's checkpoints into memory now, so the "
                          "live pages respond instantly during the demo."):
    todo = [p for p in [orig_ckpt, oracle_ckpt, sel_seed_dir / "plain.pt",
                        *epu_options(sel_seed_dir).values()] if p.exists()]
    bar = st.sidebar.progress(0.0, text="Caching checkpoints…")
    for i, ck in enumerate(todo, 1):
        load_model(str(ck))
        bar.progress(i / len(todo), text=f"Cached {ck.name}")
    bar.empty()
    st.sidebar.success(f"{len(todo)} models cached — live pages are now instant.")


# ===========================================================================
# PAGE — Home
# ===========================================================================
if page == "Home":
    topbar("Archive", "home")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("B.Tech Major Project · Medical AI · Privacy · Explainability")
        display("Unlearning that keeps the model honest about "
                "<em>where it looks</em>")
        st.markdown(f'<div class="proj-title">{PROJECT_TITLE}</div>',
                    unsafe_allow_html=True)
        lede(
            "Five simulated hospitals jointly train a Vision Transformer to read "
            "chest X-rays — without sharing a single image. When one hospital "
            "exercises its right to be forgotten, we remove its influence "
            "<b>without a full retrain</b>, audit the forgetting with "
            "membership-inference attacks, and — the part accuracy alone cannot "
            "see — keep the model's <b>visual evidence</b> anchored where it was. "
            "Every number on this site is a real measurement from the trained "
            "models you can run in the live workspaces.")

        # Badges are peer metadata — none is more important than another, so
        # none is accented. Blue is reserved for the project's contribution.
        st.markdown(
            '<span class="badge">DeiT-Small · Vision Transformer</span>'
            '<span class="badge">NIH ChestX-ray14 · 112,120 images · 14 findings</span>'
            '<span class="badge">FedAvg · 5 clients · non-IID</span>'
            '<span class="badge">3 seeds · oracle-benchmarked</span>',
            unsafe_allow_html=True)

        if sum_main:
            r = {x["config"]: x for x in sum_main["tables"]["comparison"]}
            st.markdown("")
            kpi_row([
                ("Diagnostic quality", f"{mean_of(r['original']['macro_auc']):.3f}",
                 "macro-AUROC of the federated model (0.5 = chance)"),
                ("Membership trace",
                 (f"{audit[('-', 'original')]['positive_auc'][0]:.3f} → "
                  f"{audit[('A', 'finetune_retain')]['positive_auc'][0]:.3f}")
                 if HAVE_PAPER else "—",
                 "calibrated audit AUC, Original → after unlearning "
                 "(0.5 = chance): reduced, not removed"),
                ("Explanation preserved",
                 (f"{stab_a['finetune_retain']['expl_iou_to_original'][0]:.2f} → "
                  f"{stab_a['epu@1']['expl_iou_to_original'][0]:.2f}")
                 if HAVE_PAPER else "—",
                 "overlap of attention maps with the Original, Plain → EPU"),
                ("Cost of unlearning",
                 f"{mean_of(r['finetune_retain']['speedup_vs_oracle']):.1f}× faster",
                 "than retraining from scratch"),
            ])

        section("The pipeline")
        # Blue marks what THIS project contributes; grey is standard federated
        # learning that already existed before this work.
        pipeline = flow([
            ("stage 01", "5 clients<br>non-IID shards", False),
            ("stage 02", "FedAvg<br>up to 30 rounds", False),
            ("stage 03", "Original<br>model", False),
            ("stage 04", "Unlearn<br>client 0", True),
            ("stage 05", "Evaluate<br>4 axes", True),
        ])
        panel([("#2563eb", "this project's contribution"),
               ("#74777f", "standard federated learning")], pipeline,
              "How the five stages connect",
              ["Five clients hold patient-disjoint, non-IID shards of "
               "ChestX-ray14. Images never leave the client.",
               "FedAvg runs up to 30 rounds with early stopping: local training, "
               "then a data-weighted average of the returned weights.",
               "The result is the <strong>Original model</strong> — the artefact "
               "unlearning must edit.",
               "Client 0 requests removal. Three procedures run: Oracle "
               "(retrain), Plain (fine-tune), EPU (fine-tune + explanation anchor).",
               "Every model is scored on four axes, against both the Original "
               "and the oracle, across three seeds."])
        if (PAPER / "figures" / "fig_overview.png").exists():
            st.image(str(PAPER / "figures" / "fig_overview.png"), width="stretch")
            st.caption("Block diagram of the study. (a) Data, patient-level split, "
                       "non-IID partition and FedAvg training give the Original "
                       "model. (b) After the erasure request the Oracle is retrained "
                       "without the client; in EPU each remaining client compares "
                       "the rollout map of a trainable copy with that of the frozen "
                       "Original and adds this explanation loss, weighted by λ, to "
                       "the task loss (λ = 0 gives Plain). (c) Models and metrics "
                       "compared. The gradient-ascent baseline is not drawn.")

        section("How to read this archive")
        st.markdown(
            "**Learn (01–04)** builds the ideas in order: how the model is trained "
            "without sharing data, what unlearning means and how forgetting is "
            "audited, the proposed EPU objective, and how a transformer explains "
            "itself.  \n**Live** runs the real trained checkpoints on your machine "
            "— diagnosis with attention heatmaps, then a before/after unlearning "
            "comparison.  \n**Evidence** gives the measured multi-seed results with "
            "an honest verdict, plus every hyperparameter exactly as it was run.")
        takeaway("This project's claim is <i>not</i> \"a better accuracy number.\" "
                 "It is that <b>explanation stability is a first-class, measurable "
                 "objective of unlearning</b> — and we built the pipeline that "
                 "measures it against both the pre-unlearning model and a "
                 "retrain oracle.")
        nextbox()
    with side:
        toc(["The pipeline", "How to read this archive"])


# ===========================================================================
# PAGE — 01 Federated Learning
# ===========================================================================
elif page == "01 · Federated Learning":
    topbar("Concepts", "federated learning")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Fundamental Concept · 01")
        display("Federated Learning &amp; FedAvg")
        lede("Hospitals cannot pool chest X-rays — privacy law and ethics forbid "
             "it. Federated learning trains one shared model while every image "
             "stays inside the hospital that owns it. The only thing that ever "
             "leaves a hospital is a set of <b>model weights</b>.")

        rounds = flow([
            ("server", "broadcast<br>global model w<sub>t</sub>", True),
            ("clients 1–5", "local training<br>on private data", False),
            ("upload", "weights only<br>never images", False),
            ("merge", "weighted average<br>&rarr; w<sub>t+1</sub>", True),
        ])
        panel([("#2563eb", "server-side — aggregation"),
               ("#74777f", "client-side — data never leaves")], rounds,
              "One round of FedAvg",
              ["The server broadcasts the current global model to all five clients.",
               "Each client trains locally for two epochs on its own X-rays. No "
               "data is transmitted at any point.",
               "Clients return <strong>weight updates only</strong> — the privacy "
               "guarantee of the protocol.",
               "The server merges them into the next global model by a "
               "data-weighted average, and the round repeats 30 times."])

        section("The aggregation rule")
        formula(r"w_{t+1} \;=\; \sum_{k=1}^{K} \frac{n_k}{n}\; w_k^{(t)}",
                "FedAvg: the next global model is a data-weighted average of the "
                "clients' locally trained models.")
        st.markdown("$n_k$ is client $k$'s sample count, $n=\\sum_k n_k$. A hospital "
                    "with more data pulls the average harder — proportional "
                    "representation, nothing more.")
        st.markdown("**Worked example.** One weight has local values "
                    "$0.90,\\,1.10,\\,1.00,\\,0.80,\\,1.20$ and the clients hold "
                    "$20\\%,25\\%,15\\%,10\\%,30\\%$ of the data:")
        formula(r"w_{t+1}=.2(0.9)+.25(1.1)+.15(1.0)+.10(0.8)+.30(1.2)=1.045",
                "One parameter, merged. All 22 million DeiT weights are averaged "
                "this way, every round.")

        section("The hard part: non-IID data")
        st.markdown(
            "Real hospitals see different populations — one more cardiac, another "
            "more trauma. We simulate this with a **Dirichlet (α = 0.5)** "
            "partition, giving each client a realistically skewed label mix. "
            "Non-IID data makes FedAvg genuinely harder (client updates pull in "
            "different directions), which is why we report it explicitly rather "
            "than testing on an easy uniform split.")
        st.markdown("**This project's federated setup**")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Clients | 5, patient-disjoint, Dirichlet α = 0.5 |\n"
            "| Rounds × local epochs | 30 × 2 |\n"
            "| Backbone | DeiT-Small (ImageNet-pretrained, 1-channel input) |\n"
            "| Optimizer | Adam, warmup + cosine schedule |\n"
            "| Loss | BCE with per-class positive weighting (class imbalance) |\n"
            "| Model selection | best checkpoint + early stopping |")
        if sum_main:
            r = {x["config"]: x for x in sum_main["tables"]["comparison"]}
            st.markdown(f"**Measured outcome** — the resulting *Original* model scores "
                        f"**macro-AUROC {r['original']['macro_auc']}** across three "
                        f"seeds on the held-out test set.")
        takeaway("FedAvg gives us the <b>Original model</b> — the starting point "
                 "that unlearning must edit. Data never moves; only weights do.")
        nextbox()
    with side:
        toc(["The aggregation rule", "The hard part: non-IID data"])


# ===========================================================================
# PAGE — 02 Unlearning & Forgetting
# ===========================================================================
elif page == "02 · Unlearning & Forgetting":
    topbar("Concepts", "unlearning &amp; forgetting")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Fundamental Concept · 02")
        display("Machine Unlearning &amp; the Right to be Forgotten")
        lede("Privacy regulation lets a participant withdraw consent after "
             "training. The model must then behave <b>as if that participant had "
             "never taken part</b>. The brute-force answer — retrain from scratch "
             "on everyone else — is correct by definition, and far too expensive "
             "to repeat for every withdrawal.")

        if sum_main:
            r = {x["config"]: x for x in sum_main["tables"]["comparison"]}
            kpi_row([
                ("Cost of the correct answer",
                 f"{mean_of(r['oracle']['wall_time_s'])/3600:.1f} h",
                 "measured oracle retrain, per seed, on a T4 GPU"),
                ("Cost of warm-start unlearning",
                 f"{mean_of(r['finetune_retain']['wall_time_s'])/3600:.1f} h",
                 f"{mean_of(r['finetune_retain']['speedup_vs_oracle']):.1f}× cheaper "
                 "— this is why unlearning exists"),
            ])

        three = grid3([
            ("reference", "Oracle<br>retrain from scratch", False),
            ("baseline", "Plain<br>warm-start finetune", False),
            ("proposed", "EPU<br>finetune + anchor", True),
        ])
        panel([("#2563eb", "our proposed method"),
               ("#74777f", "comparison arm")], three,
              "Three procedures, one controlled comparison",
              ["<strong>Oracle</strong> retrains on the four retained clients. "
               "Forgetting is perfect by construction — this is the reference "
               "everything is judged against.",
               "<strong>Plain</strong> warm-starts from the Original and fine-tunes "
               "on retained clients. Efficient, but nothing protects its explanations.",
               "<strong>EPU</strong> runs the identical fine-tune plus one added "
               "explanation-consistency term.",
               "All three see identical data, seeds and schedules — so any gap "
               "between Plain and EPU is attributable to that single term and "
               "nothing else."])

        section("Auditing whether the client was forgotten")
        st.markdown(
            "You cannot open a network and point at a forgotten client's "
            "influence, so we **attack** the model instead. A "
            "**membership-inference attack** (Shokri et al., 2017) exploits the "
            "fact that models behave differently on data they trained on, and "
            "tries to tell the forgotten client's training images from held-out "
            "test images. We report the AUC of that separation:")
        st.markdown(
            "- **AUC well above 0.5** — the model still carries a detectable "
            "trace of the client.\n"
            "- **AUC ≈ 0.5** — this attack cannot tell the two groups apart. "
            "That is *not* proof of forgetting: a weak attack reads 0.5 on "
            "every model.\n"
            "- The **oracle**, which never saw the client, is the reference a "
            "perfectly unlearned model should resemble.")
        st.markdown(
            "Three attacks of increasing strength were applied to the saved "
            "checkpoints: the plain **loss threshold**; the loss **calibrated "
            "per image by the oracle's loss** on the same image (Watson et al., "
            "2022); and that calibrated score restricted to the **findings "
            "present** in the image. The calibrated attacks need the retrained "
            "oracle, so they are an auditor's tool, not an outside attacker's.")
        if HAVE_PAPER:
            _o = audit[("-", "original")]
            _p = audit[("A", "finetune_retain")]
            _e = audit[("A", "epu@1")]
            kpi_row([
                ("Loss-threshold attack · Original", f"{_o['loss_auc'][0]:.3f}",
                 "at chance before anything is removed — this attack is "
                 "uninformative here"),
                ("Calibrated audit · Original", f"{_o['positive_auc'][0]:.3f}",
                 "positive-finding attack: a weak but real trace of client 0"),
                ("Calibrated audit · after unlearning",
                 f"{_p['positive_auc'][0]:.3f} / {_e['positive_auc'][0]:.3f}",
                 "Plain / EPU λ=1, setting A: lower, not at chance, and the "
                 "same for both"),
            ])

        section("The honest nuance")
        st.markdown(
            "With the loss-threshold attack used during the runs, **every** model "
            "scores ≈ 0.50 — the Original included, because a well-regularised "
            "model trained on 100k+ images barely memorises individual samples. "
            "An attack that reads 0.5 before unlearning cannot show that "
            "unlearning worked.\n\n"
            "The calibrated audit does detect something. What it shows, as "
            "measured:\n"
            "- **Fine-tuning reduces the trace but does not remove it.** The "
            "audit AUC falls in all 24 comparisons with the Original, to "
            "0.55–0.57 in setting A and 0.53–0.55 in setting B.\n"
            "- **EPU and Plain do not differ measurably** under fine-tuning: "
            "anchoring to the Original shows no sign of retaining more of the "
            "client.\n"
            "- **Part of the signal is the client's label mix**, not its "
            "individual records: computed within a single finding, the "
            "Original's AUC drops to about 0.55.\n"
            "- **After gradient ascent, EPU keeps slightly more of the trace** "
            "than plain repair (Results page).\n\n"
            "So the claim is *partial* forgetting, audited on a subset of the "
            "data with three seeds and no shadow models. We also verify the "
            "*procedure*: the forgotten client's data is excluded from every "
            "step of Plain and EPU.")
        takeaway("Unlearning is a <b>three-way trade-off</b> (forget · keep "
                 "accuracy · stay cheap). This project adds — and measures — a "
                 "fourth axis: <b>keep the explanation</b>.")
        nextbox()
    with side:
        toc(["Auditing whether the client was forgotten", "The honest nuance"])


# ===========================================================================
# PAGE — 03 EPU
# ===========================================================================
elif page == "03 · EPU — proposed method":
    topbar("Concepts", "EPU — proposed method")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Proposed Method · 03")
        display("EPU — Explanation-Preserving Unlearning")
        lede("Fine-tuning on the retained clients recovers accuracy — but nothing "
             "stops it from quietly <b>relocating the evidence</b> the model uses. "
             "A radiologist who trusted the model because it highlighted the right "
             "lung region cannot trust a model that now highlights somewhere else, "
             "even if the AUROC is identical. EPU makes that stability an explicit "
             "optimization target.")

        # Everything EPU ADDS is blue; L_task is the only component Plain
        # unlearning already had, so it is the only grey box.
        loss = flow([
            ("added by EPU", "frozen Original<br>reference maps", True),
            ("added by EPU", "L<sub>exp</sub><br>MSE of attention", True),
            ("already in Plain", "L<sub>task</sub><br>weighted BCE", False),
            ("added by EPU", "L = L<sub>task</sub><br>+ λ·L<sub>exp</sub>", True),
        ])
        panel([("#2563eb", "added by EPU"),
               ("#74777f", "already present in plain unlearning")], loss,
              "How the objective is assembled",
              ["A frozen copy of the Original model produces reference attention "
               "maps on the retained-client images.",
               "The trainable model produces its own maps on the same inputs; "
               "<strong>L<sub>exp</sub></strong> is the mean-squared error between them.",
               "<strong>L<sub>task</sub></strong> is the ordinary imbalance-weighted "
               "diagnosis loss that recovers accuracy after removing the client.",
               "The sum is minimised by gradient descent. Setting <strong>λ = 0 "
               "recovers Plain exactly</strong> — one knob, one added term, no "
               "other change."])

        section("The objective")
        formula(r"L \;=\; L_{\text{task}} \;+\; \lambda \cdot L_{\text{exp}}",
                "The complete EPU objective. λ = 0 recovers plain unlearning exactly.")
        formula(r"L_{\text{exp}} \;=\; \big\lVert\, \mathrm{rollout}(f_\theta, x)"
                r" \;-\; \mathrm{rollout}(f_{\text{orig}}, x) \,\big\rVert_2^2",
                "The explanation term: squared error between the current model's "
                "attention map and the frozen Original's, on the same image.")
        st.markdown(
            "- **$L_{task}$** — the usual diagnosis loss (imbalance-weighted BCE) "
            "on the **retained** clients: recovers accuracy after removing the "
            "forgotten client.\n"
            "- **$L_{exp}$** — mean-squared error between the current model's "
            "attention-rollout map and the **frozen Original model's** map on the "
            "same image: anchors *where the model looks*.\n"
            "- **$\\lambda$** — the strength of the anchor. **λ = 0 recovers Plain "
            "exactly**, so EPU is a controlled generalisation: one knob, one added "
            "term, no other change.")

        section("Why the anchor is the Original, not the oracle")
        st.markdown(
            "\"Preserving explanations\" means preserving them **relative to the "
            "model the clinicians already validated** — the Original. The oracle "
            "defines correct *forgetting*; the Original defines correct "
            "*explanation*. We measure fidelity to both, and they agree closely "
            "because the Original and oracle attend similarly (cosine ≈ 0.96).")

        section("Why λ = 1 was too weak")
        st.markdown(
            "Measured on our runs: $L_{task} \\approx 0.30$ and $L_{exp} \\approx "
            "0.011$ (attention maps are already min-max normalised, so their MSE "
            "is small). At **λ = 1** the explanation term contributes")
        formula(r"\frac{1 \times 0.011}{0.30 + 0.011} \;\approx\; 3.5\%",
                "Share of the total gradient signal carried by the explanation term "
                "at λ = 1 — far too small to steer training.")
        st.markdown(
            "of the gradient signal — a whisper. Balance ($\\lambda L_{exp} "
            "\\approx L_{task}$) needs $\\lambda \\approx 0.30/0.011 \\approx 27$. "
            "That arithmetic *predicted* the λ-sweep experiment (λ = 10, 30, 100), "
            "and the sweep behaved exactly as predicted — the explanation gain over "
            "Plain appears only once λ reaches the tens (see Results).")
        takeaway("EPU changes <b>one line of the loss</b>. That minimalism is the "
                 "point: the entire measured difference between Plain and EPU is "
                 "the price and value of explanation preservation.")
        nextbox()
    with side:
        toc(["The objective", "Why the anchor is the Original, not the oracle",
             "Why λ = 1 was too weak"])


# ===========================================================================
# PAGE — 04 Attention Rollout
# ===========================================================================
elif page == "04 · Attention Rollout":
    topbar("Concepts", "attention rollout")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Fundamental Concept · 04")
        display("Attention Rollout — how the transformer explains itself")
        lede("A Vision Transformer cuts the 224×224 X-ray into <b>196 patches</b> "
             "(14×14 grid) plus one classification token, and every layer lets each "
             "token attend to all others. Attention Rollout composes those "
             "layer-by-layer attention matrices into a single map: <b>how much did "
             "each image patch ultimately influence the decision?</b>")

        roll = flow([
            ("step 1", "average<br>the heads", False),
            ("step 2", "add identity<br>A + I", False),
            ("step 3", "row<br>normalise", False),
            ("step 4", "multiply<br>across layers", False),
            ("step 5", "CLS row<br>14×14 map", True),
        ])
        panel([("#2563eb", "the explanation map EPU anchors"),
               ("#74777f", "intermediate transform")], roll,
              "The rollout recipe (Abnar & Zuidema, 2020)",
              ["Average the attention heads of each layer into one matrix per layer.",
               "Add the identity — tokens also pass information through the "
               "residual connection, not only through attention.",
               "Row-normalise so each row remains a proper distribution of influence.",
               "Multiply the layer matrices together; influence composes across depth.",
               "Read the CLS row over the 196 patch tokens, reshape to 14×14 and "
               "min-max normalise. That grid, upsampled over the X-ray, is the "
               "heatmap in the live workspaces."])

        section("The composition")
        formula(r"R \;=\; \tilde A_L \,\tilde A_{L-1} \cdots \tilde A_1,"
                r"\qquad \tilde A_\ell = \mathrm{norm}(A_\ell + I)",
                "Attention composes multiplicatively through depth; each layer's "
                "matrix is head-averaged, residual-corrected and row-normalised first.")

        section("Why rollout and not Grad-CAM")
        st.markdown(
            "- It is **model-intrinsic**: computed from the attention the ViT "
            "actually used, no auxiliary gradients or target class required — "
            "natural for multi-label chest X-rays.\n"
            "- It is **differentiable** in our implementation, so the EPU loss can "
            "push gradients *through the explanation itself* during unlearning — "
            "the property the whole method depends on.\n"
            "- Honest caveat: rollout is an approximation of information flow, and "
            "we use it consistently for training and evaluation, so all comparisons "
            "are like-for-like. We also measure with four metrics (cosine, Pearson, "
            "IoU, MSE) rather than trusting a single number.")
        takeaway("Rollout turns 12 layers × 6 heads of attention into one 14×14 "
                 "map of <b>visual evidence</b> — differentiable, so EPU can "
                 "optimise it, and measurable, so the evidence pages can score it.")
        nextbox()
    with side:
        toc(["The composition", "Why rollout and not Grad-CAM"])


# ===========================================================================
# PAGE — Live Diagnosis
# ===========================================================================
elif page == "Diagnosis":
    topbar("Live workspace", "diagnosis")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Live · real trained checkpoint · CPU inference")
        display("Diagnosis — the federated model at work")
        lede("This is the <b>Original</b> model produced by FedAvg (page 01), "
             "running live on your machine. Pick or upload an X-ray in the sidebar: "
             "the model predicts all 14 findings and the attention-rollout heatmap "
             "(page 04) shows exactly which regions drove the decision.")

        section("Run the model")
        st.caption("The first run loads the checkpoint from disk and takes a few "
                   "seconds; every run after that is cached and near-instant.")
        pil_img, cap = get_image()
        if pil_img is None:
            st.info("Select a sample X-ray in the sidebar to begin.")
        elif not orig_ckpt.exists():
            st.error(f"Checkpoint not found: {orig_ckpt}")
        else:
            thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                            help="A finding is flagged when its probability exceeds "
                                 "this value. Lower = more sensitive, more false alarms.")
            with st.spinner("Running DeiT-Small on CPU…"):
                disp, probs, rmap = run_inference(str(orig_ckpt), cap, pil_img)

            c1, c2 = st.columns(2, gap="medium")
            c1.image(disp, caption=f"Input — {cap}", width="stretch", clamp=True)
            c2.image(overlay_heatmap(disp, rmap),
                     caption="Attention rollout — the model's visual evidence",
                     width="stretch")

            st.markdown("**Predicted findings**")
            order = np.argsort(probs)[::-1]
            flagged = [(NIH_LABELS[i], probs[i]) for i in order if probs[i] >= thr]
            if flagged:
                for name, p in flagged:
                    st.progress(float(p), text=f"{name} — {p:.2f}")
            else:
                st.caption("No finding exceeds the threshold.")
            with st.expander("All 14 findings"):
                for i in order:
                    st.progress(float(probs[i]), text=f"{NIH_LABELS[i]} — {probs[i]:.2f}")
            if cap in gt_map:
                st.markdown("**Radiologist label (ground truth)**")
                st.markdown("".join(f'<span class="gt-chip">{g.strip()}</span>'
                                    for g in gt_map[cap].split("|")),
                            unsafe_allow_html=True)
            st.caption(f"deit_small_patch16_224 · {seed_choice} · multi-label sigmoid "
                       "outputs · heatmap computed live from this image.")

            # ---- per-image verdict, computed live for whichever X-ray is shown
            if cap in gt_map:
                section("Reading this case")
                gt_set = [g.strip() for g in gt_map[cap].split("|")
                          if g.strip() in NIH_LABELS]
                rank_of = {NIH_LABELS[i]: r + 1 for r, i in enumerate(order)}
                import pandas as pd
                if gt_set:
                    st.markdown("**Did the model surface the labelled finding?**")
                    st.dataframe(pd.DataFrame([{
                        "Labelled finding": g,
                        "Model probability": f"{probs[NIH_LABELS.index(g)]:.2f}",
                        "Rank of 14": rank_of[g],
                        "Flagged at this threshold":
                            "yes" if probs[NIH_LABELS.index(g)] >= thr else "no",
                    } for g in gt_set]), hide_index=True, width="stretch")
                    hits = sum(1 for g in gt_set
                               if probs[NIH_LABELS.index(g)] >= thr)
                    st.markdown(f"The model surfaced **{hits} of {len(gt_set)}** "
                                f"labelled finding(s) at threshold {thr:.2f}.")
                else:
                    st.markdown("This study carries **no positive finding** in the "
                                "label file (*No Finding*), so every flag below is "
                                "unlabelled by definition.")

                extra = [n for n, _ in flagged if n not in gt_set]
                if extra and gt_set:
                    joint, total = cooccurrence(str(DEFAULT_CSV))
                    lines = []
                    for f_name in extra:
                        best, pct, cnt, tot = None, 0.0, 0, 0
                        for g in gt_set:
                            t = total.get(g, 0)
                            c = joint.get(g, {}).get(f_name, 0)
                            if t and c / t >= pct:
                                best, pct, cnt, tot = g, c / t, c, t
                        if best:
                            lines.append(
                                f"- **{f_name}** appears together with *{best}* in "
                                f"**{pct*100:.0f}%** of labelled studies "
                                f"({cnt:,} of {tot:,}).")
                    if lines:
                        st.markdown(
                            "**The additional flags are plausible confusions, not "
                            "arbitrary ones.** Measured over the full ChestX-ray14 "
                            "label file:\n" + "\n".join(lines) +
                            "\n\nThese are opacity-type findings that overlap "
                            "radiologically, and the labels themselves reflect that "
                            "overlap. Combined with the sensitivity-first operating "
                            "point below, several flags for one labelled finding is "
                            "the expected behaviour.")

            section("How to read the heatmap")
            st.markdown(
                "Bright regions are the patches the classification token attended "
                "to most, aggregated across all 12 transformer layers (page 04). "
                "This is the model's **visual evidence** — the reason a finding was "
                "flagged. It is the quantity unlearning must not disturb, and the "
                "next page checks exactly that.")

            section("How to read the operating point")
            if sum_main:
                _o = {x["config"]: x for x in sum_main["tables"]["comparison"]}["original"]
                _prec, _rec = mean_of(_o["precision"]), mean_of(_o["recall"])
                kpi_row([
                    ("Recall (sensitivity)", f"{_rec:.2f}",
                     "share of true findings the model surfaces"),
                    ("Precision", f"{_prec:.2f}",
                     f"≈ 1 in {round(1/_prec)} flagged findings is in the label set"),
                    ("Macro-AUROC", f"{mean_of(_o['macro_auc']):.3f}",
                     "ranking quality — what the model is actually good at"),
                    ("mAP", f"{mean_of(_o['mAP']):.3f}",
                     "average precision across the 14 findings"),
                ])
            st.markdown(
                "**The model is deliberately tuned for sensitivity, not precision.** "
                "Training uses class-imbalance weighting (`pos_weight`), which "
                "up-weights positive examples so rare findings are not ignored — "
                "bought at the cost of over-flagging. Expect **several flagged "
                "findings for one labelled finding**: that is the configured "
                "operating point, not a malfunction, and it is the right bias for a "
                "screening aid where a missed finding costs more than one a "
                "radiologist dismisses.\n\n"
                "**The 0.5 threshold is arbitrary.** Macro-AUROC measures how well "
                "the model *ranks* findings; the slider above turns that ranking "
                "into decisions. Moving it trades recall against precision — which "
                "is a clinical choice, not a property of the model.\n\n"
                "**Absent labels are not proven negatives.** ChestX-ray14 labels "
                "are NLP-mined from radiology reports and are known to be "
                "incomplete, so a flagged finding missing from the label set is not "
                "necessarily a false positive. Related findings also co-occur "
                "heavily — Atelectasis, Infiltration and Consolidation are all "
                "opacity-type patterns that overlap radiologically and in the "
                "labels themselves.")
            takeaway("What you just saw is the model that unlearning must edit — "
                     "accurate <i>and</i> visually accountable. The next page edits "
                     "it and checks both survive.")
        nextbox()
    with side:
        toc(["Run the model", "Reading this case", "How to read the heatmap",
             "How to read the operating point"])


# ===========================================================================
# PAGE — Live Unlearning
# ===========================================================================
elif page == "Unlearning":
    topbar("Live workspace", "unlearning")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Live · before → after · same image")
        display("Unlearning — watch what changes, and what doesn't")
        lede("Client 0 has asked to be forgotten. Below, the <b>Original</b> model "
             "and an <b>unlearned</b> model score the same X-ray. If unlearning is "
             "trustworthy, the attention stays anchored (explanation preserved) "
             "while the membership trace of the forgotten client goes down "
             "(page 02). Switch methods — including the λ-sweep EPU "
             "variants — and compare.")

        methods = {}
        plain_p = sel_seed_dir / "plain.pt"
        if plain_p.exists():
            methods["Plain unlearning"] = plain_p
        methods.update(epu_options(sel_seed_dir))
        if oracle_ckpt.exists():
            methods["Oracle (retrain from scratch)"] = oracle_ckpt

        section("Before → after on one image")
        if not methods:
            st.error(f"No unlearned checkpoints found in `{sel_seed_dir}`.")
        else:
            method = st.selectbox("Unlearned model to compare against the Original",
                                  list(methods.keys()))
            pil_img, cap = get_image()
            if pil_img is None:
                st.info("Select a sample X-ray in the sidebar to compare.")
            elif not orig_ckpt.exists():
                st.error(f"Original checkpoint not found: {orig_ckpt}")
            else:
                with st.spinner("Scoring both models on CPU…"):
                    disp, op, omap = run_inference(str(orig_ckpt), cap, pil_img)
                    _, up, umap = run_inference(str(methods[method]), cap, pil_img)
                    a, b = omap.flatten(), umap.flatten()
                    cos = float(np.dot(a, b) /
                                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

                c1, c2 = st.columns(2, gap="medium")
                c1.image(overlay_heatmap(disp, omap), caption="Original — attention",
                         width="stretch")
                c2.image(overlay_heatmap(disp, umap),
                         caption=f"{method} — attention", width="stretch")

                k1, k2 = st.columns([1, 1.4], gap="medium")
                with k1:
                    kpi("Attention similarity · this image", f"{cos:.3f}",
                        "cosine of the two rollout maps · ≈ 1.0 means the same "
                        "visual evidence")
                with k2:
                    st.markdown("**Prediction shift — top findings**")
                    import pandas as pd
                    order = np.argsort(op)[::-1][:5]
                    st.dataframe(pd.DataFrame({
                        "Finding": [NIH_LABELS[i] for i in order],
                        "Original": [f"{op[i]:.2f}" for i in order],
                        "Unlearned": [f"{up[i]:.2f}" for i in order],
                        "Δ": [f"{up[i]-op[i]:+.2f}" for i in order],
                    }), hide_index=True, width="stretch")

                section("Learning vs unlearning — what changed on each axis")
                if sel_summary:
                    _rows = {r["config"]: r
                             for r in sel_summary["tables"]["comparison"]}
                    if method.startswith("Plain"):
                        _key = "finetune_retain"
                    elif method.startswith("Oracle"):
                        _key = "oracle"
                    else:
                        _lam = re.findall(r"λ=([\d.]+)", method)
                        _key = f"epu@{float(_lam[0]):g}" if _lam else None

                    if _key in _rows and "original" in _rows:
                        _o, _u = _rows["original"], _rows[_key]

                        def _ba(metric, fmt="{:.3f}"):
                            a, b = mean_of(_o[metric]), mean_of(_u[metric])
                            return fmt.format(a), fmt.format(b), b - a

                        a1, b1, d1 = _ba("macro_auc")
                        a2, b2, d2 = _ba("mia_auc")
                        a3, b3, d3 = _ba("pred_dist_to_oracle")
                        h_o = mean_of(_o["wall_time_s"]) / 3600
                        h_u = mean_of(_u["wall_time_s"]) / 3600
                        kpi_row([
                            ("Diagnosis · macro-AUROC", f"{a1} → {b1}",
                             f"Δ {d1:+.3f} — must <b>not</b> change: the model has "
                             "to stay a good classifier"),
                            ("Loss-threshold MIA AUC", f"{a2} → {b2}",
                             f"Δ {d2:+.3f} — at chance (0.50) before and after, "
                             "so this attack says nothing; see the calibrated "
                             "audit on page 02"),
                            ("Behaviour · distance to oracle", f"{a3} → {b3}",
                             f"Δ {d3:+.3f} — this <b>should</b> move: the model "
                             "shifts toward one that never saw the client"),
                            ("Cost · training time",
                             f"{h_o:.1f} h → {h_u:.1f} h",
                             f"{mean_of(_u['speedup_vs_oracle']):.1f}× the oracle's "
                             "speed — the practical reason to unlearn"),
                        ])
                        st.markdown(
                            "Read the four axes together: **three of them should "
                            "barely move and one of them should.** Diagnosis and "
                            "the loss-threshold membership signal stay put — the "
                            "model remains just as good a classifier, and that "
                            "simple attack could not single out the forgotten "
                            "client even before unlearning (the stronger audit on "
                            "page 02 can). What genuinely changes is the model's "
                            "*behaviour relative to the oracle*: it moves toward the "
                            "model that never saw client 0. Explanation fidelity, "
                            "the fourth axis, is the one this project adds — and it "
                            "is measured in full on the **Results & Verdict** page.")
                    else:
                        st.info("Select a method with matching summary metrics to "
                                "see the axis-by-axis comparison.")

                section("The measured story across three seeds")
                if sel_summary:
                    import pandas as pd
                    nice = {"original": "Original", "finetune_retain": "Plain",
                            "oracle": "Oracle"}
                    st.dataframe(pd.DataFrame([{
                        "Model": nice.get(r["config"],
                                          r["config"].replace("epu@", "EPU λ=")),
                        "Macro-AUROC": r["macro_auc"],
                        "MIA AUC (loss threshold)": r["mia_auc"],
                        "Expl. cosine → oracle": r["expl_cos_to_oracle"],
                        "Speedup vs oracle": r["speedup_vs_oracle"],
                    } for r in sel_summary["tables"]["comparison"]]),
                        hide_index=True, width="stretch")
                    st.caption("How to read: the loss-threshold MIA is ≈ 0.50 "
                               "for every model, the Original included, so it does "
                               "not verify forgetting (page 02 has the calibrated "
                               "audit) · cosine to the oracle is ≈ 0.96 for all · "
                               "speedup > 1 ⇒ "
                               "cheaper than the oracle's retrain. Mean ± std over "
                               "seeds 42/43/44.")
                section("How to read this comparison")
                _st = per_config_stats(str(SWEEP_EXP))
                if _st and "oracle" in _st:
                    kpi_row([
                        ("Plain — cosine → Original",
                         f"{_st['finetune_retain']['expl_cos_to_original'][0]:.4f}",
                         "a warm-started fine-tune stays close to the Original"),
                        ("Oracle — cosine → Original",
                         f"{_st['oracle']['expl_cos_to_original'][0]:.4f}",
                         "an independent run from scratch is the *least* similar"),
                        ("EPU λ=100 — cosine → Original",
                         f"{_st['epu@100']['expl_cos_to_original'][0]:.4f}",
                         "the anchor holds the reasoning in place"),
                    ])
                st.markdown(
                    "**Three things worth knowing before you compare the two "
                    "images above.**\n\n"
                    "1. **The number beside the images is for *this one image*; "
                    "the table is a 96-image average over three seeds.** A single "
                    "radiograph can sit above or below the average — the table is "
                    "the claim, the image is the illustration.\n"
                    "2. **The Oracle will look the *most* different, and that is "
                    "correct.** It is an entirely independent training run from a "
                    "different random initialisation, so it reaches a different "
                    "solution with different attention — even though it is the "
                    "gold standard for forgetting. Being the reference for "
                    "*forgetting* does not make it the reference for *appearance*. "
                    "Measured across the sweep, the oracle is in fact **further** "
                    "from the Original than Plain is.\n"
                    "3. **Small prediction shifts are the expected outcome.** "
                    "Unlearning removes one client of five from a model trained on "
                    "over 100,000 images; the remaining four still support the same "
                    "findings. A large swing would suggest instability, not "
                    "successful forgetting.")
                takeaway("Unlearning is <b>shown</b> as a three-part exhibit: the "
                         "procedure excluded the client, the calibrated audit "
                         "finds less of its trace than before, and the attention "
                         "you see above barely moved. Forget (in part) · keep "
                         "accuracy · keep reasoning.")
        nextbox()
    with side:
        toc(["Before → after on one image",
             "Learning vs unlearning — what changed on each axis",
             "The measured story across three seeds",
             "How to read this comparison"])


# ===========================================================================
# PAGE — Results & Verdict
# ===========================================================================
elif page == "Results & Verdict":
    topbar("Evidence", "results &amp; verdict")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Evidence · three seeds · mean ± std")
        display("Results &amp; the honest verdict: did EPU beat Plain?")
        lede("Two fine-tuning settings were run. <b>Setting A</b> (the main "
             "experiment, λ = 0.5 and 1) is mild and keeps accuracy; <b>setting B</b> "
             "(the λ-sweep, λ = 1 to 100) fine-tunes harder. Against the retrain "
             "oracle EPU and Plain are indistinguishable in both. Against the "
             "pre-unlearning model — the objective EPU optimises — EPU is more "
             "stable in both. A stronger membership audit and a second unlearning "
             "baseline, gradient ascent, were added afterwards. Below is exactly "
             "what was measured.")

        if sum_main and sum_sweep:
            m_rows = {r["config"]: r for r in sum_main["tables"]["comparison"]}
            s_rows = {r["config"]: r for r in sum_sweep["tables"]["comparison"]}

            section("Act 1 — Setting A: no gain against the oracle, a gain against the Original")
            kpi_row([
                ("Plain — explanation cos → oracle",
                 m_rows["finetune_retain"]["expl_cos_to_oracle"],
                 "already very high without any help"),
                ("Best EPU (λ=0.5) — cos → oracle",
                 m_rows["epu@0.5"]["expl_cos_to_oracle"],
                 "statistically identical — no gain on this reference"),
            ])
            st.markdown("> Finding recorded by the run, against the oracle "
                        f"reference: *{sum_main['findings']['statement']}*")
            if HAVE_PAPER:
                st.markdown(
                    "The run logged similarity to the oracle only. Similarity to "
                    "the **Original** was computed afterwards from the saved "
                    "checkpoints, with the same code and probe images (the "
                    "computation reproduces the logged oracle similarities):")
                kpi_row([
                    ("Plain — IoU → Original",
                     pm(stab_a["finetune_retain"]["expl_iou_to_original"]),
                     "the attended regions drift although accuracy does not"),
                    ("EPU λ=0.5 — IoU → Original",
                     pm(stab_a["epu@0.5"]["expl_iou_to_original"]),
                     "higher in every seed"),
                    ("EPU λ=1 — IoU → Original",
                     pm(stab_a["epu@1"]["expl_iou_to_original"]),
                     "higher in every seed, macro-AUROC unchanged"),
                ])
                st.markdown(
                    "So setting A is not a null result on the stated objective: "
                    "EPU keeps the maps closer to the pre-unlearning model at the "
                    "operating point that preserves utility.")

            section("Act 2 — λ-sweep: the predicted effect appears")
            import pandas as pd
            ab = sum_sweep["tables"]["ablation_lambda"]
            st.dataframe(pd.DataFrame([{
                "Config": r["config"].replace("epu@", "EPU λ="),
                "λ": r["lambda"],
                "Expl. cosine → oracle": r["expl_cos_to_oracle"],
                "Gain vs Plain": f"{r['preservation_gain_vs_plain']:+.4f}",
                "MIA AUC": r["mia_auc"],
                "Macro-AUROC": r["macro_auc"],
            } for r in ab]), hide_index=True, width="stretch")
            st.markdown(
                f"- Plain's cosine in this harder regime: "
                f"**{s_rows['finetune_retain']['expl_cos_to_oracle']}**\n"
                f"- Every strong-λ EPU (10, 30, 100) lands **above** Plain, while "
                f"λ = 1 does not — the **dose-response** the theory predicted.\n"
                f"- The loss-threshold MIA is ≈ 0.50 everywhere (as it is for the "
                f"Original — see the membership audit below) and accuracy is "
                f"within a fraction of a point of Plain.\n")
            st.markdown(f"> Recorded finding: *{sum_sweep['findings']['statement']}*")

            section("Why the two campaigns report different numbers")
            st.markdown(
                "The same **Plain** baseline appears in both campaigns with "
                "different values. That is expected, and it is worth stating "
                "plainly before comparing anything across the two tables.")
            import pandas as pd
            st.dataframe(pd.DataFrame([
                {"Plain unlearning": "Macro-AUROC",
                 "Main run": m_rows["finetune_retain"]["macro_auc"],
                 "λ-sweep run": s_rows["finetune_retain"]["macro_auc"]},
                {"Plain unlearning": "Wall-clock time (s)",
                 "Main run": m_rows["finetune_retain"]["wall_time_s"],
                 "λ-sweep run": s_rows["finetune_retain"]["wall_time_s"]},
                {"Plain unlearning": "Speedup vs oracle",
                 "Main run": m_rows["finetune_retain"]["speedup_vs_oracle"],
                 "λ-sweep run": s_rows["finetune_retain"]["speedup_vs_oracle"]},
                {"Plain unlearning": "Expl. cosine → oracle",
                 "Main run": m_rows["finetune_retain"]["expl_cos_to_oracle"],
                 "λ-sweep run": s_rows["finetune_retain"]["expl_cos_to_oracle"]},
            ]), hide_index=True, width="stretch")
            st.markdown(
                "**The cause is one deliberate change: the λ-sweep uses a longer, "
                "stronger fine-tune** — 5 rounds at a constant learning rate of "
                "1 × 10⁻⁴, against the main run's shorter, gentler schedule. Three "
                "consequences follow, all of them arithmetic rather than surprise:\n\n"
                "1. **Wall-time rises**, so the speedup over the oracle falls from "
                "≈ 5× to ≈ 2×. More fine-tuning simply costs more.\n"
                "2. **Macro-AUROC drops ≈ 1.7 points.** A more aggressive fine-tune "
                "moves further from the converged Original, and on a reduced client "
                "set that costs a little accuracy.\n"
                "3. **Explanation drift increases** — which is precisely the point. "
                "The page-03 arithmetic showed the anchor only binds at λ ≈ 27, and "
                "a regime with more drift is what gives the anchor something to "
                "correct. The sweep was designed to create the conditions under "
                "which the effect is measurable.\n\n"
                "**Comparisons are therefore always made *within* a campaign.** "
                "Plain and every EPU variant inside the λ-sweep share identical "
                "data, seeds, schedule and learning rate — the only difference is "
                "the explanation term. Nothing in the verdict compares a number "
                "from one campaign against a number from the other.")
            _sp = m_rows["original"]["speedup_vs_oracle"]
            st.markdown(
                '<div class="take"><b>One number that surprises people.</b> The '
                '<i>Original</i> model shows a speedup of '
                f'<b>{_sp}</b> — below 1, i.e. <i>slower</i> than the oracle. That '
                'is arithmetically correct: the Original trains on all five clients '
                'for 30 rounds, while the oracle retrains on only the four retained '
                'clients for the same 30 rounds. Fewer clients means less work per '
                'round, so the oracle is the cheaper of the two full trainings. The '
                'speedup column exists to measure the <i>unlearning</i> procedures, '
                'and against the oracle those are 2–5× cheaper.</div>',
                unsafe_allow_html=True)

            section("Explanation fidelity measured against two references")
            st.markdown(
                "Explanation fidelity has **two legitimate reference points**, and "
                "they answer different questions:\n"
                "- **vs the Original** — *did we preserve the reasoning of the model "
                "clinicians validated?* This is the objective EPU is built for.\n"
                "- **vs the oracle** — *does the model reason like one that never saw "
                "the client?* This is the definition of correct forgetting.\n\n"
                "Act 1 and Act 2 above report the **oracle** reference. Here is the "
                "**Original** reference — the project's stated objective — with all "
                "four similarity metrics, computed per seed from `final_results.csv`.")

            stats = per_config_stats(str(SWEEP_EXP))
            if stats:
                import pandas as pd
                nice = {"original": "Original", "finetune_retain": "Plain",
                        "oracle": "Oracle", "epu@1": "EPU λ=1", "epu@10": "EPU λ=10",
                        "epu@30": "EPU λ=30", "epu@100": "EPU λ=100"}

                def cell(cfg, m, d=4):
                    v = stats.get(cfg, {}).get(m)
                    return "—" if v is None else f"{v[0]:.{d}f} ± {v[1]:.{d}f}"

                order = ["original", "finetune_retain", "epu@1", "epu@10",
                         "epu@30", "epu@100", "oracle"]
                st.dataframe(pd.DataFrame([{
                    "Model": nice.get(c, c),
                    "Cosine ↑": cell(c, "expl_cos_to_original"),
                    "Pearson ↑": cell(c, "expl_pearson_to_original"),
                    "IoU ↑": cell(c, "expl_iou_to_original"),
                    "MSE ↓": cell(c, "expl_mse_to_original"),
                } for c in order if c in stats]), hide_index=True, width="stretch")
                st.caption("Explanation similarity to the **Original** model, "
                           "λ-sweep run, mean ± std over seeds 42/43/44. ↑ higher "
                           "is better, ↓ lower is better.")

                gains = paired_gain(str(SWEEP_EXP), "expl_cos_to_original")
                if gains:
                    plain_sd = stats["finetune_retain"]["expl_cos_to_original"][1]
                    st.markdown("**Paired gain over Plain, seed by seed**")
                    rows_pg = []
                    for c in ["epu@1", "epu@10", "epu@30", "epu@100"]:
                        if c not in gains:
                            continue
                        g_mean, _g_sd, g_pos, g_per = gains[c]
                        rows_pg.append({
                            "Config": nice.get(c, c),
                            "Mean gain": f"{g_mean:+.4f}",
                            "Effect size": (f"{g_mean/plain_sd:.1f}× Plain's seed std"
                                            if plain_sd else "—"),
                            "Positive in every seed": "yes" if g_pos else "no",
                            "Per-seed": ", ".join(f"{x:+.4f}" for x in g_per),
                        })
                    st.dataframe(pd.DataFrame(rows_pg), hide_index=True,
                                 width="stretch")
                    st.caption(f"Pairing by seed removes between-seed variance. "
                               f"Plain's own across-seed std on this metric is "
                               f"{plain_sd:.4f}.")

                kpi_row([
                    ("Plain — cosine → Original",
                     f"{stats['finetune_retain']['expl_cos_to_original'][0]:.4f}",
                     "explanation drift left uncorrected"),
                    ("EPU λ=100 — cosine → Original",
                     f"{stats['epu@100']['expl_cos_to_original'][0]:.4f}",
                     "the largest anchor tested"),
                    ("IoU → Original",
                     f"{stats['finetune_retain']['expl_iou_to_original'][0]:.2f} → "
                     f"{stats['epu@100']['expl_iou_to_original'][0]:.2f}",
                     "Plain → EPU λ=100: the attended regions overlap far more"),
                    ("MSE → Original",
                     f"{stats['finetune_retain']['expl_mse_to_original'][0]:.4f} → "
                     f"{stats['epu@100']['expl_mse_to_original'][0]:.4f}",
                     "error against the Original's maps, ~14× smaller"),
                ])

                # The same numbers as the two tables above, drawn. Static PNGs
                # built by demo/make_figs.py from the λ-sweep CSV — plotted, not
                # re-measured, so the page and the report cannot drift apart.
                c1, c2 = st.columns(2, gap="medium")
                if (PRES_FIGS / "fig_dose_response.png").exists():
                    c1.image(str(PRES_FIGS / "fig_dose_response.png"),
                             width="stretch")
                    c1.caption("Dose-response in λ — both metrics rise "
                               "monotonically and clear Plain's ±1 s.d. band at "
                               "every setting. λ-sweep campaign.")
                if (PRES_FIGS / "fig_paired_gain.png").exists():
                    c2.image(str(PRES_FIGS / "fig_paired_gain.png"),
                             width="stretch")
                    c2.caption("The same comparison seed by seed rather than "
                               "averaged — every paired gain is positive and "
                               "above Plain's own run-to-run noise.")

                c3, c4 = st.columns(2, gap="medium")
                if (PRES_FIGS / "fig_forgetting_accuracy.png").exists():
                    c3.image(str(PRES_FIGS / "fig_forgetting_accuracy.png"),
                             width="stretch")
                    c3.caption("What the fidelity gain costs — the "
                               "loss-threshold attack stays on the chance line "
                               "(it is there for the Original too) and "
                               "macro-AUROC is flat across the sweep. "
                               "λ-sweep campaign.")
                if (PRES_FIGS / "fig_cost.png").exists():
                    c4.image(str(PRES_FIGS / "fig_cost.png"), width="stretch")
                    c4.caption("Cost of unlearning against a full retrain. "
                               "Main campaign — the timings come from the "
                               "longer fine-tuning schedule, so they are not "
                               "comparable with the sweep's.")

            section("Membership audit — a stronger test of forgetting")
            if HAVE_PAPER:
                import pandas as pd
                _names = {"original": "Original", "finetune_retain": "Plain",
                          "ga_only": "Ascent only", "ga": "GA",
                          "ga_epu@1": "GA + EPU λ=1"}
                _order = [("-", "original"), ("A", "finetune_retain"),
                          ("A", "epu@0.5"), ("A", "epu@1"),
                          ("B", "finetune_retain"), ("B", "epu@1"), ("B", "epu@10"),
                          ("B", "epu@30"), ("B", "epu@100")]
                st.dataframe(pd.DataFrame([{
                    "Setting": "—" if k[0] == "-" else k[0],
                    "Model": _names.get(k[1], k[1].replace("epu@", "EPU λ=")),
                    "Loss AUC": pm(audit[k]["loss_auc"]),
                    "Calibrated AUC": pm(audit[k]["calibrated_auc"]),
                    "Positive-finding AUC": pm(audit[k]["positive_auc"]),
                    "Same-finding AUC": pm(audit[k]["within_finding_auc"]),
                    "TPR at 5% FPR": pm(audit[k]["positive_tpr_at_5fpr"]),
                } for k in _order if k in audit]), hide_index=True, width="stretch")
                st.caption("Members: the forgotten client's training images; "
                           "non-members: test images, from the first of the twelve "
                           "NIH archives (292–1,000 members and 601–820 non-members "
                           "per seed, no patient in both groups). The oracle is the "
                           "reference model of the calibrated attacks. Chance: AUC "
                           "0.5, TPR 0.05. Mean ± std over three seeds.")
                st.markdown(
                    "- The **loss** and plain **calibrated** attacks sit near "
                    "chance for every model, the Original included.\n"
                    "- The **positive-finding** attack detects the forgotten "
                    f"client in the Original (AUC "
                    f"{audit[('-', 'original')]['positive_auc'][0]:.3f}). After "
                    "unlearning it is lower in all 24 comparisons, and the "
                    "decrease excludes zero at the 95% level in 8 of them "
                    "(patient-level bootstrap).\n"
                    "- **EPU minus Plain** averages between −0.017 and +0.010, and "
                    "its interval includes zero in 15 of 18 seed-level "
                    "comparisons. At 5% false positives the paired difference "
                    "includes zero in all 18.\n"
                    "- **Within a single finding** the Original falls to about "
                    "0.55 and the unlearned models to 0.51–0.53: part of the "
                    "signal is the client's label mix.")
            else:
                st.info("paper/forgetting_attacks.csv not found.")

            section("Second baseline — gradient ascent at the departing client")
            if HAVE_PAPER:
                st.markdown(
                    "To test whether the explanation term helps beyond plain "
                    "fine-tuning, it was added to a second unlearning method, "
                    "after Halimi et al. (2022). Client 0 first takes projected "
                    "**gradient-ascent** steps on its own data (24 steps inside "
                    "an L2 ball of radius 0.5% of the weight norm around the "
                    "Original); the remaining clients then repair the model with "
                    "the fine-tuning of setting A, without the explanation term "
                    "(**GA**) or with it (**GA + EPU**, λ = 1). Unlike Plain and "
                    "EPU, this method reads the forgotten client's data.")
                import pandas as pd

                def _ga_row(label, res, akey):
                    return {"Model": label,
                            "Macro-AUROC": pm(res["macro_auc"]),
                            "Cosine → Original": pm(res["expl_cos_to_original"]),
                            "IoU → Original": pm(res["expl_iou_to_original"]),
                            "IoU → oracle": pm(res["expl_iou_to_oracle"]),
                            "Audit AUC": pm(audit[akey]["positive_auc"])}

                _sa = per_config_stats(str(FULL_EXP))
                _plain = dict(_sa["finetune_retain"], **stab_a["finetune_retain"])
                _epu1 = dict(_sa["epu@1"], **stab_a["epu@1"])
                st.dataframe(pd.DataFrame([
                    _ga_row("Plain (setting A)", _plain, ("A", "finetune_retain")),
                    _ga_row("EPU λ=1 (setting A)", _epu1, ("A", "epu@1")),
                    _ga_row("Ascent only", ga_res["ga_only"], ("GA", "ga_only")),
                    _ga_row("GA", ga_res["ga"], ("GA", "ga")),
                    _ga_row("GA + EPU λ=1", ga_res["ga_epu@1"], ("GA", "ga_epu@1")),
                ]), hide_index=True, width="stretch")
                st.caption("Mean ± std over seeds 42/43/44. Audit AUC: the "
                           "positive-finding membership attack of the table above "
                           "(0.5 = chance). The ascent runs reused the Original and "
                           "Oracle checkpoints, whose metrics were reproduced "
                           "exactly, on a different host.")
                st.markdown(
                    "- **Ascent alone damages the model**: macro-AUROC falls to "
                    f"{ga_res['ga_only']['macro_auc'][0]:.3f} and the audit AUC "
                    "goes *below* chance — the forgotten images become "
                    "recognisable by their unusually high loss.\n"
                    "- **Repair restores utility**, and the maps end about as far "
                    "from the Original's as after Plain fine-tuning.\n"
                    "- **Adding the explanation term gives the same stability "
                    "gain** as over Plain: IoU "
                    f"{ga_res['ga']['expl_iou_to_original'][0]:.3f} → "
                    f"{ga_res['ga_epu@1']['expl_iou_to_original'][0]:.3f}, in all "
                    "three seeds, with utility unchanged and no gain in oracle "
                    "fidelity.\n"
                    "- **One result is not in EPU's favour.** After ascent, "
                    "GA + EPU keeps slightly more of the membership trace than GA "
                    f"({audit[('GA', 'ga_epu@1')]['positive_auc'][0]:.3f} against "
                    f"{audit[('GA', 'ga')]['positive_auc'][0]:.3f}); the difference "
                    "is positive in every seed and its interval excludes zero in "
                    "two of three. An anchor to the original model works against "
                    "an unlearning step that moves the model away from it.")
            else:
                st.info("paper/ga_results.csv not found.")

            section("The verdict — stated the way a reviewer would accept it")
            st.markdown(
                "1. **On the project's stated objective — preserving the Original's "
                "explanations — EPU is better in every seed and setting.** In "
                "setting B the margin (+0.016 to +0.023 cosine) is 2.4–3.3× Plain's "
                "own across-seed standard deviation and all four metrics agree "
                "(IoU 0.60 → 0.87). In setting A, which keeps accuracy, IoU rises "
                "from 0.70 to 0.82. Added to gradient ascent, 0.71 → 0.82. The "
                "effect is monotonic in λ.\n"
                "2. **It costs no accuracy relative to Plain.** Macro-AUROC is "
                "within a fraction of a point of Plain everywhere. Setting B "
                "itself costs about 1.7 points, for Plain and EPU alike.\n"
                "3. **On the oracle reference, EPU ≈ Plain.** No reliable gain; "
                "both references are shown, and we do not pick the flattering "
                "one.\n"
                "4. **Forgetting is partial, and equal for Plain and EPU under "
                "fine-tuning.** The calibrated audit finds a weak trace of the "
                "client that fine-tuning lowers but does not remove.\n"
                "5. **After gradient ascent EPU retains slightly more of that "
                "trace.** Stability and forgetting then have to be traded off "
                "through λ.\n"
                "6. **Limits.** One dataset, one backbone, five clients, one "
                "forgotten client, three seeds; with three seeds the paired tests "
                "are indicative only.")

            note_html = (
                '<div class="take"><b>How to interpret this result.</b><br><br>'
                '<b>1. The effect is by construction — the experiment establishes '
                'its price.</b> EPU\'s added loss term is the distance to the '
                'Original\'s attention maps, so the <i>direction</i> of the gain is '
                'by design. What was not guaranteed, and is what this study '
                'measures, is the cost: macro-AUROC stays within a fraction of a '
                'point of Plain, and under fine-tuning the membership audit shows '
                'no extra trace of the client. After gradient ascent a small extra '
                'trace does appear. The clean dose-response from λ = 1 to λ = 100 '
                'confirms the mechanism is the loss term itself, not a seed '
                'artefact.<br><br>'
                '<b>2. The two references diverge — a measured finding.</b> The '
                'oracle\'s own cosine to the Original is '
                f'{stats["oracle"]["expl_cos_to_original"][0]:.4f}, <i>lower</i> '
                f'than Plain\'s {stats["finetune_retain"]["expl_cos_to_original"][0]:.4f}. '
                'A model retrained without the client attends differently from '
                'the Original of its own accord. '
                '"Preserve the validated reasoning" and "behave like a model that '
                'never saw the client" are distinct objectives, and this work '
                'quantifies the gap between them instead of assuming they coincide '
                '— which is why both references are reported throughout.</div>'
            ) if stats else ""
            st.markdown(note_html, unsafe_allow_html=True)
        else:
            st.warning("Result summaries not found — check the experiment folders.")

        section("Figures written by the training run itself")
        st.markdown(
            "These four are the training job's own output, untouched. They all "
            "measure explanation similarity **to the oracle** — the reference on "
            "which EPU and Plain are statistically indistinguishable — so the "
            "λ-ablation below looks flat *by design*, and does not contradict "
            "the charts above. Those plot fidelity **to the Original model**, "
            "which is the objective EPU actually optimises. Same runs, same "
            "CSVs, two different reference models; both are reported.")
        fig_dir = sel_exp / "figures"
        figs = {"Explanation & forgetting — vs oracle": "fig_oracle_comparison.png",
                "λ ablation — fidelity to the oracle": "fig_lambda_ablation.png",
                "Forgetting ↔ explanation trade-off — vs oracle":
                    "fig_tradeoff.png",
                "Runtime & speedup": "fig_runtime.png"}
        cols = st.columns(2, gap="medium")
        shown = 0
        for title_, fname in figs.items():
            fp = fig_dir / fname
            if fp.exists():
                cols[shown % 2].image(str(fp), caption=title_, width="stretch")
                shown += 1
        if sel_summary:
            with st.expander("Full comparison table — every metric, every model"):
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "Model": r["config"], "Macro-AUROC": r["macro_auc"],
                    "Micro-AUROC": r["micro_auc"], "mAP": r["mAP"],
                    "F1": r["macro_f1"], "Precision": r["precision"],
                    "Recall": r["recall"], "MIA": r["mia_auc"],
                    "Forget loss": r["forget_loss"],
                    "Cos→oracle": r["expl_cos_to_oracle"],
                    "Dist→oracle": r["pred_dist_to_oracle"],
                    "Speedup": r["speedup_vs_oracle"],
                } for r in sel_summary["tables"]["comparison"]]),
                    hide_index=True, width="stretch")

        section("Why the evaluation can be trusted")
        st.markdown(
            "- **Patient-disjoint split** — no patient appears in both train and "
            "test, so no leakage inflates AUROC.\n"
            "- **Identical regimes** — Plain and EPU share data, seeds, LR and "
            "schedule; only the explanation term differs.\n"
            "- **Three seeds, mean ± std** — one lucky seed cannot carry a claim.\n"
            "- **Pre-registered bar** — what \"EPU wins\" would mean was fixed "
            "*before* the sweep ran, and the verdict is judged against it.\n"
            "- **Self-describing checkpoints** — every .pt embeds its config and seed.")
        takeaway("Honesty is the strategy: a predicted, dose-responsive gain in "
                 "stability, a null on oracle fidelity, partial forgetting and one "
                 "trade-off after gradient ascent are <b>more credible</b> to any "
                 "panel than an inflated claim — and they are what the data shows.")
        nextbox()
    with side:
        toc(["Act 1 — Setting A: no gain against the oracle, a gain against the Original",
             "Act 2 — λ-sweep: the predicted effect appears",
             "Why the two campaigns report different numbers",
             "Explanation fidelity measured against two references",
             "Membership audit — a stronger test of forgetting",
             "Second baseline — gradient ascent at the departing client",
             "The verdict — stated the way a reviewer would accept it",
             "Figures written by the training run itself",
             "Why the evaluation can be trusted"])


# ===========================================================================
# PAGE — Training Setup & Reproducibility
# ===========================================================================
elif page == "Training Setup":
    topbar("Evidence", "training setup")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Methods · every hyperparameter, as run")
        display("Training setup &amp; reproducibility")
        lede("Every value below is the one the reported runs actually used — the "
             "frozen configuration, not a recommendation. The environment block is "
             "read live from the run's own <code>reproducibility.json</code>, so it "
             "cannot drift from what produced the results.")

        section("Data & partitioning")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Dataset | NIH ChestX-ray14 — 112,120 frontal radiographs, 14 findings |\n"
            "| Labels | multi-label (a study may carry several findings) |\n"
            "| Image size | 224 × 224, grayscale, 1 channel |\n"
            "| Validation / test split | 15% / 15%, **patient-disjoint** (no leakage) |\n"
            "| Clients | 5 |\n"
            "| Partition | Dirichlet, α = 0.5 (non-IID) |\n"
            "| Batch size | 32 |\n"
            "| Image cache | decode-once, in-memory |")

        section("Model")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Backbone | DeiT-Small — `deit_small_patch16_224` |\n"
            "| Pretraining | ImageNet, then adapted to 1-channel input |\n"
            "| Patch size | 16 (→ a 14 × 14 grid of 196 patches) |\n"
            "| Output head | 14 sigmoid logits |\n"
            "| Parameters | ≈ 22 M |")

        section("Federated training")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Aggregation | FedAvg (data-weighted parameter average) |\n"
            "| Rounds × local epochs | 30 × 2 |\n"
            "| Optimizer | Adam |\n"
            "| Learning rate | 2 × 10⁻⁴ |\n"
            "| LR schedule | 3 warmup rounds, then cosine decay |\n"
            "| Weight decay | 1 × 10⁻⁴ |\n"
            "| Loss | BCEWithLogits + per-class positive weighting |\n"
            "| pos_weight cap | 20.0 (stops rare classes dominating) |\n"
            "| Model selection | best validation checkpoint |\n"
            "| Early stopping | patience 5 rounds |\n"
            "| Decision threshold | 0.5 |")

        section("Unlearning")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Forgotten client | client 0 |\n"
            "| Oracle | full retrain from scratch on the 4 retained clients |\n"
            "| Plain | warm-start fine-tune on retained clients |\n"
            "| EPU | identical fine-tune + λ·L_exp |\n"
            "| Fine-tune rounds | 5 |\n"
            "| Fine-tune LR | 1 × 10⁻⁴, constant (identical for Plain and EPU) |\n"
            "| λ — main run | 0.5, 1.0 |\n"
            "| λ — sweep | 1, 10, 30, 100 |")

        section("Explanation measurement")
        st.markdown(
            "| Setting | Value |\n|---|---|\n"
            "| Method | attention rollout (head-average, +identity, row-normalised) |\n"
            "| Fused attention | disabled, so per-layer attention is capturable |\n"
            "| Probe images | 96, fixed across all models and seeds |\n"
            "| Map normalisation | min–max to [0, 1] |\n"
            "| Metrics | cosine, Pearson, IoU (threshold 0.5), MSE |\n"
            "| References | vs the **Original** (preservation) and vs the "
            "**oracle** (correct forgetting) |")

        section("Environment — read from the run record")
        rep = load_repro(str(sel_exp))
        if rep:
            pk = rep.get("packages", {})
            cu = rep.get("cuda", {})
            kpi_row([
                ("GPU", cu.get("device_name", "—"),
                 f"{cu.get('total_memory_gb', 0):.1f} GB · CUDA available: "
                 f"{cu.get('available')}"),
                ("Git commit", rep.get("git_commit", "—"),
                 "exact code that produced these results"),
                ("Seeds", ", ".join(str(s) for s in rep.get("seeds", [])),
                 "every metric is a mean ± std over these"),
                ("Run finished", str(rep.get("timestamp_utc", "—"))[:10],
                 "UTC date of the reported run"),
            ])
            _os, _kern, _glibc, _inst, _cpus = describe_platform(rep)
            st.markdown("**Machine**")
            st.markdown(
                "| Component | Value |\n|---|---|\n"
                f"| Operating system | {_os} |\n"
                f"| Kernel | {_kern} |\n"
                f"| Instance type | {_inst or '—'}"
                f"{' (1 × T4, ' + str(_cpus) + ' vCPU)' if _inst else ''} |\n"
                f"| GPU | {cu.get('device_name','—')} · "
                f"{cu.get('total_memory_gb', 0):.1f} GB |\n"
                f"| Platform string (as recorded) | `{rep.get('platform','—')}` |")
            st.markdown("**Software**")
            st.markdown(
                "| Component | Version |\n|---|---|\n"
                f"| Python | {rep.get('python','—')} |\n"
                f"| PyTorch | {pk.get('torch','—')} |\n"
                f"| torchvision | {pk.get('torchvision','—')} |\n"
                f"| timm | {pk.get('timm','—')} |\n"
                f"| NumPy | {pk.get('numpy','—')} |\n"
                f"| pandas | {pk.get('pandas','—')} |\n"
                f"| scikit-learn | {pk.get('sklearn','—')} |")
            st.caption(
                f"Source: reproducibility.json, written by the run itself "
                f"({exp_choice}). Every version above is recorded verbatim. The "
                f"**operating system and instance type are derived**, not "
                f"recorded — Python logs the kernel and glibc rather than the "
                f"distribution name, so “Ubuntu” follows from the Canonical "
                f"`-aws` kernel flavour and glibc {_glibc}, and the instance from "
                f"1 × Tesla T4 with {_cpus} vCPU. The raw platform string is shown "
                f"above so the inference can be checked.")
        else:
            st.warning(f"reproducibility.json not found in {sel_exp}")

        section("How to reproduce")
        st.markdown(
            "1. Fetch NIH ChestX-ray14 and place it under `data/nih/`.\n"
            "2. Check out the commit shown above.\n"
            "3. Run `python scripts/run_final.py` — it executes all three seeds, "
            "both unlearning procedures and the full evaluation, writing the same "
            "CSV/JSON/figures this site reads.\n"
            "4. Every checkpoint embeds its own config and seed, so any single "
            "model can be rebuilt and re-scored without the YAML files.")
        takeaway("Nothing here is a rounded-off recollection: the tables are the "
                 "frozen configs, and the environment block is read from the run's "
                 "own record. That is what makes the numbers <b>auditable</b>.")
    with side:
        toc(["Data & partitioning", "Model", "Federated training", "Unlearning",
             "Explanation measurement", "Environment — read from the run record",
             "How to reproduce"])

# ===========================================================================
# PAGE — Related Work & Research Gap
# ===========================================================================
elif page == "Related Work & Gap":
    topbar("Context", "related work &amp; gap")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Positioning · what exists, and what does not")
        display("Related work &amp; the research gap")
        lede("The literature relevant to this project falls into four strata, each "
             "solving one part of the problem. Read together they show a consistent "
             "omission: no located work treats <b>explanation stability as a "
             "measured objective of unlearning</b>.")

        section("Stratum A — Federated learning & machine unlearning")
        st.markdown(
            "| Work | Mechanism | Limitation | Relation to this project |\n"
            "|---|---|---|---|\n"
            "| McMahan et al., 2017 — *Communication-Efficient Learning of Deep "
            "Networks from Decentralized Data* | FedAvg: local SGD + weighted "
            "averaging | Degrades under non-IID; no unlearning | Adopted as our "
            "training protocol |\n"
            "| Cao & Yang, 2015 — *Towards Making Systems Forget* | "
            "Statistical-query decomposition | Limited to simple models | We target "
            "a deep Vision Transformer |\n"
            "| Bourtoule et al., 2021 — *Machine Unlearning* (SISA) | Sharded, "
            "isolated training + partial retrain | Storage overhead; not federated | "
            "We unlearn in FL without sharding |\n"
            "| Liu et al., 2021 — *FedEraser* | Replay + calibrate stored updates | "
            "Not explanation-aware | We add and measure explanation fidelity |")
        st.markdown(
            "**Relevance.** This stratum supplies the training protocol and shows "
            "that efficient client-level unlearning is feasible. But every method "
            "defines success purely as data-influence removal plus accuracy "
            "recovery — none asks whether the model's explanations survive.")

        section("Stratum B — Vision Transformers for chest X-ray")
        st.markdown(
            "| Work | Mechanism | Limitation | Relation to this project |\n"
            "|---|---|---|---|\n"
            "| Dosovitskiy et al., 2021 — *ViT* | Patch embeddings + self-attention "
            "| Data-hungry | Backbone family |\n"
            "| Touvron et al., 2021 — *DeiT* | Distillation + strong augmentation | "
            "Still needs pretraining | Our chosen backbone |\n"
            "| Wang et al., 2017 — *ChestX-ray8/14* | Weakly-supervised multi-label "
            "CNN | Label noise; centralized | Our dataset, classified with a ViT "
            "under FL |\n"
            "| Rajpurkar et al., 2017 — *CheXNet* | DenseNet-121 classifier | "
            "Centralized; no unlearning or XAI | We add federation, unlearning and "
            "explanation fidelity |")

        section("Stratum C — Explainability & attention interpretation")
        st.markdown(
            "| Work | Mechanism | Limitation | Relation to this project |\n"
            "|---|---|---|---|\n"
            "| Abnar & Zuidema, 2020 — *Quantifying Attention Flow* | Attention "
            "rollout across layers | Approximate attribution | Our explanation "
            "operator, made differentiable |\n"
            "| Selvaraju et al., 2017 — *Grad-CAM* | Gradient-weighted activation "
            "maps | CNN-oriented, class-specific | Motivates preservation; less "
            "suited to multi-label ViT |\n"
            "| Tjoa & Guan, 2021 — *Survey on XAI: Toward Medical XAI* | Taxonomy "
            "of medical XAI | Survey; proposes no method | Motivates "
            "explanation-aware unlearning |")
        st.markdown(
            "**Relevance.** These works treat explanation as something to "
            "*visualise after training*. This project treats it as something to "
            "*optimise during unlearning* — the key conceptual move.")

        section("Stratum D — Privacy verification")
        st.markdown(
            "| Work | Mechanism | Limitation | Relation to this project |\n"
            "|---|---|---|---|\n"
            "| Shokri et al., 2017 — *Membership Inference Attacks* | Membership "
            "inference | Assumes shadow models | Repurposed as our "
            "forgetting-verification metric |")

        section("The research gap")
        st.markdown(
            "| Dimension | State of the art | This project |\n"
            "|---|---|---|\n"
            "| Unlearning objective | Remove influence, restore accuracy | "
            "**Additionally preserve explanation fidelity** |\n"
            "| Explanation during unlearning | Not considered | An explicit, "
            "differentiable training term |\n"
            "| Model family | Predominantly CNNs | Vision Transformer (DeiT-Small) |\n"
            "| Explanation measurement | Qualitative visualisation | Quantitative: "
            "cosine, Pearson, IoU, MSE — against two references |\n"
            "| Forgetting verification | Assumed or loss-based | "
            "Oracle-calibrated membership audit with patient-level bootstrap "
            "intervals |\n"
            "| Reference standard | Often absent | Retrain-from-scratch oracle |\n"
            "| Reproducibility | Frequently single-run | 3 seeds, fixed regime, "
            "full artefact logging |")
        takeaway("The unoccupied position is the intersection: <b>client-level "
                 "federated unlearning + an explicit attention-fidelity objective + "
                 "quantitative measurement + oracle-referenced verification + "
                 "multi-seed reproducibility</b>, applied to a ViT chest X-ray "
                 "classifier.")
        nextbox()
    with side:
        toc(["Stratum A — Federated learning & machine unlearning",
             "Stratum B — Vision Transformers for chest X-ray",
             "Stratum C — Explainability & attention interpretation",
             "Stratum D — Privacy verification", "The research gap"])


# ===========================================================================
# PAGE — Scope & Design Decisions
# ===========================================================================
elif page == "Scope & Design Decisions":
    topbar("Context", "scope &amp; design decisions")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Methodology · why the study is built this way")
        display("Scope &amp; design decisions")
        lede("Each choice below was made deliberately, for a stated reason, and "
             "each one is what makes the measured comparison valid. Together they "
             "define a complete, self-contained study: the experimental design is "
             "fixed, executed and reported in full.")

        section("The controlled-comparison design")
        st.markdown(
            "**Decision.** One forgotten client (client 0), one dataset, one "
            "backbone, and within each of the two settings one fine-tuning "
            "schedule — held fixed across every arm.\n\n"
            "**Why.** The scientific question is what the explanation term does. "
            "Holding everything else constant means the only variable between "
            "Plain and EPU is a single loss term, so the measured difference is "
            "attributable to that term and nothing else. Varying several factors "
            "at once would have produced a larger study with weaker conclusions.")

        section("Five clients, Dirichlet α = 0.5")
        st.markdown(
            "**Decision.** A five-client, patient-disjoint, non-IID federated "
            "simulation.\n\n"
            "**Why.** This is the standard configuration in the federated-learning "
            "literature and is deliberately *harder* than a uniform split: client "
            "updates genuinely conflict, so FedAvg is tested under realistic "
            "heterogeneity. A simulation also isolates the method from "
            "infrastructure noise — stragglers and dropped connections would "
            "confound the measurement without testing the contribution.")

        section("Three seeds, reported as mean ± std")
        st.markdown(
            "**Decision.** The full pipeline — original, oracle, both unlearning "
            "procedures, complete evaluation — repeated under seeds 42, 43 and 44.\n\n"
            "**Why.** Three independent repetitions are what allow every number to "
            "be reported with a standard deviation and every comparison to be "
            "**paired by seed**, which is how the explanation result is shown to "
            "hold in each run individually rather than only on average. Each seed "
            "costs roughly 10 GPU-hours, so this is the design that buys real "
            "statistical statements within the compute available.")

        section("Attention rollout as the explanation operator")
        st.markdown(
            "**Decision.** Attention rollout, used identically for training and "
            "for evaluation.\n\n"
            "**Why.** It is model-intrinsic to the Vision Transformer, "
            "class-agnostic (which suits multi-label chest X-rays), and — "
            "decisively — differentiable, so it can live inside the training loss. "
            "Using the same operator on both sides keeps every comparison "
            "like-for-like. Robustness is covered by reporting four independent "
            "similarity metrics (cosine, Pearson, IoU, MSE) rather than relying on "
            "any single one.")

        section("Reporting both reference points")
        st.markdown(
            "**Decision.** Explanation fidelity is reported against the "
            "**Original** *and* against the **oracle**, on every metric.\n\n"
            "**Why.** They answer different questions — preservation of validated "
            "reasoning versus equivalence to a model that never saw the client — "
            "and they do not always agree. Reporting both, including the reference "
            "on which EPU shows no advantage, is what makes the result on the "
            "stated objective credible rather than selected.")

        section("A second unlearning baseline")
        st.markdown(
            "**Decision.** The explanation term was also added to gradient ascent "
            "at the departing client, a published client-unlearning method, in "
            "setting A.\n\n"
            "**Why.** A gain shown only over plain fine-tuning could be a property "
            "of that one procedure. The same gain in stability appears after "
            "gradient ascent — and so does a small cost in the membership audit, "
            "which is reported with it.")

        section("Project maturity")
        kpi_row([
            ("Technology readiness", "TRL 4",
             "validated in a laboratory setting on a public benchmark"),
            ("Sustainable Development Goals", "SDG 3 · 9",
             "good health & well-being; industry, innovation & infrastructure"),
            ("Intended outcome", "Journal paper",
             "the contribution is a method plus an evaluation framework"),
        ])
        st.markdown(
            "The study is complete within the scope defined above: the pipeline "
            "runs end to end, the comparison is controlled, and the results are "
            "reproducible from the recorded commit and seeds. Extending the same "
            "machinery to larger federations, other imaging modalities or clinical "
            "validation would be a separate programme of work — outside this "
            "project's defined scope, and not required for its conclusions to hold.")
        takeaway("Every design choice here was made to protect one thing: that the "
                 "measured difference between Plain and EPU is caused by the "
                 "explanation term and nothing else. That is what makes the result "
                 "<b>defensible</b>.")
        nextbox()
    with side:
        toc(["The controlled-comparison design", "Five clients, Dirichlet α = 0.5",
             "Three seeds, reported as mean ± std",
             "Attention rollout as the explanation operator",
             "Reporting both reference points", "A second unlearning baseline",
             "Project maturity"])


# ===========================================================================
# PAGE — References
# ===========================================================================
elif page == "References":
    topbar("Context", "references")
    art, side = st.columns([3, 1], gap="large")
    with art:
        kicker("Bibliography")
        display("References")
        lede("Works cited across this archive, in IEEE style.")

        section("Bibliography")
        refs = [
            "B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "
            "“Communication-Efficient Learning of Deep Networks from Decentralized "
            "Data,” in <i>Proc. AISTATS</i>, 2017.",
            "G. Liu, X. Ma, Y. Yang, C. Wang, and J. Liu, “FedEraser: Enabling "
            "Efficient Client-Level Data Removal from Federated Learning Models,” "
            "in <i>Proc. IEEE/ACM IWQoS</i>, 2021.",
            "L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, "
            "A. Travers, B. Zhang, D. Lie, and N. Papernot, “Machine Unlearning,” "
            "in <i>Proc. IEEE Symposium on Security and Privacy</i>, 2021.",
            "Y. Cao and J. Yang, “Towards Making Systems Forget with Machine "
            "Unlearning,” in <i>Proc. IEEE Symposium on Security and Privacy</i>, 2015.",
            "A. Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers "
            "for Image Recognition at Scale,” in <i>Proc. ICLR</i>, 2021.",
            "H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and "
            "H. Jégou, “Training Data-Efficient Image Transformers &amp; "
            "Distillation through Attention,” in <i>Proc. ICML</i>, 2021.",
            "S. Abnar and W. Zuidema, “Quantifying Attention Flow in Transformers,” "
            "in <i>Proc. ACL</i>, 2020.",
            "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and "
            "D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via "
            "Gradient-Based Localization,” in <i>Proc. IEEE ICCV</i>, 2017.",
            "X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "
            "“ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on "
            "Weakly-Supervised Classification and Localization of Common Thorax "
            "Diseases,” in <i>Proc. IEEE CVPR</i>, 2017.",
            "P. Rajpurkar et al., “CheXNet: Radiologist-Level Pneumonia Detection "
            "on Chest X-Rays with Deep Learning,” arXiv:1711.05225, 2017.",
            "R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership "
            "Inference Attacks Against Machine Learning Models,” in <i>Proc. IEEE "
            "Symposium on Security and Privacy</i>, 2017.",
            "E. Tjoa and C. Guan, “A Survey on Explainable Artificial Intelligence "
            "(XAI): Toward Medical XAI,” <i>IEEE Transactions on Neural Networks "
            "and Learning Systems</i>, 2021.",
        ]
        st.markdown(
            "<ol style='padding-left:1.2rem'>" +
            "".join(f"<li style='margin-bottom:10px;font-size:14px;line-height:1.6;"
                    f"color:var(--on-surface-variant)'>{r}</li>" for r in refs) +
            "</ol>", unsafe_allow_html=True)

        section("Software & data")
        st.markdown(
            "| Resource | Detail |\n|---|---|\n"
            "| Dataset | NIH ChestX-ray14, National Institutes of Health Clinical "
            "Center, public release |\n"
            "| Backbone weights | `deit_small_patch16_224`, ImageNet-pretrained, "
            "via `timm` |\n"
            "| Frameworks | PyTorch, timm, torchvision, scikit-learn, NumPy, pandas |\n"
            "| Code | this project's repository at the commit shown on the Training "
            "Setup page |")
        takeaway("Every claim in this archive traces to one of the works above, to "
                 "the project's own measured outputs, or is explicitly labelled as "
                 "engineering judgement.")
    with side:
        toc(["Bibliography", "Software & data"])

st.markdown(
    f'<div class="site-foot">{PROJECT_TITLE} · DeiT-Small · '
    'NIH ChestX-ray14 · 3 seeds · oracle-benchmarked · research prototype (TRL 4) · '
    'all computation local, CPU-only.</div>', unsafe_allow_html=True)
