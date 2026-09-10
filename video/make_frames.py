import fitz, os, base64
OUT = "/Users/harshulkuhar/Documents/REPOSITORIES/rextract-aaai27/video/frames"
W, H = 1920, 1080
INK, MUT, LINE = "#14181d", "#5b6672", "#d7dde3"
GRN, RED, BLU, BG = "#2f8f5b", "#c0392b", "#2d6cb5", "#fbfcfd"
F = "Helvetica, Arial, sans-serif"

def render(name, body, bg=BG):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{F}">'
           f'<rect width="{W}" height="{H}" fill="{bg}"/>{body}</svg>')
    p = f"/tmp/_f_{name}.svg"; open(p, "w").write(svg)
    d = fitz.open(p); d[0].get_pixmap(matrix=fitz.Matrix(1, 1)).save(f"{OUT}/{name}.png")
    os.remove(p); print("  ", name + ".png")

def t(x, y, s, size=40, fill=INK, anchor="start", weight="normal", spacing="0"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" letter-spacing="{spacing}">{s}</text>')

# ---------- 1. TITLE ----------
b  = f'<rect x="0" y="0" width="{W}" height="14" fill="{GRN}"/>'
b += t(160, 300, "RE-XTRACT", 122, INK, weight="700", spacing="-2")
b += t(160, 392, "Unanimity-Gated Document Extraction", 58, MUT)
b += t(160, 462, "with Calibrated Abstention", 58, MUT)
b += f'<line x1="160" y1="546" x2="620" y2="546" stroke="{LINE}" stroke-width="3"/>'
b += t(160, 626, "Harshul Kuhar &#183; Harshit Oberoi", 40, INK)
b += t(160, 684, "Housing.com, Gurugram, India", 36, MUT)
b += t(160, 940, "AAAI-27 Demonstration Track", 34, GRN, weight="600", spacing="2")
render("01_title", b)

# ---------- 2. THE PROBLEM ----------
b  = t(160, 190, "THE ASYMMETRY", 32, GRN, weight="700", spacing="4")
b += t(160, 320, "Wrong information costs more", 78, INK, weight="700")
b += t(160, 412, "than missing information.", 78, INK, weight="700")
# two cards
b += f'<rect x="160" y="530" width="760" height="330" rx="14" fill="#fdf1ef" stroke="{RED}" stroke-width="3"/>'
b += t(210, 610, "WRONG VALUE", 30, RED, weight="700", spacing="3")
b += t(210, 682, "Silently republished as the", 38, INK)
b += t(210, 734, "project&#8217;s official status.", 38, INK)
b += t(210, 812, "Nobody knows to check it.", 36, MUT)
b += f'<rect x="1000" y="530" width="760" height="330" rx="14" fill="#eef6f1" stroke="{GRN}" stroke-width="3"/>'
b += t(1050, 610, "MISSING VALUE", 30, GRN, weight="700", spacing="3")
b += t(1050, 682, "Flagged, re-queued, and", 38, INK)
b += t(1050, 734, "read by a human.", 38, INK)
b += t(1050, 812, "Recoverable.", 36, MUT)
b += t(160, 960, "A VLM cannot tell you which one it has just produced.", 40, MUT)
render("02_problem", b)

# ---------- 3. ARCHITECTURE ----------
img = base64.b64encode(open("/Users/harshulkuhar/Documents/REPOSITORIES/rextract-aaai27/paper/data/rextract_flow.png","rb").read()).decode()
iw, ih = 1512, 1368
sc = 880 / ih
b  = t(150, 140, "ONE GATE CYCLE, RUN TWICE", 44, INK, weight="700", spacing="1")
b += f'<image x="140" y="180" width="{iw*sc:.0f}" height="{ih*sc:.0f}" xlink:href="data:image/png;base64,{img}"/>'
x0 = 140 + iw*sc + 90
b += t(x0, 300, "Stage 1", 46, GRN, weight="700")
b += t(x0, 362, "Document classification", 38, INK)
b += t(x0, 414, "4 certificate types, or reject", 34, MUT)
b += f'<line x1="{x0}" y1="470" x2="{W-150}" y2="470" stroke="{LINE}" stroke-width="3"/>'
b += t(x0, 552, "Stage 2", 46, BLU, weight="700")
b += t(x0, 614, "Date &amp; detail extraction", 38, INK)
b += t(x0, 666, "that type&#8217;s schema", 34, MUT)
b += f'<line x1="{x0}" y1="722" x2="{W-150}" y2="722" stroke="{LINE}" stroke-width="3"/>'
b += t(x0, 800, "Same three gates.", 38, INK, weight="600")
b += t(x0, 852, "Same abstention rule.", 38, INK, weight="600")
render("03_architecture", b)

# ---------- 4. RESULTS ----------
b  = t(150, 150, "RESULTS", 34, GRN, weight="700", spacing="4")
b += t(150, 232, "143 documents &#183; 47 projects &#183; 7 states", 44, MUT)
def row(y, label, a, bb, bold=False, big=False):
    s = 46 if not big else 54
    w = "700" if bold else "normal"
    o  = t(190, y, label, s, INK, weight=w)
    o += t(1180, y, a, s, MUT, anchor="end", weight=w)
    o += t(1720, y, bb, s, GRN if bold else INK, anchor="end", weight="700")
    return o
b += t(1180, 330, "Generator&#8211;verifier", 34, MUT, anchor="end", spacing="1")
b += t(1720, 330, "RE-XTRACT", 34, INK, anchor="end", weight="700", spacing="1")
b += f'<line x1="150" y1="366" x2="1770" y2="366" stroke="{INK}" stroke-width="3"/>'
b += t(190, 440, "Stage 1  &#183;  Document classification", 34, GRN, weight="700", spacing="2")
b += row(514, "Accuracy", "85.3%", "95.1%", bold=True, big=True)
b += row(586, "Macro-F1", ".804", ".955")
b += row(658, "Out-of-schema recall", ".32", ".93", bold=True)
b += f'<line x1="150" y1="712" x2="1770" y2="712" stroke="{LINE}" stroke-width="2"/>'
b += t(190, 786, "Stage 2  &#183;  Date extraction, 113 gold dates", 34, BLU, weight="700", spacing="2")
b += row(866, "Accuracy", "88.5%", "95.6%", bold=True, big=True)
b += f'<line x1="150" y1="928" x2="1770" y2="928" stroke="{INK}" stroke-width="3"/>'
render("04_results", b)

# ---------- 5. CLOSING ----------
b  = f'<rect x="0" y="0" width="{W}" height="14" fill="{GRN}"/>'
b += t(160, 300, "RE-XTRACT", 96, INK, weight="700", spacing="-2")
b += t(160, 376, "Unanimity-gated extraction with calibrated abstention", 42, MUT)
b += f'<line x1="160" y1="470" x2="620" y2="470" stroke="{LINE}" stroke-width="3"/>'
b += t(160, 570, "Live demo", 34, GRN, weight="700", spacing="3")
b += t(160, 640, "taxonomic-unadversely-zonia.ngrok-free.dev", 46, INK)
b += t(160, 760, "Code", 34, GRN, weight="700", spacing="3")
b += t(160, 830, "github.com/elarahq/rextract_aaai27", 46, INK)
b += t(160, 980, "Harshul Kuhar &#183; Harshit Oberoi &#183; Housing.com", 34, MUT)
render("05_closing", b)
