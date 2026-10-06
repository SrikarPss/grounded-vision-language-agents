"""Generate deterministic PNGs for the synthetic sample fixture.

Run once to populate data/samples/images/.  All images are built with
PIL only (no external assets) so they are fully reproducible offline.

Coverage (50 tasks):
  - UI / web interactions  (click, type, select, scroll)  — 25 tasks
  - Visual QA              (answer)                        — 25 tasks
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMG_DIR = HERE / "images"
IMG_DIR.mkdir(exist_ok=True)

W, H = 320, 200   # standard canvas


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _canvas(bg: tuple = (245, 245, 245)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)


def _button(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
            label: str, bg=(70, 130, 180), fg=(255, 255, 255), radius: int = 6) -> None:
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=bg, outline=(40, 80, 120), width=2)
    font = _font(14)
    bbox = d.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x + (w - tw) // 2, y + (h - th) // 2), label, fill=fg, font=font)


def _input_field(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                 placeholder: str = "", value: str = "") -> None:
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    text = value or placeholder
    color = (120, 120, 120) if not value else (30, 30, 30)
    d.text((x + 8, y + (h - 14) // 2), text, fill=color, font=_font(13))


def _label(d: ImageDraw.ImageDraw, x: int, y: int, text: str,
           color=(50, 50, 50), size: int = 13) -> None:
    d.text((x, y), text, fill=color, font=_font(size))


def _save(img: Image.Image, name: str) -> Path:
    p = IMG_DIR / name
    img.save(p, format="PNG")
    return p


# ---------------------------------------------------------------------------
# UI / web images (25)
# ---------------------------------------------------------------------------

def ui_login():
    img, d = _canvas((230, 240, 255))
    _label(d, 10, 8, "Sign in to your account", size=15, color=(30, 30, 80))
    _label(d, 10, 42, "Email")
    _input_field(d, 10, 58, 200, 28, "you@example.com")
    _label(d, 10, 96, "Password")
    _input_field(d, 10, 112, 200, 28, "••••••••")
    _button(d, 10, 152, 100, 32, "Log in", bg=(60, 100, 200))
    _button(d, 120, 152, 100, 32, "Cancel", bg=(160, 160, 160))
    return _save(img, "login.png")


def ui_submit():
    img, d = _canvas((230, 255, 230))
    _label(d, 10, 8, "Submit your response", size=15, color=(30, 80, 30))
    _label(d, 10, 42, "Your answer:")
    _input_field(d, 10, 58, 290, 28, "Type here…")
    _label(d, 10, 100, "Agree to terms?")
    d.rectangle([10, 116, 26, 132], fill=(255, 255, 255), outline=(100, 100, 100), width=2)
    _label(d, 34, 115, "I agree to the Terms of Service", size=12)
    _button(d, 10, 152, 120, 32, "Submit", bg=(40, 160, 40))
    _button(d, 140, 152, 80, 32, "Reset", bg=(180, 80, 40))
    return _save(img, "submit.png")


def ui_search():
    img, d = _canvas()
    _label(d, 10, 8, "Search", size=16, color=(30, 30, 30))
    _input_field(d, 10, 40, 240, 32, "Search products…")
    _button(d, 258, 40, 50, 32, "🔍 Go")
    _label(d, 10, 90, "Recent searches:", size=12, color=(100, 100, 100))
    for i, term in enumerate(["laptop", "keyboard", "monitor"]):
        _label(d, 20, 110 + i * 20, f"• {term}", size=12, color=(70, 70, 200))
    return _save(img, "search.png")


def ui_dropdown():
    img, d = _canvas()
    _label(d, 10, 10, "Sort results by:", size=14)
    d.rectangle([10, 36, 200, 64], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 44, "Price: Low to High", size=13)
    d.polygon([(186, 46), (196, 46), (191, 56)], fill=(80, 80, 80))
    for i, opt in enumerate(["Relevance", "Price: Low to High",
                              "Price: High to Low", "Customer Rating"]):
        y = 68 + i * 26
        bg = (220, 230, 255) if i == 1 else (255, 255, 255)
        d.rectangle([10, y, 200, y + 24], fill=bg, outline=(200, 200, 200))
        _label(d, 18, y + 5, opt, size=12)
    return _save(img, "dropdown.png")


def ui_navbar():
    img, d = _canvas((40, 60, 100))
    for i, item in enumerate(["Home", "Products", "About", "Contact"]):
        x = 10 + i * 76
        bg = (255, 255, 255) if item == "Home" else (40, 60, 100)
        fg = (30, 30, 30) if item == "Home" else (220, 220, 220)
        d.rectangle([x, 10, x + 68, 40], fill=bg, outline=(100, 120, 160))
        _label(d, x + 8, 18, item, color=fg, size=13)
    d.rectangle([0, 50, W, 200], fill=(245, 245, 245))
    _label(d, 10, 70, "Welcome to our store!", size=15, color=(30, 30, 80))
    _label(d, 10, 100, "Browse our latest collection below.", size=12)
    return _save(img, "navbar.png")


def ui_signup():
    img, d = _canvas((250, 230, 255))
    _label(d, 10, 8, "Create Account", size=16, color=(80, 30, 120))
    _label(d, 10, 42, "Full Name")
    _input_field(d, 10, 58, 290, 26)
    _label(d, 10, 92, "Email")
    _input_field(d, 10, 108, 290, 26)
    _label(d, 10, 140, "Password")
    _input_field(d, 10, 156, 290, 26, "Min. 8 characters")
    _button(d, 90, W // 2 - 20, 140, 32, "Sign Up", bg=(120, 40, 180))
    return _save(img, "signup.png")


def ui_cart():
    img, d = _canvas()
    _label(d, 10, 8, "Shopping Cart  (2 items)", size=15, color=(30, 30, 30))
    d.line([0, 32, W, 32], fill=(200, 200, 200), width=1)
    items = [("Mechanical Keyboard", "$89.99"), ("USB-C Hub", "$34.99")]
    for i, (name, price) in enumerate(items):
        y = 44 + i * 44
        d.rectangle([6, y, W - 6, y + 36], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 14, y + 10, name, size=13)
        _label(d, W - 70, y + 10, price, size=13, color=(180, 40, 40))
    _button(d, 10, 148, 140, 36, "Add to Cart", bg=(255, 153, 0), fg=(30, 30, 30))
    _button(d, 162, 148, 148, 36, "Proceed to Checkout", bg=(40, 160, 40))
    return _save(img, "cart.png")


def ui_email_form():
    img, d = _canvas()
    _label(d, 10, 8, "Contact Us", size=16, color=(30, 30, 80))
    _label(d, 10, 40, "Your Email Address")
    _input_field(d, 10, 58, 290, 28, "email@domain.com")
    _label(d, 10, 96, "Subject")
    _input_field(d, 10, 112, 290, 28, "Enter subject")
    _label(d, 10, 148, "Message")
    d.rectangle([10, 164, 300, 190], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    _label(d, 18, 170, "Type your message here…", size=12, color=(160, 160, 160))
    return _save(img, "email_form.png")


def ui_password():
    img, d = _canvas()
    _label(d, 10, 8, "Change Password", size=15, color=(30, 30, 80))
    for i, lbl in enumerate(["Current Password", "New Password", "Confirm Password"]):
        y = 42 + i * 46
        _label(d, 10, y, lbl)
        _input_field(d, 10, y + 16, 260, 26, "••••••••")
    _button(d, 10, 166, 140, 28, "Update Password", bg=(60, 100, 200))
    return _save(img, "password.png")


def ui_checkbox():
    img, d = _canvas()
    _label(d, 10, 10, "Account Preferences", size=15, color=(30, 30, 80))
    opts = [
        (True,  "Receive newsletter"),
        (False, "Enable two-factor authentication"),
        (True,  "Show profile publicly"),
        (False, "Dark mode"),
    ]
    for i, (checked, text) in enumerate(opts):
        y = 44 + i * 34
        d.rectangle([10, y, 26, y + 16], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
        if checked:
            d.line([12, y + 8, 17, y + 13], fill=(40, 160, 40), width=3)
            d.line([17, y + 13, 24, y + 4], fill=(40, 160, 40), width=3)
        _label(d, 34, y + 1, text, size=13)
    return _save(img, "checkbox.png")


def ui_close():
    img, d = _canvas()
    _label(d, 10, 10, "Confirmation Dialog", size=15, color=(30, 30, 80))
    d.rectangle([10, 38, W - 10, H - 10], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    d.text((W - 30, 14), "✕", fill=(160, 60, 60), font=_font(18))
    _label(d, 20, 52, "Are you sure you want to delete", size=13)
    _label(d, 20, 70, "this item? This cannot be undone.", size=13)
    _button(d, 20, 110, 100, 30, "Cancel", bg=(160, 160, 160))
    _button(d, 134, 110, 120, 30, "Delete", bg=(200, 50, 50))
    return _save(img, "close.png")


def ui_pagination():
    img, d = _canvas()
    _label(d, 10, 10, "Search Results  (Page 3 of 8)", size=14)
    for i in range(1, 6):
        x = 10 + (i - 1) * 52
        bg = (60, 100, 200) if i == 3 else (240, 240, 240)
        fg = (255, 255, 255) if i == 3 else (60, 60, 60)
        _button(d, x, 50, 44, 32, str(i), bg=bg, fg=fg)
    _button(d, 10, 104, 80, 30, "◀ Prev", bg=(240, 240, 240), fg=(60, 60, 60))
    _button(d, 100, 104, 80, 30, "Next ▶", bg=(60, 100, 200))
    return _save(img, "pagination.png")


def ui_download():
    img, d = _canvas((230, 255, 245))
    _label(d, 10, 10, "File Downloads", size=15, color=(20, 80, 60))
    files = [("report_2024.pdf", "1.2 MB"), ("data.csv", "340 KB"), ("slides.pptx", "5.7 MB")]
    for i, (name, size) in enumerate(files):
        y = 44 + i * 40
        _label(d, 10, y + 4, f"📄 {name}", size=13)
        _label(d, 220, y + 4, size, size=11, color=(120, 120, 120))
        _button(d, 248, y, 60, 26, "⬇ Save", bg=(40, 160, 120), fg=(255, 255, 255))
    return _save(img, "download.png")


def ui_delete():
    img, d = _canvas()
    _label(d, 10, 10, "Manage Files", size=15, color=(30, 30, 80))
    files = ["project_draft.docx", "old_backup.zip", "temp_notes.txt"]
    for i, name in enumerate(files):
        y = 44 + i * 42
        d.rectangle([6, y, W - 6, y + 34], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 14, y + 9, name, size=13)
        _button(d, W - 76, y + 5, 64, 24, "🗑 Delete", bg=(200, 50, 50), fg=(255, 255, 255))
    return _save(img, "delete.png")


def ui_save():
    img, d = _canvas()
    _label(d, 10, 10, "Edit Profile", size=15, color=(30, 30, 80))
    _label(d, 10, 44, "Display Name")
    _input_field(d, 10, 60, 290, 26, value="Alex Johnson")
    _label(d, 10, 96, "Bio")
    d.rectangle([10, 112, 300, 148], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    _label(d, 18, 118, "Software engineer & coffee enthusiast.", size=12)
    _button(d, 10, 160, 80, 28, "💾 Save", bg=(60, 100, 200))
    _button(d, 100, 160, 80, 28, "Discard", bg=(160, 160, 160))
    return _save(img, "save.png")


def ui_upload():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 10, "Upload Document", size=15, color=(30, 30, 80))
    d.rectangle([20, 44, W - 20, 130], fill=(255, 255, 255),
                outline=(100, 140, 220), width=2)
    _label(d, 80, 60, "⬆", size=28, color=(100, 140, 220))
    _label(d, 60, 96, "Drag & drop or click to browse", size=12, color=(120, 120, 160))
    _button(d, 80, 144, 160, 32, "📂 Choose File", bg=(80, 120, 200))
    return _save(img, "upload.png")


def ui_filter():
    img, d = _canvas()
    _label(d, 10, 8, "Filter Products", size=15, color=(30, 30, 80))
    _label(d, 10, 36, "Category")
    d.rectangle([10, 52, 200, 76], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 58, "All Categories  ▾", size=13)
    _label(d, 10, 86, "Price Range")
    d.rectangle([10, 102, 200, 126], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 108, "$0 – $500  ▾", size=13)
    _label(d, 10, 136, "In Stock Only")
    d.rectangle([10, 152, 26, 168], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    _button(d, 60, 150, 120, 30, "Apply Filters", bg=(60, 100, 200))
    return _save(img, "filter.png")


def ui_sort():
    img, d = _canvas()
    _label(d, 10, 8, "Sort by:", size=14, color=(50, 50, 50))
    options = ["Name (A–Z)", "Date Modified", "File Size", "Type"]
    for i, opt in enumerate(options):
        y = 36 + i * 36
        selected = (i == 1)
        bg = (220, 230, 255) if selected else (255, 255, 255)
        d.rectangle([10, y, 220, y + 28], fill=bg, outline=(180, 180, 200), width=2)
        dot = "●" if selected else "○"
        _label(d, 18, y + 6, f"{dot}  {opt}", size=13,
               color=(60, 80, 180) if selected else (60, 60, 60))
    return _save(img, "sort.png")


def ui_settings():
    img, d = _canvas()
    _label(d, 10, 8, "⚙  Settings", size=16, color=(30, 30, 30))
    items = ["Account", "Notifications", "Privacy", "Help & Support"]
    for i, item in enumerate(items):
        y = 40 + i * 36
        d.rectangle([6, y, W - 6, y + 28], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 16, y + 7, item, size=13)
        _label(d, W - 26, y + 7, "›", size=16, color=(160, 160, 160))
    return _save(img, "settings.png")


def ui_profile():
    img, d = _canvas((240, 245, 255))
    d.ellipse([10, 16, 70, 76], fill=(180, 200, 240), outline=(100, 130, 200), width=3)
    _label(d, 22, 36, "AJ", size=20, color=(60, 80, 160))
    _label(d, 80, 20, "Alex Johnson", size=15, color=(30, 30, 80))
    _label(d, 80, 42, "alex.j@example.com", size=12, color=(100, 100, 100))
    _label(d, 80, 60, "Member since 2021", size=11, color=(140, 140, 140))
    d.line([0, 90, W, 90], fill=(200, 200, 210), width=1)
    _button(d, 10, 104, 140, 30, "Edit Profile", bg=(60, 100, 200))
    _button(d, 160, 104, 140, 30, "Log Out", bg=(200, 60, 60))
    return _save(img, "profile.png")


def ui_scroll():
    img, d = _canvas()
    _label(d, 10, 8, "News Feed", size=15, color=(30, 30, 80))
    headlines = [
        "Scientists discover new exoplanet",
        "Stock markets reach record high",
        "New AI model breaks benchmarks",
        "Local team wins championship",
    ]
    for i, h in enumerate(headlines):
        y = 36 + i * 36
        d.line([0, y - 4, W, y - 4], fill=(220, 220, 220), width=1)
        _label(d, 10, y + 2, h, size=12)
        _label(d, 10, y + 18, "2 hours ago", size=10, color=(150, 150, 150))
    d.rectangle([W - 14, 0, W, H], fill=(230, 230, 230))
    d.rectangle([W - 13, 20, W - 1, 60], fill=(150, 150, 150), outline=(120, 120, 120))
    return _save(img, "scroll.png")


def ui_terms():
    img, d = _canvas()
    _label(d, 10, 8, "Terms and Conditions", size=14, color=(30, 30, 80))
    lines = [
        "1. You agree to use this service lawfully.",
        "2. We respect your privacy.",
        "3. Content may not be reproduced.",
        "4. Service may change without notice.",
    ]
    for i, line in enumerate(lines):
        _label(d, 10, 36 + i * 22, line, size=11, color=(60, 60, 60))
    d.rectangle([10, 130, 26, 146], fill=(255, 255, 255), outline=(100, 100, 100), width=2)
    _label(d, 34, 130, "I have read and accept the Terms", size=12)
    _button(d, 10, 158, 140, 30, "Continue", bg=(60, 100, 200))
    return _save(img, "terms.png")


def ui_notification():
    img, d = _canvas()
    _label(d, 10, 8, "🔔  Notifications  (3 unread)", size=14, color=(30, 30, 80))
    items = [
        ("Alice liked your post", "2 min ago", True),
        ("Bob started following you", "1 hr ago", True),
        ("Your report is ready", "3 hrs ago", True),
        ("System update completed", "Yesterday", False),
    ]
    for i, (msg, time, unread) in enumerate(items):
        y = 38 + i * 38
        bg = (230, 238, 255) if unread else (255, 255, 255)
        d.rectangle([6, y, W - 6, y + 30], fill=bg, outline=(200, 208, 230))
        if unread:
            d.ellipse([W - 20, y + 11, W - 10, y + 21], fill=(60, 100, 200))
        _label(d, 14, y + 6, msg, size=12)
        _label(d, 14, y + 20, time, size=10, color=(140, 140, 140))
    return _save(img, "notification.png")


def ui_rating():
    img, d = _canvas()
    _label(d, 10, 8, "Rate this product", size=15, color=(30, 30, 80))
    _label(d, 10, 36, "Wireless Headphones XZ-400", size=13)
    for i in range(5):
        x = 10 + i * 40
        color = (255, 200, 0) if i < 4 else (210, 210, 210)
        d.polygon([
            (x + 20, 70), (x + 26, 88), (x + 40, 88),
            (x + 28, 98), (x + 33, 116), (x + 20, 106),
            (x + 7, 116), (x + 12, 98), (x, 88), (x + 14, 88)
        ], fill=color, outline=(180, 150, 0))
    _label(d, 10, 130, "4 out of 5 stars", size=12, color=(100, 100, 100))
    _button(d, 10, 156, 120, 30, "Submit Rating", bg=(60, 100, 200))
    return _save(img, "rating.png")


# ---------------------------------------------------------------------------
# Visual QA images (25)
# ---------------------------------------------------------------------------

def qa_diagram():
    """Existing H2O molecule."""
    img, d = _canvas((255, 255, 220))
    _label(d, 10, 8, "Identify the molecule:", size=14, color=(30, 30, 80))
    cx, cy = 160, 110
    d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(200, 200, 255), outline=(60, 60, 180), width=3)
    _label(d, cx - 8, cy - 10, "O", size=18, color=(60, 60, 180))
    for dx in [-60, 60]:
        d.ellipse([cx + dx - 14, cy + 20, cx + dx + 14, cy + 48],
                  fill=(255, 200, 200), outline=(180, 60, 60), width=2)
        _label(d, cx + dx - 6, cy + 26, "H", size=14, color=(180, 60, 60))
        d.line([cx + dx // 3, cy + 12, cx + dx - 2, cy + 24], fill=(80, 80, 80), width=2)
    _label(d, 10, 170, 'Formula: H₂O  (water)', size=12, color=(80, 80, 80))
    return _save(img, "diagram.png")


def qa_barchart():
    img, d = _canvas((250, 250, 255))
    _label(d, 10, 6, "Monthly Sales ($k)  — Q1 2024", size=13, color=(30, 30, 80))
    bars = [("Jan", 42, (100, 149, 237)), ("Feb", 68, (100, 200, 100)), ("Mar", 55, (237, 149, 100))]
    max_val = 80
    for i, (month, val, color) in enumerate(bars):
        x = 30 + i * 90
        bar_h = int((val / max_val) * 110)
        y0 = 160 - bar_h
        d.rectangle([x, y0, x + 60, 160], fill=color, outline=(80, 80, 80))
        _label(d, x + 18, y0 - 18, f"${val}k", size=11)
        _label(d, x + 20, 164, month, size=12)
    d.line([20, 30, 20, 165], fill=(80, 80, 80), width=2)
    d.line([20, 165, 295, 165], fill=(80, 80, 80), width=2)
    return _save(img, "barchart.png")


def qa_piechart():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 6, "Budget Allocation 2024", size=14, color=(30, 30, 80))
    cx, cy, r = 120, 115, 80
    slices = [
        ("Engineering", 40, (100, 149, 237), 0),
        ("Marketing",   25, (100, 200, 100), 144),
        ("Operations",  20, (237, 200, 100), 234),
        ("R&D",         15, (200, 100, 200), 306),
    ]
    for name, pct, color, start in slices:
        extent = pct * 3.6
        d.pieslice([cx - r, cy - r, cx + r, cy + r],
                   start=start, end=start + extent, fill=color, outline=(255, 255, 255), width=2)
    for i, (name, pct, color, _) in enumerate(slices):
        y = 28 + i * 36
        d.rectangle([220, y, 236, y + 14], fill=color, outline=(80, 80, 80))
        _label(d, 242, y, f"{name} {pct}%", size=11)
    return _save(img, "piechart.png")


def qa_thermometer():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 8, "What temperature is shown?", size=13, color=(30, 30, 80))
    tx, ty = 160, 100
    d.ellipse([tx - 18, ty + 50, tx + 18, ty + 86], fill=(220, 60, 60), outline=(160, 40, 40), width=3)
    d.rectangle([tx - 8, ty - 60, tx + 8, ty + 68], fill=(255, 255, 255), outline=(160, 160, 180), width=2)
    d.rectangle([tx - 5, ty + 10, tx + 5, ty + 66], fill=(220, 60, 60))
    for i, temp in enumerate([0, 20, 40, 60, 80, 100]):
        y = ty + 8 - i * 12
        d.line([tx + 8, y, tx + 16, y], fill=(80, 80, 80), width=1)
        _label(d, tx + 18, y - 6, f"{temp}°", size=9, color=(60, 60, 60))
    _label(d, 10, 170, "Scale: Celsius", size=11, color=(120, 120, 120))
    return _save(img, "thermometer.png")


def qa_clock():
    img, d = _canvas()
    _label(d, 10, 6, "What time does the clock show?", size=13, color=(30, 30, 80))
    cx, cy, r = 160, 110, 72
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(60, 60, 60), width=4)
    for h in range(12):
        angle = math.radians(h * 30 - 90)
        x1, y1 = cx + (r - 10) * math.cos(angle), cy + (r - 10) * math.sin(angle)
        x2, y2 = cx + (r - 4) * math.cos(angle), cy + (r - 4) * math.sin(angle)
        d.line([x1, y1, x2, y2], fill=(60, 60, 60), width=2)
        _label(d, cx + (r - 22) * math.cos(angle) - 6,
               cy + (r - 22) * math.sin(angle) - 8, str(h or 12), size=11)
    hour_a = math.radians(3 * 30 - 90)  # 3 o'clock
    min_a = math.radians(0 * 6 - 90)    # :00
    d.line([cx, cy, cx + 44 * math.cos(hour_a), cy + 44 * math.sin(hour_a)],
           fill=(30, 30, 30), width=5)
    d.line([cx, cy, cx + 60 * math.cos(min_a), cy + 60 * math.sin(min_a)],
           fill=(60, 60, 60), width=3)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(30, 30, 30))
    return _save(img, "clock.png")


def qa_trafficlight():
    img, d = _canvas((200, 200, 200))
    _label(d, 10, 8, "What does the traffic light show?", size=13)
    d.rectangle([130, 30, 190, 170], fill=(40, 40, 40), outline=(20, 20, 20), width=3)
    lights = [(60, 60, 60), (60, 60, 60), (40, 200, 40)]
    centers = [60, 100, 140]
    for i, (color, cy) in enumerate(zip(lights, centers)):
        d.ellipse([140, cy, 180, cy + 40], fill=color, outline=(20, 20, 20), width=2)
    _label(d, 10, 80, "Red", size=12, color=(180, 60, 60))
    _label(d, 10, 110, "Yellow", size=12, color=(180, 160, 0))
    _label(d, 10, 140, "Green ←", size=12, color=(40, 160, 40))
    return _save(img, "trafficlight.png")


def qa_flag():
    img, d = _canvas()
    _label(d, 10, 8, "What country does this flag represent?", size=12, color=(30, 30, 80))
    # French tricolour
    fw, fh = 180, 110
    fx, fy = 70, 40
    d.rectangle([fx, fy, fx + fw // 3, fy + fh], fill=(0, 35, 149))
    d.rectangle([fx + fw // 3, fy, fx + 2 * fw // 3, fy + fh], fill=(255, 255, 255))
    d.rectangle([fx + 2 * fw // 3, fy, fx + fw, fy + fh], fill=(237, 41, 57))
    d.rectangle([fx, fy, fx + fw, fy + fh], outline=(80, 80, 80), width=2)
    _label(d, 10, 165, "Hint: blue, white, red vertical stripes", size=11, color=(100, 100, 100))
    return _save(img, "flag.png")


def qa_shapes():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 8, "How many triangles are in the image?", size=13, color=(30, 30, 80))
    positions = [(60, 80), (150, 80), (240, 80), (105, 130), (195, 130)]
    for px, py in positions:
        d.polygon([(px, py + 40), (px - 25, py + 80), (px + 25, py + 80)],
                  fill=(100, 160, 240), outline=(40, 80, 180), width=2)
    return _save(img, "shapes.png")


def qa_colors():
    img, d = _canvas()
    _label(d, 10, 8, "What color is the largest circle?", size=13, color=(30, 30, 80))
    d.ellipse([60, 40, 200, 180], fill=(220, 60, 60), outline=(160, 30, 30), width=3)
    d.ellipse([10, 130, 70, 190], fill=(60, 120, 220), outline=(30, 60, 160), width=2)
    d.ellipse([230, 40, 310, 120], fill=(60, 180, 60), outline=(30, 120, 30), width=2)
    return _save(img, "colors.png")


def qa_count():
    img, d = _canvas((255, 250, 240))
    _label(d, 10, 8, "How many stars are visible?", size=13, color=(30, 30, 80))
    positions = [(60, 60), (130, 45), (200, 70), (80, 120), (160, 110),
                 (240, 90), (110, 160), (185, 155)]
    for px, py in positions:
        pts = []
        for k in range(10):
            r = 16 if k % 2 == 0 else 7
            a = math.radians(k * 36 - 90)
            pts.extend([px + r * math.cos(a), py + r * math.sin(a)])
        d.polygon(pts, fill=(255, 210, 0), outline=(200, 160, 0), width=1)
    return _save(img, "count.png")


def qa_equation():
    img, d = _canvas((245, 255, 245))
    _label(d, 10, 8, "What is the result of this equation?", size=13, color=(30, 30, 80))
    _label(d, 40, 60, "3 × 7 + 4 = ?", size=32, color=(30, 30, 30))
    _label(d, 40, 120, "Show your reasoning:", size=12, color=(100, 100, 100))
    _label(d, 40, 140, "3 × 7 = 21   →   21 + 4 = 25", size=13, color=(60, 100, 60))
    return _save(img, "equation.png")


def qa_direction():
    img, d = _canvas()
    _label(d, 10, 8, "Which direction is the arrow pointing?", size=13, color=(30, 30, 80))
    cx, cy = 160, 110
    d.polygon([(cx, 50), (cx + 30, 90), (cx + 12, 90), (cx + 12, 170),
               (cx - 12, 170), (cx - 12, 90), (cx - 30, 90)],
              fill=(60, 100, 200), outline=(30, 60, 140), width=2)
    return _save(img, "direction.png")


def qa_co2():
    img, d = _canvas((255, 255, 220))
    _label(d, 10, 8, "Name this molecule:", size=14, color=(30, 30, 80))
    cx, cy = 160, 100
    for dx in [-80, 80]:
        d.ellipse([cx + dx - 18, cy - 18, cx + dx + 18, cy + 18],
                  fill=(255, 160, 160), outline=(180, 60, 60), width=3)
        _label(d, cx + dx - 8, cy - 9, "O", size=16, color=(160, 40, 40))
        d.line([cx + dx // 4, cy, cx + dx - 20, cy], fill=(80, 80, 80), width=2)
    d.ellipse([cx - 18, cy - 18, cx + 18, cy + 18],
              fill=(200, 200, 200), outline=(80, 80, 80), width=3)
    _label(d, cx - 8, cy - 9, "C", size=16, color=(60, 60, 60))
    _label(d, 10, 155, "Linear molecule  |  3 atoms", size=11, color=(100, 100, 100))
    return _save(img, "co2.png")


def qa_element():
    img, d = _canvas((240, 248, 240))
    _label(d, 10, 8, "What chemical element is shown?", size=13, color=(30, 30, 80))
    d.rectangle([70, 40, 250, 160], fill=(255, 215, 0), outline=(180, 150, 0), width=4)
    _label(d, 130, 50, "Au", size=40, color=(120, 90, 0))
    _label(d, 108, 110, "Gold", size=16, color=(100, 70, 0))
    _label(d, 80, 130, "Atomic Number: 79", size=12, color=(120, 100, 0))
    return _save(img, "element.png")


def qa_map():
    img, d = _canvas((180, 210, 255))
    _label(d, 10, 8, "Which continent is highlighted?", size=13, color=(30, 30, 80))
    # Africa silhouette (rough)
    pts = [
        150, 30, 190, 32, 210, 50, 215, 80, 200, 110, 210, 140,
        195, 165, 170, 175, 145, 165, 125, 150, 110, 120, 115, 90,
        130, 60, 140, 42
    ]
    d.polygon(pts, fill=(100, 200, 100), outline=(40, 140, 40), width=3)
    _label(d, 148, 98, "?", size=22, color=(30, 80, 30))
    _label(d, 10, 170, "Largest continent by area on this view", size=11, color=(60, 80, 120))
    return _save(img, "map.png")


def qa_chart_highest():
    img, d = _canvas((250, 255, 250))
    _label(d, 10, 6, "Which city has the highest temperature?", size=12, color=(30, 30, 80))
    data = [("NYC", 22, (100, 149, 237)), ("LA", 31, (237, 149, 100)),
            ("CHI", 18, (149, 200, 149)), ("HOU", 28, (200, 149, 237))]
    max_val = 40
    for i, (city, val, color) in enumerate(data):
        x = 20 + i * 72
        bar_h = int((val / max_val) * 110)
        y0 = 155 - bar_h
        d.rectangle([x, y0, x + 52, 155], fill=color, outline=(80, 80, 80))
        _label(d, x + 14, y0 - 16, f"{val}°C", size=10)
        _label(d, x + 14, 158, city, size=11)
    d.line([12, 28, 12, 158], fill=(80, 80, 80), width=2)
    d.line([12, 158, 302, 158], fill=(80, 80, 80), width=2)
    return _save(img, "chart_highest.png")


def qa_largest_slice():
    img, d = _canvas((255, 248, 240))
    _label(d, 10, 6, "Which department has the largest budget?", size=11, color=(30, 30, 80))
    cx, cy, r = 115, 115, 80
    slices = [
        ("Engineering", 45, (100, 149, 237), 0),
        ("Marketing",   20, (237, 149, 100), 162),
        ("HR",          15, (149, 200, 100), 234),
        ("Legal",       20, (200, 149, 237), 288),
    ]
    for name, pct, color, start in slices:
        extent = pct * 3.6
        d.pieslice([cx - r, cy - r, cx + r, cy + r],
                   start=start, end=start + extent, fill=color, outline=(255, 255, 255), width=2)
    for i, (name, pct, color, _) in enumerate(slices):
        y = 28 + i * 36
        d.rectangle([216, y, 232, y + 14], fill=color, outline=(80, 80, 80))
        _label(d, 238, y, f"{name} {pct}%", size=11)
    return _save(img, "largest_slice.png")


def qa_logo_color():
    img, d = _canvas()
    _label(d, 10, 8, "What color is the company logo?", size=13, color=(30, 30, 80))
    d.ellipse([95, 50, 225, 150], fill=(220, 30, 30), outline=(160, 20, 20), width=4)
    _label(d, 120, 78, "ACME", size=28, color=(255, 255, 255))
    _label(d, 75, 162, "Official brand color shown above", size=12, color=(100, 100, 100))
    return _save(img, "logo_color.png")


def qa_next_number():
    img, d = _canvas((245, 245, 255))
    _label(d, 10, 8, "What is the next number in the sequence?", size=12, color=(30, 30, 80))
    nums = [2, 4, 8, 16, 32, "?"]
    for i, n in enumerate(nums):
        x = 20 + i * 48
        color = (60, 100, 200) if str(n) != "?" else (200, 60, 60)
        d.rounded_rectangle([x, 60, x + 38, 100], radius=6,
                             fill=(220, 230, 255) if str(n) != "?" else (255, 220, 220),
                             outline=color, width=2)
        _label(d, x + 8, 70, str(n), size=16, color=color)
    _label(d, 10, 120, "Pattern: each number is doubled", size=12, color=(100, 100, 100))
    return _save(img, "sequence.png")


def qa_planet():
    img, d = _canvas((20, 20, 60))
    _label(d, 10, 8, "Which planet is shown?", size=13, color=(200, 200, 255))
    cx, cy = 160, 115
    d.ellipse([cx - 60, cy - 60, cx + 60, cy + 60],
              fill=(210, 180, 120), outline=(180, 150, 90), width=3)
    d.ellipse([cx - 100, cy - 22, cx + 100, cy + 22],
              outline=(200, 170, 110), width=6, fill=None)
    for star in [(40, 40), (280, 60), (30, 160), (290, 150), (150, 170), (60, 100)]:
        d.ellipse([star[0] - 2, star[1] - 2, star[0] + 2, star[1] + 2], fill=(255, 255, 255))
    _label(d, 10, 170, "Famous for its ring system", size=11, color=(160, 160, 200))
    return _save(img, "planet.png")


def qa_weather():
    img, d = _canvas((135, 185, 235))
    _label(d, 10, 8, "What weather condition is shown?", size=13, color=(20, 20, 80))
    d.ellipse([80, 50, 180, 130], fill=(255, 255, 200), outline=(240, 220, 0), width=4)
    for angle in range(0, 360, 45):
        a = math.radians(angle)
        x1 = 130 + 56 * math.cos(a)
        y1 = 90 + 56 * math.sin(a)
        x2 = 130 + 72 * math.cos(a)
        y2 = 90 + 72 * math.sin(a)
        d.line([x1, y1, x2, y2], fill=(240, 220, 0), width=4)
    _label(d, 40, 155, "Sky condition: clear and bright", size=12, color=(20, 40, 80))
    return _save(img, "weather.png")


def qa_fruit():
    img, d = _canvas((240, 255, 240))
    _label(d, 10, 8, "How many apples are shown?", size=13, color=(30, 60, 30))
    positions = [(70, 80), (160, 70), (250, 85), (115, 130), (205, 125)]
    for px, py in positions:
        d.ellipse([px - 28, py - 28, px + 28, py + 28],
                  fill=(220, 40, 40), outline=(160, 20, 20), width=2)
        d.line([px, py - 28, px, py - 42], fill=(60, 120, 30), width=3)
        d.arc([px - 4, py - 44, px + 10, py - 32], start=200, end=340,
              fill=(60, 120, 30), width=2)
    return _save(img, "fruit.png")


# ---------------------------------------------------------------------------
# New UI images (15)
# ---------------------------------------------------------------------------

def ui_modal():
    img, d = _canvas()
    _label(d, 10, 6, "Confirmation Dialog", size=14, color=(30, 30, 80))
    d.rectangle([20, 28, W - 20, H - 8], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    d.rectangle([20, 28, W - 20, 54], fill=(220, 50, 50))
    _label(d, 28, 34, "Confirm Delete", size=14, color=(255, 255, 255))
    _label(d, 30, 62, "Are you sure you want to delete this", size=12, color=(50, 50, 50))
    _label(d, 30, 80, "item? This action cannot be undone.", size=12, color=(50, 50, 50))
    _button(d, 32, 136, 90, 30, "Confirm", bg=(200, 50, 50))
    _button(d, 136, 136, 90, 30, "Cancel", bg=(140, 140, 140))
    return _save(img, "modal.png")


def ui_form_contact():
    img, d = _canvas((230, 245, 255))
    _label(d, 10, 4, "Contact Us", size=16, color=(30, 30, 80))
    _label(d, 10, 32, "Name")
    _input_field(d, 10, 48, 290, 24)
    _label(d, 10, 78, "Email")
    _input_field(d, 10, 94, 290, 24, "your@email.com")
    _label(d, 10, 124, "Message")
    d.rectangle([10, 140, 300, 168], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    _label(d, 18, 146, "Type your message here…", size=11, color=(160, 160, 160))
    _button(d, 168, 176, 130, 22, "Send Message", bg=(60, 100, 200))
    return _save(img, "form_contact.png")


def ui_table():
    img, d = _canvas()
    _label(d, 10, 2, "User Directory", size=14, color=(30, 30, 80))
    d.rectangle([6, 20, W - 6, 40], fill=(60, 100, 200))
    for ci, h in enumerate(["Name", "Age", "Role"]):
        _label(d, 14 + ci * 100, 26, h, size=12, color=(255, 255, 255))
    rows = [("Alice", "28", "Admin"), ("Bob", "34", "Editor"), ("Carol", "25", "Viewer")]
    for ri, (name, age, role) in enumerate(rows):
        y = 42 + ri * 34
        bg = (245, 247, 255) if ri % 2 == 0 else (255, 255, 255)
        d.rectangle([6, y, W - 6, y + 32], fill=bg, outline=(210, 210, 220))
        for ci, val in enumerate([name, age, role]):
            _label(d, 14 + ci * 100, y + 9, val, size=12)
    _label(d, 10, 148, "Page 1 of 3", size=11, color=(120, 120, 120))
    _button(d, W - 86, 144, 80, 24, "Next ▶", bg=(60, 100, 200))
    return _save(img, "table.png")


def ui_tabs():
    img, d = _canvas()
    tab_names = ["Description", "Reviews", "Specs", "Shipping"]
    for i, name in enumerate(tab_names):
        x = 6 + i * 77
        active = (i == 0)
        bg = (255, 255, 255) if active else (225, 225, 230)
        color = (30, 60, 180) if active else (80, 80, 80)
        d.rectangle([x, 8, x + 74, 38], fill=bg, outline=(180, 180, 200))
        if active:
            d.line([x, 8, x + 74, 8], fill=(60, 100, 200), width=3)
        _label(d, x + 4, 17, name, size=11, color=color)
    d.rectangle([6, 38, W - 6, H - 6], fill=(255, 255, 255), outline=(180, 180, 200))
    _label(d, 16, 50, "Overview of the product's main features", size=12, color=(60, 60, 60))
    _label(d, 16, 68, "and benefits. Compatible with all major", size=12, color=(60, 60, 60))
    _label(d, 16, 86, "platforms. Ships in 2-3 business days.", size=12, color=(60, 60, 60))
    return _save(img, "tabs.png")


def ui_breadcrumb():
    img, d = _canvas()
    _label(d, 10, 6, "You are here:", size=12, color=(100, 100, 100))
    items = ["Home", "Electronics", "Laptops", "MacBook Pro"]
    x = 10
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        color = (50, 50, 50) if is_last else (60, 100, 200)
        _label(d, x, 30, item, size=13, color=color)
        bbox = d.textbbox((0, 0), item, font=_font(13))
        tw = bbox[2] - bbox[0]
        x += tw + 4
        if not is_last:
            _label(d, x, 30, " ›", size=13, color=(150, 150, 150))
            x += 14
    _label(d, 10, 64, "MacBook Pro 14-inch (2024)", size=15, color=(30, 30, 30))
    _label(d, 10, 90, "Apple M3 Pro chip, 18 GB memory, 512 GB SSD", size=11, color=(80, 80, 80))
    return _save(img, "breadcrumb.png")


def ui_accordion():
    img, d = _canvas()
    _label(d, 10, 2, "Frequently Asked Questions", size=13, color=(30, 30, 80))
    faqs = [
        ("What is your return policy?", True, "We offer 30-day returns on all unused items."),
        ("How long does shipping take?", False, None),
        ("Do you offer warranties?", False, None),
    ]
    y = 24
    for q, expanded, ans in faqs:
        h = 52 if expanded else 32
        bg = (235, 242, 255) if expanded else (252, 252, 252)
        d.rectangle([6, y, W - 6, y + h], fill=bg, outline=(200, 200, 220))
        arrow = "▼" if expanded else "▶"
        _label(d, 14, y + 8, f"{arrow}  {q}", size=12, color=(30, 30, 80))
        if expanded and ans:
            _label(d, 28, y + 30, ans, size=11, color=(60, 60, 60))
        y += h + 5
    return _save(img, "accordion.png")


def ui_step_wizard():
    img, d = _canvas((240, 245, 255))
    _label(d, 10, 2, "Create Account — Step 2 of 4", size=13, color=(30, 30, 80))
    steps = [("1", "Account", False, True), ("2", "Address", True, False),
             ("3", "Payment", False, False), ("4", "Review", False, False)]
    for i, (num, name, active, done) in enumerate(steps):
        x = 8 + i * 76
        bg = (40, 160, 40) if done else (60, 100, 200) if active else (200, 200, 210)
        fg = (255, 255, 255) if (done or active) else (80, 80, 80)
        _button(d, x, 22, 72, 26, f"{num}. {name}", bg=bg, fg=fg)
    _label(d, 14, 62, "Shipping Address", size=14, color=(30, 30, 80))
    _label(d, 14, 86, "Street Address")
    _input_field(d, 14, 102, 280, 26)
    _button(d, 10, 158, 72, 28, "◀ Back", bg=(150, 150, 150))
    _button(d, 228, 158, 82, 28, "Next ▶", bg=(60, 100, 200))
    return _save(img, "step_wizard.png")


def ui_lang_dropdown():
    img, d = _canvas()
    _label(d, 10, 8, "Select Language", size=14, color=(30, 30, 80))
    d.rectangle([10, 34, 210, 60], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 42, "English", size=13)
    d.polygon([(192, 44), (204, 44), (198, 54)], fill=(80, 80, 80))
    langs = [("English", True), ("Spanish", False), ("French", False), ("German", False)]
    for i, (lang, selected) in enumerate(langs):
        y = 62 + i * 28
        bg = (220, 230, 255) if selected else (255, 255, 255)
        d.rectangle([10, y, 210, y + 26], fill=bg, outline=(200, 200, 210))
        mark = "✓ " if selected else "   "
        _label(d, 18, y + 6, f"{mark}{lang}", size=12,
               color=(30, 60, 180) if selected else (60, 60, 60))
    return _save(img, "dropdown_lang.png")


def ui_sidebar():
    img, d = _canvas()
    d.rectangle([0, 0, 96, H], fill=(28, 38, 68))
    _label(d, 8, 8, "MyApp", size=13, color=(180, 200, 255))
    d.line([0, 30, 96, 30], fill=(50, 64, 100))
    items = [("Dashboard", True), ("Analytics", False), ("Reports", False), ("Settings", False)]
    for i, (item, active) in enumerate(items):
        y = 38 + i * 36
        if active:
            d.rectangle([0, y, 96, y + 28], fill=(60, 100, 200))
        _label(d, 8, y + 7, item, size=11,
               color=(255, 255, 255) if active else (160, 175, 210))
    d.rectangle([96, 0, W, H], fill=(248, 249, 252))
    _label(d, 106, 16, "Dashboard", size=15, color=(30, 30, 80))
    _label(d, 106, 42, "Welcome back, Alex!", size=12, color=(80, 80, 80))
    _label(d, 106, 62, "Here's your overview for today.", size=11, color=(120, 120, 120))
    return _save(img, "sidebar.png")


def ui_banner():
    img, d = _canvas((255, 198, 50))
    d.text((W - 26, 4), "✕", fill=(160, 100, 0), font=_font(16))
    _label(d, 10, 16, "Summer Sale", size=24, color=(110, 40, 0))
    _label(d, 10, 50, "Up to 50% off select items!", size=14, color=(100, 40, 0))
    _button(d, 10, 82, 118, 32, "Shop Now →", bg=(190, 70, 10), fg=(255, 255, 255))
    _label(d, 10, 128, "Limited time only. Ends June 30.", size=12, color=(120, 60, 0))
    return _save(img, "banner.png")


def ui_toast():
    img, d = _canvas((240, 242, 245))
    _label(d, 10, 8, "Application", size=13, color=(80, 80, 80))
    d.rounded_rectangle([14, 50, W - 14, 140], radius=8,
                        fill=(36, 155, 80), outline=(20, 110, 55), width=2)
    _label(d, 32, 72, "✓", size=22, color=(255, 255, 255))
    _label(d, 64, 68, "Profile updated", size=14, color=(255, 255, 255))
    _label(d, 64, 90, "successfully!", size=14, color=(255, 255, 255))
    d.text((W - 32, 56), "✕", fill=(180, 230, 200), font=_font(16))
    return _save(img, "toast.png")


def ui_date_picker():
    img, d = _canvas()
    d.rectangle([6, 4, W - 6, 28], fill=(60, 100, 200))
    _label(d, 70, 9, "March  2025", size=13, color=(255, 255, 255))
    _label(d, 14, 10, "◀", size=13, color=(200, 220, 255))
    _label(d, W - 26, 10, "▶", size=13, color=(200, 220, 255))
    headers = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
    cw = 44
    for i, h in enumerate(headers):
        _label(d, 10 + i * cw, 32, h, size=10, color=(100, 100, 140))
    # March 2025 starts on Saturday (index 6)
    day = 1
    for pos in range(6, 6 + 31):
        col = pos % 7
        row = pos // 7
        x = 10 + col * cw
        y = 50 + row * 26
        if day == 15:
            d.ellipse([x - 1, y - 2, x + 22, y + 18], fill=(60, 100, 200))
            _label(d, x + 3, y, str(day), size=11, color=(255, 255, 255))
        elif day == 22:
            d.ellipse([x - 1, y - 2, x + 22, y + 18], fill=(200, 215, 245))
            _label(d, x + 3, y, str(day), size=11, color=(30, 60, 180))
        else:
            _label(d, x + 3, y, str(day), size=11, color=(50, 50, 50))
        day += 1
    return _save(img, "date_picker.png")


def ui_slider():
    img, d = _canvas()
    _label(d, 10, 14, "Price Range Filter", size=14, color=(30, 30, 80))
    _label(d, 10, 44, "$0", size=12, color=(80, 80, 80))
    _label(d, W - 40, 44, "$500", size=12, color=(80, 80, 80))
    rx, rw = 30, W - 60
    d.rectangle([rx, 72, rx + rw, 80], fill=(200, 200, 210), outline=(180, 180, 195))
    fill_x = rx + int(150 / 500 * rw)
    d.rectangle([rx, 72, fill_x, 80], fill=(60, 100, 200))
    d.ellipse([fill_x - 10, 66, fill_x + 10, 86], fill=(60, 100, 200), outline=(30, 60, 150), width=2)
    _label(d, fill_x - 16, 90, "$150", size=12, color=(60, 100, 200))
    _label(d, 10, 126, "Selected range: $0 – $150", size=13, color=(50, 50, 50))
    return _save(img, "slider.png")


def ui_progress_bar():
    img, d = _canvas()
    _label(d, 10, 14, "Uploading document.pdf...", size=13, color=(30, 30, 80))
    _label(d, 10, 44, "65% complete", size=12, color=(60, 100, 60))
    d.rounded_rectangle([10, 66, W - 10, 92], radius=6, fill=(210, 215, 225), outline=(185, 190, 205))
    fill_w = int(0.65 * (W - 24))
    d.rounded_rectangle([12, 68, 12 + fill_w, 90], radius=4, fill=(50, 160, 60))
    _label(d, 10, 102, "Size: 4.3 MB / 6.6 MB", size=11, color=(100, 100, 100))
    _label(d, 10, 118, "Estimated time remaining: 12 seconds", size=11, color=(100, 100, 100))
    _button(d, W - 88, 148, 80, 28, "Cancel", bg=(180, 60, 60))
    return _save(img, "progress_bar.png")


def ui_tooltip():
    img, d = _canvas()
    _label(d, 10, 14, "API Settings", size=14, color=(30, 30, 80))
    _label(d, 10, 46, "API Key")
    _input_field(d, 10, 62, 238, 28, "sk-xxxx-xxxx-xxxx")
    d.ellipse([256, 64, 278, 86], fill=(90, 130, 210), outline=(55, 95, 175), width=2)
    _label(d, 264, 68, "?", size=14, color=(255, 255, 255))
    d.polygon([(264, 100), (274, 100), (269, 90)], fill=(28, 28, 28))
    d.rounded_rectangle([60, 100, W - 4, 158], radius=6, fill=(28, 28, 28), outline=(20, 20, 20))
    _label(d, 68, 108, "Your API key can be found", size=11, color=(240, 240, 240))
    _label(d, 68, 126, "in account settings.", size=11, color=(240, 240, 240))
    return _save(img, "tooltip.png")


# ---------------------------------------------------------------------------
# New Visual QA images (15)
# ---------------------------------------------------------------------------

def qa_line_chart():
    img, d = _canvas((248, 248, 255))
    _label(d, 10, 4, "Monthly Revenue 2024 ($k)", size=12, color=(30, 30, 80))
    data = [("Jan", 20), ("Feb", 35), ("Mar", 28), ("Apr", 45), ("May", 40), ("Jun", 55)]
    max_v = 60
    xs, ys = [], []
    for i, (month, val) in enumerate(data):
        x = 28 + i * 46
        y = 158 - int(val / max_v * 118)
        xs.append(x); ys.append(y)
        _label(d, x - 8, 162, month, size=9, color=(80, 80, 80))
        _label(d, x - 8, y - 14, str(val), size=9, color=(60, 60, 160))
    for i in range(len(xs) - 1):
        d.line([xs[i], ys[i], xs[i + 1], ys[i + 1]], fill=(60, 100, 200), width=2)
    for x, y in zip(xs, ys):
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 100, 200), outline=(30, 60, 140))
    d.line([20, 30, 20, 160], fill=(80, 80, 80), width=2)
    d.line([20, 160, W - 10, 160], fill=(80, 80, 80), width=2)
    return _save(img, "line_chart.png")


def qa_histogram():
    img, d = _canvas((255, 248, 240))
    _label(d, 10, 4, "Score Distribution", size=14, color=(30, 30, 80))
    bars = [("0-10", 3), ("10-20", 7), ("20-30", 12), ("30-40", 8), ("40-50", 4)]
    max_v = 14
    for i, (rng, cnt) in enumerate(bars):
        x = 18 + i * 56
        bar_h = int(cnt / max_v * 112)
        y0 = 160 - bar_h
        d.rectangle([x, y0, x + 50, 160], fill=(100, 149, 237), outline=(60, 100, 200))
        _label(d, x + 16, y0 - 14, str(cnt), size=10, color=(60, 60, 60))
        _label(d, x + 4, 163, rng, size=8, color=(80, 80, 80))
    d.line([12, 28, 12, 162], fill=(80, 80, 80), width=2)
    d.line([12, 162, W - 8, 162], fill=(80, 80, 80), width=2)
    return _save(img, "histogram.png")


def qa_venn():
    img, d = _canvas((248, 255, 248))
    _label(d, 10, 4, "Skill Set Venn Diagram", size=13, color=(30, 30, 80))
    d.ellipse([18, 38, 188, 158], fill=(210, 225, 255), outline=(60, 100, 200), width=2)
    d.ellipse([132, 38, 302, 158], fill=(255, 225, 210), outline=(200, 100, 60), width=2)
    _label(d, 24, 168, "Machine Learning", size=10, color=(40, 80, 180))
    _label(d, 200, 168, "Data Analysis", size=10, color=(180, 80, 40))
    _label(d, 28, 78, "Python", size=10, color=(40, 80, 180))
    _label(d, 36, 96, "R", size=10, color=(40, 80, 180))
    _label(d, 148, 74, "Data", size=10, color=(50, 50, 50))
    _label(d, 134, 92, "Statistics", size=10, color=(50, 50, 50))
    _label(d, 234, 78, "SQL", size=10, color=(180, 80, 40))
    _label(d, 228, 96, "Excel", size=10, color=(180, 80, 40))
    return _save(img, "venn.png")


def qa_periodic_fe():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 6, "Identify this periodic table element:", size=12, color=(30, 30, 80))
    d.rectangle([68, 36, 252, 168], fill=(255, 140, 55), outline=(180, 95, 25), width=4)
    _label(d, 86, 42, "26", size=14, color=(110, 55, 0))
    _label(d, 116, 52, "Fe", size=46, color=(80, 30, 0))
    _label(d, 110, 118, "Iron", size=18, color=(100, 50, 0))
    _label(d, 76, 142, "Atomic Mass: 55.85", size=12, color=(110, 55, 0))
    return _save(img, "periodic_fe.png")


def qa_ruler():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 6, "What length is the blue object?", size=13, color=(30, 30, 80))
    rx, ry, rw = 20, 90, W - 40
    obj_w = int(7.5 / 10 * rw)
    d.rectangle([rx, 54, rx + obj_w, 88], fill=(180, 210, 255), outline=(80, 120, 200), width=2)
    _label(d, rx + obj_w // 2 - 18, 63, "Object", size=11, color=(40, 80, 180))
    d.rectangle([rx, ry, rx + rw, ry + 28], fill=(255, 238, 170), outline=(160, 120, 40), width=2)
    for i in range(11):
        x = rx + int(i / 10 * rw)
        tick_h = 18 if i % 5 == 0 else 10
        d.line([x, ry, x, ry + tick_h], fill=(80, 50, 10), width=1)
        if i % 5 == 0:
            _label(d, x - 4, ry + 18, str(i), size=9, color=(60, 40, 0))
    _label(d, rx + rw - 14, ry + 18, "cm", size=9, color=(60, 40, 0))
    _label(d, 10, 148, "Measured length: 7.5 cm", size=12, color=(60, 60, 60))
    return _save(img, "ruler.png")


def qa_protractor():
    img, d = _canvas((250, 255, 255))
    _label(d, 10, 4, "What angle is shown on the protractor?", size=12, color=(30, 30, 80))
    cx, cy, r = 160, 158, 104
    d.pieslice([cx - r, cy - r, cx + r, cy + r],
               start=180, end=360, fill=(228, 235, 255), outline=(100, 130, 220), width=2)
    d.line([cx - r, cy, cx + r, cy], fill=(80, 80, 80), width=2)
    for deg in range(0, 181, 15):
        a = math.radians(180 - deg)
        x1 = cx + (r - 6) * math.cos(a); y1 = cy - (r - 6) * math.sin(a)
        x2 = cx + r * math.cos(a);       y2 = cy - r * math.sin(a)
        d.line([x1, y1, x2, y2], fill=(80, 80, 80), width=1)
        if deg % 45 == 0:
            xl = cx + (r - 20) * math.cos(a); yl = cy - (r - 20) * math.sin(a)
            _label(d, xl - 7, yl - 7, str(deg), size=9, color=(60, 60, 60))
    a65 = math.radians(180 - 65)
    d.line([cx, cy, int(cx + r * math.cos(a65)), int(cy - r * math.sin(a65))],
           fill=(200, 50, 50), width=3)
    _label(d, cx + 28, cy - 52, "65°", size=14, color=(200, 50, 50))
    _label(d, 10, 172, "Type: acute (< 90°)", size=11, color=(80, 80, 80))
    return _save(img, "protractor.png")


def qa_compass():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 4, "Which direction does the needle point?", size=12, color=(30, 30, 80))
    cx, cy, r = 160, 108, 72
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 240), outline=(80, 80, 80), width=3)
    for lbl, ax, ay in [("N", cx - 6, cy - r + 2), ("S", cx - 6, cy + r - 16),
                        ("E", cx + r - 14, cy - 7), ("W", cx - r + 4, cy - 7)]:
        _label(d, ax, ay, lbl, size=13, color=(60, 60, 60))
    # NE needle (45° from north = upper-right)
    offset = int(r * 0.7 * math.sin(math.radians(45)))
    nx, ny = cx + offset, cy - offset
    sx, sy = cx - int(offset * 0.6), cy + int(offset * 0.6)
    d.line([cx, cy, nx, ny], fill=(210, 50, 50), width=5)
    d.line([cx, cy, sx, sy], fill=(80, 80, 80), width=4)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(30, 30, 30))
    _label(d, 10, 172, "Red tip → North-East", size=11, color=(160, 60, 60))
    return _save(img, "compass.png")


def qa_calendar():
    img, d = _canvas()
    d.rectangle([0, 0, W, 24], fill=(60, 100, 200))
    _label(d, 80, 5, "November  2024", size=14, color=(255, 255, 255))
    headers = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
    cw = 44
    for i, h in enumerate(headers):
        _label(d, 10 + i * cw, 28, h, size=10, color=(80, 80, 120))
    # November 2024 starts on Friday (index 5)
    day = 1
    for pos in range(5, 5 + 30):
        col = pos % 7
        row = pos // 7
        x = 10 + col * cw
        y = 46 + row * 26
        _label(d, x + 6, y, str(day), size=11, color=(50, 50, 50))
        day += 1
    return _save(img, "calendar.png")


def qa_speedometer():
    img, d = _canvas((28, 28, 48))
    _label(d, 10, 4, "What speed is shown on the gauge?", size=12, color=(190, 195, 255))
    cx, cy, r = 160, 120, 86
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 38), outline=(80, 80, 105), width=4)
    for spd in range(0, 121, 20):
        frac = spd / 120
        a = math.radians(210 - frac * 240)
        x1 = cx + (r - 8) * math.cos(a); y1 = cy - (r - 8) * math.sin(a)
        x2 = cx + r * math.cos(a);       y2 = cy - r * math.sin(a)
        d.line([x1, y1, x2, y2], fill=(150, 150, 160), width=2)
        xl = cx + (r - 22) * math.cos(a); yl = cy - (r - 22) * math.sin(a)
        _label(d, xl - 8, yl - 7, str(spd), size=9, color=(150, 150, 160))
    na = math.radians(210 - (75 / 120) * 240)
    d.line([cx, cy, int(cx + (r - 12) * math.cos(na)), int(cy - (r - 12) * math.sin(na))],
           fill=(220, 55, 55), width=3)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(70, 70, 90))
    _label(d, cx - 16, cy + 18, "75", size=20, color=(230, 230, 240))
    _label(d, cx - 12, cy + 42, "mph", size=12, color=(150, 150, 170))
    _label(d, 10, 170, "Max: 120 mph", size=11, color=(120, 120, 140))
    return _save(img, "speedometer.png")


def qa_battery():
    img, d = _canvas()
    _label(d, 10, 8, "What is the battery status?", size=13, color=(30, 30, 80))
    bx, by, bw, bh = 80, 48, 160, 80
    d.rectangle([bx, by, bx + bw, by + bh], fill=(235, 235, 235), outline=(60, 60, 60), width=3)
    d.rectangle([bx + bw, by + bh // 3, bx + bw + 12, by + 2 * bh // 3], fill=(60, 60, 60))
    fill_w = int(0.75 * (bw - 6))
    d.rectangle([bx + 3, by + 3, bx + 3 + fill_w, by + bh - 3], fill=(38, 195, 75))
    bltx = bx + bw // 2
    d.polygon([(bltx - 8, by + 12), (bltx + 6, by + 12), (bltx - 2, by + bh // 2 - 2),
               (bltx + 10, by + bh // 2 - 2), (bltx - 6, by + bh - 14),
               (bltx + 2, by + bh // 2 + 2), (bltx - 10, by + bh // 2 + 2)],
              fill=(255, 238, 50))
    _label(d, bx + bw // 2 - 18, by + bh + 8, "75%", size=14, color=(38, 155, 60))
    _label(d, bx + bw // 2 - 36, by + bh + 28, "⚡ Charging", size=12, color=(70, 70, 70))
    return _save(img, "battery.png")


def qa_signal_bars():
    img, d = _canvas()
    _label(d, 10, 8, "How many signal bars are filled?", size=13, color=(30, 30, 80))
    heights = [18, 34, 50, 66, 82]
    filled = 4
    bx = 80
    for i, h in enumerate(heights):
        x = bx + i * 34
        yb = 170
        color = (38, 160, 75) if i < filled else (195, 195, 200)
        outline = (20, 120, 50) if i < filled else (160, 160, 165)
        d.rectangle([x, yb - h, x + 22, yb], fill=color, outline=outline)
    _label(d, 76, 178, "4 out of 5 bars filled", size=11, color=(38, 140, 60))
    _label(d, 10, 60, "Signal Strength", size=13, color=(50, 50, 50))
    return _save(img, "signal_bars.png")


def qa_math_fraction():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 6, "What fraction is shown? Simplify if possible.", size=12, color=(30, 30, 80))
    cx = W // 2
    _label(d, cx - 18, 44, "3", size=54, color=(30, 30, 120))
    d.line([cx - 44, 112, cx + 44, 112], fill=(30, 30, 120), width=5)
    _label(d, cx - 18, 120, "4", size=54, color=(30, 30, 120))
    _label(d, 10, 180, "Already in simplest form.  0.75 as decimal.", size=11, color=(100, 100, 100))
    return _save(img, "math_fraction.png")


def qa_dna():
    img, d = _canvas((230, 248, 255))
    _label(d, 10, 4, "What biological molecule is illustrated?", size=12, color=(30, 30, 80))
    for x in range(20, W - 10, 4):
        t = (x - 20) / 28
        y1 = int(105 + 52 * math.sin(t))
        y2 = int(105 - 52 * math.sin(t))
        d.ellipse([x - 3, y1 - 3, x + 3, y1 + 3], fill=(60, 100, 200))
        d.ellipse([x - 3, y2 - 3, x + 3, y2 + 3], fill=(200, 75, 55))
        if int(t * 9) % 5 == 0:
            d.line([x, y1, x, y2], fill=(80, 170, 100), width=2)
    _label(d, 10, 166, "DNA  (Deoxyribonucleic Acid)  — 2 strands", size=11, color=(55, 55, 55))
    return _save(img, "dna.png")


def qa_balance():
    img, d = _canvas((250, 255, 250))
    _label(d, 10, 4, "Which side of the balance scale is heavier?", size=12, color=(30, 30, 80))
    cx = W // 2
    d.line([cx, 40, cx, 148], fill=(100, 80, 40), width=4)
    d.ellipse([cx - 6, 36, cx + 6, 48], fill=(80, 60, 20))
    # Tilted beam: left lower (heavier)
    d.line([cx - 100, 76, cx + 100, 54], fill=(100, 80, 40), width=4)
    # Left pan (lower, 5 kg)
    lx, ly = cx - 100, 76
    d.line([lx, ly, lx, ly + 38], fill=(120, 100, 60), width=2)
    d.ellipse([lx - 32, ly + 36, lx + 32, ly + 54], fill=(175, 158, 98), outline=(120, 100, 60), width=2)
    _label(d, lx - 16, ly + 42, "5 kg", size=12, color=(60, 40, 20))
    # Right pan (higher, 3 kg)
    rx, ry = cx + 100, 54
    d.line([rx, ry, rx, ry + 38], fill=(120, 100, 60), width=2)
    d.ellipse([rx - 32, ry + 36, rx + 32, ry + 54], fill=(175, 158, 98), outline=(120, 100, 60), width=2)
    _label(d, rx - 16, ry + 42, "3 kg", size=12, color=(60, 40, 20))
    _label(d, 10, 166, "Left (5 kg) is heavier → left side tilts down", size=11, color=(70, 70, 70))
    return _save(img, "balance.png")


def qa_scatter():
    img, d = _canvas((250, 250, 255))
    _label(d, 10, 2, "Height (cm) vs Weight (kg)", size=13, color=(30, 30, 80))
    d.line([30, 22, 30, 162], fill=(80, 80, 80), width=2)
    d.line([30, 162, W - 8, 162], fill=(80, 80, 80), width=2)
    _label(d, 2, 86, "H", size=10, color=(80, 80, 80))
    _label(d, 2, 98, "t", size=10, color=(80, 80, 80))
    _label(d, 110, 168, "Weight (kg)", size=10, color=(80, 80, 80))
    cluster = [(60, 128), (76, 116), (90, 108), (104, 100), (116, 92),
               (128, 84), (140, 76), (152, 70), (164, 62), (176, 55), (188, 48)]
    for px, py in cluster:
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(60, 100, 200), outline=(30, 60, 140))
    outliers = [(50, 58), (200, 138)]
    for px, py in outliers:
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(200, 55, 55), outline=(140, 30, 30))
    _label(d, W - 58, 150, "r > 0", size=11, color=(38, 155, 60))
    _label(d, 10, 180, "Blue = cluster (11 pts)  Red = outliers (2 pts)", size=9, color=(80, 80, 80))
    return _save(img, "scatter.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    # UI / web (original 24)
    ui_login, ui_submit, ui_search, ui_dropdown, ui_navbar,
    ui_signup, ui_cart, ui_email_form, ui_password, ui_checkbox,
    ui_close, ui_pagination, ui_download, ui_delete, ui_save,
    ui_upload, ui_filter, ui_sort, ui_settings, ui_profile,
    ui_scroll, ui_terms, ui_notification, ui_rating,
    # QA / visual (original 22)
    qa_diagram, qa_barchart, qa_piechart, qa_thermometer, qa_clock,
    qa_trafficlight, qa_flag, qa_shapes, qa_colors, qa_count,
    qa_equation, qa_direction, qa_co2, qa_element, qa_map,
    qa_chart_highest, qa_largest_slice, qa_logo_color,
    qa_next_number, qa_planet, qa_weather, qa_fruit,
    # New UI images (15)
    ui_modal, ui_form_contact, ui_table, ui_tabs, ui_breadcrumb,
    ui_accordion, ui_step_wizard, ui_lang_dropdown, ui_sidebar,
    ui_banner, ui_toast, ui_date_picker, ui_slider, ui_progress_bar,
    ui_tooltip,
    # New VQA images (15)
    qa_line_chart, qa_histogram, qa_venn, qa_periodic_fe, qa_ruler,
    qa_protractor, qa_compass, qa_calendar, qa_speedometer, qa_battery,
    qa_signal_bars, qa_math_fraction, qa_dna, qa_balance, qa_scatter,
]


def main() -> None:
    for fn in GENERATORS:
        path = fn()
        print(f"  wrote {path.name}")
    print(f"\n{len(GENERATORS)} images written to {IMG_DIR}")


if __name__ == "__main__":
    main()
