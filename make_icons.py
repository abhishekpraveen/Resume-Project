"""
Generates small flat-style illustration icons used in the Streamlit app,
so the app never needs emojis or an internet connection for images.
"""
from PIL import Image, ImageDraw
import math
import os

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

def new_canvas(size=300, bg=None):
    img = Image.new("RGBA", (size, size), bg if bg else (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)

def rounded_bg(draw, size, color1, color2):
    # simple vertical gradient circle-ish backdrop
    for y in range(size):
        t = y / size
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

def make_document_icon():
    size = 300
    img, draw = new_canvas(size)
    rounded_bg(draw, size, (108, 99, 255), (77, 171, 247))
    # round the corners by masking
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size, size], radius=48, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    # paper
    draw.rounded_rectangle([85, 55, 215, 245], radius=14, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([85, 55, 145, 100], radius=10, fill=(255, 214, 102, 255))
    # lines of "text"
    for i, w in enumerate([90, 70, 100, 60, 85]):
        y = 130 + i * 20
        draw.rounded_rectangle([105, y, 105 + w, y + 8], radius=4, fill=(190, 190, 220, 255))
    img.save(os.path.join(OUT, "resume_icon.png"))

def make_magnifier_icon():
    size = 300
    img, draw = new_canvas(size)
    rounded_bg(draw, size, (255, 145, 158), (255, 200, 55))
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size, size], radius=48, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    # magnifier glass
    draw.ellipse([70, 60, 190, 180], outline=(255, 255, 255, 255), width=16)
    draw.ellipse([85, 75, 175, 165], fill=(255, 255, 255, 90))
    draw.line([172, 162, 230, 220], fill=(255, 255, 255, 255), width=20)
    img.save(os.path.join(OUT, "analyze_icon.png"))

def make_success_icon():
    size = 300
    img, draw = new_canvas(size)
    rounded_bg(draw, size, (56, 217, 169), (72, 187, 255))
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size, size], radius=48, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    draw.ellipse([55, 55, 245, 245], fill=(255, 255, 255, 230))
    # checkmark
    draw.line([(95, 155), (135, 195), (210, 105)], fill=(56, 175, 120, 255), width=18, joint="curve")
    img.save(os.path.join(OUT, "success_icon.png"))

def make_chart_icon():
    size = 300
    img, draw = new_canvas(size)
    rounded_bg(draw, size, (255, 132, 124), (167, 119, 227))
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size, size], radius=48, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    bars = [(80, 200, 60, (255, 214, 102)),
            (150, 150, 60, (255, 255, 255)),
            (220, 175, 60, (120, 231, 194))]
    for x, h, w, color in bars:
        draw.rounded_rectangle([x, 230 - h, x + w, 230], radius=8, fill=color + (255,))
    img.save(os.path.join(OUT, "chart_icon.png"))

def make_upload_icon():
    size = 300
    img, draw = new_canvas(size)
    rounded_bg(draw, size, (77, 171, 247), (108, 99, 255))
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, size, size], radius=48, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([70, 170, 230, 230], radius=14, fill=(255, 255, 255, 255))
    # arrow
    draw.polygon([(150, 60), (110, 110), (135, 110), (135, 170), (165, 170), (165, 110), (190, 110)],
                 fill=(255, 214, 102, 255))
    img.save(os.path.join(OUT, "upload_icon.png"))

def make_banner():
    w, h = 1000, 220
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for x in range(w):
        t = x / w
        # multi-color horizontal gradient: purple -> blue -> teal -> gold
        stops = [(108, 99, 255), (77, 171, 247), (56, 217, 169), (255, 200, 87)]
        seg = t * (len(stops) - 1)
        i = min(int(seg), len(stops) - 2)
        local_t = seg - i
        c1, c2 = stops[i], stops[i + 1]
        r = int(c1[0] + (c2[0] - c1[0]) * local_t)
        g = int(c1[1] + (c2[1] - c1[1]) * local_t)
        b = int(c1[2] + (c2[2] - c1[2]) * local_t)
        draw.line([(x, 0), (x, h)], fill=(r, g, b, 255))
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, w, h], radius=36, fill=255)
    img.putalpha(mask)
    img.save(os.path.join(OUT, "banner.png"))

make_document_icon()
make_magnifier_icon()
make_success_icon()
make_chart_icon()
make_upload_icon()
make_banner()
print("Icons generated in", OUT)
