"""Write a tiny VobSub (.idx + .sub) pair from text, for testing the bitmap
subtitle path. ffmpeg can't encode bitmap subtitles from text, so the DVD SPU
packets are built by hand: 2-bit RLE, interlaced fields, one display control
sequence and one stop sequence per cue, each wrapped in an MPEG-PS pack.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 480


def _rle_line(px):
    """One scanline of 2-bit pixels -> DVD RLE nibbles (a list of ints 0..15)."""
    out, i, n = [], 0, len(px)
    while i < n:
        c, j = px[i], i
        while j < n and px[j] == c and j - i < 255:
            j += 1
        run = j - i
        if j == n and run > 63:          # run to end of line: 14 zero bits + colour
            code, bits = c, 16
        elif run < 4:
            code, bits = (run << 2) | c, 4
        elif run < 16:
            code, bits = (run << 2) | c, 8
        elif run < 64:
            code, bits = (run << 2) | c, 12
        else:
            code, bits = (run << 2) | c, 16
        for k in range(bits // 4 - 1, -1, -1):
            out.append((code >> (4 * k)) & 0xF)
        i = j
    if len(out) % 2:
        out.append(0)                    # byte-align every line
    return out


def _nibbles_to_bytes(nb):
    return bytes((nb[k] << 4) | nb[k + 1] for k in range(0, len(nb), 2))


def _spu(text, dur):
    font = ImageFont.load_default(size=40)
    img = Image.new("L", (W - 40, 70), 0)
    ImageDraw.Draw(img).text((10, 10), text, fill=255, font=font)
    w, h = img.size
    px = [[1 if img.getpixel((x, y)) > 127 else 0 for x in range(w)] for y in range(h)]
    top = b"".join(_nibbles_to_bytes(_rle_line(px[y])) for y in range(0, h, 2))
    bot = b"".join(_nibbles_to_bytes(_rle_line(px[y])) for y in range(1, h, 2))
    x1, y1 = 20, H - 100
    x2, y2 = x1 + w - 1, y1 + h - 1
    top_off = 4
    bot_off = top_off + len(top)
    dcsq1_off = bot_off + len(bot)
    cmds1 = (bytes([0x03, 0x00, 0x10])            # palette: pattern=1 (white), bg=0
             + bytes([0x04, 0x00, 0xF0])          # alpha: pattern opaque, bg clear
             + bytes([0x05, x1 >> 4, ((x1 & 0xF) << 4) | (x2 >> 8), x2 & 0xFF,
                      y1 >> 4, ((y1 & 0xF) << 4) | (y2 >> 8), y2 & 0xFF])
             + bytes([0x06]) + top_off.to_bytes(2, "big") + bot_off.to_bytes(2, "big")
             + bytes([0x01, 0xFF]))
    dcsq2_off = dcsq1_off + 4 + len(cmds1)
    delay = int(dur * 90000 / 1024)
    dcsq1 = (0).to_bytes(2, "big") + dcsq2_off.to_bytes(2, "big") + cmds1
    dcsq2 = delay.to_bytes(2, "big") + dcsq2_off.to_bytes(2, "big") + bytes([0x02, 0xFF])
    body = top + bot + dcsq1 + dcsq2
    return (len(body) + 4).to_bytes(2, "big") + dcsq1_off.to_bytes(2, "big") + body


def _pts(t):
    v = int(t * 90000)
    return bytes([0x21 | ((v >> 29) & 0x0E), (v >> 22) & 0xFF, 0x01 | ((v >> 14) & 0xFE),
                  (v >> 7) & 0xFF, 0x01 | ((v << 1) & 0xFE)])


def _packs(spu, t):
    """Split one SPU across 2048-byte PS packs (private stream 1, substream 0x20)."""
    out, first, pos = b"", True, 0
    while pos < len(spu):
        pack = b"\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01\x01\x89\xc3\xf8"
        hdr = b"\x81\x80\x05" + _pts(t) if first else b"\x81\x00\x00"
        room = 2048 - len(pack) - 6 - len(hdr) - 1
        chunk = spu[pos:pos + room]
        pes_body = hdr + b"\x20" + chunk
        pes = b"\x00\x00\x01\xbd" + len(pes_body).to_bytes(2, "big") + pes_body
        blob = pack + pes
        pad = 2048 - len(blob)
        if pad >= 6:
            blob += b"\x00\x00\x01\xbe" + (pad - 6).to_bytes(2, "big") + b"\xff" * (pad - 6)
        elif pad:
            blob += b"\xff" * pad
        out += blob
        pos += len(chunk)
        first = False
    return out


def write_vobsub(base, cues):
    """cues: [(start_s, end_s, text)] -> base.idx + base.sub"""
    sub, idx_lines = b"", []
    for st, en, text in cues:
        ms = int(round(st * 1000))
        h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        idx_lines.append(f"timestamp: {h:02d}:{m:02d}:{s:02d}:{ms:03d}, filepos: {len(sub):09x}")
        sub += _packs(_spu(text, en - st), st)
    palette = ", ".join(["000000", "ffffff"] + ["808080"] * 14)
    with open(base + ".idx", "w") as f:
        f.write("# VobSub index file, v7 (do not modify this line!)\n"
                f"size: {W}x{H}\npalette: {palette}\n\nid: en, index: 0\n"
                + "\n".join(idx_lines) + "\n")
    with open(base + ".sub", "wb") as f:
        f.write(sub)
