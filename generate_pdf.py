"""
TrueFrame AI — Professional PDF Report Generator (Fixed)

FIXES:
  - temporal_max_real_run  (was: temporal_max_run)
  - temporal_confidence_adj (was: temporal_conf_adj / temporal_conf_adj)
  - temporal_fused_sequence (was: temporal_fused)
  - temporal_slope now read correctly
  - avg_compression_level  (was: avg_compression)
  - Page-2 stamp cy moved to 218 to avoid footer overlap
  - Sign-off line moved to y=228 — safe on A4
  - report_id defined once at top of generate_pdf
"""

from fpdf import FPDF, XPos, YPos
import datetime
import os
import math

# ── Colour palette ─────────────────────────────────────────────────────────
C_BG         = (4,   7,  15)
C_WHITE      = (255, 255, 255)
C_REAL       = (0,   195, 115)
C_AI         = (235,  75,  40)
C_ACCENT     = (61,  111, 255)
C_MUTED      = (90,  110, 150)
C_TEXT       = (25,   40,  65)
C_BGREAL     = (220, 255, 238)
C_BGAI       = (255, 228, 218)
C_BGGREY     = (242, 245, 252)
C_BGLIGHT    = (249, 251, 255)
C_AMBER      = (185, 120,   0)
C_BGYELL     = (255, 252, 228)
C_BORDER     = (215, 225, 242)
C_STAMP_R    = (0,   160,  90)
C_STAMP_A    = (210,  55,  25)
C_TEMPORAL   = (100,  60, 180)
C_BGTEMPORAL = (240, 235, 255)

PAGE_W   = 210
LEFT     = 16
RIGHT    = 16
USABLE_W = PAGE_W - LEFT - RIGHT   # 178 mm
FOOTER_H = 16


def _safe(text):
    if not isinstance(text, str):
        text = str(text)
    table = {
        '\u2014': ' - ', '\u2013': '-', '\u2022': '*', '\u2019': "'",
        '\u2018': "'",   '\u201c': '"', '\u201d': '"', '\u2026': '...',
        '\u2713': '[Y]', '\u2717': '[N]', '\u00a0': ' ', '\u2012': '-',
        '\u2015': '-',   '\u00b7': '.', '\u2014': ' - ',
    }
    for ch, r in table.items():
        text = text.replace(ch, r)
    return text.encode('latin-1', errors='replace').decode('latin-1')


class TrueFramePDF(FPDF):
    def __init__(self, is_authentic):
        super().__init__()
        self.is_authentic = is_authentic
        self.v_color  = C_REAL   if is_authentic else C_AI
        self.bg_color = C_BGREAL if is_authentic else C_BGAI
        self.set_margins(LEFT, 22, RIGHT)
        self.set_auto_page_break(auto=True, margin=FOOTER_H + 4)

    def header(self):
        self.set_fill_color(*C_BG)
        self.rect(0, 0, PAGE_W, 20, style='F')
        self.set_fill_color(*self.v_color)
        self.rect(0, 0, 5, 20, style='F')

        self.set_xy(10, 5)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*C_WHITE)
        self.cell(50, 7, 'TrueFrame AI', new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font('Helvetica', '', 7.5)
        self.set_text_color(140, 160, 200)
        self.cell(60, 7, '  Video Authenticity Report', new_x=XPos.RIGHT, new_y=YPos.TOP)

        ts = datetime.datetime.now().strftime('%d %b %Y  %H:%M UTC')
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*C_MUTED)
        self.set_xy(150, 5)
        self.cell(44, 4, _safe(ts), align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_xy(150, 10)
        self.cell(44, 4, 'Page {} of 2'.format(self.page_no()), align='R')
        self.set_xy(LEFT, 22)

    def footer(self):
        self.set_y(-FOOTER_H)
        self.set_fill_color(*C_BG)
        self.rect(0, self.get_y(), PAGE_W, FOOTER_H, style='F')
        self.set_fill_color(*self.v_color)
        self.rect(0, self.get_y(), PAGE_W, 1.2, style='F')
        self.set_y(-11)
        self.set_font('Helvetica', '', 6.5)
        self.set_text_color(80, 100, 130)
        self.cell(0, 5,
                  'TrueFrame AI  |  TrueFrameAI is an AI and can make mistakes. '
                  'Please double-check its responses  |  For informational purposes only  |  '
                  'Not a substitute for expert forensic review',
                  align='C')

    def draw_stamp(self, cx, cy, radius=22):
        col          = C_STAMP_R if self.is_authentic else C_STAMP_A
        label_top    = 'TRUEFRAME AI'
        label_bottom = 'VERIFIED' if self.is_authentic else 'FLAGGED'
        label_main   = 'AUTHENTIC' if self.is_authentic else 'AI GENERATED'
        label_sub    = 'VIDEO CONTENT'

        self.set_draw_color(*col)
        self.set_line_width(1.2)
        self.ellipse(cx - radius, cy - radius, radius * 2, radius * 2, style='D')
        r2 = radius - 3.5
        self.set_line_width(0.5)
        self.ellipse(cx - r2, cy - r2, r2 * 2, r2 * 2, style='D')

        self.set_font('Helvetica', 'B', 5.5)
        self.set_text_color(*col)
        r_txt     = radius - 1.8
        arc_span  = math.radians(140)
        start_ang = math.radians(270) - arc_span / 2
        chars     = list(label_top)
        n         = len(chars)
        for i, ch in enumerate(chars):
            a  = start_ang + arc_span * (i / max(n - 1, 1))
            lx = cx + r_txt * math.cos(a) - 1.2
            ly = cy + r_txt * math.sin(a) - 1.5
            self.set_xy(lx, ly)
            self.cell(2.4, 3, ch, align='C')

        arc_span2 = math.radians(100)
        start2    = math.radians(90) - arc_span2 / 2
        chars2    = list(label_bottom)
        n2        = len(chars2)
        for i, ch in enumerate(chars2):
            a  = start2 + arc_span2 * (i / max(n2 - 1, 1))
            lx = cx + r_txt * math.cos(a) - 1.2
            ly = cy + r_txt * math.sin(a) - 1.5
            self.set_xy(lx, ly)
            self.cell(2.4, 3, ch, align='C')

        font_size = 7 if self.is_authentic else 5.8
        self.set_font('Helvetica', 'B', font_size)
        self.set_text_color(*col)
        self.set_xy(cx - 18, cy - 5)
        self.cell(36, 5, label_main, align='C')

        self.set_font('Helvetica', '', 5)
        self.set_xy(cx - 18, cy + 1)
        self.cell(36, 3, label_sub, align='C')

        self.set_draw_color(*col)
        self.set_line_width(0.3)
        self.line(cx - 10, cy - 0.5, cx - 4, cy - 0.5)
        self.line(cx + 4,  cy - 0.5, cx + 10, cy - 0.5)

        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.set_text_color(*C_TEXT)


def _section_head(pdf, text, color=None):
    c = color or C_ACCENT
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*c)
    pdf.cell(USABLE_W, 6, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*c)
    pdf.set_line_width(0.35)
    pdf.line(LEFT, pdf.get_y(), LEFT + USABLE_W, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_text_color(*C_TEXT)
    pdf.set_x(LEFT)
    pdf.ln(2.5)


def _kv(pdf, key, val, shade):
    bg = C_BGGREY if shade else C_BGLIGHT
    pdf.set_fill_color(*bg)
    pdf.set_text_color(*C_TEXT)
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.cell(52, 7, _safe(key),
             fill=True, border='LTB', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.cell(USABLE_W - 52, 7, _safe(str(val)),
             fill=True, border='RTB', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)


def _temporal_kv(pdf, key, val, shade):
    bg = C_BGTEMPORAL if shade else C_BGLIGHT
    pdf.set_fill_color(*bg)
    pdf.set_text_color(*C_TEXT)
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.cell(74, 7, _safe(key),
             fill=True, border='LTB', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.cell(USABLE_W - 74, 7, _safe(str(val)),
             fill=True, border='RTB', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)


def _space_left(pdf):
    """Remaining vertical space before footer (mm)."""
    return 297 - FOOTER_H - 4 - pdf.get_y()


def generate_pdf(filename, file_size, result):
    label        = result['label']
    confidence   = result['confidence']
    detail       = result['detail']
    is_authentic = (label == 'AUTHENTIC')
    v_col        = C_REAL   if is_authentic else C_AI
    bg_col       = C_BGREAL if is_authentic else C_BGAI

    os.makedirs('static/results/trueframe', exist_ok=True)
    base     = os.path.splitext(filename)[0]
    pdf_path = 'static/results/trueframe/' + base + '_report.pdf'

    pdf = TrueFramePDF(is_authentic=is_authentic)

    # Single definition of report_id used throughout
    report_id = 'TF-{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    # ================================================================
    # PAGE 1
    # ================================================================
    pdf.add_page()

    # 1a. Verdict banner
    BANNER_H = 42
    y0       = pdf.get_y()

    pdf.set_fill_color(*bg_col)
    pdf.set_draw_color(*v_col)
    pdf.set_line_width(1.4)
    pdf.rect(LEFT, y0, USABLE_W, BANNER_H, style='FD')
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    pdf.set_fill_color(*v_col)
    pdf.rect(LEFT, y0, 6, BANNER_H, style='F')

    pdf.set_xy(28, y0 + 5)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(*v_col)
    pdf.cell(USABLE_W - 14, 13, label, align='L',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(28, y0 + 19)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(*C_MUTED)
    meta = 'Confidence: {:.1f}%  |  {} frames analysed  |  {}s processing time'.format(
        confidence, result['frames_count'], result['processing_time'])
    pdf.cell(USABLE_W - 14, 6, _safe(meta), align='L',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    t_flag = detail.get('temporal_flag', False)
    t_note = 'Temporal Inconsistency Detected' if t_flag else 'Temporally Consistent'
    t_icon = '[!]' if t_flag else '[OK]'
    pdf.set_xy(28, y0 + 27)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(*C_TEMPORAL)
    pdf.cell(USABLE_W - 14, 5,
             _safe('Temporal Analysis: {}  {}'.format(t_icon, t_note)),
             align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(28, y0 + 34)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(130, 150, 180)
    pdf.cell(USABLE_W - 14, 5,
             _safe('Report ID: {}  |  Generated: {}'.format(
                 report_id,
                 datetime.datetime.now().strftime('%d %B %Y, %I:%M %p'))),
             align='L')

    pdf.set_xy(LEFT, y0 + BANNER_H + 6)
    pdf.set_text_color(*C_TEXT)

    # 1b. Video details
    _section_head(pdf, 'Video Details')

    def fmt_sz(b):
        if b >= 1_048_576: return '{:.2f} MB'.format(b / 1_048_576)
        if b >= 1_024:     return '{:.2f} KB'.format(b / 1_024)
        return '{} B'.format(b)

    # FIX: read avg_compression_level (matches predict_hybrid output)
    avg_comp = detail.get('avg_compression_level', 0.0)
    if avg_comp >= 0.60:
        comp_label = 'Heavy  (bias correction applied)'
    elif avg_comp >= 0.30:
        comp_label = 'Moderate  (bias correction applied)'
    else:
        comp_label = 'Low / None'

    rows = [
        ('File Name',         filename),
        ('File Size',         fmt_sz(file_size)),
        ('Frames Sampled',    str(result['frames_count'])),
        ('Processing Time',   '{} seconds'.format(result['processing_time'])),
        ('Faces Detected',
            'Yes ({} frame(s))'.format(detail.get('face_count', 0))
            if detail.get('faces_detected')
            else 'No  (full-frame analysis applied)'),
        ('Compression Level', '{:.3f}  —  {}'.format(avg_comp, comp_label)),
        ('Analysis Date',     datetime.datetime.now().strftime('%B %d, %Y')),
        ('Analysis Time',     datetime.datetime.now().strftime('%I:%M:%S %p')),
        ('Report ID',         report_id),
    ]
    for i, (k, v) in enumerate(rows):
        _kv(pdf, k, v, shade=(i % 2 == 0))
    pdf.ln(6)

    # 1c. Confidence bar
    _section_head(pdf, 'Confidence Score')
    conf_pct = float(confidence)

    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*C_TEXT)
    pdf.cell(90, 6, 'Overall Detection Confidence',
             new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*v_col)
    pdf.cell(USABLE_W - 90, 6, '{:.1f}%'.format(conf_pct), align='R',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)

    BAR_H = 12
    bx    = LEFT
    by    = pdf.get_y() + 1
    bw    = USABLE_W

    pdf.set_fill_color(*C_BGGREY)
    pdf.rect(bx, by, bw, BAR_H, style='F')
    fill_w = bw * min(conf_pct, 100) / 100
    pdf.set_fill_color(*v_col)
    pdf.rect(bx, by, fill_w, BAR_H, style='F')

    if fill_w > 20:
        pdf.set_xy(bx + 2, by + 2)
        pdf.set_font('Helvetica', 'B', 7)
        pdf.set_text_color(*C_WHITE)
        pdf.cell(fill_w - 4, 8, '{:.0f}%'.format(conf_pct), align='R')

    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.4)
    pdf.rect(bx, by, bw, BAR_H, style='D')
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.2)

    pdf.set_xy(LEFT, by + BAR_H + 3)

    if conf_pct >= 85:
        ci = 'HIGH confidence — strong, consistent signal across the majority of frames.'
    elif conf_pct >= 70:
        ci = 'MODERATE-HIGH confidence — clear dominant signal with minor frame-level variation.'
    else:
        ci = 'MODERATE confidence — mixed signals; verdict reflects the overall balance of evidence.'

    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(USABLE_W, 5, _safe(ci), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*C_TEXT)

    # 1d. Stamp — safe absolute position, page 1
    pdf.draw_stamp(cx=175, cy=232, radius=21)

    # ================================================================
    # PAGE 2
    # ================================================================
    pdf.add_page()

    # 2a. Temporal sequence analysis
    _section_head(pdf, 'Temporal Sequence Analysis  (LSTM-Style)', color=C_TEMPORAL)

    # FIX: all keys now match predict_hybrid.py output exactly
    t_variance    = detail.get('temporal_variance',        0.0)
    t_instability = detail.get('temporal_instability',     0.0)
    t_flips       = detail.get('temporal_flips',           0)
    t_max_run     = detail.get('temporal_max_real_run',    0)   # FIXED key
    t_slope       = detail.get('temporal_slope',           0.0) # FIXED key (was missing)
    t_conf_adj    = detail.get('temporal_confidence_adj',  0.0) # FIXED key
    t_verd_raw    = detail.get('temporal_verdict',         None)
    t_verd_str    = t_verd_raw if t_verd_raw else 'Deferred to ensemble engine'

    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(*C_TEXT)
    expl = (
        'Per-frame confidence scores from both models are fused into a single authenticity '
        'sequence. LSTM-inspired statistics are computed over that sequence to detect the '
        'frame-to-frame inconsistency that generative models produce but real cameras do not. '
        'This analysis gates the final verdict.'
    )
    pdf.multi_cell(USABLE_W, 5.2, _safe(expl), align='J',
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)
    pdf.ln(2)

    t_rows = [
        ('Temporal Variance',
         '{:.4f}  {}'.format(t_variance,
             '[HIGH — inconsistency]' if t_variance > 0.10 else '[Normal]')),
        ('Frame Instability',
         '{:.4f}  {}'.format(t_instability,
             '[HIGH — abrupt shifts]' if t_instability > 0.18 else '[Normal]')),
        ('Class Flip Count',
         '{}  {}'.format(t_flips,
             '[EXCESSIVE]' if t_flips > 5 else '[Normal]')),
        ('Max Authentic Run',
         '{} consecutive frames'.format(t_max_run)),
        ('Sequence Slope',
         '{:+.5f}  {}'.format(t_slope,
             '[Drifting AI-ward]' if t_slope < -0.015 else '[Stable]')),
        ('Confidence Adjustment',
         '{:+.1f} points applied to base confidence'.format(t_conf_adj)),
        ('Temporal Verdict',  t_verd_str),
    ]
    for i, (k, v) in enumerate(t_rows):
        _temporal_kv(pdf, k, v, shade=(i % 2 == 0))

    pdf.set_x(LEFT)
    pdf.ln(5)

    # Sparkline chart
    # FIX: read temporal_fused_sequence (was temporal_fused)
    seq = detail.get('temporal_fused_sequence', [])
    if seq and _space_left(pdf) > 28:
        _section_head(pdf, 'Frame-by-Frame Fused Authenticity Sequence',
                      color=C_TEMPORAL)
        pdf.set_x(LEFT)
        pdf.set_font('Helvetica', '', 6.5)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(USABLE_W, 4,
                 'Each bar = one sampled frame.  '
                 'Height = fused authenticity score  (higher = more authentic).',
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(LEFT)

        CHART_H = 14
        chart_x = LEFT
        chart_y = pdf.get_y() + 1
        chart_w = USABLE_W
        bar_w   = chart_w / max(len(seq), 1)

        pdf.set_fill_color(*C_BGGREY)
        pdf.rect(chart_x, chart_y, chart_w, CHART_H, style='F')

        mid_y = chart_y + CHART_H * 0.5
        pdf.set_draw_color(*C_MUTED)
        pdf.set_line_width(0.25)
        dash_len = 2.0; gap_len = 1.5; x_cursor = chart_x
        while x_cursor < chart_x + chart_w:
            x_end = min(x_cursor + dash_len, chart_x + chart_w)
            pdf.line(x_cursor, mid_y, x_end, mid_y)
            x_cursor += dash_len + gap_len
        pdf.set_line_width(0.2)

        for j, val in enumerate(seq):
            bh  = CHART_H * float(val)
            bx  = chart_x + j * bar_w
            by  = chart_y + CHART_H - bh
            col = C_REAL if val >= 0.5 else C_AI
            pdf.set_fill_color(*col)
            pdf.rect(bx, by, max(bar_w - 0.4, 0.4), bh, style='F')

        pdf.set_draw_color(*C_BORDER)
        pdf.set_line_width(0.4)
        pdf.rect(chart_x, chart_y, chart_w, CHART_H, style='D')
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.2)

        leg_y = chart_y + CHART_H + 2
        pdf.set_fill_color(*C_REAL)
        pdf.rect(LEFT, leg_y, 5, 3, style='F')
        pdf.set_xy(LEFT + 6, leg_y)
        pdf.set_font('Helvetica', '', 6.5)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(24, 3, 'Authentic', new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.set_fill_color(*C_AI)
        pdf.rect(LEFT + 34, leg_y, 5, 3, style='F')
        pdf.set_xy(LEFT + 40, leg_y)
        pdf.cell(30, 3, 'AI-generated signal',
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_xy(LEFT, leg_y + 6)

    pdf.set_x(LEFT)
    pdf.ln(3)

    # 2b. Verdict interpretation
    _section_head(pdf, 'Verdict Interpretation')
    if is_authentic:
        interp = (
            'An AUTHENTIC verdict means TrueFrame AI found no credible evidence that this '
            'video was artificially generated or that any faces were digitally manipulated. '
            'The video exhibits statistical properties consistent with footage captured by a '
            'physical recording device. Temporal analysis confirmed frame-to-frame consistency '
            'characteristic of real camera footage. Where social-media compression was detected, '
            'the AI-detector score was corrected to prevent false positives.'
        )
    else:
        interp = (
            'An AI GENERATED verdict means TrueFrame AI detected significant markers of '
            'artificial synthesis. Modern AI generators produce distinctive statistical '
            'fingerprints at the pixel level. The temporal sequence analysis additionally '
            'detected frame-to-frame inconsistencies characteristic of generative models '
            'that sample each frame semi-independently. Both spatial and temporal evidence '
            'informed this classification after accounting for any compression artefacts.'
        )
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*C_TEXT)
    pdf.multi_cell(USABLE_W, 5.8, _safe(interp), align='J',
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)
    pdf.ln(4)

    # 2c. Detection notes
    _section_head(pdf, 'Detection Notes')
    notes_rows = [
        ('Authenticity Score',
         '{:.1f}% of frames scored authentic'.format(detail.get('rn_real_ratio', 0))),
        ('Manipulation Score',
         '{:.1f}% of frames showed manipulation signals'.format(detail.get('rn_fake_ratio', 0))),
        ('Synthesis Rate',
         '{:.1f}% of frames flagged for AI synthesis'.format(detail.get('ai_ai_ratio', 0))),
        ('Avg. Auth. Confidence',  '{:.1f}%'.format(detail.get('rn_real_avg', 0))),
        ('Avg. Synth. Confidence', '{:.1f}%'.format(detail.get('ai_ai_avg', 0))),
        ('Face Analysis',
            'Detected in {} frame(s) — face-crop analysis applied'.format(
                detail.get('face_count', 0))
            if detail.get('faces_detected')
            else 'No faces detected — full-frame analysis applied'),
        ('Decision Rule',
            result.get('reason_code', 'N/A').replace('_', ' ').title()),
    ]
    for i, (k, v) in enumerate(notes_rows):
        _kv(pdf, k, v, shade=(i % 2 == 0))
    pdf.set_x(LEFT)
    pdf.ln(4)

    # 2d. Disclaimer
    if _space_left(pdf) < 40:
        pdf.add_page()
        pdf.set_x(LEFT)

    pdf.set_fill_color(*C_BGYELL)
    pdf.set_draw_color(*C_AMBER)
    pdf.set_line_width(0.7)
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(*C_AMBER)
    pdf.cell(USABLE_W, 6, 'DISCLAIMER',
             border='LTR', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*C_TEXT)
    pdf.multi_cell(
        USABLE_W, 5,
        'This report is generated by an automated AI system for informational purposes only. '
        'No automated detection system achieves 100% accuracy on all content. Results represent '
        'the best available algorithmic assessment and should not be used as the sole basis for '
        'legal, journalistic, or forensic decisions. TrueFrame AI accepts no liability for '
        'actions taken based solely on this report. For high-stakes applications, consult '
        'qualified human experts and conduct independent verification.',
        fill=True, border='LBR', align='J',
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(LEFT)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    # 2e. Stamp — FIX: moved up to cy=218 so it clears the footer (footer starts ~y=277)
    pdf.draw_stamp(cx=175, cy=218, radius=21)

    # Sign-off — FIX: moved to y=232 to sit below stamp without overlapping footer
    pdf.set_xy(LEFT, 232)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(USABLE_W, 4,
             _safe('TrueFrame AI  |  Report ID: {}  |  {}'.format(
                 report_id,
                 datetime.datetime.now().strftime('%d %B %Y'))),
             align='C')

    pdf.output(pdf_path)
    return pdf_path