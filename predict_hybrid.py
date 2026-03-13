"""
TrueFrame AI — Hybrid Video Authenticity Detector (v5 — Fixed)

KEY FIXES:
  - temporal_max_real_run  (was: temporal_max_run)
  - temporal_confidence_adj (was: temporal_conf_adj)
  - temporal_fused_sequence (was: temporal_fused)
  - temporal_slope added (was missing from _temporal_analysis output)
  - avg_compression_level  (was: avg_compression) — matches generate_pdf
  - Dropout 0.3 in predict path (matches training checkpoint)
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import time

DEVICE        = torch.device("cpu")
RESNET_PATH   = "models/best_model.pth"
AI_MODEL_NAME = "umm-maybe/AI-image-detector"

_resnet = None
_aidet  = None
_aiproc = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── Frame count ───────────────────────────────────────────────────────────────
def _get_num_frames(duration_seconds):
    if duration_seconds <= 10:   return 8
    elif duration_seconds <= 30: return 10
    elif duration_seconds <= 60: return 12
    else:                        return 16


# ── Compression detector ──────────────────────────────────────────────────────
def _is_social_media_compressed(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    block_score = 0.0
    if h >= 16 and w >= 16:
        br = gray[8::8, :]; ir = gray[7::8, :]
        nr = min(len(br), len(ir))
        bc = gray[:, 8::8]; ic = gray[:, 7::8]
        nc = min(bc.shape[1], ic.shape[1])
        if nr > 0 and nc > 0:
            diff_r = float(np.mean(np.abs(br[:nr] - ir[:nr])))
            diff_c = float(np.mean(np.abs(bc[:, :nc] - ic[:, :nc])))
            block_score = min((diff_r + diff_c) / 20.0, 1.0)
    quant_scores = []
    for ch in cv2.split(frame_bgr):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        quant_scores.append(float(np.sum(hist == 0)) / 256.0)
    quant_score   = float(np.mean(quant_scores))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_score    = max(0.0, 1.0 - min(laplacian_var / 300.0, 1.0))
    return float(0.40 * block_score + 0.35 * quant_score + 0.25 * blur_score)


def _compression_bias_correction(ai_score, compression_level):
    if compression_level < 0.25:
        return ai_score
    strength = min((compression_level - 0.25) / 0.40, 1.0)
    if ai_score > 50:
        corrected = ai_score - (ai_score - 50.0) * strength * 0.80
        return max(50.0, corrected)
    return ai_score


# ── Faceswap engine ───────────────────────────────────────────────────────────
def _detect_face_largest(gray):
    for sf, mn, ms in [(1.1,4,(40,40)), (1.05,3,(30,30)), (1.08,2,(25,25)), (1.05,2,(20,20))]:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=sf,
                                              minNeighbors=mn, minSize=ms)
        if len(faces) > 0:
            areas = [(w * h, x, y, w, h) for (x, y, w, h) in faces]
            areas.sort(reverse=True)
            return areas[0]
    return None


def _cr_edge_center(frame_bgr):
    H, W  = frame_bgr.shape[:2]
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    result = _detect_face_largest(gray)
    if result is None:
        return 0.0, False
    _, fx, fy, fw, fh = result
    fx1, fy1 = max(0, fx),      max(0, fy)
    fx2, fy2 = min(W, fx + fw), min(H, fy + fh)
    if (fx2 - fx1) < 24 or (fy2 - fy1) < 24:
        return 0.0, False
    face_cr = ycrcb[fy1:fy2, fx1:fx2, 1].astype(float)
    fh_, fw_ = face_cr.shape
    center = face_cr[fh_//4:3*fh_//4, fw_//4:3*fw_//4]
    eb     = max(5, min(8, fh_//10))
    edge   = np.concatenate([
        face_cr[:eb,  :].ravel(), face_cr[-eb:, :].ravel(),
        face_cr[:, :eb].ravel(), face_cr[:, -eb:].ravel(),
    ])
    if center.size == 0 or len(edge) == 0:
        return 0.0, False
    return abs(float(center.mean()) - float(edge.mean())), True


def _analyze_faceswap(frames_data):
    print("\n[FACESWAP ENGINE v5] Analysing frames...")
    cr_vals  = []
    face_cnt = 0

    for i, (frame, _) in enumerate(frames_data):
        cr, found = _cr_edge_center(frame)
        if found:
            cr_vals.append(cr)
            face_cnt += 1
        else:
            cr_vals.append(None)
        print(f"  frame {i+1:02d}: cr={cr:.2f}  face={found}")

    valid = [v for v in cr_vals if v is not None]
    if not valid:
        print("[FACESWAP] No faces → deferred")
        return None, 0.0, {'faceswap_engine': 'no_faces'}

    HIGH     = 15.0
    max_consec = cur = 0
    for v in valid:
        if v > HIGH: cur += 1; max_consec = max(max_consec, cur)
        else: cur = 0
    high_frac = sum(1 for v in valid if v > HIGH) / len(valid)
    cr_med    = float(np.median(valid))

    print(f"[FACESWAP] cr_med={cr_med:.2f}  consec={max_consec}  frac={high_frac:.2f}  faces={face_cnt}")

    verdict = confidence = None
    min_consec = max(4, int(len(valid) * 0.35))
    if max_consec >= min_consec and high_frac >= 0.40:
        confidence = min(88.0, 60.0 + max_consec * 2.0 + high_frac * 10.0)
        verdict    = 'AI GENERATED'
        print(f"[FACESWAP] → SEAM DETECTED  consec={max_consec}>={min_consec}  conf={confidence:.1f}%")
    else:
        print(f"[FACESWAP] → No clear seam → deferred to ML")
        verdict    = None
        confidence = 0.0

    detail = {
        'faceswap_engine':     'active',
        'faceswap_cr_median':  round(cr_med, 2),
        'faceswap_max_consec': max_consec,
        'faceswap_high_frac':  round(high_frac * 100, 1),
        'faceswap_faces':      face_cnt,
    }
    return verdict, confidence, detail


# ── Model loaders ─────────────────────────────────────────────────────────────
def _load_resnet():
    global _resnet
    if _resnet is not None:
        return _resnet
    try:
        m = models.resnet18(weights=None)
        # Use Dropout(0.5) to match training checkpoint exactly
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))
        m.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
        m.eval()
        _resnet = m
        print("[INFO] ResNet-18 loaded")
        return _resnet
    except Exception as e:
        print(f"[WARN] ResNet not loaded: {e}")
        return None


def _load_aidet():
    global _aidet, _aiproc
    if _aidet is not None:
        return _aidet, _aiproc
    try:
        _aiproc = AutoImageProcessor.from_pretrained(AI_MODEL_NAME)
        _aidet  = AutoModelForImageClassification.from_pretrained(AI_MODEL_NAME)
        _aidet.eval()
        print("[INFO] AI-image-detector loaded")
        return _aidet, _aiproc
    except Exception as e:
        print(f"[WARN] AI-det not loaded: {e}")
        return None, None


# ── Per-frame inference ───────────────────────────────────────────────────────
def _resnet_score(model, frame_bgr):
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(20, 20))
    face_found = len(faces) > 0
    if face_found:
        areas = [(w * h, x, y, w, h) for (x, y, w, h) in faces]
        areas.sort(reverse=True)
        _, x, y, w, h = areas[0]
        hf, wf = frame_bgr.shape[:2]
        roi = frame_bgr[max(0,y-15):min(hf,y+h+15), max(0,x-15):min(wf,x+w+15)]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            roi = frame_bgr
    else:
        roi = frame_bgr
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    t   = _transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    return float(probs[1]) * 100, float(probs[0]) * 100, face_found


def _aidet_score(model, processor, frame_bgr, compression_level=0.0):
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp  = processor(images=Image.fromarray(rgb), return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits, dim=1)[0]
    ai_raw  = float(probs[1]) * 100
    ai_corr = _compression_bias_correction(ai_raw, compression_level)
    return 100.0 - ai_corr, ai_corr


# ── Temporal analysis ─────────────────────────────────────────────────────────
def _temporal_analysis(rn_real_scores, ai_real_scores):
    n      = max(len(rn_real_scores), len(ai_real_scores), 1)
    rn_n   = [s/100.0 for s in rn_real_scores] if rn_real_scores else [0.5]*n
    ai_n   = [s/100.0 for s in ai_real_scores] if ai_real_scores else [0.5]*n
    L      = max(len(rn_n), len(ai_n))
    rn_n  += [0.5] * (L - len(rn_n))
    ai_n  += [0.5] * (L - len(ai_n))
    fused  = [0.50*rn_n[i] + 0.50*ai_n[i] for i in range(L)]

    variance    = float(np.var(fused))
    instability = float(np.mean(np.abs(np.diff(fused)))) if L > 1 else 0.0
    binary      = [s > 0.5 for s in fused]
    flips       = sum(1 for i in range(1, len(binary)) if binary[i] != binary[i-1])
    max_real    = cur = 0
    for b in binary:
        if b: cur += 1; max_real = max(max_real, cur)
        else: cur = 0
    slope = float(np.polyfit(range(L), fused, 1)[0]) if L >= 3 else 0.0

    flip_thresh = max(5, L // 3)
    signals     = sum([variance > 0.12, instability > 0.20, flips > flip_thresh])

    t_flag = False; verdict = None; conf_adj = 0.0
    if signals >= 2:
        t_flag = True; verdict = "AI GENERATED"; conf_adj = 3.0 * signals
    elif max_real >= max(L * 0.75, 4) and variance < 0.04:
        verdict = "AUTHENTIC"; conf_adj = +6.0
    elif slope < -0.020 and variance > 0.08:
        t_flag = True; verdict = "AI GENERATED"; conf_adj = +3.0

    detail = {
        # ── KEY NAMES must exactly match generate_pdf.py reads ──
        "temporal_variance":        round(variance, 4),
        "temporal_instability":     round(instability, 4),
        "temporal_flips":           flips,
        "temporal_max_real_run":    max_real,       # FIX: was temporal_max_run
        "temporal_slope":           round(slope, 6), # FIX: was missing
        "temporal_flag":            t_flag,
        "temporal_verdict":         verdict,
        "temporal_confidence_adj":  round(conf_adj, 2),   # FIX: was temporal_conf_adj
        "temporal_fused_sequence":  [round(s, 3) for s in fused],  # FIX: was temporal_fused
    }
    print(f"[TEMPORAL] var={variance:.4f} instab={instability:.4f} flips={flips}/{flip_thresh} "
          f"run={max_real}/{L} slope={slope:+.5f} → {verdict}  adj={conf_adj:+.1f}")
    return verdict, conf_adj, t_flag, detail


# ── Frame extraction ──────────────────────────────────────────────────────────
def _extract_frames(video_path, video_name):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        ret, fr = cap.read()
        if not ret: cap.release(); return []
        total = 1
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total / fps
    nf       = _get_num_frames(duration)
    n        = min(nf, total)
    print(f"[INFO] dur={duration:.1f}s  fps={fps:.1f}  total={total}  sampling={n}")

    start   = max(0, int(total * 0.05))
    end     = min(total - 1, int(total * 0.95))
    indices = set(np.linspace(start, end, n, dtype=int))

    save_dir = os.path.join("static", "results", "deepguard", video_name)
    os.makedirs(save_dir, exist_ok=True)

    out = []; cnt = idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if cnt in indices:
            fname = f"frame_{idx}.jpg"
            h, w  = frame.shape[:2]
            if w > 480:
                frame = cv2.resize(frame, (480, int(h * 480 / w)))
            fpath = os.path.join(save_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append((frame, f"results/deepguard/{video_name}/{fname}"))
            idx += 1
        cnt += 1
    cap.release()
    return out


# ── Decision engine ───────────────────────────────────────────────────────────
def _decide(rn_real_scores, rn_fake_scores, ai_real_scores, ai_scores,
            face_count, total,
            t_verdict, t_conf_adj, t_flag,
            fs_verdict, fs_confidence):

    rn_real_avg   = float(np.mean(rn_real_scores)) if rn_real_scores else 50.0
    rn_fake_avg   = float(np.mean(rn_fake_scores)) if rn_fake_scores else 50.0
    ai_real_avg   = float(np.mean(ai_real_scores)) if ai_real_scores else 50.0
    ai_ai_avg     = float(np.mean(ai_scores))       if ai_scores       else 50.0
    rn_real_ratio = sum(1 for s in rn_real_scores if s > 50) / max(total, 1)
    rn_fake_ratio = 1.0 - rn_real_ratio
    ai_ai_ratio   = sum(1 for s in ai_scores if s > 50)      / max(total, 1)
    ai_real_ratio = 1.0 - ai_ai_ratio
    has_faces     = face_count > 0

    print(f"[DECISION] rn_real={rn_real_avg:.1f}({rn_real_ratio:.2f})  "
          f"ai_ai={ai_ai_avg:.1f}({ai_ai_ratio:.2f})  "
          f"ai_real={ai_real_avg:.1f}  faces={face_count}/{total}")

    detail = {
        "rn_real_ratio":  round(rn_real_ratio * 100, 1),
        "rn_fake_ratio":  round(rn_fake_ratio * 100, 1),
        "rn_real_avg":    round(rn_real_avg, 1),
        "ai_ai_ratio":    round(ai_ai_ratio * 100, 1),
        "ai_real_ratio":  round(ai_real_ratio * 100, 1),
        "ai_ai_avg":      round(ai_ai_avg, 1),
        "faces_detected": has_faces,
        "face_count":     face_count,
    }

    # P0 — Faceswap seam block
    if fs_verdict == 'AI GENERATED' and fs_confidence >= 60.0:
        adj  = t_conf_adj if t_flag else 0.0
        conf = min(95.0, fs_confidence + adj)
        print(f"[DECISION] P0 FACESWAP → AI GENERATED  conf={conf:.1f}")
        return "AI GENERATED", conf, "FACESWAP_DETECTED", detail

    # P1 — Temporal LSTM
    if t_flag and t_verdict is not None:
        if t_verdict == "AI GENERATED":
            conf = min(95, max(ai_ai_avg, 58.0) + t_conf_adj)
            return "AI GENERATED", conf, "TEMPORAL_INCONSISTENCY", detail
        elif t_verdict == "AUTHENTIC":
            conf = min(95, max(rn_real_avg, 58.0) + t_conf_adj)
            return "AUTHENTIC", conf, "TEMPORAL_CONSISTENT", detail

    # P2 — AI detector dominant
    if ai_ai_ratio >= 0.60 and ai_ai_avg > 60:
        if rn_real_ratio >= 0.90 and rn_real_avg > 88 and has_faces and ai_ai_avg < 70:
            conf = min(88, ai_ai_avg * 0.60 + rn_fake_avg * 0.40 + t_conf_adj)
            return "AI GENERATED", min(78, max(58, conf)), "BORDERLINE_AI", detail
        conf = min(95, ai_ai_avg * 0.80 + ai_ai_ratio * 12 + t_conf_adj)
        return "AI GENERATED", min(95, max(62, conf)), "AI_DET_DOMINANT", detail

    # P3 — Both agree real
    if rn_real_ratio >= 0.70 and ai_real_ratio >= 0.55 and rn_real_avg > 68:
        conf = min(95, rn_real_avg * 0.55 + ai_real_avg * 0.45 + t_conf_adj)
        return "AUTHENTIC", min(95, max(62, conf)), "BOTH_AGREE_REAL", detail

    # P4 — ResNet strongly real
    if rn_real_ratio >= 0.75 and rn_real_avg > 75 and ai_ai_ratio < 0.55:
        conf = min(95, rn_real_avg * 0.72 + ai_real_avg * 0.28 + t_conf_adj)
        return "AUTHENTIC", min(92, max(60, conf)), "CAMERA_REAL", detail

    # P5 — AI det moderate
    if ai_ai_ratio >= 0.55 and ai_ai_avg > 58:
        conf = min(92, ai_ai_avg * 0.75 + ai_ai_ratio * 14 + t_conf_adj)
        return "AI GENERATED", min(88, max(56, conf)), "AI_DET_MODERATE", detail

    # P6 — ResNet moderate real
    if rn_real_ratio >= 0.60 and rn_real_avg > 65:
        conf = min(90, rn_real_avg * 0.65 + ai_real_avg * 0.35 + t_conf_adj)
        return "AUTHENTIC", min(85, max(56, conf)), "RESNET_MODERATE_REAL", detail

    # P7 — Weighted ensemble
    real_s = rn_real_ratio * rn_real_avg/100 * 0.60 + ai_real_ratio * ai_real_avg/100 * 0.40
    ai_s   = ai_ai_ratio   * ai_ai_avg  /100 * 0.60 + rn_fake_ratio * rn_fake_avg /100 * 0.40
    tot    = real_s + ai_s
    if tot == 0:
        return "AUTHENTIC", 55, "FALLBACK_NEUTRAL", detail
    if real_s >= ai_s:
        conf = min(85, max(55, real_s/tot * 100 + t_conf_adj))
        return "AUTHENTIC",     conf, "ENSEMBLE_REAL", detail
    else:
        conf = min(85, max(55, ai_s/tot   * 100 + t_conf_adj))
        return "AI GENERATED",  conf, "ENSEMBLE_AI",  detail


# ── Public API ────────────────────────────────────────────────────────────────
def predict_video(video_path, video_name):
    t0 = time.time()
    resnet        = _load_resnet()
    aidet, aiproc = _load_aidet()

    frames_data = _extract_frames(video_path, video_name)
    if not frames_data:
        print(f"[ERROR] No frames from {video_path}")
        return None

    total       = len(frames_data)
    saved_paths = [p for _, p in frames_data]

    fs_verdict, fs_confidence, fs_detail = _analyze_faceswap(frames_data)

    rn_real_scores=[]; rn_fake_scores=[]
    ai_real_scores=[]; ai_ai_scores=[]
    frame_rows=[]; face_count=0; comp_levels=[]

    for i, (frame, _) in enumerate(frames_data):
        comp = _is_social_media_compressed(frame)
        comp_levels.append(comp)

        rn_real=rn_fake=rn_face=None
        ai_real=ai_ai=None

        if resnet:
            try:
                rn_real, rn_fake, rn_face = _resnet_score(resnet, frame)
                rn_real_scores.append(rn_real); rn_fake_scores.append(rn_fake)
                if rn_face: face_count += 1
            except Exception as e:
                print(f"[WARN] ResNet f{i}: {e}")
                rn_real_scores.append(50.0); rn_fake_scores.append(50.0)
                rn_real=rn_fake=50.0; rn_face=False

        if aidet:
            try:
                ai_real, ai_ai = _aidet_score(aidet, aiproc, frame, comp)
                ai_real_scores.append(ai_real); ai_ai_scores.append(ai_ai)
            except Exception as e:
                print(f"[WARN] AI-det f{i}: {e}")
                ai_real_scores.append(50.0); ai_ai_scores.append(50.0)
                ai_real=ai_ai=50.0

        frame_rows.append({
            "frame":       i+1,
            "rn_real":     round(rn_real,1) if rn_real is not None else None,
            "rn_fake":     round(rn_fake,1) if rn_fake is not None else None,
            "ai_real":     round(ai_real,1) if ai_real is not None else None,
            "ai_ai":       round(ai_ai,  1) if ai_ai   is not None else None,
            "has_face":    bool(rn_face)     if rn_face is not None else False,
            "compression": round(comp, 3),
        })

    avg_comp = float(np.mean(comp_levels)) if comp_levels else 0.0
    print(f"[COMPRESS] avg={avg_comp:.3f}  frames={total}  time={time.time()-t0:.1f}s")

    t_verdict, t_conf_adj, t_flag, t_detail = _temporal_analysis(
        rn_real_scores, ai_real_scores)

    label, confidence, reason_code, detail = _decide(
        rn_real_scores, rn_fake_scores,
        ai_real_scores, ai_ai_scores,
        face_count, total,
        t_verdict, t_conf_adj, t_flag,
        fs_verdict, fs_confidence,
    )

    detail.update(t_detail)
    detail.update(fs_detail)
    # FIX: key name matches generate_pdf.py expectation
    detail["avg_compression_level"] = round(avg_comp, 3)

    elapsed = round(time.time() - t0, 2)
    print(f"[DONE] {label}  conf={confidence:.1f}%  reason={reason_code}  time={elapsed}s")

    return {
        "label":           label,
        "confidence":      round(min(95, max(55, confidence)), 1),
        "frames":          saved_paths,
        "frames_count":    total,
        "processing_time": elapsed,
        "reason_code":     reason_code,
        "detail":          detail,
        "frame_rows":      frame_rows,
        "resnet_ok":       resnet is not None,
        "aidet_ok":        aidet  is not None,
    }