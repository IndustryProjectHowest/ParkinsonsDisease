import os
import tempfile
import uuid
import cv2
import subprocess
import pickle

import numpy as np
import streamlit as st
from tensorflow import keras

from detectron2_utils import setup_detectron2, process_frame
from snippet_utils    import generate_snippets

IMG_SIZE        = 224
MAX_SEQ_LENGTH  = 50
NUM_FEATURES    = 2048

st.sidebar.header("1) Detection Settings")
det_thresh = st.sidebar.slider(
    "Detectron2 confidence threshold",
    0.0, 1.0, 0.5, 0.01
)

st.sidebar.header("2) Snippet Settings")
snip_thresh_bb  = st.sidebar.slider(
    "BBox-change threshold",
    0.0, 0.2, 0.15, 1e-6
)
snip_iou        = st.sidebar.slider(
    "Merge‐overlap IOU threshold",
    0.0, 1.0, 0.01, 1e-6
)

st.sidebar.header("3) Classification Settings")
pos_thresh = st.sidebar.slider(
    "Positive‐class threshold",
    0.0, 1.0, 0.42, 0.01
)

def make_predictor():
    return setup_detectron2(
        model_path="model_final.pth",
        num_classes=2,
        score_thresh=det_thresh
    )

@st.cache_resource
def load_tf_models():
 
    with open("label_processor.pkl", "rb") as f:
        label_lookup = pickle.load(f)
    class_vocab = label_lookup.get_vocabulary()

 
    feat_ex = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

 
    seq_mod = keras.models.load_model("my_sequence_model.h5", compile=False)

    return class_vocab, feat_ex, seq_mod

CLASS_VOCAB, feature_extractor, sequence_model = load_tf_models()


def prepare_frames(path):
    cap    = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        h, w = fr.shape[:2]
        m    = min(h, w)
   
        fr   = fr[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
        fr   = cv2.resize(fr, (IMG_SIZE, IMG_SIZE))
        frames.append(fr[..., ::-1])  # BGR→RGB
    cap.release()
    return np.array(frames)


def classify_snippet(path):
    frames = prepare_frames(path)
    n      = len(frames)
    if n == 0:
        return "N/A", 0.0, 0.0

    feats = feature_extractor.predict(frames, verbose=0)
    mask  = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
    mask[0, :n] = True

    padded = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    padded[0, :n, :] = feats[:MAX_SEQ_LENGTH]

    probs = sequence_model.predict([padded, mask], verbose=0)[0]
    pos_idx = CLASS_VOCAB.index("positive")
    neg_idx = CLASS_VOCAB.index("negative")
    ppos    = float(probs[pos_idx])
    pneg    = float(probs[neg_idx])

    if ppos >= pos_thresh:
        label = "positive"
    else:
        label = "negative"

    return label, ppos, pneg

st.title("🐶🐽 Dog-Sniff Tank Classifier + Snippet Analyzer")

# session‐state slots
st.session_state.setdefault("orig_blob",  None)
st.session_state.setdefault("orig_path",  None)
st.session_state.setdefault("proc_blob",  None)
st.session_state.setdefault("snippets",   None)

# ---- step 1: upload ----
if not st.session_state.orig_blob:
    uploaded = st.file_uploader("▶️ 1) Upload video", type=["mp4","avi","mov"])
    if uploaded:
        data = uploaded.read()
        st.session_state.orig_blob = data

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(data); tmp.close()
        st.session_state.orig_path = tmp.name

        st.subheader("Original Preview")
        st.video(data)

# ---- step 2: run Detectron2 & draw boxes ----
if st.session_state.orig_path and not st.session_state.proc_blob:
    if st.button("🦾 2) Run Detection & Draw"):
        predictor = make_predictor()

        cap   = cv2.VideoCapture(st.session_state.orig_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w,h   = int(cap.get(3)), int(cap.get(4))

        tdir    = tempfile.mkdtemp()
        raw_out = os.path.join(tdir, f"{uuid.uuid4().hex}_raw.mp4")
        enc_out = os.path.join(tdir, f"{uuid.uuid4().hex}_h264.mp4")

        writer = cv2.VideoWriter(
            raw_out,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w,h)
        )
        bar  = st.progress(0)
        info = st.empty()

        for i in range(total):
            ret, fr = cap.read()
            if not ret:
                break
            out = process_frame(fr, predictor)
            writer.write(out)
            bar.progress((i+1)/total)
            info.text(f"Frame {i+1}/{total}")
        cap.release()
        writer.release()

        # re-encode for browser
        subprocess.run([
            "ffmpeg","-y","-i", raw_out,
            "-c:v","libx264","-preset","fast","-crf","23",
            enc_out
        ], check=True)

        st.session_state.proc_blob = open(enc_out,"rb").read()
        st.success("✅ Detection done!")

# ---- show original vs processed ----
if st.session_state.proc_blob:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.video(st.session_state.orig_blob)
    with c2:
        st.subheader("Processed")
        st.video(st.session_state.proc_blob)

# ---- step 3: generate snippets ----
if st.session_state.proc_blob and st.session_state.snippets is None:
    if st.button("🎬 3) Generate Snippets"):
        with st.spinner("Snipping…"):
            clips = generate_snippets(
                st.session_state.orig_path,
                make_predictor(),
                conf_thresh=det_thresh,
                threshold_bb=snip_thresh_bb,
                iou_threshold=snip_iou
            )
        st.session_state.snippets = clips
        st.success(f"{len(clips)} snippets created!")

# ---- step 4: display & classify each snippet ----
if st.session_state.snippets:
    st.subheader(f"🎥 Snippets ({len(st.session_state.snippets)}) & Classification")
    for idx, path in enumerate(st.session_state.snippets):
        label, ppos, pneg = classify_snippet(path)
        with st.expander(f"Snippet #{idx+1}: {os.path.basename(path)}"):
            st.video(open(path,"rb").read())
            st.markdown(
                f"**Predicted**: {label}  \n"
                f"• positive: {ppos:.1%}  \n"
                f"• negative: {pneg:.1%}"
            )
            st.download_button(
                f"Download Snippet #{idx+1}",
                open(path,"rb").read(),
                file_name=os.path.basename(path),
                mime="video/mp4"
            )
