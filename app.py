import io
import os
import re
import zipfile
import numpy as np
import streamlit as st
import soundfile as sf

import torch
import lameenc

from qwen_tts import Qwen3TTSModel  # official package API


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


# -----------------------------
# Text chunking (10k+ chars)
# -----------------------------
def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\r\n", "\n", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\。\！\？\n])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def make_silence(sr: int, ms: int) -> np.ndarray:
    n = int(sr * (ms / 1000.0))
    return np.zeros(n, dtype=np.float32)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / max(peak, 1e-8)
    return x


# -----------------------------
# MP3 encoding (no ffmpeg)
# -----------------------------
def float_to_int16_pcm(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def encode_mp3_mono(audio_float32: np.ndarray, sr: int, bitrate_kbps: int = 192) -> bytes:
    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate_kbps))
    enc.set_in_sample_rate(int(sr))
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(float_to_int16_pcm(audio_float32))
    mp3 += enc.flush()
    return bytes(mp3)  # Streamlit requires bytes


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "chapter"


# -----------------------------
# Model loading (qwen-tts)
# -----------------------------
def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


@st.cache_resource(show_spinner=False)
def load_qwen_tts():
    device_map, dtype = pick_device_and_dtype()

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        dtype=dtype,
    )

    try:
        speakers = model.get_supported_speakers()
    except Exception:
        speakers = []

    try:
        languages = model.get_supported_languages()
    except Exception:
        languages = []

    return model, speakers, languages, device_map, str(dtype)


# -----------------------------
# Session state for persistent output
# -----------------------------
def init_state():
    if "out_single_name" not in st.session_state:
        st.session_state.out_single_name = None
    if "out_single_mp3" not in st.session_state:
        st.session_state.out_single_mp3 = None  # bytes

    if "out_batch_zip" not in st.session_state:
        st.session_state.out_batch_zip = None  # bytes
    if "out_batch_files" not in st.session_state:
        st.session_state.out_batch_files = []  # list of (name, bytes)

init_state()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Haseeb's TTS", layout="wide")
st.title("🎧 Haseeb's TTS")
st.caption("Audiobook Generator • MP3 Output • Batch Mode • Language • Voices • Instruction Control")

# Torch sanity check
try:
    _ = torch.tensor([1.0])
except Exception as e:
    st.error(f"PyTorch failed to initialize: {e}")
    st.stop()

with st.spinner("Loading model (first run can take a while)…"):
    tts_model, supported_speakers, supported_langs, device_map, dtype_str = load_qwen_tts()

colA, colB = st.columns([2, 1], gap="large")

with colB:
    st.subheader("Controls")
    st.caption(f"Device: `{device_map}` • dtype: `{dtype_str}`")

    fallback_langs = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
    lang_options = supported_langs if supported_langs else fallback_langs
    language = st.selectbox("Language", options=lang_options, index=0)

    fallback_speakers = ["Vivian", "Ryan"]
    spk_options = supported_speakers if supported_speakers else fallback_speakers
    speaker = st.selectbox("Speaker / Voice", options=spk_options, index=0)

    instruct = st.text_area(
        "Instruction (style/emotion/pacing)",
        value="Warm, clear narration. Medium pace. Slightly expressive.",
        height=90,
    ).strip()

    st.markdown("### Long Text Settings")
    max_chars = st.slider("Chunk size (characters)", 600, 3000, 1400, 100)
    gap_ms = st.slider("Silence between chunks (ms)", 0, 1200, 250, 50)

    st.markdown("### Generation Parameters")
    max_new_tokens = st.slider("max_new_tokens", 256, 8192, 4096, 256)

    st.markdown("### MP3 Export")
    mp3_bitrate = st.selectbox("MP3 bitrate (kbps)", [96, 128, 160, 192, 256, 320], index=3)
    do_normalize = st.checkbox("Normalize output audio", value=True)

    st.divider()
    if st.button("Clear Output", use_container_width=True):
        st.session_state.out_single_name = None
        st.session_state.out_single_mp3 = None
        st.session_state.out_batch_zip = None
        st.session_state.out_batch_files = []
        st.success("Output cleared.")


with colA:
    st.subheader("Input")

    mode = st.radio("Mode", ["Single chapter", "Batch (multiple .txt)"], horizontal=True)

    progress = st.progress(0)
    status = st.empty()

    def synth_one_mp3(text: str, label: str, base_prog: float, span_prog: float) -> bytes:
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            raise ValueError("No text chunks produced.")

        stitched = None
        sr_out = None

        for i, chunk in enumerate(chunks, start=1):
            status.write(f"{label}: chunk {i}/{len(chunks)}")

            wavs, sr = tts_model.generate_custom_voice(
                text=chunk,
                language=language if language else "Auto",
                speaker=speaker,
                instruct=instruct if instruct else "",
                max_new_tokens=int(max_new_tokens),
            )

            audio = np.asarray(wavs[0], dtype=np.float32)
            if do_normalize:
                audio = normalize_audio(audio)

            if stitched is None:
                stitched = audio
                sr_out = int(sr)
            else:
                if gap_ms > 0:
                    stitched = np.concatenate([stitched, make_silence(sr_out, gap_ms), audio])
                else:
                    stitched = np.concatenate([stitched, audio])

            frac = i / len(chunks)
            progress.progress(int((base_prog + frac * span_prog) * 100))

        return encode_mp3_mono(stitched, sr_out, bitrate_kbps=int(mp3_bitrate))

    # -------- Single --------
    if mode == "Single chapter":
        input_type = st.radio("Input type", ["Paste text", "Upload .txt"], horizontal=True)
        text = ""

        if input_type == "Paste text":
            text = st.text_area("Chapter text", height=380, placeholder="Paste your chapter text here…")
        else:
            f = st.file_uploader("Upload a .txt file", type=["txt"])
            if f is not None:
                text = f.read().decode("utf-8", errors="ignore")

        st.write(f"**Characters:** {len(text):,}")
        st.divider()

        if st.button("Generate MP3", type="primary", use_container_width=True):
            if not text.strip():
                st.error("Please provide some text.")
                st.stop()

            progress.progress(0)
            status.write("Starting…")

            try:
                mp3_bytes = synth_one_mp3(text, "Single", 0.0, 1.0)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

            status.write("✅ Done.")

            # Save to persistent output
            st.session_state.out_single_name = "audiobook_chapter.mp3"
            st.session_state.out_single_mp3 = bytes(mp3_bytes)

            # Clear batch output (optional)
            st.session_state.out_batch_zip = None
            st.session_state.out_batch_files = []

    # -------- Batch --------
    else:
        st.markdown("Upload multiple `.txt` files (each file = one chapter).")
        files = st.file_uploader("Upload chapter .txt files", type=["txt"], accept_multiple_files=True)

        st.divider()

        if st.button("Generate MP3s (Batch)", type="primary", use_container_width=True):
            if not files:
                st.error("Please upload at least one .txt file.")
                st.stop()

            progress.progress(0)
            status.write("Starting batch…")

            zip_buf = io.BytesIO()
            previews = []

            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                n = len(files)
                for idx, f in enumerate(files, start=1):
                    raw = f.read().decode("utf-8", errors="ignore")
                    base = sanitize_filename(os.path.splitext(f.name)[0])
                    mp3_name = f"{base}.mp3"

                    base_prog = (idx - 1) / n
                    span_prog = 1.0 / n

                    try:
                        mp3_bytes = synth_one_mp3(raw, f"{idx}/{n} {base}", base_prog, span_prog)
                    except Exception as e:
                        st.error(f"Failed on '{f.name}': {e}")
                        st.stop()

                    mp3_b = bytes(mp3_bytes)
                    zf.writestr(mp3_name, mp3_b)
                    previews.append((mp3_name, mp3_b))

            status.write("✅ Batch complete.")
            zip_bytes = zip_buf.getvalue()

            # Save to persistent output
            st.session_state.out_batch_zip = zip_bytes
            st.session_state.out_batch_files = previews

            # Clear single output (optional)
            st.session_state.out_single_name = None
            st.session_state.out_single_mp3 = None

    # -----------------------------
    # Persistent Output Panel
    # -----------------------------
    st.divider()
    st.subheader("Output")

    if (
        st.session_state.out_single_mp3 is None
        and st.session_state.out_batch_zip is None
        and len(st.session_state.out_batch_files) == 0
    ):
        st.info("No output yet. Generate audio and it will appear here.")
    else:
        # Single output
        if st.session_state.out_single_mp3 is not None:
            st.markdown("### Single Result")
            st.audio(st.session_state.out_single_mp3, format="audio/mp3")
            st.download_button(
                "Download MP3",
                data=st.session_state.out_single_mp3,
                file_name=st.session_state.out_single_name or "audiobook_chapter.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )

        # Batch output
        if st.session_state.out_batch_zip is not None:
            st.markdown("### Batch Results")
            st.download_button(
                "Download ZIP (all MP3s)",
                data=st.session_state.out_batch_zip,
                file_name="audiobook_mp3_batch.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.markdown("#### Individual MP3s")
            for name, mp3_b in st.session_state.out_batch_files:
                with st.expander(name, expanded=False):
                    st.audio(mp3_b, format="audio/mp3")
                    st.download_button(
                        f"Download {name}",
                        data=mp3_b,
                        file_name=name,
                        mime="audio/mpeg",
                        use_container_width=True,
                        key=f"dl_{name}",
                    )
