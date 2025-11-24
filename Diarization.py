
# -------------------------
# Imports
# -------------------------
import os, csv, logging, warnings, signal, subprocess, tempfile, \
       multiprocessing as mp
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# Torch + audio processing stack
import torch, torchaudio, soundfile as sf
from tqdm import tqdm

# PyAnnote for state-of-the-art speaker diarization
from pyannote.audio import Pipeline

# Whisper for ASR transcription
import whisper

# ----------------------------- Config -------------------------------
# HuggingFace auth token for diarization model
HUGGINGFACE_TOKEN = "YOURTOKEN"

# Input directory containing raw MP3 or WAV recordings
INPUT_DIR   = "Input"
# Output directory to store diarization results and extracted speaker tracks
OUTPUT_DIR  = "Output"

# Limits number of files processed; None = all files
MAX_FILES   = None

# Whisper model name (medium.en fits well on GPU with ~5GB VRAM)
WHISPER_MODEL = "medium.en"
# Maximum diarization → Whisper transcription jobs running simultaneously
MAX_INFLIGHT  = 3

# CSV summary output files
SPEAKER_CSV     = os.path.join(OUTPUT_DIR, "speaker_report.csv")
SEGMENT_CSV     = os.path.join(OUTPUT_DIR, "segment_transcripts.csv")
SPEAKERFILE_CSV = os.path.join(OUTPUT_DIR, "speakerfile_transcripts.csv")

# Question-word detection for estimating which speaker asks questions
QUESTION_WORDS = {
    "how","what","when","where","why","do","does","did","are","is",
    "have","has","can","could","would","will","on","rate","scale",
}

# ------------------------- Logging / warnings -----------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("diarization.log"),
              logging.StreamHandler()],
)

# ---------------------- Whisper worker init -------------------------
# Whisper is loaded only once by a separate worker process.
WHISPER = None
def init_whisper_worker() -> None:
    """
    Loads a Whisper model on GPU-1.
    This avoids re-loading Whisper for each audio file.
    """
    global WHISPER
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    WHISPER = whisper.load_model(WHISPER_MODEL, device=dev)
    logging.info(f"[Whisper] medium.en on {dev}")

# ------------------ Timeout helper for Whisper ---------------------
# Whisper transcription occasionally gets stuck; use SIGALRM timeout.
class Timeout(Exception):
    pass
def _handle_alarm(_signum, _frame):
    raise Timeout
signal.signal(signal.SIGALRM, _handle_alarm)

def safe_transcribe(audio_tensor):
    """Safely run Whisper with a 5-minute timeout."""
    signal.alarm(300)
    try:
        txt = WHISPER.transcribe(
            audio_tensor,
            fp16=False, language="en", task="transcribe"
        )["text"].strip()
    finally:
        signal.alarm(0)
    return txt

# -------------------- MP3 → WAV conversion -------------------------
# Converts any MP3 into a 16kHz mono WAV file using ffmpeg.
def mp3_to_wav_16k(mp3_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", mp3_path,
        "-ac", "1", "-ar", "16000", tmp.name
    ]
    if subprocess.run(cmd).returncode != 0 or os.path.getsize(tmp.name) == 0:
        return None
    return tmp.name

# ------------------ Write extracted speaker tracks ------------------
def write_track(wf, sr, segs, path):
    """
    Concatenates diarization-based time segments and writes
    them into a new WAV file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not segs:
        logging.warning(f"[SKIP] empty track for {path}")
        return path, 0.0

    clip = torch.cat([wf[:, int(a*sr):int(b*sr)] for a, b in segs], dim=1)
    pcm = (clip.clamp(-1, 1) * 32767).to(torch.int16).squeeze().numpy()

    sf.write(path, pcm, sr, subtype="PCM_16")
    return path, pcm.shape[-1] / sr

# ------------------ Whisper on segment lists ------------------------
def whisper_segments(wav_path, sr, segs0, segs1, base):
    """
    Performs Whisper transcription on diarized segment
    and computes question ratios per speaker.
    """
    def analyse(seg_list):
        rows, q = [], 0
        for a, b in seg_list:
            clip, _ = torchaudio.load(
                wav_path, frame_offset=int(a*sr),
                num_frames=int((b-a)*sr))
            text = safe_transcribe(clip.squeeze().float().numpy())
            if text and (text.endswith("?")
                         or text.split()[0].lower() in QUESTION_WORDS):
                q += 1
            rows.append((f"{base}.wav", f"{a:.2f}", f"{b:.2f}", text))
        return q / len(seg_list) if seg_list else 0.0, rows

    qr0, r0 = analyse(segs0)
    qr1, r1 = analyse(segs1)
    return qr0, qr1, r0 + r1

# ---------- Whisper full speaker-only WAV to CSV --------------------
def fullfile_to_csv(path, label, orig_base):
    """
    Transcribes the entire extracted speaker track 
    and appends a row into SPEAKERFILE_CSV.
    """
    txt = safe_transcribe(torchaudio.load(path)[0].squeeze().float().numpy())
    info = torchaudio.info(path)
    dur = info.num_frames / info.sample_rate
    with open(SPEAKERFILE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{orig_base}.wav", label,
            os.path.basename(path), "0.00", f"{dur:.2f}", txt
        ])
        f.flush()

# --------------------------- Main -----------------------------------
def main():
    """Main batch-processing driver."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track previously processed files to allow resuming
    processed = set()
    if os.path.exists(SPEAKER_CSV):
        with open(SPEAKER_CSV) as f:
            for row in csv.DictReader(f):
                processed.add(f"{row['file']}|{row['label']}")

    # Ensure CSVs exist with header rows
    for p, hdr in [
        (SPEAKER_CSV, ["file","label","longest","first","most_turns","most_qratio",
                       "votes_for_nurse","nurse_confident",
                       "speaker0","speaker1",
                       "turns_spk0","turns_spk1","qr_spk0","qr_spk1"]),
        (SEGMENT_CSV, ["file","label","start","end","text"]),
        (SPEAKERFILE_CSV,["file","label","speaker_wav","start","end","text"]),
    ]:
        if not os.path.exists(p):
            with open(p, "w", newline="") as f:
                csv.writer(f).writerow(hdr)

    # Initialize PyAnnote diarization model
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    ).to(device0)
    pipe._segmentation.batch_size = 256
    pipe._embedding.batch_size    = 512

    # Whisper worker pool (1 worker keeps GPU VRAM usage controlled)
    wpool = ProcessPoolExecutor(max_workers=1, initializer=init_whisper_worker)

    inflight, futures = 0, {}

    # Build file list 
    audio_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        if os.path.commonpath([root, OUTPUT_DIR]) == OUTPUT_DIR:
            continue
        for fn in files:
            if fn.lower().endswith((".mp3", ".wav")):
                audio_files.append(os.path.join(root, fn))

    if MAX_FILES:
        audio_files = audio_files[:MAX_FILES]

    # Main diarization loop
    for path in tqdm(audio_files, desc="Diar"):
        label = "pos" if "/pos/" in path else "neg"
        base  = os.path.splitext(os.path.basename(path))[0]
        if f"{base}|{label}" in processed:
            continue

        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        # Convert files to mono 16k WAV for consistency
        if path.lower().endswith(".mp3"):
            wav = mp3_to_wav_16k(path)
            if wav is None:
                logging.error(f"[SKIP] bad MP3 {path}")
                continue
        else:
            wav = path

        # Skip tiny files
        if os.path.getsize(wav) < 3*16000*2:
            continue

        # ----- RUN DIARIZATION -----
        diar = pipe(wav, num_speakers=2)

        # Save RTTM output
        with open(os.path.join(out_dir, f"{base}.rttm"), "w") as fp:
            diar.write_rttm(fp)

        # Collect diarization metadata
        dur, turns, first_seen = defaultdict(float), defaultdict(int), {}
        first_lab, t0 = None, float("inf")
        for seg,_,lab in diar.itertracks(yield_label=True):
            if lab == "_": continue
            dur[lab]+=seg.duration
            turns[lab]+=1
            if seg.start<t0:
                first_lab,t0 = lab,seg.start
            if seg.start<first_seen.get(lab,float("inf")):
                first_seen[lab]=seg.start
        if len(dur) < 2:
            continue

        labs = sorted(dur,key=dur.get,reverse=True)[:2]
        labs.sort(key=lambda l:first_seen[l])
        spk0, spk1 = labs

        segs0 = [(s.start,s.end) for s,_,l in diar.itertracks(yield_label=True) if l==spk0]
        segs1 = [(s.start,s.end) for s,_,l in diar.itertracks(yield_label=True) if l==spk1]
        wf, sr = torchaudio.load(wav)

        # Transcribe segments asynchronously
        fut = wpool.submit(whisper_segments, wav, sr, segs0, segs1, base)
        futures[fut] = (label, base, out_dir, wf, sr,
                        segs0, segs1, dur, turns, first_lab, spk0, spk1)
        inflight += 1

        # Limit parallelism to keep Whisper GPU usage manageable
        while inflight >= MAX_INFLIGHT:
            done,_ = wait(futures, return_when=FIRST_COMPLETED)
            for d in done:
                handle_done(d, futures, wpool)
                inflight -= 1

    # Handle remaining futures
    for fut in list(futures):
        handle_done(fut, futures, wpool)

    wpool.shutdown()

# ----------------- Result handler ----------------------------------
def handle_done(fut, store, wpool):
    """
    Consumes a finished Whisper, computes nurse/patient labeling,
    extracts speaker tracks, and writes CSV rows.
    """
    (label, base, out_dir, wf, sr,
     segs0, segs1, dur, turns, first_lab, spk0, spk1) = store.pop(fut)

    try:
        qr0, qr1, seg_rows = fut.result()
    except Timeout:
        logging.error(f"[Whisper TIMEOUT] {base}")
        qr0 = qr1 = 0.0; seg_rows=[]
    except Exception as e:
        logging.error(f"[Whisper ERR] {base}: {e}")
        qr0 = qr1 = 0.0; seg_rows=[]

    # Voting scheme to infer who is the nurse (more questions, earlier speaking, etc.)
    votes = Counter([
        max(dur,key=dur.get),
        first_lab,
        max(turns,key=turns.get),
        spk0 if qr0>qr1 else spk1
    ])
    nurse_label, vote_ct = votes.most_common(1)[0]
    nurse_conf = vote_ct >= 3

    speaker_files=[]

    # If confident, write nurse and patient files explicitly
    if nurse_conf:
        nurse = nurse_label
        patient = spk1 if nurse==spk0 else spk0
        n_path,_ = write_track(wf,sr,segs0 if nurse==spk0 else segs1,
                               os.path.join(out_dir,f"{base}_nurse.wav"))
        p_path,_ = write_track(wf,sr,segs1 if nurse==spk0 else segs0,
                               os.path.join(out_dir,f"{base}_patient.wav"))
        speaker_files += [(n_path,label,base),(p_path,label,base)]
    else:
        # If not confident, output _speaker0 and _speaker1
        s0_path,_ = write_track(wf,sr,segs0,
                                os.path.join(out_dir,f"{base}_speaker0.wav"))
        s1_path,_ = write_track(wf,sr,segs1,
                                os.path.join(out_dir,f"{base}_speaker1.wav"))
        speaker_files += [(s0_path,label,base),(s1_path,label,base)]

    # Transcribe full speaker files
    for p,lbl,orig in speaker_files:
        wpool.submit(fullfile_to_csv, p, lbl, orig)

    # Append summary to speaker_report.csv
    with open(SPEAKER_CSV,"a",newline="") as f:
        csv.writer(f).writerow([
            base,label,max(dur,key=dur.get),first_lab,
            max(turns,key=turns.get),
            spk0 if qr0>qr1 else spk1,
            vote_ct,nurse_conf,
            spk0,spk1,turns[spk0],turns[spk1],
            f"{qr0:.2f}",f"{qr1:.2f}"
        ]); f.flush()
    with open(SEGMENT_CSV,"a",newline="") as f:
        w=csv.writer(f)
        for r in seg_rows:
            w.writerow([r[0],label,r[1],r[2],r[3]]); f.flush()

    logging.info(f"[DONE] {base} votes={vote_ct} conf={nurse_conf}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
    print("Processing finished.")

