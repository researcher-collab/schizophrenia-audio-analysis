
# -------------------------
# Imports
# -------------------------

import os
import sys
import csv
import logging
import warnings
from typing import Dict, List

import numpy as np
import librosa
from scipy.signal import find_peaks

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from torchvggish import vggish, vggish_input
import opensmile
import requests
import parselmouth
import re

# ------------------------------------------------------------------
# Device & precision
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    filename="audio_feature_extraction.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# =========================
# Utility Functions
# =========================

def ensure_array(value):
    """
    Ensure `value` is a 1D numpy array.

    Parameters
    ----------
    value : Any

    Returns
    -------
    np.ndarray
    """
    if isinstance(value, (np.ndarray, list)):
        return np.array(value)
    return np.array([value])


def pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply a simple pre-emphasis filter to boost high frequencies.

    y[n] = x[n] - coeff * x[n-1]

    Parameters
    ----------
    audio : np.ndarray
        1D audio signal.
    coeff : float
        Pre-emphasis coefficient.

    Returns
    -------
    np.ndarray
        Pre-emphasized audio.
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def load_audio(audio_filename: str, sr: int = 16000):
    """
    Load an audio file as mono at the desired sampling rate.

    Parameters
    ----------
    audio_filename : str
        Path to audio file (WAV/MP3/etc.).
    sr : int
        Target sampling rate (Hz).

    Returns
    -------
    audio : np.ndarray
        1D mono waveform.
    sr : int
        Sampling rate actually used.

    Raises
    ------
    ValueError
        If the audio is empty.
    Exception
        Any load error is logged and re-raised.
    """
    try:
        logging.info(f"Loading audio file: {audio_filename}")
        audio, _ = librosa.load(audio_filename, sr=sr)
        if len(audio) == 0:
            raise ValueError("Empty audio data")
        logging.info(f"Audio loaded successfully, length: {len(audio)} samples.")
        return audio, sr
    except Exception as e:
        logging.error(f"Error loading {audio_filename}: {e}")
        raise


def download_vggish_model_files():
    """
    Download VGGish model and PCA parameter files if missing,
    and set environment variables for torchvggish.

    This is only done once per environment and cached locally.
    """
    model_url = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
    pca_params_url = "https://storage.googleapis.com/audioset/vggish_pca_params.npz"
    model_path = "vggish_model.ckpt"
    pca_params_path = "vggish_pca_params.npz"

    if not os.path.exists(model_path):
        logging.info("Downloading VGGish model file …")
        with open(model_path, "wb") as f:
            f.write(requests.get(model_url).content)
        logging.info("VGGish model file downloaded.")

    if not os.path.exists(pca_params_path):
        logging.info("Downloading VGGish PCA parameters file …")
        with open(pca_params_path, "wb") as f:
            f.write(requests.get(pca_params_url).content)
        logging.info("VGGish PCA parameters file downloaded.")

    os.environ["VGGISH_MODEL"] = model_path
    os.environ["VGGISH_PCA_PARAMS"] = pca_params_path


def calculate_n_fft(audio_length: int, desired_n_fft: int = 2048) -> int:
    """
    Choose an n_fft that fits within the audio length.

    For very short clips, this picks the largest power of two not exceeding
    the audio length, with a minimum of 32.

    Parameters
    ----------
    audio_length : int
        Number of samples in the audio.
    desired_n_fft : int
        Default FFT size for typical audio.

    Returns
    -------
    int
        Selected n_fft.
    """
    def next_power_of_two(x):
        return 2 ** int(np.floor(np.log2(x)))

    if audio_length >= desired_n_fft:
        return desired_n_fft
    max_n_fft = next_power_of_two(audio_length)
    return max_n_fft if max_n_fft >= 32 else audio_length


def pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Zero-pad or truncate a signal to exactly `target_length`.

    Parameters
    ----------
    audio : np.ndarray
        1D signal.
    target_length : int
        Desired length in samples.

    Returns
    -------
    np.ndarray
    """
    pad = target_length - len(audio)
    return np.pad(audio, (0, pad), mode="constant") if pad > 0 else audio


# ------------------------------------------------------------------
# Feature helpers (MFCC, spectral, pitch, etc.)
# ------------------------------------------------------------------

def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = None,
) -> Dict[str, float]:
    """
    Compute MFCC mean and standard deviation across time.

    Returns:
        MFCC_mean_0..n_mfcc-1 and MFCC_std_0..n_mfcc-1
    """
    mfcc_features: Dict[str, float] = {}
    try:
        if hop_length is None:
            hop_length = max(1, n_fft // 2)

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        if mfcc.shape[1] < 1:
            raise ValueError("MFCC empty")

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        for i in range(n_mfcc):
            mfcc_features[f"MFCC_mean_{i}"] = float(mfcc_mean[i])
            mfcc_features[f"MFCC_std_{i}"] = float(mfcc_std[i])
    except Exception as e:
        logging.error(f"Error in MFCC extraction: {e}")
        for i in range(n_mfcc):
            mfcc_features[f"MFCC_mean_{i}"] = np.nan
            mfcc_features[f"MFCC_std_{i}"] = np.nan
    return mfcc_features


def extract_spectral_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = None,
) -> Dict[str, float]:
    """
    Extract summary spectral features: HNR, loudness, spectral flux, ZCR,
    alpha ratio, Hammarberg index, and RMS.

    Returns
    -------
    dict
        Keys: HNR, Loudness, Spectral_Flux, ZCR, Alpha_Ratio,
              Hammarberg_Index, RMS
    """
    feats = {k: np.nan for k in [
        "HNR", "Loudness", "Spectral_Flux", "ZCR",
        "Alpha_Ratio", "Hammarberg_Index", "RMS"
    ]}
    try:
        if hop_length is None:
            hop_length = max(1, n_fft // 2)

        S = np.abs(librosa.stft(audio, n_fft=n_fft,
                                hop_length=hop_length, window="hamming"))
        if S.shape[1] == 0:
            return feats

        # Harmonic/noise decomposition for a simple HNR proxy
        y_h, y_p = librosa.effects.hpss(audio)
        feats["HNR"] = (
            librosa.feature.rms(
                y=y_h, frame_length=n_fft, hop_length=hop_length
            ).mean()
            / (librosa.feature.rms(
                y=y_p, frame_length=n_fft, hop_length=hop_length
            ).mean() + 1e-10)
        )

        feats["Loudness"] = librosa.feature.rms(
            y=audio, frame_length=n_fft, hop_length=hop_length
        ).mean()

        onset_env = librosa.onset.onset_strength(
            S=S, sr=sr, hop_length=hop_length
        )
        feats["Spectral_Flux"] = float(onset_env.mean())

        feats["ZCR"] = float(librosa.feature.zero_crossing_rate(
            audio, frame_length=n_fft, hop_length=hop_length
        )[0].mean())

        # Alpha ratio: <1 kHz vs >1 kHz energy
        freq_1k = int(1000 * n_fft / sr)
        if freq_1k < S.shape[0]:
            feats["Alpha_Ratio"] = float(
                (S[:freq_1k, :].sum(0) / (S[freq_1k:, :].sum(0) + 1e-10)).mean()
            )

        # Hammarberg index: <2 kHz vs >2 kHz spectral peak
        freq_2k = int(2000 * n_fft / sr)
        if freq_2k < S.shape[0]:
            feats["Hammarberg_Index"] = float(
                (S[:freq_2k, :].max(0) / (S[freq_2k:, :].max(0) + 1e-10)).mean()
            )

        feats["RMS"] = float(librosa.feature.rms(
            y=audio, frame_length=n_fft, hop_length=hop_length
        )[0].mean())
    except Exception as e:
        logging.error(f"Error in spectral feature extraction: {e}")
    return feats


def extract_pitch_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = None,
) -> Dict[str, float]:
    """
    Pitch statistics using YIN fundamental frequency estimation.

    Returns
    -------
    dict
        Keys: Pitch_mean, Pitch_std
    """
    feats = {"Pitch_mean": np.nan, "Pitch_std": np.nan}
    try:
        if hop_length is None:
            hop_length = max(1, n_fft // 2)
        f0 = librosa.yin(
            audio, fmin=50, fmax=500, sr=sr,
            frame_length=n_fft, hop_length=hop_length
        )
        f0 = f0[f0 > 0]
        if f0.size:
            feats["Pitch_mean"] = float(f0.mean())
            feats["Pitch_std"] = float(f0.std())
    except Exception as e:
        logging.error(f"Error in pitch feature extraction: {e}")
    return feats


def extract_jitter_shimmer(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Estimate jitter and shimmer using Praat via parselmouth.

    Returns
    -------
    dict
        Keys: Jitter, Shimmer (Praat local measures)
        NaN if fully unvoiced or failure.
    """
    feats = {"Jitter": np.nan, "Shimmer": np.nan}
    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)

        # Default Praat voice range
        pitch = snd.to_pitch(
            time_step=0.0,
            voicing_threshold=0.7,
            pitch_floor=75,
            pitch_ceiling=500,
        )

        pp = parselmouth.praat.call(
            [snd, pitch],
            "To PointProcess (periodic, cc)", 75, 500
        )

        feats["Jitter"] = float(parselmouth.praat.call(
            pp, "Get jitter (local)",
            0, 0,       # whole signal
            0.0001,     # min pitch period
            0.02,       # max pitch period
            1.3         # max period factor
        ))

        feats["Shimmer"] = float(parselmouth.praat.call(
            [snd, pp], "Get shimmer (local)",
            0, 0,
            0.0001, 0.02,
            1.3, 1.6
        ))

    except Exception as e:
        logging.error(f"Error in jitter/shimmer extraction: {e}")

    return feats


def extract_formant_features(
    audio: np.ndarray,
    sr: int = 16000,
    order: int = 8,
    n_fft: int = 2048,
    hop_length: int = None,
) -> Dict[str, float]:
    """
    LPC-based formant estimation: mean F1, F2, F3 across frames.

    Returns
    -------
    dict
        Keys: F1_mean, F2_mean, F3_mean
    """
    feats = {"F1_mean": np.nan, "F2_mean": np.nan, "F3_mean": np.nan}
    try:
        if hop_length is None:
            hop_length = max(1, n_fft // 2)
        frames = librosa.util.frame(
            audio, frame_length=n_fft, hop_length=hop_length
        ).T
        if frames.shape[0] == 0:
            return feats

        window = np.hamming(n_fft)
        stack: List[np.ndarray] = []

        for fr in frames:
            fr = (fr - fr.mean()) * window
            a = librosa.lpc(fr, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angs = np.angle(roots)
            form = np.sort(angs * sr / (2 * np.pi))
            form = form[(form > 90) & (form < 5000)]
            if len(form) >= 3:
                stack.append(form[:3])

        if stack:
            arr = np.vstack(stack)
            feats["F1_mean"], feats["F2_mean"], feats["F3_mean"] = arr.mean(0)
    except Exception as e:
        logging.error(f"Error in formant extraction: {e}")
    return feats


def estimate_speaking_rate(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Estimate speaking rate using energy peaks as a proxy for syllables.

    Returns
    -------
    dict
        Keys: Speaking_Rate (syllables/s), Num_Syllables (count)
    """
    feats = {"Speaking_Rate": np.nan, "Num_Syllables": 0}
    try:
        n_fft = calculate_n_fft(len(audio))
        hop = n_fft // 2
        energy = librosa.feature.rms(
            y=audio, frame_length=n_fft, hop_length=hop
        )[0]
        peaks, _ = find_peaks(energy, prominence=0.01)
        num = len(peaks)
        feats["Num_Syllables"] = int(num)
        feats["Speaking_Rate"] = float(
            num / (len(audio) / sr + 1e-10)
        )
    except Exception as e:
        logging.error(f"Error in speaking rate estimation: {e}")
    return feats


def extract_pause_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract pause statistics using energy-based voice activity detection.

    Returns
    -------
    dict
        Keys:
            Total_Pause_Duration (seconds),
            Num_Pauses (count),
            Pause_Rate (pauses/second)
    """
    feats = {
        "Total_Pause_Duration": np.nan,
        "Num_Pauses": 0,
        "Pause_Rate": np.nan,
    }
    try:
        ivl = librosa.effects.split(audio, top_db=20)
        total = len(audio) / sr
        if ivl.size == 0:
            return feats

        voiced = (ivl[:, 1] - ivl[:, 0]).sum() / sr
        pauses = total - voiced
        n_p = max(0, len(ivl) - 1)
        feats["Total_Pause_Duration"] = float(pauses)
        feats["Num_Pauses"] = int(n_p)
        feats["Pause_Rate"] = float(n_p / (total + 1e-10))
    except Exception as e:
        logging.error(f"Error in pause feature extraction: {e}")
    return feats


# ------------------------------------------------------------------
# Model preload (GPU-aware)
# ------------------------------------------------------------------

GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")


def preload_models():
    """
    Load HuBERT (on GPU if available) and VGGish (on CPU).

    Returns
    -------
    hubert : HubertModel
        HuBERT base model in eval mode.
    vg : torch.nn.Module
        VGGish model in eval mode with `vg.device` set.
    """
    logging.info("Loading HuBERT …")
    hubert = (
        HubertModel
        .from_pretrained("facebook/hubert-base-ls960")
        .to(GPU)
        .eval()
    )

    logging.info("Loading VGGish (CPU) …")
    download_vggish_model_files()
    vg = vggish().eval()  # VGGish is lightweight, kept on CPU
    vg.device = CPU       # torchvggish uses this attribute in forward()
    return hubert, vg


# ------------------------------------------------------------------
# HuBERT & VGGish extractors
# ------------------------------------------------------------------

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)


@torch.no_grad()
def extract_hubert_features(
    audio: np.ndarray,
    hubert_model,
    sr: int = 16000,
    layer: int = 6,
) -> Dict[str, float]:
    """
    Extract HuBERT hidden representation (mean pooled) from a specific layer.

    Returns
    -------
    dict
        Keys: HuBERT_mean_0..767
    """
    feats = {f"HuBERT_mean_{i}": np.nan for i in range(768)}
    try:
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
            sr = 16000

        # Ensure minimum duration for HuBERT (4k samples ≈ 0.25s at 16 kHz)
        audio = pad_audio(audio, 4000)

        inp = feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values.to(device)

        hidden = hubert_model(
            inp,
            output_hidden_states=True
        ).hidden_states[layer]

        vec = hidden.squeeze(0).mean(0).cpu().numpy()
        for i, v in enumerate(vec):
            feats[f"HuBERT_mean_{i}"] = float(v)
    except Exception as e:
        logging.error(f"Error in HuBERT extraction: {e}")
    return feats


@torch.no_grad()
def extract_vggish_features(
    audio: np.ndarray,
    vggish_model,
    sr: int = 16000,
) -> Dict[str, float]:
    """
    Compute VGGish embedding (mean across patches).

    Returns
    -------
    dict
        Keys: VGGish_mean_0..127
    """
    feats = {f"VGGish_mean_{i}": np.nan for i in range(128)}
    try:
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
            sr = 16000

        # VGGish expects ~0.96 s chunks
        audio = pad_audio(audio, int(0.96 * sr))

        batch = vggish_input.waveform_to_examples(audio, sr)
        if batch.shape[0] == 0:
            return feats

        inp = torch.as_tensor(
            batch,
            dtype=torch.float32,
            device=vggish_model.device,
        )
        emb = vggish_model(inp).cpu().numpy()
        vec = emb.mean(0)
        for i, v in enumerate(vec):
            feats[f"VGGish_mean_{i}"] = float(v)
    except Exception as e:
        logging.error(f"Error in VGGish extraction: {e}")
    return feats


# ------------------------------------------------------------------
# OpenSMILE (eGeMAPS) features
# ------------------------------------------------------------------

def extract_opensmile_features(audio_file: str) -> Dict[str, float]:
    """
    Extract eGeMAPS functionals via OpenSMILE.

    Returns
    -------
    dict
        All OpenSMILE features for the file, plus:
          Jitter  – jitterLocal_sma3nz_amean
          Shimmer – shimmerLocaldB_sma3nz_amean
    """
    feat: Dict[str, float] = {}
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        res = smile.process_file(audio_file)
        row = res.iloc[0]

        # copy everything
        feat.update(row.to_dict())

        # Expose canonical names for jitter & shimmer
        feat["Jitter"] = row.get(
            "jitterLocal_sma3nz_amean", np.nan
        )
        feat["Shimmer"] = row.get(
            "shimmerLocaldB_sma3nz_amean", np.nan
        )

    except Exception as e:
        logging.error(f"Error in OpenSMILE extraction: {e}")
    return feat


# ------------------------------------------------------------------
# Filename parsing (ID, race, gender, role)
# ------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"""
    ^
    (?P<id>[^_]+)           # 1 – arbitrary ID (no "_")
    _
    (?P<race>[BW])          # 2 – B or W
    _
    (?P<gender>[MF])        # 3 – M or F
    (?:_
       (?P<role>[^_]+)      # 4 – nurse|patient|speaker0… or similar
    )?
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ------------------------------------------------------------------
# Full per-file pipeline
# ------------------------------------------------------------------

def extract_features(
    audio_file: str,
    root_dir: str,
    hubert_model,
    vggish_model,
    mfcc_coeffs: int = 13,
    desired_n_fft: int = 2048,
) -> Dict[str, float]:
    """
    Run the full feature extraction pipeline on a single audio file.

    The filename is parsed as:
        <id>_<race>_<gender>[_<role>].ext

    Examples:
        12345_B_M_patient.wav
        98765_W_F_nurse.wav

    Parameters
    ----------
    audio_file : str
        Path to audio file.
    root_dir : str
        Root directory for relative path computation.
    hubert_model : HubertModel
    vggish_model : torch.nn.Module
    mfcc_coeffs : int
        Number of MFCC coefficients.
    desired_n_fft : int
        Default FFT window size.

    Returns
    -------
    dict
        All extracted features for this file.
    """
    basename = os.path.basename(audio_file)
    stem = os.path.splitext(basename)[0]
    m = FILENAME_RE.match(stem)

    # Graceful fallback if the name does not match the pattern
    id_ = m.group("id") if m else stem
    race = m.group("race") if m else "U"        # U = Unknown
    gender = m.group("gender") if m else "U"
    role = m.group("role") if m else "unspecified"

    feats: Dict[str, float] = {
        "File": os.path.relpath(audio_file, root_dir),
        "ID": id_,
        "Race": race.upper(),
        "Gender": gender.upper(),
        "Role": role.lower(),
    }

    try:
        audio, sr = load_audio(audio_file)
        audio = pre_emphasis(audio)

        n_fft = calculate_n_fft(len(audio), desired_n_fft)
        hop = n_fft // 2

        # Classical acoustic features
        feats.update(extract_mfcc(audio, sr, mfcc_coeffs, n_fft, hop))
        feats.update(extract_spectral_features(audio, sr, n_fft, hop))
        feats.update(extract_pitch_features(audio, sr, n_fft, hop))
        feats.update(extract_jitter_shimmer(audio, sr))
        feats.update(extract_formant_features(audio, sr, 8, n_fft, hop))
        feats.update(estimate_speaking_rate(audio, sr))
        feats.update(extract_pause_features(audio, sr))

        # Deep embeddings
        feats.update(extract_hubert_features(audio, hubert_model, sr))
        feats.update(extract_vggish_features(audio, vggish_model, sr))

        # OpenSMILE (file-based)
        feats.update(extract_opensmile_features(audio_file))

        logging.info(f"Feature extraction completed for {audio_file}")
    except Exception as e:
        logging.error(f"Unhandled exception while extracting {audio_file}: {e}")

    return feats


# ------------------------------------------------------------------
# Batch driver
# ------------------------------------------------------------------

def process_audio_files(
    root_dir: str,
    hubert_model,
    vggish_model,
    output_csv: str = "extracted_features.csv",
    mfcc_coeffs: int = 13,
    desired_n_fft: int = 2048,
):
    """
    Walk `root_dir`, find all WAV/MP3 files, extract features,
    and write them to a CSV.

    Parameters
    ----------
    root_dir : str
        Root directory to search for audio files.
    hubert_model : HubertModel
    vggish_model : torch.nn.Module
    output_csv : str
        Destination CSV file path.
    mfcc_coeffs : int
    desired_n_fft : int
    """
    audio_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(root_dir)
        for f in files
        if f.lower().endswith((".wav", ".mp3"))
    ]
    total = len(audio_files)
    logging.info(f"{total} audio files found in {root_dir}")

    # Base columns
    fieldnames = ["File", "ID", "Race", "Gender", "Role"]

    # Probe first file to discover all feature keys
    if total:
        probe = extract_features(
            audio_files[0],
            root_dir,
            hubert_model,
            vggish_model,
            mfcc_coeffs,
            desired_n_fft,
        )
        fieldnames += [k for k in probe if k not in fieldnames]

    # Open once, stream rows, and force flush to disk after each write
    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        cf.flush()
        os.fsync(cf.fileno())

        for idx, path in enumerate(audio_files, 1):
            logging.info(f"Processing {idx}/{total}: {path}")
            row = extract_features(
                path,
                root_dir,
                hubert_model,
                vggish_model,
                mfcc_coeffs,
                desired_n_fft,
            )
            writer.writerow(row)

            # Ensure progress is not lost on crash / interruption
            cf.flush()
            os.fsync(cf.fileno())

            print(f"Processed {idx}/{total}: {path}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Load models once (shared across all files)
        hubert_model, vggish_model = preload_models()

        # TODO: UPDATE THESE PATHS FOR YOUR ENVIRONMENT
        ROOT_DIR = "ROOT"
        OUTPUT_CSV = "OUT.csv"

        process_audio_files(
            root_dir=ROOT_DIR,
            hubert_model=hubert_model,
            vggish_model=vggish_model,
            output_csv=OUTPUT_CSV,
            mfcc_coeffs=13,
            desired_n_fft=2048,
        )
        logging.info("All audio files processed successfully.")
        print("All audio files processed successfully.")
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        print(f"Script terminated due to: {e}")
