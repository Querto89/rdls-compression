# compressor.py
import os
import numpy as np
import imageio.v3 as iio
import subprocess


def make_compress(img: np.ndarray, img_name: str, kodek: str):
    img_name = img_name.replace(".png", "")
    path = "data/compressed/"
    os.makedirs(path, exist_ok=True)

    # Upewnij się, że obraz ma typ uint8 i zakres 0–255
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

    match kodek:
        # ------------------ JPEG 2000 ------------------
        case "JPEG 2000":
            out_path = os.path.join(path, img_name + ".jp2")
            try:
                import glymur
                if img.ndim == 2:
                    glymur.Jp2k(out_path, data=img)
                elif img.ndim == 3 and img.shape[2] in (3, 4):
                    glymur.Jp2k(out_path, data=img[:, :, :3])
                else:
                    raise ValueError("Nieobsługiwany format obrazu JP2.")
            except ImportError:
                print("[WARN] Brak biblioteki 'glymur' – używam ffmpeg do zapisu JP2.")
                _save_jpeg2000_ffmpeg(img, out_path)
            except Exception as e:
                print(f"[WARN] Nie udało się zapisać JPEG 2000 przez glymur ({e}) – używam ffmpeg.")
                _save_jpeg2000_ffmpeg(img, out_path)
            path = out_path

        # ------------------ JPEG XL ------------------
        case "JPEG XL":
            out_path = os.path.join(path, img_name + ".jxl")
            os.makedirs("data/tmp", exist_ok=True)
            tmp_png = f"data/tmp/tmp_jxl.png"
            iio.imwrite(tmp_png, img)

            try:
                # Użycie ffmpeg do JPEG XL
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", tmp_png,
                    "-c:v", "libjxl",
                    out_path
                ], check=True)
            except Exception as e:
                print(f"[ERROR] Nie udało się zapisać JPEG XL przez ffmpeg ({e}) – zapisuję jako PNG.")
                iio.imwrite(out_path.replace(".jxl", ".png"), img)
                out_path = out_path.replace(".jxl", ".png")
            finally:
                if os.path.exists(tmp_png):
                    os.remove(tmp_png)

            path = out_path

        # ------------------ HEVC ------------------
        case "HEVC":
            os.makedirs("data/tmp", exist_ok=True)
            tmp_path = f"data/tmp/{img_name}.png"
            out_path = os.path.join(path, img_name + ".hevc")
            iio.imwrite(tmp_path, img)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-c:v", "libx265",
                "-preset", "slow",
                "-x265-params", "lossless=1",
                out_path
            ], check=True)
            path = out_path
            os.remove(tmp_path)

        # ------------------ VVC ------------------
        case "VVC":
            os.makedirs("data/tmp", exist_ok=True)
            tmp_path = f"data/tmp/{img_name}.png"
            yuv_path = f"data/tmp/{img_name}.yuv"
            out_path = os.path.join(path, img_name + ".vvc")
            iio.imwrite(tmp_path, img)
            height, width = img.shape[:2]

            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-pix_fmt", "yuv420p",
                yuv_path
            ], check=True)

            subprocess.run([
                "vvencapp",
                "-i", yuv_path,
                "-s", f"{width}x{height}",
                "-o", out_path
            ], check=True)

            for f in (tmp_path, yuv_path):
                if os.path.exists(f):
                    os.remove(f)

            path = out_path

    # --- Obliczenia ---
    size = os.path.getsize(path)
    bpp = size * 8 / (img.shape[0] * img.shape[1])
    compression_ratio = img.nbytes / size

    return {
        "size": size,
        "bpp": bpp,
        "compression_ratio": compression_ratio
    }


def _save_jpeg2000_ffmpeg(img: np.ndarray, out_path: str):
    """Pomocnicza funkcja: zapis JP2 przez ffmpeg, jeśli glymur nie działa."""
    os.makedirs("data/tmp", exist_ok=True)
    tmp_png = "data/tmp/tmp_jp2.png"
    iio.imwrite(tmp_png, img)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_png,
            "-c:v", "jpeg2000",
            "-pix_fmt", "rgb24",
            out_path
        ], check=True)
    except Exception as e:
        print(f"[ERROR] ffmpeg nie zapisał JPEG2000: {e}")
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)
