#compressor.py
import os
import imageio.v3 as iio
import numpy as np
import subprocess

def make_compress(img: np.ndarray, img_name: str, kodek):
    img_name = img_name.replace(".png", "")
    path = "data/compress/"

    match(kodek):
        case "JPEG 2000":
            #Kompresja pliku do JPEG2000
            path = path+img_name+".jp2"
            iio.imwrite(path, img, format="JP2")
        case "JPEG XL":
            #Kompresja pliku do JPEGXL
            path = path+img_name+".jxl"
            iio.imwrite(path, img, format="JXL")
        case "HEVC":
            #Kompresja HEVC
            tmp_path = "data/tmp/"+img_name+".png"
            iio.imwrite(tmp_path, img)
            path = "data/compress/" + img_name + ".hevc"
            cmd = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-c:v", "libx265",
            "-preset", "slow",
            "-x265-params", "lossless=1",
            path
            ]
            subprocess.run(cmd, check=True)
        case "VVC":
            #Kompresja VVC
            tmp_path = "data/tmp/"+img_name+".png"
            iio.imwrite(tmp_path, img)
            yuv_path = "data/tmp/" + img_name + ".yuv"
            path = path + img_name + ".vvc"
            height, width = img.shape[:2]

            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-pix_fmt", "yuv420p",
                yuv_path
            ], check=True)

            cmd = [
                "vvencapp",
                "-i", yuv_path,
                "-s", f"{width}x{height}",
                "-o", path
            ]
            subprocess.run(cmd, check=True)

    size = os.path.getsize(path)
    bpp = size * 8 / (img.shape[0] * img.shape[1])
    compression_ratio = (img.nbytes) / size
    
    for f in [path] + ([tmp_path] if "tmp_path" in locals() else []):
        try:
            os.remove(f)
        except OSError:
            pass
    
    return {
        "size": size,
        "bpp": bpp,
        "compression_ratio": compression_ratio
    }