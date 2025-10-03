#compressor.py
import os
import imageio.v3 as iio
import numpy as np
import subprocess

def make_compress(img: np.ndarray, img_name: str):
    img_name = img_name.replace(".png", "")
    path = "data/compress/"
    
    #Kompresja pliku do JPEG2000
    jp2_path = path+img_name+".jp2"
    iio.imwrite(jp2_path, img, format="JP2")
    jp2_size = os.path.getsize(jp2_path)
    jp2_bpp = jp2_size * 8 / (img.shape[0] * img.shape[1])
    jp2_compression_ratio = (img.nbytes) / jp2_size

    #Kompresja pliku do JPEGXL
    jxl_path = path+img_name+".jxl"
    iio.imwrite(jxl_path, img, format="JXL")
    jxl_size = os.path.getsize(jxl_path)
    jxl_bpp = jxl_size * 8 / (img.shape[0] * img.shape[1])
    jxl_compression_ratio = (img.nbytes) / jxl_size

    tmp_path = "data/tmp/"+img_name+".png"
    iio.imwrite(tmp_path, img)

    #Kompresja HEVC
    hevc_path = "data/compress/" + img_name + ".hevc"
    cmd = [
    "ffmpeg", "-y",
    "-i", tmp_path,
    "-c:v", "libx265",
    "-preset", "slow",
    "-x265-params", "lossless=1",
    hevc_path
    ]
    subprocess.run(cmd, check=True)
    hevc_size = os.path.getsize(hevc_path)
    hevc_bpp = hevc_size * 8 / (img.shape[0] * img.shape[1])
    hevc_compression_ratio = (img.nbytes) / hevc_size

    #Kompresja VVC
    vvc_path = path + img_name + ".vvc"
    yuv_path = "data/tmp/" + img_name + ".yuv"
    height, width = img.shape[:2]

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-pix_fmt", "yuv420p",
        yuv_path
    ], check=True)

    cmd = [
        "vvencapp",
        "-i", tmp_path,
        "-s", f"{width}x{height}",
        "-o", vvc_path
    ]
    subprocess.run(cmd, check=True)
    vvc_size = os.path.getsize(vvc_path)
    vvc_bpp = vvc_size * 8 / (img.shape[0] * img.shape[1])
    vvc_compression_ratio = (img.nbytes) / vvc_size
    
    for f in [jp2_path, jxl_path, tmp_path, hevc_path, vvc_path]:
        try:
            os.remove(f)
        except OSError:
            pass
    
    return {
        "bpp_jp2": jp2_bpp,
        "compression_ratio_jp2": jp2_compression_ratio,
        "bpp_jxl": jxl_bpp,
        "compression_ratio_jxl": jxl_compression_ratio,
        "bpp_hevc": hevc_bpp,
        "compression_ratio_hevc": hevc_compression_ratio,
        "bpp_vvc": vvc_bpp,
        "compression_ratio_vvc": vvc_compression_ratio
    }