# Yardimci Fonksiyonlar Modulu
# Goruntu yukleme, kaydetme ve analiz fonksiyonlari

import cv2
import numpy as np
import os
from typing import Tuple, List


def load_image(path: str) -> np.ndarray:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {path}")
    
    # RGBA görüntüleri de okuyabilmek için UNCHANGED kullan
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    
    # RGBA görüntüyü BGR'ye dönüştür (alpha kanalını kaldır)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # Gri tonlamalı görüntüyü BGR'ye dönüştür
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image


def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    
    # Dizin yoksa oluştur
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Dosya uzantısına göre kaydet
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 10)]
    else:
        params = []
    
    success = cv2.imwrite(path, image, params)
    
    if success:
        print(f"[INFO] Görüntü kaydedildi: {path}")
    else:
        print(f"[HATA] Görüntü kaydedilemedi: {path}")
    
    return success


def get_image_info(image: np.ndarray) -> dict:
    
    info = {
        "shape": image.shape,
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
        "size_mb": round(image.nbytes / (1024 * 1024), 2),
        "min_value": int(np.min(image)),
        "max_value": int(np.max(image)),
        "mean_value": round(float(np.mean(image)), 2)
    }
    return info


def calculate_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    
    # Görüntüleri aynı boyuta getir
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    return cv2.PSNR(original, enhanced)


def calculate_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    
    # Görüntüleri aynı boyuta getir
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Gri tonlamaya çevir
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        enhanced_gray = enhanced
    
    # SSIM parametreleri
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Gaussian blur uygula
    original_gray = original_gray.astype(np.float64)
    enhanced_gray = enhanced_gray.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(original_gray, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(enhanced_gray, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(original_gray ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(enhanced_gray ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(original_gray * enhanced_gray, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def estimate_blur_level(image: np.ndarray) -> Tuple[float, str]:
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 50:
        description = "Çok bulanık"
    elif laplacian_var < 100:
        description = "Bulanık"
    elif laplacian_var < 500:
        description = "Orta keskinlik"
    else:
        description = "Keskin"
    
    return laplacian_var, description


def estimate_noise_level(image: np.ndarray) -> Tuple[float, str]:
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Yüksek frekans bileşenlerini al (gürültü genelde yüksek frekansta)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(np.float64) - blurred.astype(np.float64))
    noise_level = np.std(noise)
    
    if noise_level < 3:
        description = "Gürültüsüz"
    elif noise_level < 8:
        description = "Az gürültülü"
    elif noise_level < 15:
        description = "Orta gürültülü"
    else:
        description = "Çok gürültülü"
    
    return noise_level, description


def estimate_brightness(image: np.ndarray) -> Tuple[float, str]:
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 50:
        description = "Çok karanlık"
    elif mean_brightness < 100:
        description = "Karanlık"
    elif mean_brightness < 180:
        description = "Normal"
    else:
        description = "Parlak"
    
    return mean_brightness, description


def analyze_image(image: np.ndarray) -> dict:
    
    blur_val, blur_desc = estimate_blur_level(image)
    noise_val, noise_desc = estimate_noise_level(image)
    bright_val, bright_desc = estimate_brightness(image)
    
    return {
        "basic_info": get_image_info(image),
        "blur": {"value": blur_val, "description": blur_desc},
        "noise": {"value": noise_val, "description": noise_desc},
        "brightness": {"value": bright_val, "description": bright_desc}
    }


def list_images_in_directory(directory: str, 
                              extensions: List[str] = None) -> List[str]:
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    extensions = [ext.lower() for ext in extensions]
    
    image_files = []
    
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)
