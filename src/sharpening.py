# Keskinlestirme Modulu
# Unsharp Masking teknigi ile goruntu keskinlestirme

import cv2
import numpy as np


def unsharp_mask(image: np.ndarray,
                 kernel_size: tuple = (5, 5),
                 sigma: float = 1.0,
                 amount: float = 1.5,
                 threshold: int = 0) -> np.ndarray:
    
    # Adım 1: Gaussian blur uygula (bulanık versiyon oluştur)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Adım 2: Orijinal ve bulanık görüntü arasındaki farkı hesapla
    # Bu fark, kenar bilgisini içerir
    if threshold > 0:
        # Eşik değeri varsa, küçük farkları yoksay (gürültü filtreleme)
        diff = image.astype(np.float32) - blurred.astype(np.float32)
        mask = np.abs(diff) > threshold
        diff = diff * mask
    else:
        diff = image.astype(np.float32) - blurred.astype(np.float32)
    
    # Adım 3: Keskinleştirilmiş görüntüyü oluştur
    # sharpened = original + amount * (original - blurred)
    sharpened = image.astype(np.float32) + amount * diff
    
    # Değerleri 0-255 aralığına sınırla ve uint8'e dönüştür
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def laplacian_sharpening(image: np.ndarray, 
                         strength: float = 1.0) -> np.ndarray:
    
    # Görüntüyü float'a dönüştür
    image_float = image.astype(np.float32)
    
    # Laplacian filtresi uygula
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
    
    # Keskinleştirilmiş görüntü = orijinal - strength * laplacian
    # (Laplacian negatif kenarlar için negatif değer verir)
    sharpened = image_float - strength * laplacian
    
    # Değerleri sınırla ve uint8'e dönüştür
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def kernel_sharpening(image: np.ndarray, 
                      intensity: str = "medium") -> np.ndarray:
    
    # Keskinleştirme kernel'ları
    kernels = {
        "light": np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32),
        
        "medium": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        
        "strong": np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
    }
    
    kernel = kernels.get(intensity, kernels["medium"])
    
    # Konvolüsyon uygula
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def adaptive_sharpening(image: np.ndarray,
                        blur_threshold: float = 100.0) -> np.ndarray:
    
    # Bulanıklık seviyesini Laplacian varyansı ile hesapla
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Bulanıklık seviyesine göre keskinleştirme miktarını belirle
    if laplacian_var < blur_threshold * 0.5:
        # Çok bulanık - güçlü keskinleştirme
        amount = 2.5
    elif laplacian_var < blur_threshold:
        # Orta bulanık - orta keskinleştirme
        amount = 1.5
    else:
        # Az bulanık veya keskin - hafif keskinleştirme
        amount = 0.8
    
    return unsharp_mask(image, amount=amount)
