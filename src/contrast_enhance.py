# Kontrast ve Parlaklik Iyilestirme Modulu
# CLAHE ve Gamma Correction teknikleriyle goruntu iyilestirme

import cv2
import numpy as np


def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    
    # BGR'den LAB renk uzayına dönüştür
    # LAB uzayı, parlaklık (L) ve renk (A, B) bilgisini ayırır
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # L kanalını ayır (parlaklık kanalı)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # CLAHE nesnesini oluştur
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    # CLAHE'yi sadece L kanalına uygula
    # Bu sayede renk bilgisi bozulmaz
    l_enhanced = clahe.apply(l_channel)
    
    # Kanalları birleştir
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # LAB'den BGR'ye geri dönüştür
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    
    # Lookup table oluştur (performans optimizasyonu)
    # Her 0-255 değeri için gamma dönüşümünü önceden hesapla
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    # Lookup table kullanarak gamma düzeltmesi uygula
    return cv2.LUT(image, table)


def auto_gamma_correction(image: np.ndarray, 
                          target_brightness: int = 128) -> np.ndarray:
    
    # Görüntüyü gri tonlamaya çevir ve ortalama parlaklığı hesapla
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Mevcut ortalama parlaklık
    current_brightness = np.mean(gray)
    
    # Sıfıra bölme hatasını önle
    if current_brightness == 0:
        current_brightness = 1
    
    # Otomatik gamma değeri hesapla
    # log(target/255) / log(current/255) formülü
    gamma = np.log(target_brightness / 255.0) / np.log(current_brightness / 255.0)
    
    # Gamma değerini makul aralıkta tut
    gamma = np.clip(gamma, 0.3, 3.0)
    
    return apply_gamma_correction(image, gamma)


def enhance_contrast_and_brightness(image: np.ndarray,
                                    clahe_clip_limit: float = 2.0,
                                    gamma: float = None,
                                    auto_brightness: bool = True) -> np.ndarray:
    
    # Adım 1: CLAHE ile kontrast iyileştir
    result = apply_clahe(image, clip_limit=clahe_clip_limit)
    
    # Adım 2: Gamma düzeltmesi
    if gamma is not None:
        result = apply_gamma_correction(result, gamma)
    elif auto_brightness:
        result = auto_gamma_correction(result)
    
    return result
