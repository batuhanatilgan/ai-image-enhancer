# Gurultu Azaltma Modulu\n# Non-Local Means algoritmasiyla goruntu gurultusunu azaltir

import cv2
import numpy as np


def denoise_image(image: np.ndarray, 
                  filter_strength: int = 10,
                  template_window_size: int = 7,
                  search_window_size: int = 21) -> np.ndarray:
    
    # Görüntünün renkli mi yoksa gri tonlamalı mı olduğunu kontrol et
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Renkli görüntü için fastNlMeansDenoisingColored kullan
        # Parametreler: src, dst, h, hForColorComponents, templateWindowSize, searchWindowSize
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            filter_strength,
            filter_strength,
            template_window_size,
            search_window_size
        )
    else:
        # Gri tonlamalı görüntü için fastNlMeansDenoising kullan
        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            filter_strength,
            template_window_size,
            search_window_size
        )
    
    return denoised


def denoise_video_frame(frame: np.ndarray,
                        prev_frames: list = None,
                        filter_strength: int = 4,
                        temporal_window_size: int = 5) -> np.ndarray:
    
    if prev_frames is None or len(prev_frames) < temporal_window_size - 1:
        # Yeterli önceki kare yoksa standart denoising uygula
        return denoise_image(frame, filter_strength=filter_strength)
    
    # Temporal denoising için kare listesi oluştur
    frames = prev_frames[-(temporal_window_size - 1):] + [frame]
    
    # OpenCV'nin temporal denoising fonksiyonu
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoisingColoredMulti(
            frames,
            imgToDenoiseIndex=len(frames) - 1,
            temporalWindowSize=temporal_window_size,
            h=filter_strength,
            hForColorComponents=filter_strength
        )
    else:
        denoised = cv2.fastNlMeansDenoisingMulti(
            frames,
            imgToDenoiseIndex=len(frames) - 1,
            temporalWindowSize=temporal_window_size,
            h=filter_strength
        )
    
    return denoised
