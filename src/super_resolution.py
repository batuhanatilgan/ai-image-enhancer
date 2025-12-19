# Super Cozunurluk Modulu
# OpenCV DNN ile goruntu cozunurlugunu artirma (EDSR, FSRCNN)

import cv2
import numpy as np
import os
import urllib.request


# Model indirme URL'leri (resmi OpenCV modelleri)
MODEL_URLS = {
    "edsr": {
        2: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
        3: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb",
        4: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
    },
    "fsrcnn": {
        2: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
        3: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
        4: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
    },
    "espcn": {
        2: "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
        3: "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
        4: "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
    },
    "lapsrn": {
        2: "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb",
        4: "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb",
        8: "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb"
    }
}


def download_model(model_name: str, scale: int, models_dir: str) -> str:
    
    model_name = model_name.lower()
    
    if model_name not in MODEL_URLS:
        raise ValueError(f"Desteklenmeyen model: {model_name}. "
                        f"Desteklenen modeller: {list(MODEL_URLS.keys())}")
    
    if scale not in MODEL_URLS[model_name]:
        raise ValueError(f"{model_name} için desteklenmeyen ölçek: {scale}. "
                        f"Desteklenen ölçekler: {list(MODEL_URLS[model_name].keys())}")
    
    # Model dosya adı
    filename = f"{model_name.upper()}_x{scale}.pb"
    model_path = os.path.join(models_dir, filename)
    
    # Model zaten varsa indirme
    if os.path.exists(model_path):
        print(f"[INFO] Model zaten mevcut: {model_path}")
        return model_path
    
    # Dizin yoksa oluştur
    os.makedirs(models_dir, exist_ok=True)
    
    # Modeli indir
    url = MODEL_URLS[model_name][scale]
    print(f"[INFO] Model indiriliyor: {url}")
    print(f"[INFO] Hedef: {model_path}")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Model başarıyla indirildi: {model_path}")
    except Exception as e:
        print(f"[HATA] Model indirilemedi: {e}")
        raise
    
    return model_path


class SuperResolution:
    
    
    def __init__(self, 
                 model_name: str = "edsr",
                 scale: int = 2,
                 models_dir: str = "./models"):
        
        self.model_name = model_name.lower()
        self.scale = scale
        self.models_dir = models_dir
        self.sr = None
        
        # Modeli yükle
        self._load_model()
    
    def _load_model(self):
        
        # dnn_superres modülünün varlığını kontrol et
        if not hasattr(cv2, 'dnn_superres'):
            raise ImportError(
                "OpenCV dnn_superres modülü bulunamadı. "
                "Lütfen opencv-contrib-python paketini yükleyin:\n"
                "pip install opencv-contrib-python"
            )
        
        # Model dosyasını indir veya yolunu al
        model_path = download_model(
            self.model_name, 
            self.scale, 
            self.models_dir
        )
        
        # SuperResolution nesnesini oluştur
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        # Modeli oku
        self.sr.readModel(model_path)
        
        # Model ve ölçeği ayarla
        self.sr.setModel(self.model_name, self.scale)
        
        print(f"[INFO] Model yüklendi: {self.model_name} x{self.scale}")
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        
        if self.sr is None:
            raise RuntimeError("Model yüklenmemiş. Lütfen sınıfı tekrar başlatın.")
        
        # Görüntüyü yükselt
        result = self.sr.upsample(image)
        
        return result


def upscale_image(image: np.ndarray,
                  model_name: str = "edsr",
                  scale: int = 2,
                  models_dir: str = "./models") -> np.ndarray:
    
    sr = SuperResolution(
        model_name=model_name,
        scale=scale,
        models_dir=models_dir
    )
    return sr.upscale(image)


def bicubic_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    
    height, width = image.shape[:2]
    new_size = (width * scale, height * scale)
    
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
