# AI Image Enhancer - Ana Pipeline
# Dusuk kaliteli guvenlik kamerasi goruntulerini iyilestirme araci
# Kullanim: python main.py --input test.jpg --output output/

import argparse
import os
import sys
import time
from src.noise_reduction import denoise_image
from src.contrast_enhance import enhance_contrast_and_brightness
from src.sharpening import unsharp_mask
from src.super_resolution import SuperResolution, bicubic_upscale
from src.utils import (
    load_image, save_image, analyze_image, 
    get_image_info, list_images_in_directory
)


# Goruntu iyilestirme sinifi
class ImageEnhancer:
    
    # Sinif baslatma fonksiyonu
    def __init__(self,
                 model_name: str = "edsr",
                 scale: int = 2,
                 models_dir: str = "./models",
                 # Gürültü azaltma parametreleri
                 denoise_strength: int = 10,
                 # Kontrast parametreleri
                 clahe_clip_limit: float = 2.0,
                 gamma: float = None,
                 auto_brightness: bool = True,
                 # Keskinleştirme parametreleri
                 sharpen_amount: float = 1.5,
                 sharpen_threshold: int = 0,
                 # Pipeline kontrolleri
                 enable_denoise: bool = True,
                 enable_contrast: bool = True,
                 enable_sharpen: bool = True,
                 enable_super_res: bool = True):
        self.model_name = model_name
        self.scale = scale
        self.models_dir = models_dir
        
        # Parametre kayıt
        self.denoise_strength = denoise_strength
        self.clahe_clip_limit = clahe_clip_limit
        self.gamma = gamma
        self.auto_brightness = auto_brightness
        self.sharpen_amount = sharpen_amount
        self.sharpen_threshold = sharpen_threshold
        
        # Pipeline kontrolleri
        self.enable_denoise = enable_denoise
        self.enable_contrast = enable_contrast
        self.enable_sharpen = enable_sharpen
        self.enable_super_res = enable_super_res
        
        # Süper çözünürlük modelini yükle
        self.sr = None
        if self.enable_super_res:
            try:
                self.sr = SuperResolution(
                    model_name=model_name,
                    scale=scale,
                    models_dir=models_dir
                )
            except Exception as e:
                print(f"[UYARI] Süper çözünürlük modeli yüklenemedi: {e}")
                print("[UYARI] Bicubic interpolasyon kullanılacak.")
                self.sr = None
    
    # Goruntyu iyilestirir
    def enhance(self, image_or_path):
        import numpy as np
        
        # Zamanlama
        start_time = time.time()
        
        # Görüntüyü yükle
        if isinstance(image_or_path, str):
            print(f"\n[1/5] Görüntü yükleniyor: {image_or_path}")
            image = load_image(image_or_path)
        else:
            print("\n[1/5] Görüntü numpy array olarak alındı")
            image = image_or_path.copy()
        
        original = image.copy()
        results = {"original": original}
        
        # Görüntü analizi
        print("[INFO] Görüntü analiz ediliyor...")
        analysis = analyze_image(image)
        print(f"       Boyut: {analysis['basic_info']['width']}x{analysis['basic_info']['height']}")
        print(f"       Bulanıklık: {analysis['blur']['description']}")
        print(f"       Gürültü: {analysis['noise']['description']}")
        print(f"       Parlaklık: {analysis['brightness']['description']}")
        results["analysis"] = analysis
        
        # Adım 1: Gürültü Azaltma
        if self.enable_denoise:
            print(f"\n[2/5] Gürültü azaltma uygulanıyor (strength={self.denoise_strength})...")
            image = denoise_image(image, filter_strength=self.denoise_strength)
            results["denoised"] = image.copy()
        else:
            print("\n[2/5] Gürültü azaltma atlandı")
        
        # Adım 2: Kontrast ve Parlaklık İyileştirme
        if self.enable_contrast:
            print(f"\n[3/5] Kontrast iyileştirme uygulanıyor (CLAHE clip={self.clahe_clip_limit})...")
            image = enhance_contrast_and_brightness(
                image,
                clahe_clip_limit=self.clahe_clip_limit,
                gamma=self.gamma,
                auto_brightness=self.auto_brightness
            )
            results["contrast_enhanced"] = image.copy()
        else:
            print("\n[3/5] Kontrast iyileştirme atlandı")
        
        # Adım 3: Keskinleştirme
        if self.enable_sharpen:
            print(f"\n[4/5] Keskinleştirme uygulanıyor (amount={self.sharpen_amount})...")
            image = unsharp_mask(
                image,
                amount=self.sharpen_amount,
                threshold=self.sharpen_threshold
            )
            results["sharpened"] = image.copy()
        else:
            print("\n[4/5] Keskinleştirme atlandı")
        
        # Adım 4: Süper Çözünürlük
        if self.enable_super_res:
            print(f"\n[5/5] Süper çözünürlük uygulanıyor ({self.model_name} x{self.scale})...")
            if self.sr is not None:
                image = self.sr.upscale(image)
            else:
                print("       Bicubic interpolasyon kullanılıyor...")
                image = bicubic_upscale(image, self.scale)
            results["super_res"] = image.copy()
        else:
            print("\n[5/5] Süper çözünürlük atlandı")
        
        # Sonuç
        results["enhanced"] = image
        
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        
        print(f"\n[TAMAMLANDI] İşlem süresi: {elapsed_time:.2f} saniye")
        print(f"             Orijinal boyut: {original.shape[1]}x{original.shape[0]}")
        print(f"             Yeni boyut: {image.shape[1]}x{image.shape[0]}")
        
        return results
    
    # Dizindeki tum goruntuleri isler
    def process_directory(self, input_dir, output_dir, suffix="_enhanced"):
        # Görüntü dosyalarını listele
        image_files = list_images_in_directory(input_dir)
        
        if not image_files:
            print(f"[UYARI] Dizinde görüntü bulunamadı: {input_dir}")
            return []
        
        print(f"[INFO] {len(image_files)} görüntü bulundu")
        
        # Çıkış dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(image_files)}] İşleniyor: {os.path.basename(image_path)}")
            print('='*60)
            
            try:
                # Görüntüyü işle
                result = self.enhance(image_path)
                
                # Çıkış dosya adı
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                ext = os.path.splitext(image_path)[1]
                output_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
                
                # Kaydet
                save_image(result["enhanced"], output_path)
                
                results.append({
                    "input": image_path,
                    "output": output_path,
                    "success": True,
                    "time": result["elapsed_time"]
                })
                
            except Exception as e:
                print(f"[HATA] Görüntü işlenemedi: {e}")
                results.append({
                    "input": image_path,
                    "output": None,
                    "success": False,
                    "error": str(e)
                })
        
        # Özet
        successful = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}")
        print(f"[ÖZET] {successful}/{len(results)} görüntü başarıyla işlendi")
        print('='*60)
        
        return results


# Komut satiri argumanlarini ayristirir
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Image Enhancer - Düşük kaliteli görüntüleri iyileştirme aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )
    
    # Temel argümanlar
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Giriş görüntü dosyası veya dizini"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./output_images",
        help="Çıkış dizini (varsayılan: ./output_images)"
    )
    
    # Model ayarları
    parser.add_argument(
        "--model", "-m",
        default="edsr",
        choices=["edsr", "fsrcnn", "espcn", "lapsrn"],
        help="Süper çözünürlük modeli (varsayılan: edsr)"
    )
    
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        choices=[2, 3, 4, 8],
        help="Ölçekleme faktörü (varsayılan: 2)"
    )
    
    # İşlem parametreleri
    parser.add_argument(
        "--denoise-strength",
        type=int,
        default=10,
        help="Gürültü azaltma gücü (varsayılan: 10)"
    )
    
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE kontrast sınırı (varsayılan: 2.0)"
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Gamma değeri (varsayılan: otomatik)"
    )
    
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=1.5,
        help="Keskinleştirme miktarı (varsayılan: 1.5)"
    )
    
    # İşlem kontrolleri
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Gürültü azaltmayı devre dışı bırak"
    )
    
    parser.add_argument(
        "--no-contrast",
        action="store_true",
        help="Kontrast iyileştirmeyi devre dışı bırak"
    )
    
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Keskinleştirmeyi devre dışı bırak"
    )
    
    parser.add_argument(
        "--no-super-res",
        action="store_true",
        help="Süper çözünürlüğü devre dışı bırak"
    )
    
    # Diğer seçenekler
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Sadece görüntü analizi yap, işlem yapma"
    )
    
    parser.add_argument(
        "--suffix",
        default="_enhanced",
        help="Çıkış dosya adı eki (varsayılan: _enhanced)"
    )
    
    return parser.parse_args()


# Ana fonksiyon
def main():
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("     AI IMAGE ENHANCER - Görüntü İyileştirme Sistemi")
    print("="*60)
    
    # Giriş kontrolü
    if not os.path.exists(args.input):
        print(f"[HATA] Giriş bulunamadı: {args.input}")
        sys.exit(1)
    
    # Sadece analiz modu
    if args.analyze_only:
        print("\n[ANALIZ MODU]")
        if os.path.isfile(args.input):
            image = load_image(args.input)
            analysis = analyze_image(image)
            print(f"\nDosya: {args.input}")
            print(f"Boyut: {analysis['basic_info']['width']}x{analysis['basic_info']['height']}")
            print(f"Kanallar: {analysis['basic_info']['channels']}")
            print(f"Dosya boyutu: {analysis['basic_info']['size_mb']} MB")
            print(f"\nBulanıklık: {analysis['blur']['description']} (değer: {analysis['blur']['value']:.2f})")
            print(f"Gürültü: {analysis['noise']['description']} (değer: {analysis['noise']['value']:.2f})")
            print(f"Parlaklık: {analysis['brightness']['description']} (değer: {analysis['brightness']['value']:.2f})")
        else:
            print(f"[HATA] Analiz için tek bir dosya belirtilmelidir")
            sys.exit(1)
        return
    
    # Enhancer oluştur
    print("\n[CONFIG]")
    print(f"  Model: {args.model.upper()} x{args.scale}")
    print(f"  Gürültü azaltma: {'Aktif' if not args.no_denoise else 'Devre dışı'}")
    print(f"  Kontrast: {'Aktif' if not args.no_contrast else 'Devre dışı'}")
    print(f"  Keskinleştirme: {'Aktif' if not args.no_sharpen else 'Devre dışı'}")
    print(f"  Süper çözünürlük: {'Aktif' if not args.no_super_res else 'Devre dışı'}")
    
    enhancer = ImageEnhancer(
        model_name=args.model,
        scale=args.scale,
        models_dir="./models",
        denoise_strength=args.denoise_strength,
        clahe_clip_limit=args.clahe_clip,
        gamma=args.gamma,
        sharpen_amount=args.sharpen_amount,
        enable_denoise=not args.no_denoise,
        enable_contrast=not args.no_contrast,
        enable_sharpen=not args.no_sharpen,
        enable_super_res=not args.no_super_res
    )
    
    # Tek dosya mı, dizin mi?
    if os.path.isfile(args.input):
        # Tek dosya işle
        result = enhancer.enhance(args.input)
        
        # Çıkış dosya adı
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        ext = os.path.splitext(args.input)[1]
        output_path = os.path.join(args.output, f"{base_name}{args.suffix}{ext}")
        
        # Kaydet
        save_image(result["enhanced"], output_path)
        
    else:
        # Dizin işle
        enhancer.process_directory(args.input, args.output, args.suffix)
    
    print("\n[BİTTİ] İşlem tamamlandı!")


if __name__ == "__main__":
    main()
