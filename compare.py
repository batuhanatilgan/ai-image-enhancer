# Goruntu Karsilastirma Araci
# Orijinal ve iyilestirilmis goruntuleri karsilastirir
# Kullanim: python compare.py --original input.jpg --enhanced output.jpg

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils import (
    load_image, save_image, calculate_psnr, calculate_ssim,
    analyze_image, get_image_info
)


def create_side_by_side(original: np.ndarray, 
                        enhanced: np.ndarray,
                        title_original: str = "Orijinal",
                        title_enhanced: str = "İyileştirilmiş") -> np.ndarray:
    
    # Boyutları eşitle (yükseklik bazında)
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    # Hedef yükseklik (büyük olanı kullan)
    target_height = max(h1, h2)
    
    # Orijinali yeniden boyutlandır
    if h1 != target_height:
        scale = target_height / h1
        original = cv2.resize(original, (int(w1 * scale), target_height))
    
    # İyileştirilmişi yeniden boyutlandır
    if h2 != target_height:
        scale = target_height / h2
        enhanced = cv2.resize(enhanced, (int(w2 * scale), target_height))
    
    # Başlık ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    # Başlık yüksekliği
    text_height = 40
    
    # Orijinal için başlık
    orig_with_title = np.zeros((original.shape[0] + text_height, original.shape[1], 3), dtype=np.uint8)
    orig_with_title[:text_height, :] = bg_color
    cv2.putText(orig_with_title, title_original, (10, 30), font, font_scale, text_color, thickness)
    orig_with_title[text_height:, :] = original
    
    # İyileştirilmiş için başlık
    enh_with_title = np.zeros((enhanced.shape[0] + text_height, enhanced.shape[1], 3), dtype=np.uint8)
    enh_with_title[:text_height, :] = bg_color
    cv2.putText(enh_with_title, title_enhanced, (10, 30), font, font_scale, text_color, thickness)
    enh_with_title[text_height:, :] = enhanced
    
    # Ayırıcı çizgi
    separator = np.zeros((orig_with_title.shape[0], 5, 3), dtype=np.uint8)
    separator[:, :] = (100, 100, 100)
    
    # Birleştir
    combined = np.hstack([orig_with_title, separator, enh_with_title])
    
    return combined


def create_detailed_comparison(original: np.ndarray,
                               enhanced: np.ndarray,
                               output_path: str = None,
                               show: bool = True) -> None:
    
    # BGR'den RGB'ye dönüştür (matplotlib için)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    # Kalite metrikleri
    # İyileştirilmişi orijinal boyutuna getir (PSNR hesaplaması için)
    enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    psnr = calculate_psnr(original, enhanced_resized)
    ssim = calculate_ssim(original, enhanced_resized)
    
    # Görüntü bilgileri
    orig_info = get_image_info(original)
    enh_info = get_image_info(enhanced)
    
    # Analiz
    orig_analysis = analyze_image(original)
    enh_analysis = analyze_image(enhanced)
    
    # Grafik oluştur
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[4, 1, 1])
    
    # Orijinal görüntü
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_rgb)
    ax1.set_title(f'Orijinal\n{orig_info["width"]}x{orig_info["height"]} px', fontsize=14)
    ax1.axis('off')
    
    # İyileştirilmiş görüntü
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced_rgb)
    ax2.set_title(f'İyileştirilmiş\n{enh_info["width"]}x{enh_info["height"]} px', fontsize=14)
    ax2.axis('off')
    
    # Histogramlar - Orijinal
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        ax3.plot(hist, color=color, alpha=0.7)
    ax3.set_xlim([0, 256])
    ax3.set_title('Orijinal Histogram', fontsize=12)
    ax3.set_xlabel('Piksel Değeri')
    ax3.set_ylabel('Frekans')
    
    # Histogramlar - İyileştirilmiş
    ax4 = fig.add_subplot(gs[1, 1])
    for i, color in enumerate(colors):
        hist = cv2.calcHist([enhanced], [i], None, [256], [0, 256])
        ax4.plot(hist, color=color, alpha=0.7)
    ax4.set_xlim([0, 256])
    ax4.set_title('İyileştirilmiş Histogram', fontsize=12)
    ax4.set_xlabel('Piksel Değeri')
    ax4.set_ylabel('Frekans')
    
    # Metrikler
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    metrics_text = (
        f"KARŞILAŞTIRMA METRİKLERİ\n"
        f"{'='*60}\n\n"
        f"Çözünürlük Artışı: {orig_info['width']}x{orig_info['height']} → "
        f"{enh_info['width']}x{enh_info['height']} "
        f"({enh_info['width']/orig_info['width']:.1f}x)\n\n"
        f"Bulanıklık: {orig_analysis['blur']['description']} → "
        f"{enh_analysis['blur']['description']}\n"
        f"Gürültü: {orig_analysis['noise']['description']} → "
        f"{enh_analysis['noise']['description']}\n"
        f"Parlaklık: {orig_analysis['brightness']['description']} → "
        f"{enh_analysis['brightness']['description']}\n\n"
        f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"
    )
    
    ax5.text(0.5, 0.5, metrics_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Kaydet
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Karşılaştırma kaydedildi: {output_path}")
    
    # Göster
    if show:
        plt.show()
    else:
        plt.close()


def create_zoom_comparison(original: np.ndarray,
                           enhanced: np.ndarray,
                           crop_region: tuple = None,
                           zoom_factor: float = 2.0,
                           output_path: str = None,
                           show: bool = True) -> None:
    
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    # Ölçek oranı
    scale_x = w2 / w1
    scale_y = h2 / h1
    
    # Kırpma bölgesi (varsayılan: merkez)
    if crop_region is None:
        crop_size = min(w1, h1) // 4
        x = (w1 - crop_size) // 2
        y = (h1 - crop_size) // 2
        crop_region = (x, y, crop_size, crop_size)
    
    x, y, cw, ch = crop_region
    
    # Orijinalden kırp
    orig_crop = original[y:y+ch, x:x+cw]
    
    # İyileştirilmişten kırp (ölçekli koordinatlar)
    ex = int(x * scale_x)
    ey = int(y * scale_y)
    ecw = int(cw * scale_x)
    ech = int(ch * scale_y)
    enh_crop = enhanced[ey:ey+ech, ex:ex+ecw]
    
    # Yakınlaştır
    orig_zoomed = cv2.resize(orig_crop, None, fx=zoom_factor, fy=zoom_factor, 
                             interpolation=cv2.INTER_NEAREST)
    enh_zoomed = cv2.resize(enh_crop, None, fx=zoom_factor, fy=zoom_factor,
                            interpolation=cv2.INTER_NEAREST)
    
    # RGB'ye dönüştür
    orig_zoomed_rgb = cv2.cvtColor(orig_zoomed, cv2.COLOR_BGR2RGB)
    enh_zoomed_rgb = cv2.cvtColor(enh_zoomed, cv2.COLOR_BGR2RGB)
    
    # Grafik oluştur
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(orig_zoomed_rgb)
    axes[0].set_title(f'Orijinal (Yakınlaştırılmış {zoom_factor}x)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(enh_zoomed_rgb)
    axes[1].set_title(f'İyileştirilmiş (Yakınlaştırılmış {zoom_factor}x)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Yakınlaştırılmış karşılaştırma kaydedildi: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description="Görüntü Karşılaştırma Aracı - Orijinal ve iyileştirilmiş görüntüleri karşılaştırır",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--original", "-o",
        required=True,
        help="Orijinal görüntü yolu"
    )
    
    parser.add_argument(
        "--enhanced", "-e",
        required=True,
        help="İyileştirilmiş görüntü yolu"
    )
    
    parser.add_argument(
        "--save", "-s",
        default=None,
        help="Karşılaştırma görüntüsünü kaydet"
    )
    
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Basit yan yana karşılaştırma oluştur"
    )
    
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Yakınlaştırılmış karşılaştırma göster"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Grafiği gösterme (sadece kaydet)"
    )
    
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("     GÖRÜNTÜ KARŞILAŞTIRMA ARACI")
    print("="*60)
    
    # Görüntüleri yükle
    print(f"\n[INFO] Orijinal yükleniyor: {args.original}")
    original = load_image(args.original)
    
    print(f"[INFO] İyileştirilmiş yükleniyor: {args.enhanced}")
    enhanced = load_image(args.enhanced)
    
    # Görüntü bilgileri
    orig_info = get_image_info(original)
    enh_info = get_image_info(enhanced)
    
    print(f"\n[INFO] Orijinal boyut: {orig_info['width']}x{orig_info['height']}")
    print(f"[INFO] İyileştirilmiş boyut: {enh_info['width']}x{enh_info['height']}")
    print(f"[INFO] Ölçek: {enh_info['width']/orig_info['width']:.2f}x")
    
    # Kalite metrikleri
    enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    psnr = calculate_psnr(original, enhanced_resized)
    ssim = calculate_ssim(original, enhanced_resized)
    
    print(f"\n[METRİKLER]")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    show = not args.no_show
    
    if args.side_by_side:
        # Basit yan yana
        combined = create_side_by_side(original, enhanced)
        if args.save:
            save_image(combined, args.save)
        if show:
            cv2.imshow("Karşılaştırma", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.zoom:
        # Yakınlaştırılmış karşılaştırma
        create_zoom_comparison(original, enhanced, output_path=args.save, show=show)
    else:
        # Detaylı karşılaştırma
        create_detailed_comparison(original, enhanced, output_path=args.save, show=show)
    
    print("\n[BİTTİ] Karşılaştırma tamamlandı!")


if __name__ == "__main__":
    main()
