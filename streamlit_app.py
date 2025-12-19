# -*- coding: utf-8 -*-
# AI Image Enhancer - Streamlit Web Arayuzu
# Guvenlik kamerasi goruntu iyilestirme sistemi
# Kullanim: streamlit run streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import io

# Proje modullerini import et
from src.noise_reduction import denoise_image
from src.contrast_enhance import enhance_contrast_and_brightness
from src.sharpening import unsharp_mask
from src.super_resolution import SuperResolution, bicubic_upscale
from src.utils import analyze_image, get_image_info


# Sayfa ayarlari
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(, unsafe_allow_html=True)


# OpenCV BGR goruntusunu PIL RGB'ye donusturur
def numpy_to_pil(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# PIL goruntusunu OpenCV BGR'ye donusturur
def pil_to_numpy(image):
    rgb = np.array(image)
    if len(rgb.shape) == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:
        return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# Video dosyasindan kareler cikarir
def extract_frames_from_video(video_path, max_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // max_frames
        frame_indices = [i * step for i in range(max_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append({
                'frame': frame,
                'index': idx,
                'time': idx / fps if fps > 0 else 0
            })
    
    cap.release()
    return frames


# Goruntu iyilestirme pipeline fonksiyonu
def process_image(image, denoise_enabled, denoise_strength, 
                  contrast_enabled, clahe_clip, gamma,
                  sharpen_enabled, sharpen_amount,
                  super_res_enabled, model_name, scale):
    
    result = image.copy()
    
    # 1. Gurultu Azaltma
    if denoise_enabled:
        result = denoise_image(result, filter_strength=denoise_strength)
    
    # 2. Kontrast Iyilestirme
    if contrast_enabled:
        gamma_val = gamma if gamma != 1.0 else None
        result = enhance_contrast_and_brightness(
            result,
            clahe_clip_limit=clahe_clip,
            gamma=gamma_val,
            auto_brightness=gamma_val is None
        )
    
    # 3. Keskinlestirme
    if sharpen_enabled:
        result = unsharp_mask(result, amount=sharpen_amount)
    
    # 4. Super Cozunurluk
    if super_res_enabled:
        try:
            sr = SuperResolution(
                model_name=model_name,
                scale=scale,
                models_dir='./models'
            )
            result = sr.upscale(result)
        except Exception as e:
            st.warning(f"Model yuklenemedi, bicubic kullaniliyor: {str(e)[:50]}")
            result = bicubic_upscale(result, scale)
    
    return result


def main():
    # Header
    st.markdown('<h1 class="main-header">✨ AI Image Enhancer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Guvenlik Kamerasi & IP Kamera Goruntu Iyilestirme Sistemi</p>', unsafe_allow_html=True)
    
    # Sidebar - Ayarlar
    with st.sidebar:
        st.header("⚙️ Islem Ayarlari")
        
        # Gurultu Azaltma
        st.subheader("🔇 Gurultu Azaltma")
        denoise_enabled = st.checkbox("Aktif", value=True, key="denoise")
        denoise_strength = st.slider("Filtre Gucu", 1, 20, 10, key="denoise_str")
        
        st.divider()
        
        # Kontrast
        st.subheader("🌓 Kontrast")
        contrast_enabled = st.checkbox("Aktif", value=True, key="contrast")
        clahe_clip = st.slider("CLAHE Limit", 0.5, 5.0, 2.0, 0.1, key="clahe")
        gamma = st.slider("Gamma (< 1 aydinlatir)", 0.3, 2.5, 1.0, 0.1, key="gamma")
        
        st.divider()
        
        # Keskinlestirme
        st.subheader("🔍 Keskinlestirme")
        sharpen_enabled = st.checkbox("Aktif", value=True, key="sharpen")
        sharpen_amount = st.slider("Keskinlik Miktari", 0.5, 3.0, 1.5, 0.1, key="sharp_amt")
        
        st.divider()
        
        # Super Cozunurluk
        st.subheader("📐 Super Cozunurluk")
        super_res_enabled = st.checkbox("Aktif", value=True, key="super_res")
        model_name = st.selectbox("Model", ["fsrcnn", "edsr", "espcn", "lapsrn"], key="model")
        scale = st.selectbox("Olcek", [2, 3, 4], key="scale")
    
    # Ana icerik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Dosya Yukle")
        uploaded_file = st.file_uploader(
            "Goruntu veya Video yukleyin",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'mkv'],
            key="uploader"
        )
    
    # Session state for images
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'enhanced_image' not in st.session_state:
        st.session_state.enhanced_image = None
    if 'video_frames' not in st.session_state:
        st.session_state.video_frames = []
    
    # Dosya yuklendi mi?
    if uploaded_file is not None:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        # Video mu?
        if file_ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            # Video islemleri
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            with st.spinner("Video kareleri cikariliyor..."):
                st.session_state.video_frames = extract_frames_from_video(tmp_path, max_frames=10)
            
            os.unlink(tmp_path)
            
            if st.session_state.video_frames:
                st.success(f"✅ {len(st.session_state.video_frames)} kare cikarildi!")
                
                with col2:
                    frame_idx = st.selectbox(
                        "Kare Secin",
                        range(len(st.session_state.video_frames)),
                        format_func=lambda x: f"Kare {x+1} ({st.session_state.video_frames[x]['time']:.1f}s)"
                    )
                    st.session_state.current_image = st.session_state.video_frames[frame_idx]['frame']
            else:
                st.error("Video okunamadi!")
        else:
            # Goruntu islemleri
            image = Image.open(uploaded_file)
            st.session_state.current_image = pil_to_numpy(image)
            st.session_state.video_frames = []
    
    # Goruntu goster ve isle
    if st.session_state.current_image is not None:
        col_orig, col_enh = st.columns(2)
        
        with col_orig:
            st.subheader("📷 Orijinal Goruntu")
            st.image(numpy_to_pil(st.session_state.current_image), use_container_width=True)
            
            # Analiz
            analysis = analyze_image(st.session_state.current_image)
            info = get_image_info(st.session_state.current_image)
            
            st.markdown(f)
        
        with col_enh:
            st.subheader("✨ Iyilestirilmis Goruntu")
            
            # Islem butonu
            if st.button("🚀 Islemi Baslat", type="primary", use_container_width=True):
                with st.spinner("Islem yapiliyor..."):
                    st.session_state.enhanced_image = process_image(
                        st.session_state.current_image,
                        denoise_enabled, denoise_strength,
                        contrast_enabled, clahe_clip, gamma,
                        sharpen_enabled, sharpen_amount,
                        super_res_enabled, model_name, scale
                    )
                st.success("✅ Islem tamamlandi!")
                st.rerun()
            
            if st.session_state.enhanced_image is not None:
                st.image(numpy_to_pil(st.session_state.enhanced_image), use_container_width=True)
                
                enh_info = get_image_info(st.session_state.enhanced_image)
                orig_info = get_image_info(st.session_state.current_image)
                
                st.markdown(f)
                
                # Indirme butonu
                pil_img = numpy_to_pil(st.session_state.enhanced_image)
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                
                st.download_button(
                    label="💾 Indir",
                    data=buf.getvalue(),
                    file_name="enhanced_image.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # Bilgi kartlari
    st.divider()
    
    cols = st.columns(4)
    with cols[0]:
        st.info("**🔇 Gurultu Azaltma**\n\nNon-Local Means algoritmasi")
    with cols[1]:
        st.info("**🌓 Kontrast**\n\nCLAHE + Gamma Correction")
    with cols[2]:
        st.info("**🔍 Keskinlik**\n\nUnsharp Masking")
    with cols[3]:
        st.info("**📐 Super Cozunurluk**\n\nEDSR / FSRCNN AI Modelleri")


if __name__ == "__main__":
    main()
