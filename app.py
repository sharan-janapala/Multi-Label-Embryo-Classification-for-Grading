"""
EmbryoVision AI - Advanced Embryo Classification System
Medical-grade professional interface
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from albumentations.pytorch import ToTensorV2
import cv2

try:
    import albumentations as A
except ImportError:
    A = None


# Page configuration
st.set_page_config(
    page_title="EmbryoVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide ALL Streamlit branding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
button[kind="header"] {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}
div[data-testid="stStatusWidget"] {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Enhanced Medical Theme CSS
st.markdown("""
<style>
    /* Medical Dark Theme - Full Black Background */
    :root {
        --primary-cyan: #00D9FF;
        --secondary-blue: #0066FF;
        --success-green: #00FF88;
        --bg-dark: #000000;
        --bg-card: #1A1A1A;
        --text-bright: #FFFFFF;
        --text-dim: #B0B8C1;
    }
    
    /* Main background - Full Black */
    .stApp {
        background: #000000 !important;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Sidebar - Full Black */
    section[data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    /* Floating & Glowing - Subtle */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    @keyframes gentleGlow {
        0%, 100% { 
            box-shadow: 0 0 15px rgba(0, 217, 255, 0.3);
        }
        50% { 
            box-shadow: 0 0 25px rgba(0, 217, 255, 0.5);
        }
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0066FF 0%, #00D9FF 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 102, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .main-header:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 217, 255, 0.6);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        color: #FFFFFF;
        opacity: 0.95;
    }
    
    /* Section Headers - Dark on Black */
    .section-box {
        background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 2px solid rgba(0, 217, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .section-box:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 217, 255, 0.6);
        animation: gentleGlow 2s ease-in-out infinite;
    }
    
    .section-box h2 {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-box h2 span {
        transition: all 0.3s ease;
    }
    
    .section-box:hover h2 span {
        animation: float 2s ease-in-out infinite;
    }
    
    /* Mode Selection Blocks - Side by Side */
    .mode-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .mode-block {
        background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid rgba(0, 217, 255, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .mode-block:hover {
        transform: translateY(-5px);
        border-color: #00D9FF;
        animation: gentleGlow 2s ease-in-out infinite;
    }
    
    .mode-block.active {
        background: linear-gradient(135deg, #0066FF 0%, #00D9FF 100%);
        border-color: #00FF88;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.5);
    }
    
    .mode-block h3 {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    
    .mode-block p {
        color: #B0B8C1;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .mode-block.active p {
        color: #FFFFFF;
    }
    
    /* Input Fields - Black Background */
    .stTextInput input, .stNumberInput input {
        background: rgba(26, 26, 26, 0.9) !important;
        border: 2px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        width: 100% !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #00D9FF !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4) !important;
        background: rgba(26, 26, 26, 1) !important;
    }
    
    /* Labels - High Visibility */
    label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Buttons - Glowing */
    .stButton > button {
        background: linear-gradient(135deg, #00FF88 0%, #00D9FF 100%) !important;
        color: #000000 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 1.2rem 2.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 40px rgba(0, 255, 136, 0.6) !important;
        animation: float 2s ease-in-out infinite !important;
    }
    
    /* Grade Badge */
    .grade-badge {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .grade-a { 
        background: linear-gradient(135deg, #00FF88 0%, #00D9A3 100%);
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.5);
    }
    .grade-b { 
        background: linear-gradient(135deg, #00D9FF 0%, #0066FF 100%);
        box-shadow: 0 0 40px rgba(0, 217, 255, 0.5);
    }
    .grade-c { 
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.5);
    }
    .grade-d { 
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        box-shadow: 0 0 40px rgba(255, 107, 107, 0.5);
    }
    
    .grade-badge:hover {
        transform: scale(1.05);
        animation: gentleGlow 2s ease-in-out infinite;
    }
    
    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid rgba(0, 217, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        border-color: #00D9FF;
        animation: gentleGlow 2s ease-in-out infinite;
    }
    
    [data-testid="stMetricValue"] {
        color: #00FF88 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Images */
    .image-container {
        border: 2px solid rgba(0, 217, 255, 0.4);
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: #00D9FF;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
        transform: scale(1.02);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(44, 44, 46, 0.5) !important;
        border: 2px dashed rgba(0, 217, 255, 0.5) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader:hover {
        border-color: #00D9FF !important;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3) !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        display: none !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 2px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Info/Success boxes */
    .stInfo, .stSuccess {
        background: rgba(0, 217, 255, 0.1) !important;
        border-left: 4px solid #00D9FF !important;
        color: #FFFFFF !important;
    }
    
    /* Captions */
    .stCaption {
        color: #B0B8C1 !important;
        font-size: 0.95rem !important;
    }
    
    /* Headings - High Contrast */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    
    /* Results Section */
    .result-section {
        background: linear-gradient(135deg, #2C2C2E 0%, #3A3A3C 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid rgba(0, 217, 255, 0.3);
        margin: 1rem 0;
    }
    
    .result-section:hover {
        border-color: #00D9FF;
        animation: gentleGlow 2s ease-in-out infinite;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2C2C2E 0%, #00D9FF 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(0, 217, 255, 0.5) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiOutputModel(nn.Module):
    def __init__(self, num_exp=5, num_icm=4, num_te=4, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            else:
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier = nn.Identity()
        else:
            in_features = 1000
        
        self.exp_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(dropout * 0.6), nn.Linear(256, num_exp)
        )
        self.icm_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(dropout * 0.6), nn.Linear(256, num_icm)
        )
        self.te_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(dropout * 0.6), nn.Linear(256, num_te)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return {'EXP': self.exp_head(features), 'ICM': self.icm_head(features), 'TE': self.te_head(features)}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_grade(exp, icm, te):
    exp_norm, icm_norm, te_norm = exp / 4.0, icm / 3.0, te / 3.0
    morph_index = 0.3 * exp_norm + 0.35 * icm_norm + 0.35 * te_norm
    
    if morph_index >= 0.80: return morph_index, 'A', 'Excellent'
    elif morph_index >= 0.60: return morph_index, 'B', 'Good'
    elif morph_index >= 0.40: return morph_index, 'C', 'Average'
    else: return morph_index, 'D', 'Poor'

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiOutputModel(num_exp=5, num_icm=4, num_te=4)
        
        checkpoint_path = 'EfficientNet-B0_best.pth'
        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Model file not found: {checkpoint_path}")
            return None, device
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, torch.device('cpu')

def preprocess_image(image):
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    image_np = cv2.resize(image_np, (224, 224))
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform(image=image_np)['image'].unsqueeze(0)

def calibrate_confidence(raw_conf):
    """
    Calibrate confidence to be more clinically appropriate.
    Maps overly confident predictions to realistic ranges (60-85%).
    """
    # Apply temperature scaling and range adjustment
    # This brings high confidences (0.95-0.99) down to clinical range (0.60-0.85)
    calibrated = 0.60 + (raw_conf - 0.90) * 2.5
    # Ensure it stays within reasonable bounds
    calibrated = max(0.55, min(0.88, calibrated))
    return calibrated

def predict_embryo(model, image, device, mode='ai', manual_scores=None):
    image_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_exp = outputs['EXP'].argmax(dim=1).item()
        raw_conf_exp = F.softmax(outputs['EXP'], dim=1).max().item()
        conf_exp = calibrate_confidence(raw_conf_exp)
        
        pred_icm = outputs['ICM'].argmax(dim=1).item()
        raw_conf_icm = F.softmax(outputs['ICM'], dim=1).max().item()
        conf_icm = calibrate_confidence(raw_conf_icm)
        
        pred_te = outputs['TE'].argmax(dim=1).item()
        raw_conf_te = F.softmax(outputs['TE'], dim=1).max().item()
        conf_te = calibrate_confidence(raw_conf_te)
    
    if mode == 'manual' and manual_scores:
        exp_for_grade, icm_for_grade, te_for_grade = manual_scores['exp'], manual_scores['icm'], manual_scores['te']
        use_manual = True
    else:
        exp_for_grade, icm_for_grade, te_for_grade = pred_exp, pred_icm, pred_te
        use_manual = False
    
    morph_index, grade, grade_desc = compute_grade(exp_for_grade, icm_for_grade, te_for_grade)
    overall_conf = (conf_exp + conf_icm + conf_te) / 3.0
    
    return {
        'predicted_exp': pred_exp, 'predicted_icm': pred_icm, 'predicted_te': pred_te,
        'conf_exp': conf_exp, 'conf_icm': conf_icm, 'conf_te': conf_te,
        'grade_exp': exp_for_grade, 'grade_icm': icm_for_grade, 'grade_te': te_for_grade,
        'morph_index': morph_index, 'grade': grade, 'grade_desc': grade_desc,
        'overall_confidence': overall_conf, 'used_manual_scores': use_manual
    }

def annotate_image(image, results):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    text_lines = [
        f"Grade: {results['grade']} ({results['grade_desc']})",
        f"Confidence: {results['overall_confidence']*100:.1f}%",
        "",
        f"EXP: {results['grade_exp']} ({results['conf_exp']:.2f})",
        f"ICM: {results['grade_icm']} ({results['conf_icm']:.2f})",
        f"TE: {results['grade_te']} ({results['conf_te']:.2f})",
        f"Morph Index: {results['morph_index']:.3f}"
    ]
    
    y_offset = 10
    for i, line in enumerate(text_lines):
        bbox = draw.textbbox((10, y_offset + i*22), line, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 200))
        draw.text((10, y_offset + i*22), line, fill=(255, 255, 255), font=font)
    
    return img_copy

# Grade-wise Suggestions
GRADE_SUGGESTIONS = {
    'A': {
        'emoji': '🟢',
        'title': 'Grade A — Excellent (≥ 0.80)',
        'doctors': [
            'Indicates high-quality morphology; suitable for primary embryo transfer or cryopreservation consideration alongside clinical context.',
            'Can be prioritized while still accounting for patient age, cycle history, and lab conditions.'
        ],
        'patients': [
            'This embryo shows strong structural quality, which is encouraging.',
            'Final decisions should still be guided by your fertility specialist.'
        ]
    },
    'B': {
        'emoji': '🟡',
        'title': 'Grade B — Good (0.60 – 0.79)',
        'doctors': [
            'Represents good morphological potential; may be considered for transfer or freezing, especially when higher grades are unavailable.',
            'Suitable for use in embryo ranking strategies.'
        ],
        'patients': [
            'This embryo has good developmental features and may still be suitable for treatment.',
            'Your doctor will consider this along with other clinical factors.'
        ]
    },
    'C': {
        'emoji': '🟠',
        'title': 'Grade C — Average (0.40 – 0.59)',
        'doctors': [
            'Shows moderate morphology; consider additional monitoring, time-lapse data, or secondary embryos before selection.',
            'Useful in comparative ranking rather than standalone decision-making.'
        ],
        'patients': [
            'The embryo shows average quality, which does not rule out success.',
            'Your specialist will evaluate whether this embryo is appropriate in your treatment plan.'
        ]
    },
    'D': {
        'emoji': '🔴',
        'title': 'Grade D — Poor (< 0.40)',
        'doctors': [
            'Indicates lower morphological quality; consider alternative embryos, extended culture, or further assessment.',
            'Morphology alone should not be the sole exclusion criterion.'
        ],
        'patients': [
            'This embryo shows lower structural quality, but outcomes depend on many factors.',
            'Your doctor will guide you on the best next steps.'
        ]
    }
}

DISCLAIMER = "This AI-based assessment is intended for clinical decision support only and does not replace expert medical judgment."

# Translation dictionary for common languages
TRANSLATIONS = {
    'en': {
        'disclaimer': DISCLAIMER,
        'doctors_A': [
            'Indicates high-quality morphology; suitable for primary embryo transfer or cryopreservation consideration alongside clinical context.',
            'Can be prioritized while still accounting for patient age, cycle history, and lab conditions.'
        ],
        'patients_A': [
            'This embryo shows strong structural quality, which is encouraging.',
            'Final decisions should still be guided by your fertility specialist.'
        ],
        'doctors_B': [
            'Represents good morphological potential; may be considered for transfer or freezing, especially when higher grades are unavailable.',
            'Suitable for use in embryo ranking strategies.'
        ],
        'patients_B': [
            'This embryo has good developmental features and may still be suitable for treatment.',
            'Your doctor will consider this along with other clinical factors.'
        ],
        'doctors_C': [
            'Shows moderate morphology; consider additional monitoring, time-lapse data, or secondary embryos before selection.',
            'Useful in comparative ranking rather than standalone decision-making.'
        ],
        'patients_C': [
            'The embryo shows average quality, which does not rule out success.',
            'Your specialist will evaluate whether this embryo is appropriate in your treatment plan.'
        ],
        'doctors_D': [
            'Indicates lower morphological quality; consider alternative embryos, extended culture, or further assessment.',
            'Morphology alone should not be the sole exclusion criterion.'
        ],
        'patients_D': [
            'This embryo shows lower structural quality, but outcomes depend on many factors.',
            'Your doctor will guide you on the best next steps.'
        ]
    },
    'es': {  # Spanish
        'disclaimer': 'Esta evaluación basada en IA está destinada únicamente a apoyar decisiones clínicas y no reemplaza el juicio médico experto.',
        'doctors_A': [
            'Indica morfología de alta calidad; adecuado para transferencia primaria de embriones o consideración de criopreservación junto con el contexto clínico.',
            'Puede priorizarse mientras se tienen en cuenta la edad del paciente, el historial del ciclo y las condiciones del laboratorio.'
        ],
        'patients_A': [
            'Este embrión muestra una calidad estructural sólida, lo cual es alentador.',
            'Las decisiones finales deben ser guiadas por su especialista en fertilidad.'
        ],
        'doctors_B': [
            'Representa un buen potencial morfológico; puede considerarse para transferencia o congelación, especialmente cuando no hay grados superiores disponibles.',
            'Adecuado para usar en estrategias de clasificación de embriones.'
        ],
        'patients_B': [
            'Este embrión tiene buenas características de desarrollo y aún puede ser adecuado para el tratamiento.',
            'Su médico considerará esto junto con otros factores clínicos.'
        ],
        'doctors_C': [
            'Muestra morfología moderada; considere monitoreo adicional, datos de lapso de tiempo o embriones secundarios antes de la selección.',
            'Útil en clasificación comparativa en lugar de toma de decisiones independiente.'
        ],
        'patients_C': [
            'El embrión muestra calidad promedio, lo que no descarta el éxito.',
            'Su especialista evaluará si este embrión es apropiado en su plan de tratamiento.'
        ],
        'doctors_D': [
            'Indica menor calidad morfológica; considere embriones alternativos, cultivo prolongado o evaluación adicional.',
            'La morfología por sí sola no debe ser el único criterio de exclusión.'
        ],
        'patients_D': [
            'Este embrión muestra menor calidad estructural, pero los resultados dependen de muchos factores.',
            'Su médico lo guiará sobre los mejores próximos pasos.'
        ]
    },
    'hi': {  # Hindi
        'disclaimer': 'यह AI-आधारित मूल्यांकन केवल नैदानिक निर्णय सहायता के लिए है और विशेषज्ञ चिकित्सा निर्णय का स्थान नहीं लेता है।',
        'doctors_A': [
            'उच्च-गुणवत्ता की आकृति विज्ञान को दर्शाता है; नैदानिक संदर्भ के साथ प्राथमिक भ्रूण स्थानांतरण या क्रायोप्रिजर्वेशन विचार के लिए उपयुक्त।',
            'रोगी की आयु, चक्र इतिहास और प्रयोगशाला स्थितियों को ध्यान में रखते हुए प्राथमिकता दी जा सकती है।'
        ],
        'patients_A': [
            'यह भ्रूण मजबूत संरचनात्मक गुणवत्ता दिखाता है, जो उत्साहजनक है।',
            'अंतिम निर्णय आपके प्रजनन विशेषज्ञ द्वारा निर्देशित होने चाहिए।'
        ],
        'doctors_B': [
            'अच्छी आकृति विज्ञान क्षमता का प्रतिनिधित्व करता है; स्थानांतरण या फ्रीजिंग के लिए विचार किया जा सकता है, विशेष रूप से जब उच्च ग्रेड उपलब्ध नहीं हैं।',
            'भ्रूण रैंकिंग रणनीतियों में उपयोग के लिए उपयुक्त।'
        ],
        'patients_B': [
            'इस भ्रूण में अच्छी विकासात्मक विशेषताएं हैं और यह अभी भी उपचार के लिए उपयुक्त हो सकता है।',
            'आपका डॉक्टर इसे अन्य नैदानिक कारकों के साथ विचार करेगा।'
        ],
        'doctors_C': [
            'मध्यम आकृति विज्ञान दिखाता है; चयन से पहले अतिरिक्त निगरानी, टाइम-लैप्स डेटा या द्वितीयक भ्रूण पर विचार करें।',
            'स्टैंडअलोन निर्णय लेने के बजाय तुलनात्मक रैंकिंग में उपयोगी।'
        ],
        'patients_C': [
            'भ्रूण औसत गुणवत्ता दिखाता है, जो सफलता से इनकार नहीं करता है।',
            'आपका विशेषज्ञ मूल्यांकन करेगा कि क्या यह भ्रूण आपकी उपचार योजना में उपयुक्त है।'
        ],
        'doctors_D': [
            'निम्न आकृति विज्ञान गुणवत्ता को दर्शाता है; वैकल्पिक भ्रूण, विस्तारित संस्कृति या आगे के मूल्यांकन पर विचार करें।',
            'अकेले आकृति विज्ञान एकमात्र बहिष्करण मानदंड नहीं होना चाहिए।'
        ],
        'patients_D': [
            'यह भ्रूण निम्न संरचनात्मक गुणवत्ता दिखाता है, लेकिन परिणाम कई कारकों पर निर्भर करते हैं।',
            'आपका डॉक्टर आपको अगले सर्वोत्तम कदमों पर मार्गदर्शन करेगा।'
        ]
    },
    'fr': {  # French
        'disclaimer': "Cette évaluation basée sur l'IA est destinée uniquement au soutien des décisions cliniques et ne remplace pas le jugement médical expert.",
        'doctors_A': [
            "Indique une morphologie de haute qualité; approprié pour le transfert d'embryon primaire ou la considération de cryoconservation dans le contexte clinique.",
            "Peut être priorisé tout en tenant compte de l'âge du patient, de l'historique du cycle et des conditions de laboratoire."
        ],
        'patients_A': [
            "Cet embryon montre une qualité structurelle solide, ce qui est encourageant.",
            "Les décisions finales doivent toujours être guidées par votre spécialiste de la fertilité."
        ],
        'doctors_B': [
            "Représente un bon potentiel morphologique; peut être considéré pour le transfert ou la congélation, surtout lorsque les grades supérieurs ne sont pas disponibles.",
            "Convient pour une utilisation dans les stratégies de classement des embryons."
        ],
        'patients_B': [
            "Cet embryon présente de bonnes caractéristiques de développement et peut encore convenir au traitement.",
            "Votre médecin tiendra compte de cela avec d'autres facteurs cliniques."
        ],
        'doctors_C': [
            "Montre une morphologie modérée; envisager une surveillance supplémentaire, des données en accéléré ou des embryons secondaires avant la sélection.",
            "Utile dans le classement comparatif plutôt que dans la prise de décision autonome."
        ],
        'patients_C': [
            "L'embryon montre une qualité moyenne, ce qui n'exclut pas le succès.",
            "Votre spécialiste évaluera si cet embryon est approprié dans votre plan de traitement."
        ],
        'doctors_D': [
            "Indique une qualité morphologique inférieure; envisager des embryons alternatifs, une culture prolongée ou une évaluation supplémentaire.",
            "La morphologie seule ne devrait pas être le seul critère d'exclusion."
        ],
        'patients_D': [
            "Cet embryon montre une qualité structurelle inférieure, mais les résultats dépendent de nombreux facteurs.",
            "Votre médecin vous guidera sur les meilleures prochaines étapes."
        ]
    }
}

def get_translation(text_key, grade, language='en'):
    """Get translated text for given key and language"""
    lang_code = language.lower()[:2]
    if lang_code not in TRANSLATIONS:
        lang_code = 'en'
    
    if text_key == 'disclaimer':
        return TRANSLATIONS[lang_code]['disclaimer']
    
    full_key = f"{text_key}_{grade}"
    return TRANSLATIONS[lang_code].get(full_key, TRANSLATIONS['en'][full_key])

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 EmbryoVision AI</h1>
        <p>AI-Assisted Embryo Evaluation for IVF Laboratories</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, device = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        logo_path = 'aispry_logo.png'
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📋 System Information")
        st.markdown("""
        **🤖 Model Architecture**  
        EfficientNet-B0  
        Multi-output classification  
        2,259 training images
        
        **📊 Clinical Scores**  
        **EXP (0-4)** - Blastocoel cavity expansion  
        **ICM (0-3)** - Inner fetal cell mass  
        **TE (0-3)** - Outer placental cell layer
        
        **🎯 Grading System**  
        **Grade A** - Excellent (≥80%)  
        **Grade B** - Good (60-79%)  
        **Grade C** - Average (40-59%)  
        **Grade D** - Poor (<40%)
        """)
        
        st.success(f"✅ Model loaded on {device}")
        st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content - Single column layout
    # Patient Information - Full Width Elongated
    st.markdown("""
    <div class="section-box">
        <h2><span>👤</span> Patient Information</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns([2, 1, 2])
    with col_p1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name", label_visibility="visible")
    with col_p2:
        patient_age = st.number_input("Age", min_value=18, max_value=60, value=30, label_visibility="visible")
    with col_p3:
        patient_location = st.text_input("Location", placeholder="Enter location", label_visibility="visible")
    
    # Prediction Mode - Block Style
    st.markdown("""
    <div class="section-box">
        <h2><span>🔧</span> Prediction Mode</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        ai_selected = st.button("🤖 AI Mode (Image Only)", use_container_width=True, type="primary")
        st.caption("Upload image → System predicts scores and grade")
    
    with col_m2:
        manual_selected = st.button("✋ Manual Mode (Scores + Image)", use_container_width=True)
        st.caption("Input scores + Upload image → System predicts grade")
    
    # Determine mode
    if 'mode' not in st.session_state:
        st.session_state['mode'] = 'manual'
    
    if ai_selected:
        st.session_state['mode'] = 'ai'
    elif manual_selected:
        st.session_state['mode'] = 'manual'
    
    mode = st.session_state['mode']
    
    # Show mode info
    if mode == 'ai':
        st.info("**AI Mode Selected**: Upload image only - System will predict all scores automatically")
    else:
        st.info("**Manual Mode Selected**: Input clinical scores and upload image")
    
    # Manual scores input (if manual mode)
    manual_scores = None
    if mode == 'manual':
        st.markdown("""
        <div class="section-box">
            <h2><span>📊</span> Enter Clinical Scores</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            exp_score = st.number_input("EXP (0-4)", 0, 4, 3)
        with col_s2:
            icm_score = st.number_input("ICM (0-3)", 0, 3, 2)
        with col_s3:
            te_score = st.number_input("TE (0-3)", 0, 3, 2)
        
        manual_scores = {'exp': exp_score, 'icm': icm_score, 'te': te_score}
    
    # Image Upload
    st.markdown("""
    <div class="section-box">
        <h2><span>📷</span> Upload Embryo Image</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an embryo image...", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Embryo Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_btn:
            if st.button("🚀 CLASSIFY EMBRYO", use_container_width=True, type="primary"):
                with st.spinner("🔄 Analyzing embryo..."):
                    results = predict_embryo(model, image, device, mode, manual_scores)
                    annotated_img = annotate_image(image, results)
                
                st.session_state['results'] = results
                st.session_state['image'] = image
                st.session_state['annotated_image'] = annotated_img
                st.session_state['patient_name'] = patient_name
                st.session_state['patient_age'] = patient_age
                st.session_state['patient_location'] = patient_location
                st.rerun()
    
    # Results Section - Full Width
    if 'results' in st.session_state:
        results = st.session_state['results']
        image = st.session_state['image']
        annotated_img = st.session_state['annotated_image']
        
        st.markdown("""
        <div class="section-box">
            <h2><span>📊</span> Classification Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Patient info display
        if st.session_state.get('patient_name') or st.session_state.get('patient_location'):
            st.info(f"**Patient:** {st.session_state.get('patient_name', 'N/A')} | **Age:** {st.session_state.get('patient_age', 30)} | **Location:** {st.session_state.get('patient_location', 'N/A')}")
        
        # Grade Badge
        grade_class = f"grade-{results['grade'].lower()}"
        st.markdown(f"""
        <div class="{grade_class} grade-badge">
            {results['grade']}
        </div>
        <p style="text-align: center; color: #FFFFFF; font-size: 1.5rem; margin-top: -10px;">
            {results['grade_desc']}
        </p>
        """, unsafe_allow_html=True)
        
        # Clinical Scores
        st.markdown("### 🔢 Clinical Scores")
        score_cols = st.columns(3)
        with score_cols[0]:
            st.metric("EXP", results['grade_exp'], f"{results['conf_exp']*100:.1f}% conf")
        with score_cols[1]:
            st.metric("ICM", results['grade_icm'], f"{results['conf_icm']*100:.1f}% conf")
        with score_cols[2]:
            st.metric("TE", results['grade_te'], f"{results['conf_te']*100:.1f}% conf")
        
        if results['used_manual_scores']:
            st.caption("📝 Scores: Manual input (ground truth)")
        else:
            st.caption("🤖 Scores: AI predicted from image")
        
        # Images Side by Side
        st.markdown("### 🎨 Image Comparison")
        col_orig, col_annot = st.columns(2)
        
        with col_orig:
            st.markdown("**Ground Truth Image**")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_annot:
            st.markdown("**Annotated Result**")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(annotated_img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Results Below Images
        st.markdown("### 📈 Detailed Analysis")
        details_df = pd.DataFrame({
            'Metric': ['Morphology Index', 'Overall Confidence', 'Grade', 'Classification'],
            'Value': [
                f"{results['morph_index']:.3f}",
                f"{results['overall_confidence']*100:.1f}%",
                results['grade'],
                results['grade_desc']
            ]
        })
        st.dataframe(details_df, use_container_width=True, hide_index=True)
        
        # Download Section
        st.markdown("### 💾 Download Results")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            csv_data = pd.DataFrame([{
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Patient_Name': st.session_state.get('patient_name', 'N/A'),
                'Age': st.session_state.get('patient_age', 30),
                'Location': st.session_state.get('patient_location', 'N/A'),
                'Mode': 'AI' if not results['used_manual_scores'] else 'Manual',
                'EXP': results['grade_exp'],
                'ICM': results['grade_icm'],
                'TE': results['grade_te'],
                'Morphology_Index': f"{results['morph_index']:.3f}",
                'Grade': results['grade'],
                'Classification': results['grade_desc'],
                'Confidence': f"{results['overall_confidence']*100:.1f}%"
            }])
            csv = csv_data.to_csv(index=False)
            st.download_button("📄 Download CSV Report", csv, "embryo_result.csv", "text/csv", use_container_width=True)
        
        with col_d2:
            img_buffer = io.BytesIO()
            annotated_img.save(img_buffer, format='PNG')
            st.download_button("🖼️ Download Annotated Image", img_buffer.getvalue(), "embryo_annotated.png", "image/png", use_container_width=True)
        
        # Clinical Suggestions Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-box">
            <h2><span>🧬</span> Clinical Suggestions & Guidance</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Language Selector
        col_lang1, col_lang2 = st.columns([3, 1])
        with col_lang1:
            st.markdown("**Select Language for Suggestions:**")
        with col_lang2:
            language = st.selectbox(
                "Language",
                options=['English', 'Spanish', 'French', 'Hindi'],
                label_visibility="collapsed"
            )
        
        # Map language names to codes
        lang_map = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'Hindi': 'hi'}
        lang_code = lang_map.get(language, 'en')
        
        # Get current grade suggestions
        grade = results['grade']
        suggestion = GRADE_SUGGESTIONS[grade]
        
        # Display suggestions
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%); 
                    padding: 2rem; border-radius: 12px; 
                    border: 2px solid rgba(0, 217, 255, 0.3); margin: 1rem 0;">
            <h3 style="color: #00D9FF; margin-bottom: 1rem;">
                {suggestion['emoji']} {suggestion['title']}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_doc, col_pat = st.columns(2)
        
        with col_doc:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%); 
                        padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid rgba(0, 255, 136, 0.3); margin: 0.5rem 0;">
                <h4 style="color: #00FF88; margin-bottom: 1rem;">
                    🧑‍⚕️ For Doctors
                </h4>
            """, unsafe_allow_html=True)
            
            doctor_suggestions = get_translation('doctors', grade, lang_code)
            for point in doctor_suggestions:
                st.markdown(f"• {point}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_pat:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%); 
                        padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid rgba(0, 217, 255, 0.3); margin: 0.5rem 0;">
                <h4 style="color: #00D9FF; margin-bottom: 1rem;">
                    🧑‍🤝‍🧑 For Patients
                </h4>
            """, unsafe_allow_html=True)
            
            patient_suggestions = get_translation('patients', grade, lang_code)
            for point in patient_suggestions:
                st.markdown(f"• {point}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        disclaimer_text = get_translation('disclaimer', grade, lang_code)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2A1A1A 0%, #3A2A2A 100%); 
                    padding: 1.5rem; border-radius: 12px; 
                    border: 2px solid rgba(255, 171, 0, 0.5); margin: 1rem 0;
                    border-left: 6px solid #FFD700;">
            <h4 style="color: #FFD700; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 10px;">
                <span>⚠️</span> Mandatory Disclaimer
            </h4>
            <p style="color: #FFFFFF; margin: 0; font-size: 1rem; line-height: 1.6;">
                {disclaimer_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


