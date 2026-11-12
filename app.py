import os
import math
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from skimage.morphology import skeletonize

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)

# -------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CORPORATIVOS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Auscultación de Pavimentos · ASTM D6433",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f6f5; }
    .header-box {
        background: linear-gradient(135deg, #143d2b 0%, #1e563d 100%);
        padding: 20px 26px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .header-box h1 { margin: 0; font-size: 24px; font-weight: 700; color: #ffffff; }
    .header-box p { margin: 4px 0 0 0; font-size: 14px; color: #d1e3da; }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #dce3df;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
    }
    .kpi-card-dark {
        background-color: #143d2b;
        color: #ffffff;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #5c6b63; font-weight: 600; }
    .kpi-label-dark { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #a8c7b8; font-weight: 600; }
    .kpi-value { font-size: 22px; font-weight: 700; color: #143d2b; margin-top: 4px; }
    .kpi-value-dark { font-size: 22px; font-weight: 700; color: #ffffff; margin-top: 4px; }
    .rec-box {
        background-color: #ffffff;
        border-left: 5px solid #143d2b;
        padding: 12px 16px;
        border-radius: 6px;
        border-top: 1px solid #dce3df;
        border-right: 1px solid #dce3df;
        border-bottom: 1px solid #dce3df;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# CONSTANTES Y MODELO
# -------------------------------------------------------------------------
RUTA_PESOS = "best.pt"
ESCALA_BASE = 0.25
ALTURA_BASE_CM = 50.0
ANGULO_ALERTA = 15
UMBRAL_BAJA = 10.0
UMBRAL_ALTA = 25.0

COLORES_BGR = {"Baja": (0, 200, 0), "Media": (0, 165, 255), "Alta": (0, 0, 255)}
HEX_SEV = {"Baja": "#2e8b3d", "Media": "#e08a00", "Alta": "#c62828", "Sin fallas": "#143d2b"}
ORDEN_SEV = {"Baja": 1, "Media": 2, "Alta": 3}

RECOMENDACION = {
    "Baja": "Sellado asfáltico con emulsión modificada",
    "Media": "Fresado localizado y parchado superficial",
    "Alta": "Parchado profundo con reposición estructural",
    "Sin fallas": "Sin intervención; monitoreo rutinario",
}

@st.cache_resource
def cargar_modelo():
    return YOLO(RUTA_PESOS)

modelo = cargar_modelo()

def clasificar_severidad(ancho_mm):
    if ancho_mm < UMBRAL_BAJA:
        return "Baja"
    elif ancho_mm <= UMBRAL_ALTA:
        return "Media"
    return "Alta"

def procesar_opencv_clasico(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilatada = cv2.dilate(bordes, kernel, iterations=1)
    salida = img_bgr.copy()
    salida[dilatada > 0] = (0, 0, 255)
    return cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)

def generar_pdf_reporte(datos, img_yolo_rgb, img_cv_rgb):
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4, rightMargin=1.8*cm, leftMargin=1.8*cm, topMargin=1.8*cm, bottomMargin=1.8*cm)
    estilos = getSampleStyleSheet()
    verde = colors.HexColor("#143d2b")

    titulo_st = ParagraphStyle("Titulo", parent=estilos["Heading1"], fontSize=15, textColor=verde, spaceAfter=4)
    sub_st = ParagraphStyle("Sub", parent=estilos["Normal"], fontSize=9.5, textColor=colors.grey, spaceAfter=14)
    h2_st = ParagraphStyle("H2", parent=estilos["Heading2"], fontSize=11.5, textColor=verde, spaceBefore=10, spaceAfter=6)
    norm_st = ParagraphStyle("Norm", parent=estilos["Normal"], fontSize=9.5, alignment=TA_JUSTIFY, leading=13)

    elementos = []
    elementos.append(Paragraph("INFORME TÉCNICO DE AUSCULTACIÓN DE PAVIMENTOS", titulo_st))
    elementos.append(Paragraph(f"Evaluación automatizada bajo estándar ASTM D6433 · Fecha: {datos['fecha']}", sub_st))

    elementos.append(Paragraph("1. Parámetros de Calibración Fotogramétrica", h2_st))
    elementos.append(Paragraph(
        f"Modo empleado: <b>{datos['modo']}</b>. Escala métrica calculada: <b>{datos['escala']:.4f} mm/px</b>. "
        f"{datos['nota_geom']}", norm_st
    ))

    elementos.append(Paragraph("2. Indicadores Resumen del Tramo Evaluado", h2_st))
    kpi_data = [
        ["Total de fallas", "Ancho máx. crítico", "Área total dañada", "Severidad global"],
        [str(datos["total"]), f"{datos['ancho_max']:.1f} mm", f"{datos['area_total']:.1f} mm²", datos["sev_global"]]
    ]
    t_kpi = Table(kpi_data, colWidths=[4.2*cm]*4)
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), verde),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
    ]))
    elementos.append(t_kpi)
    elementos.append(Spacer(1, 8))
    elementos.append(Paragraph(f"<b>Recomendación técnica de intervención:</b> {datos['recomendacion']}", norm_st))

    elementos.append(Paragraph("3. Evidencia Visual Comparativa", h2_st))
    f_yolo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    f_cv = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(img_yolo_rgb).save(f_yolo.name)
    Image.fromarray(img_cv_rgb).save(f_cv.name)

    t_imgs = Table([
        [RLImage(f_yolo.name, width=8.1*cm, height=6.2*cm), RLImage(f_cv.name, width=8.1*cm, height=6.2*cm)],
        ["Segmentación YOLOv8n-seg (ASTM D6433)", "Filtro OpenCV Clásico (Referencia)"]
    ], colWidths=[8.5*cm, 8.5*cm])
    t_imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, 1), 8.5),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey),
    ]))
    elementos.append(t_imgs)

    if not datos["df"].empty:
        elementos.append(Paragraph("4. Detalle Cuantitativo de Defectos Detectados", h2_st))
        tabla_det = [datos["df"].columns.tolist()] + datos["df"].values.tolist()
        t_det = Table(tabla_det, repeatRows=1)
        t_det.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), verde),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ]))
        elementos.append(t_det)

    doc.build(elementos)
    return tmp_pdf.name

# -------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# -------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>Auscultación de Pavimentos · ASTM D6433</h1>
    <p>Detección, medición y clasificación de severidad de fallas mediante YOLOv8n-seg y soporte fotogramétrico 3D</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Calibración & Configuración")
    modo = st.radio(
        "Modo de calibración",
        ["Soporte 3D Paramétrico (Altura y Ángulo)", "Calibración Manual con Testigo Métrico"]
    )

    alerta_geom = None
    if modo.startswith("Soporte 3D"):
        altura = st.slider("Altura del soporte (cm)", 20, 150, 50)
        angulo = st.slider("Inclinación respecto a la vertical (°)", 0, 45, 0)
        escala_mm_px = ESCALA_BASE * (altura / ALTURA_BASE_CM) / max(math.cos(math.radians(angulo)), 0.1)
        if angulo > ANGULO_ALERTA:
            alerta_geom = f"Inclinación elevada ({angulo}° > {ANGULO_ALERTA}°). Posible distorsión perspectiva."
    else:
        ref_mm = st.number_input("Longitud real testigo (mm)", min_value=10.0, value=300.0)
        ref_px = st.number_input("Longitud en imagen (px)", min_value=1.0, value=1200.0)
        escala_mm_px = ref_mm / ref_px

    st.info(f"Escala activa: **{escala_mm_px:.4f} mm/px**")
    if alerta_geom:
        st.warning(alerta_geom)

    conf = st.slider("Umbral de confianza", 0.05, 0.90, 0.25, 0.05)
    archivo_img = st.file_uploader("Cargar fotografía del pavimento", type=["jpg", "jpeg", "png"])

if archivo_img is None:
    st.info("Sube una fotografía del pavimento en el panel izquierdo para iniciar la auscultación automatizada.")
else:
    img_pil = Image.open(archivo_img).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    resultados = modelo.predict(img_bgr, conf=conf, verbose=False)[0]
    overlay = img_bgr.copy()
    filas = []

    if resultados.masks is not None:
        mascaras = resultados.masks.data.cpu().numpy()
        cajas = resultados.boxes
        for idx, m_raw in enumerate(mascaras):
            m_bin = (cv2.resize(m_raw, (w, h)) > 0.5).astype(np.uint8)
            area_px = int(np.sum(m_bin))
            if area_px == 0:
                continue

            esq = skeletonize(m_bin > 0).astype(np.uint8)
            largo_px = max(int(np.sum(esq)), 1)
            ancho_medio_px = area_px / largo_px

            dist = cv2.distanceTransform(m_bin, cv2.DIST_L2, 5)
            ancho_max_px = float(np.max(dist) * 2.0)

            largo_mm = largo_px * escala_mm_px
            ancho_max_mm = ancho_max_px * escala_mm_px
            ancho_medio_mm = ancho_medio_px * escala_mm_px
            area_mm2 = area_px * (escala_mm_px ** 2)

            sev = clasificar_severidad(ancho_max_mm)
            cls_id = int(cajas.cls[idx].item())
            clase_nombre = modelo.names.get(cls_id, "Falla")
            conf_val = float(cajas.conf[idx].item())

            color_bgr = COLORES_BGR[sev]
            coloreada = np.zeros_like(overlay)
            coloreada[m_bin > 0] = color_bgr
            overlay = cv2.addWeighted(overlay, 1.0, coloreada, 0.45, 0)

            ys, xs = np.where(m_bin > 0)
            cx, cy = int(np.mean(xs)), int(np.min(ys))
            etiqueta = f"#{idx+1} {sev} {ancho_max_mm:.1f}mm"
            cv2.putText(overlay, etiqueta, (max(cx - 40, 5), max(cy - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            filas.append({
                "ID": idx + 1,
                "Clase": clase_nombre,
                "Conf.": round(conf_val, 2),
                "Largo (mm)": round(largo_mm, 1),
                "Ancho máx (mm)": round(ancho_max_mm, 1),
                "Ancho medio (mm)": round(ancho_medio_mm, 1),
                "Área (mm²)": round(area_mm2, 1),
                "Severidad": sev
            })

    img_yolo_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    img_cv_rgb = procesar_opencv_clasico(img_bgr)
    df_defectos = pd.DataFrame(filas)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_yolo_rgb, caption="YOLOv8n-seg (Segmentación y Severidad ASTM D6433)", use_container_width=True)
    with col2:
        st.image(img_cv_rgb, caption="OpenCV Clásico (Referencia)", use_container_width=True)

    total_fallas = len(df_defectos)
    if total_fallas > 0:
        ancho_critico = float(df_defectos["Ancho máx (mm)"].max())
        area_total = float(df_defectos["Área (mm²)"].sum())
        sev_global = max(df_defectos["Severidad"], key=lambda s: ORDEN_SEV[s])
    else:
        ancho_critico, area_total, sev_global = 0.0, 0.0, "Sin fallas"

    rec_texto = RECOMENDACION[sev_global]
    color_sev = HEX_SEV[sev_global]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card-dark"><div class="kpi-label-dark">Total de fallas</div><div class="kpi-value-dark">{total_fallas}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Ancho máximo crítico</div><div class="kpi-value">{ancho_critico:.1f} mm</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Área total dañada</div><div class="kpi-value">{area_total:.1f} mm²</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Severidad global</div><div class="kpi-value" style="color:{color_sev};">{sev_global}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="rec-box"><b>RECOMENDACIÓN DE MANTENIMIENTO:</b> {rec_texto}</div>', unsafe_allow_html=True)

    st.subheader("Detalle de Defectos")
    if total_fallas > 0:
        st.dataframe(df_defectos, use_container_width=True, hide_index=True)
    else:
        st.write("No se detectaron fallas con el umbral de confianza seleccionado.")

    datos_pdf = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modo": modo,
        "escala": escala_mm_px,
        "nota_geom": alerta_geom if alerta_geom else "Geometría de captura dentro de tolerancia.",
        "total": total_fallas,
        "ancho_max": ancho_critico,
        "area_total": area_total,
        "sev_global": sev_global,
        "recomendacion": rec_texto,
        "df": df_defectos
    }
    ruta_pdf = generar_pdf_reporte(datos_pdf, img_yolo_rgb, img_cv_rgb)
    with open(ruta_pdf, "rb") as f_pdf:
        st.download_button(
            label="Descargar Reporte Técnico (PDF)",
            data=f_pdf,
            file_name=f"Reporte_ASTM_D6433_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary"
        )