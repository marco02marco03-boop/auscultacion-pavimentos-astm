# app.py — Auscultación de Pavimentos · ASTM D6433 (Streamlit Community Cloud)
# Archivos del repo: app.py, best.pt, requirements.txt, packages.txt
import io
import math
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.morphology import skeletonize
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (CondPageBreak, HRFlowable, Image, KeepTogether,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

st.set_page_config(page_title="Auscultación de Pavimentos", page_icon="🛣️", layout="wide")

# =====================================================================
# CONSTANTES
# =====================================================================
VERDE = "#143d2b"                      # verde bosque institucional
COLOR_VERDE = colors.HexColor(VERDE)
VERDE_CLARO = colors.HexColor("#e8f0ec")
TZ_LIMA = timezone(timedelta(hours=-5))
MAX_LADO = 1280                        # reduce fotos grandes (RAM del servidor gratis)
LIMITE_ORTOGONAL = 15.0                # grados
COLORES = {"Baja": (0, 200, 0), "Media": (0, 165, 255), "Alta": (0, 0, 255)}  # BGR
ORDEN = {"Baja": 1, "Media": 2, "Alta": 3}
SEV_PDF = {  # (fondo, texto)
    "Baja": (colors.HexColor("#d4edda"), "#155724"),
    "Media": (colors.HexColor("#fff3cd"), "#856404"),
    "Alta": (colors.HexColor("#f8d7da"), "#721c24"),
}
PRIORIDAD = {
    "Baja": "Preventiva",
    "Media": "Mediano Plazo (máx. 60 días)",
    "Alta": "Crítica / Inmediata (máx. 15 días)",
}
# Costos unitarios referenciales municipales (S/) — pre-inversión
PU = {
    "Baja": ("Sellado asfáltico de fisuras", "m", 18.00),
    "Media": ("Fresado y parchado superficial", "m²", 145.00),
    "Alta": ("Parchado profundo estructural", "m²", 260.00),
}
FACTOR_SANEO = 1.25                    # saneo de bordes para severidad Media/Alta

# =====================================================================
# MODELO
# =====================================================================
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")


modelo = cargar_modelo()

# =====================================================================
# DETECCIÓN, CALIBRACIÓN Y MEDICIÓN
# =====================================================================
def deteccion_clasica(img_bgr, canny_bajo=50, canny_alto=150, area_min=80):
    """Pipeline clásico: gris → Gauss → Canny → morfología → filtro de área."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(blur, canny_bajo, canny_alto)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, k, iterations=2), k)
    n, et, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    mascara = np.zeros_like(dil)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            mascara[et == i] = 255
    ov = img_bgr.copy()
    ov[mascara > 0] = (0, 0, 255)
    return cv2.addWeighted(img_bgr, 0.5, ov, 0.5, 0)


def severidad_astm(ancho_mm):
    return "Baja" if ancho_mm < 10 else ("Media" if ancho_mm <= 25 else "Alta")


def escala_soporte_3d(altura_cm, angulo_deg, ancho_px, hfov_deg):
    """Escala (mm/px) del Soporte 3D paramétrico con modelo de cámara estenopeica.

    La distancia a lo largo del eje óptico hasta el punto principal sobre el pavimento
    es d = h / cos(θ). El ancho de terreno cubierto por la imagen a esa distancia es
    2·d·tan(HFOV/2), que dividido entre el ancho en píxeles da el mm/px.
    Devuelve (escala_mm_px, distorsion_perspectiva_%) donde la distorsión es el
    alargamiento 1/cos(θ) − 1 en el eje de inclinación.
    """
    th = math.radians(angulo_deg)
    d_mm = altura_cm * 10.0 / math.cos(th)
    ancho_terreno_mm = 2.0 * d_mm * math.tan(math.radians(hfov_deg) / 2.0)
    return ancho_terreno_mm / ancho_px, (1.0 / math.cos(th) - 1.0) * 100.0


def medir_defectos(img_bgr, escala, conf):
    """Segmenta con YOLO y mide cada defecto. Devuelve (imagen anotada BGR, DataFrame)."""
    res = modelo.predict(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    anotada, filas = img_bgr.copy(), []
    if res.masks is None:
        return anotada, pd.DataFrame()
    h, w = img_bgr.shape[:2]
    for i, (pol, cls, c) in enumerate(zip(res.masks.xy, res.boxes.cls, res.boxes.conf), 1):
        if len(pol) < 3:
            continue
        nombre = res.names[int(cls)]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pol.astype(np.int32)], 255)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        esq = skeletonize(mask > 0)
        anchos = 2 * dist[esq] if esq.any() else np.array([2 * dist.max()])
        a_max, a_med = anchos.max() * escala, anchos.mean() * escala
        if any(k in nombre.lower() for k in ["pothole", "bache", "hole"]):
            (_, _), (rw, rh), _ = cv2.minAreaRect(pol.astype(np.float32))
            largo = max(rw, rh) * escala
        else:
            largo = esq.sum() * escala
        sev = severidad_astm(a_max)
        filas.append({"ID": i, "Clase": nombre, "Conf.": round(float(c), 2),
                      "Largo (mm)": round(largo, 1), "Ancho máx (mm)": round(a_max, 1),
                      "Ancho medio (mm)": round(a_med, 1),
                      "Área (mm²)": round((mask > 0).sum() * escala ** 2, 1), "Severidad": sev})
        capa = anotada.copy()
        cv2.fillPoly(capa, [pol.astype(np.int32)], COLORES[sev])
        anotada = cv2.addWeighted(capa, 0.45, anotada, 0.55, 0)
        x, y = pol.astype(int).min(axis=0)
        cv2.putText(anotada, f"#{i} {sev} {a_max:.0f}mm", (int(x), max(int(y) - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return anotada, pd.DataFrame(filas)


# =====================================================================
# INDICADORES, METRADO Y TEXTOS TÉCNICOS
# =====================================================================
def calcular_kpis(tabla):
    if tabla.empty:
        return {"n": 0, "ancho_max": 0.0, "area_mm2": 0.0, "area_m2": 0.0, "largo_m": 0.0,
                "sev": None, "prioridad": "Sin intervención inmediata (monitoreo)"}
    sev = max(tabla["Severidad"], key=lambda s: ORDEN[s])
    area_mm2 = float(tabla["Área (mm²)"].sum())
    return {"n": int(len(tabla)), "ancho_max": float(tabla["Ancho máx (mm)"].max()),
            "area_mm2": area_mm2, "area_m2": area_mm2 / 1e6,
            "largo_m": float(tabla["Largo (mm)"].sum()) / 1000.0,
            "sev": sev, "prioridad": PRIORIDAD[sev]}


def calcular_metrado(tabla):
    """Metrado y costo referencial por severidad. Baja: longitud (m). Media/Alta: área × 1.25 (m²)."""
    filas, total = [], 0.0
    for sev in ("Baja", "Media", "Alta"):
        sub = tabla[tabla["Severidad"] == sev] if not tabla.empty else tabla
        if sub.empty:
            continue
        partida, unidad, pu = PU[sev]
        if sev == "Baja":
            metrado = float(sub["Largo (mm)"].sum()) / 1000.0
        else:
            metrado = float(sub["Área (mm²)"].sum()) / 1e6 * FACTOR_SANEO
        parcial = metrado * pu
        total += parcial
        filas.append({"sev": sev, "partida": partida, "unidad": unidad,
                      "metrado": metrado, "pu": pu, "parcial": parcial})
    return filas, total


def texto_diagnostico(k):
    if k["sev"] is None:
        return ("No se detectaron fallas superficiales con el umbral de confianza empleado en el encuadre "
                "evaluado. No se identifica necesidad de intervención inmediata; se recomienda mantener el "
                "monitoreo periódico y verificar que la fotografía represente adecuadamente la superficie.")
    base = (f"Se identificaron {k['n']} falla(s) en el encuadre evaluado, con un ancho máximo crítico de "
            f"{k['ancho_max']:.1f} mm, una longitud acumulada de {k['largo_m']:.2f} m y un área dañada total de "
            f"{k['area_m2']:.4f} m². ")
    if k["sev"] == "Baja":
        det = ("La severidad global es BAJA (ancho máximo inferior a 10 mm): corresponde a fisuración incipiente "
               "sin pérdida de material ni deformación asociada. La carpeta conserva su funcionalidad, pero las "
               "fisuras permiten la infiltración de agua hacia la base y pueden evolucionar si no se sellan. Se "
               "recomienda una intervención preventiva de sellado y el monitoreo periódico de la evolución.")
    elif k["sev"] == "Media":
        det = ("La severidad global es MEDIA (ancho máximo entre 10 y 25 mm): indica fisuración abierta con riesgo "
               "de desprendimiento de bordes, infiltración y debilitamiento progresivo de la base. De no atenderse, "
               "la falla puede evolucionar a severidad alta. Se recomienda programar la reparación localizada "
               "(fresado y parchado superficial) en el mediano plazo.")
    else:
        det = ("La severidad global es ALTA (ancho máximo superior a 25 mm): se presume compromiso estructural de "
               "la carpeta y posible afectación de la base, con riesgo para la seguridad del usuario y deterioro "
               "acelerado por agua y tráfico. Se requiere intervención inmediata de saneamiento estructural y la "
               "evaluación de las causas (drenaje, capacidad portante y espesores) antes de reponer el pavimento.")
    return base + det


PROTOCOLOS = {
    "Baja": ("Sellado de fisuras (severidad Baja)", [
        "Limpieza de la fisura: barrido y soplado con aire comprimido para retirar polvo, material suelto y vegetación; la superficie debe quedar seca.",
        "Ruteo o apertura mecánica de la fisura, cuando el ancho lo permita, para formar un reservorio de sellado uniforme.",
        "Aplicación de imprimante en las caras de la fisura si lo exige el fabricante del sellador.",
        "Sellado con sellador asfáltico (emulsión modificada o sellante elastomérico en caliente), enrasado con la superficie.",
        "Espolvoreo con arena fina para evitar el pegado por tráfico y reapertura al tránsito tras el curado.",
        "Registro fotográfico y monitoreo periódico para verificar la evolución de la fisura.",
    ]),
    "Media": ("Fresado y parchado superficial (severidad Media)", [
        "Señalización y desvío del tránsito en la zona de trabajo.",
        "Delimitación de un área regular que englobe la falla, con margen de saneo en los bordes (factor 1.25 aplicado en el metrado).",
        "Corte vertical con sierra de los bordes y fresado o retiro de la carpeta asfáltica deteriorada hasta la profundidad que indique la inspección.",
        "Limpieza del fondo y de las caras laterales; eliminación de todo material suelto.",
        "Imprimación o riego de liga con emulsión asfáltica en el fondo y en las paredes de la excavación.",
        "Colocación de mezcla asfáltica en caliente en capas, extendida y compactada con plancha vibratoria o rodillo hasta la compactación especificada.",
        "Sellado perimetral de juntas con sellador asfáltico y verificación de nivel y rasante respecto del pavimento adyacente.",
    ]),
    "Alta": ("Parchado profundo estructural (severidad Alta)", [
        "Señalización, desvío del tránsito y protección de la zona de trabajo por seguridad vial.",
        "Delimitación y corte con sierra del área afectada, con margen de saneo de bordes (factor 1.25 aplicado en el metrado).",
        "Demolición y retiro de la carpeta asfáltica y excavación de la base o subbase deteriorada hasta encontrar material firme.",
        "Diagnóstico de causas (drenaje, capacidad portante, espesores) y corrección previa a la reposición: drenaje o estabilización si corresponde.",
        "Conformación y compactación de la base granular con control de humedad y de densidad de campo.",
        "Imprimación asfáltica de la base y riego de liga en las paredes de la excavación.",
        "Reposición de la carpeta con mezcla asfáltica en caliente en capas, con control de temperatura y compactación a la densidad especificada.",
        "Sellado de juntas perimetrales, control de rasante y reapertura al tránsito cuando la mezcla alcance la temperatura adecuada.",
    ]),
}


def dictamen_ortogonalidad(modo3d, angulo, distorsion):
    """Devuelve (texto, es_alerta)."""
    if modo3d:
        if angulo > LIMITE_ORTOGONAL:
            return (f"ALERTA — NO CONFORME: la inclinación de {angulo:.0f}° supera el límite de "
                    f"{LIMITE_ORTOGONAL:.0f}° para una toma cuasi-cenital. Existe distorsión de perspectiva "
                    f"(alargamiento de hasta {distorsion:.1f} % en el eje de inclinación), por lo que las "
                    f"mediciones deben considerarse solo referenciales. Se recomienda repetir la toma con el eje "
                    f"óptico lo más cercano posible a la vertical (0° a {LIMITE_ORTOGONAL:.0f}°).", True)
        return (f"CONFORME: la inclinación de {angulo:.0f}° no supera el límite de {LIMITE_ORTOGONAL:.0f}° de "
                f"ortogonalidad. La distorsión de perspectiva estimada es de {distorsion:.1f} % en el eje de "
                f"inclinación, aceptable para una evaluación fotogramétrica de la superficie.", False)
    return ("NO VERIFICABLE POR ÁNGULO: se empleó calibración manual con testigo métrico. La validez de la escala "
            "depende de que el testigo esté en el mismo plano del pavimento y de que la toma sea ortogonal "
            "(inclinación máxima recomendada: 15°). Se recomienda confirmar estas condiciones en campo.", False)


# =====================================================================
# REPORTE PDF (reportlab)
# =====================================================================
_base = getSampleStyleSheet()


def _ps(name, **kw):
    return ParagraphStyle(name, parent=_base["Normal"], **kw)


S = {
    "titulo": _ps("titulo", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.white),
    "sub": _ps("sub", fontSize=9.5, leading=13, textColor=colors.HexColor("#d8e8df")),
    "h1": _ps("h1", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=COLOR_VERDE,
              spaceBefore=8, spaceAfter=2),
    "h2": _ps("h2", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=COLOR_VERDE,
              spaceBefore=6, spaceAfter=3),
    "body": _ps("body", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY),
    "cell": _ps("cell", fontSize=8.8, leading=11.5),
    "cellb": _ps("cellb", fontName="Helvetica-Bold", fontSize=8.8, leading=11.5),
    "cellc": _ps("cellc", fontSize=8.8, leading=11.5, alignment=TA_CENTER),
    "th": _ps("th", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white,
              alignment=TA_CENTER),
    "nota": _ps("nota", fontSize=7.8, leading=10.5, textColor=colors.HexColor("#555555"),
                alignment=TA_JUSTIFY),
    "caption": _ps("caption", fontSize=8, leading=10.5, textColor=colors.HexColor("#333333"),
                   alignment=TA_JUSTIFY),
    "bullet": _ps("bullet", fontSize=9.2, leading=12.8, leftIndent=14, bulletIndent=3,
                  alignment=TA_JUSTIFY, spaceAfter=1.5),
}
ANCHO = A4[0] - 3.6 * cm   # ancho útil (márgenes de 1.8 cm)


def P(texto, estilo="cell"):
    return Paragraph(texto, S[estilo])


def sev_p(sev, estilo="cellc"):
    """Etiqueta de severidad con color de texto."""
    return Paragraph(f'<font color="{SEV_PDF[sev][1]}"><b>{sev}</b></font>', S[estilo])


def seccion(titulo):
    return [Paragraph(titulo, S["h1"]),
            HRFlowable(width="100%", thickness=1, color=COLOR_VERDE, spaceAfter=5)]


def tabla_kv(filas, anchos):
    """Tabla de dos columnas: etiqueta (fondo verde claro) | valor."""
    t = Table([[P(a, "cellb"), v if not isinstance(v, str) else P(v)] for a, v in filas],
              colWidths=anchos)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), VERDE_CLARO),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9cfc3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _imagen_pdf(img_bgr, ancho_pt, alto_max_pt):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    h, w = img_bgr.shape[:2]
    ancho, alto = ancho_pt, ancho_pt * h / w
    if alto > alto_max_pt:
        ancho, alto = alto_max_pt * w / h, alto_max_pt
    return Image(io.BytesIO(buf.tobytes()), width=ancho, height=alto)


def _fabrica_canvas(codigo):
    """Canvas con pie de página: código del informe y 'Página X de Y'."""
    class _Numerado(rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._paginas = []

        def showPage(self):
            self._paginas.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._paginas)
            for estado in self._paginas:
                self.__dict__.update(estado)
                self._pie(total)
                super().showPage()
            super().save()

        def _pie(self, total):
            self.setStrokeColor(COLOR_VERDE)
            self.setLineWidth(0.8)
            self.line(1.8 * cm, 1.5 * cm, A4[0] - 1.8 * cm, 1.5 * cm)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#555555"))
            self.drawString(1.8 * cm, 1.0 * cm, f"{codigo}  |  Informe generado automáticamente — documento referencial")
            self.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {self._pageNumber} de {total}")
    return _Numerado


def generar_pdf(r, meta):
    """Construye el Informe Técnico Pericial y devuelve los bytes del PDF."""
    tabla, k = r["tabla"], calcular_kpis(r["tabla"])
    fecha_txt = r["fecha"].strftime("%d/%m/%Y %H:%M") + " (hora de Lima)"
    story = []

    # ---- Membrete institucional y metadatos ----
    membrete = Table([[P("INFORME TÉCNICO PERICIAL DE INGENIERÍA VIAL", "titulo")],
                      [P("Auscultación de Pavimentos · ASTM D6433 / Manual de Conservación Vial (MTC)", "sub")]],
                     colWidths=[ANCHO])
    membrete.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_VERDE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 0), ("BOTTOMPADDING", (0, 1), (0, 1), 12),
    ]))
    story += [membrete, Spacer(1, 8)]

    w4 = [3.0 * cm, 5.7 * cm, 3.0 * cm, ANCHO - 11.7 * cm]
    meta_tbl = Table([
        [P("Código de informe", "cellb"), P(r["codigo"]), P("Fecha y hora", "cellb"), P(fecha_txt)],
        [P("Tramo evaluado", "cellb"), P(escape(meta["tramo"])), P("Distrito / ubicación", "cellb"), P(escape(meta["distrito"]))],
        [P("Responsable técnico", "cellb"), P(escape(meta["responsable"])), P("Sistema de análisis", "cellb"), P("YOLOv8n-seg + OpenCV")],
    ], colWidths=w4)
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), VERDE_CLARO), ("BACKGROUND", (2, 0), (2, -1), VERDE_CLARO),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9cfc3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [meta_tbl, Spacer(1, 4)]

    # ---- 1. Calibración y validación geométrica ----
    modo3d = r["modo3d"]
    texto_dict, alerta = dictamen_ortogonalidad(modo3d, r["angulo"] or 0.0, r["distorsion"] or 0.0)
    if modo3d:
        filas = [("Modo de calibración", "Soporte 3D paramétrico (altura y ángulo variables)"),
                 ("Altura del soporte", f"{r['altura']:.0f} cm"),
                 ("Inclinación respecto a la vertical", f"{r['angulo']:.0f}° (0° = toma cenital)"),
                 ("Campo de visión horizontal adoptado", f"{r['hfov']:.0f}°"),
                 ("Escala resultante", f"{r['escala']:.4f} mm/px")]
    else:
        filas = [("Modo de calibración", "Calibración manual con testigo métrico"),
                 ("Testigo métrico", f"{r['ref_mm']:.1f} mm = {r['ref_px']:.0f} px en la foto original"),
                 ("Altura / inclinación", "No aplica (no registradas en este modo)"),
                 ("Escala resultante", f"{r['escala']:.4f} mm/px")]
    caja_bg = colors.HexColor("#f8d7da") if alerta else VERDE_CLARO
    caja_tx = "#721c24" if alerta else VERDE
    dict_tbl = Table([[Paragraph(f'<font color="{caja_tx}"><b>Dictamen técnico de ortogonalidad.</b> '
                                 f'{escape(texto_dict)}</font>', S["body"])]], colWidths=[ANCHO])
    dict_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), caja_bg), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(caja_tx)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(KeepTogether(
        seccion("1. Parámetros de Calibración Fotogramétrica y Validación Geométrica 3D")
        + [tabla_kv(filas, [6.2 * cm, ANCHO - 6.2 * cm]), Spacer(1, 6), dict_tbl]))

    # ---- 2. Indicadores resumen y prioridad ----
    sev = k["sev"]
    kpi_filas = [("Total de fallas detectadas", f"{k['n']}"),
                 ("Ancho máximo crítico", f"{k['ancho_max']:.1f} mm"),
                 ("Longitud acumulada de fallas", f"{k['largo_m']:.2f} m"),
                 ("Área total dañada", f"{k['area_mm2']:,.1f} mm²  =  {k['area_m2']:.4f} m²"),
                 ("Severidad global (ASTM D6433)", sev_p(sev, "cell") if sev else "Sin fallas detectadas"),
                 ("Nivel de prioridad de atención", f"<b>{escape(k['prioridad'])}</b>")]
    kpi_tbl = tabla_kv(kpi_filas, [6.2 * cm, ANCHO - 6.2 * cm])
    if sev:
        kpi_tbl.setStyle(TableStyle([("BACKGROUND", (1, 4), (1, 5), SEV_PDF[sev][0])]))
    story.append(KeepTogether(
        seccion("2. Indicadores Resumen y Nivel de Prioridad Vial") + [kpi_tbl]))

    # ---- 3. Evidencia visual comparativa ----
    celda_w = ANCHO / 2 - 8
    img1 = _imagen_pdf(r["img_yolo"], celda_w, 8.5 * cm)
    img2 = _imagen_pdf(r["img_cv"], celda_w, 8.5 * cm)
    ev = Table([[img1, img2],
                [P("<b>Figura 1.</b> Segmentación con YOLOv8n-seg. Cada máscara se colorea según su severidad "
                   "(verde: baja, naranja: media, rojo: alta) y se rotula con ID, severidad y ancho máximo.", "caption"),
                 P("<b>Figura 2.</b> Filtro clásico OpenCV (escala de grises, desenfoque gaussiano, Canny y "
                   "morfología). Detecta bordes sin distinguir grietas de sombras, manchas o juntas, por lo que "
                   "puede generar falsos positivos.", "caption")]],
               colWidths=[ANCHO / 2, ANCHO / 2])
    ev.setStyle(TableStyle([("VALIGN", (0, 0), (-1, 0), "MIDDLE"), ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                            ("VALIGN", (0, 1), (-1, 1), "TOP"),
                            ("BOX", (0, 0), (0, -1), 0.5, colors.HexColor("#b9cfc3")),
                            ("BOX", (1, 0), (1, -1), 0.5, colors.HexColor("#b9cfc3")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether(seccion("3. Evidencia Visual Comparativa") + [ev]))

    # ---- 4. Detalle cuantitativo ----
    story.append(CondPageBreak(6 * cm))
    story += seccion("4. Detalle Cuantitativo de Defectos Detectados")
    if tabla.empty:
        story.append(P("No se detectaron defectos con el umbral de confianza empleado.", "body"))
    else:
        cab = ["ID", "Clase", "Conf.", "Largo (mm)", "Ancho máx (mm)", "Ancho medio (mm)", "Área (mm²)", "Severidad"]
        datos = [[P(c, "th") for c in cab]]
        estilos = [("BACKGROUND", (0, 0), (-1, 0), COLOR_VERDE),
                   ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9cfc3")),
                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8f6")]),
                   ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        for i, fila in enumerate(tabla.itertuples(index=False), start=1):
            datos.append([P(str(fila[0]), "cellc"), P(escape(str(fila[1]))), P(f"{fila[2]:.2f}", "cellc"),
                          P(f"{fila[3]:,.1f}", "cellc"), P(f"{fila[4]:,.1f}", "cellc"),
                          P(f"{fila[5]:,.1f}", "cellc"), P(f"{fila[6]:,.1f}", "cellc"), sev_p(fila[7])])
            estilos.append(("BACKGROUND", (7, i), (7, i), SEV_PDF[fila[7]][0]))
        t4 = Table(datos, colWidths=[1.1 * cm, 2.9 * cm, 1.7 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.8 * cm, ANCHO - 15.5 * cm],
                   repeatRows=1)
        t4.setStyle(TableStyle(estilos))
        story.append(t4)

    # ---- 5. Diagnóstico, protocolo y estimación ----
    story.append(CondPageBreak(8 * cm))
    story += seccion("5. Diagnóstico Técnico, Protocolo Constructivo y Estimación Referencial de Intervención")
    story.append(Paragraph("5.1 Diagnóstico ingenieril", S["h2"]))
    story.append(P(escape(texto_diagnostico(k)), "body"))
    story.append(Spacer(1, 3))
    story.append(P("<b>Alcance:</b> la severidad se clasifica por el ancho máximo medido sobre la máscara "
                   "segmentada (Baja: menor a 10 mm; Media: de 10 a 25 mm; Alta: mayor a 25 mm). Es una evaluación "
                   "fotogramétrica puntual del encuadre fotografiado y no reemplaza la determinación del índice PCI "
                   "completo según ASTM D6433 (unidades de muestra, tipo, extensión y profundidad de fallas).", "nota"))

    story.append(Paragraph("5.2 Protocolo constructivo recomendado", S["h2"]))
    sevs_presentes = [s for s in ("Alta", "Media", "Baja") if not tabla.empty and (tabla["Severidad"] == s).any()]
    if not sevs_presentes:
        story.append(P("No se requiere protocolo constructivo; mantener el monitoreo periódico.", "body"))
    for s in sevs_presentes:
        titulo, items = PROTOCOLOS[s]
        bloque = [Paragraph(titulo, S["cellb"]), Spacer(1, 2)]
        bloque += [Paragraph(escape(it), S["bullet"], bulletText="•") for it in items]
        bloque.append(Spacer(1, 4))
        story.append(KeepTogether(bloque))
    if sevs_presentes:
        story.append(P("Las especificaciones, tolerancias y ensayos de control deben verificarse con las "
                       "Especificaciones Técnicas Generales para Construcción de Carreteras (MTC) vigentes y con el "
                       "expediente técnico de la intervención.", "nota"))

    story.append(Spacer(1, 4))
    metrado, total = calcular_metrado(tabla)
    bloque53 = [Paragraph("5.3 Cuadro de metrado y costo referencial estimado (S/)", S["h2"])]
    if not metrado:
        bloque53.append(P("No hay metrado afectado que valorizar.", "body"))
    else:
        cab = ["Severidad", "Partida", "Unid.", "Metrado", "P.U. (S/)", "Parcial (S/)"]
        datos = [[P(c, "th") for c in cab]]
        estilos = [("BACKGROUND", (0, 0), (-1, 0), COLOR_VERDE),
                   ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b9cfc3")),
                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                   ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        for i, f in enumerate(metrado, start=1):
            detalle = ("longitud total de fisuras" if f["sev"] == "Baja"
                       else f"área × {FACTOR_SANEO:.2f} (saneo de bordes)")
            datos.append([sev_p(f["sev"]),
                          P(f'{f["partida"]}<br/><font size="7.5" color="#555555">Metrado: {detalle}</font>'),
                          P(f["unidad"], "cellc"), P(f"{f['metrado']:,.3f}", "cellc"),
                          P(f"{f['pu']:,.2f}", "cellc"), P(f"{f['parcial']:,.2f}", "cellc")])
            estilos.append(("BACKGROUND", (0, i), (0, i), SEV_PDF[f["sev"]][0]))
        n = len(datos)
        datos.append([P("<b>COSTO REFERENCIAL TOTAL (S/)</b>"), "", "", "", "", P(f"<b>{total:,.2f}</b>", "cellc")])
        estilos += [("SPAN", (0, n), (4, n)), ("BACKGROUND", (0, n), (-1, n), VERDE_CLARO)]
        t53 = Table(datos, colWidths=[2.0 * cm, 6.6 * cm, 1.4 * cm, 2.4 * cm, 2.2 * cm, ANCHO - 14.6 * cm])
        t53.setStyle(TableStyle(estilos))
        bloque53 += [t53, Spacer(1, 3),
                     P("<b>Nota:</b> los costos unitarios (sellado asfáltico S/ 18.00/m; fresado y parchado "
                       "superficial S/ 145.00/m²; parchado profundo estructural S/ 260.00/m²) son referenciales "
                       "municipales para fines de pre-inversión y no constituyen un presupuesto de obra. El metrado "
                       "corresponde únicamente al encuadre fotografiado; para presupuestar el tramo se requiere un "
                       "levantamiento de campo y la actualización de precios a la fecha de ejecución.", "nota")]
    story.append(KeepTogether(bloque53))

    # ---- 6. Conformidad y validación profesional ----
    nota_conf = Table([[P("Documento generado de forma automatizada mediante visión artificial (YOLOv8n-seg) a partir "
                          "de registro fotográfico del pavimento. Los resultados constituyen una evaluación "
                          "preliminar y deben ser verificados y validados por un profesional colegiado antes de su "
                          "uso técnico, contractual o pericial.", "body")]], colWidths=[ANCHO])
    nota_conf.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, COLOR_VERDE),
                                   ("BACKGROUND", (0, 0), (-1, -1), VERDE_CLARO),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    firmas = Table([["", "", ""],
                    [P("<b>Elaborado por</b><br/>Ing. Civil — CIP N° ____________", "cellc"), "",
                     P("<b>Revisado / Aprobado</b><br/>Ing. Civil — CIP N° ____________", "cellc")]],
                   colWidths=[7.6 * cm, ANCHO - 15.2 * cm, 7.6 * cm], rowHeights=[1.9 * cm, None])
    firmas.setStyle(TableStyle([("LINEABOVE", (0, 1), (0, 1), 0.8, colors.black),
                                ("LINEABOVE", (2, 1), (2, 1), 0.8, colors.black),
                                ("VALIGN", (0, 1), (-1, 1), "TOP")]))
    story.append(CondPageBreak(8 * cm))
    story.append(KeepTogether(seccion("6. Conformidad y Validación Profesional")
                              + [nota_conf, Spacer(1, 6), firmas]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=2.2 * cm,
                            title=f"Informe Técnico {r['codigo']}", author=meta["responsable"],
                            subject="Auscultación de Pavimentos · ASTM D6433")
    doc.build(story, canvasmaker=_fabrica_canvas(r["codigo"]))
    return buf.getvalue()


# =====================================================================
# INTERFAZ STREAMLIT
# =====================================================================
st.markdown(f"""
<style>
.banner {{background:{VERDE}; color:#fff; padding:22px 28px; border-radius:14px; margin-bottom:18px;}}
.banner h1 {{color:#fff; font-size:1.7rem; margin:0 0 6px 0; padding:0;}}
.banner p {{color:#d8e8df; margin:0; font-size:.95rem;}}
div.stButton > button[kind="primary"], div.stDownloadButton > button {{
    background:{VERDE}; color:#fff; border:1px solid {VERDE}; border-radius:8px;}}
div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button:hover {{
    background:#1f5a40; border-color:#1f5a40; color:#fff;}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{color:{VERDE};}}
h3 {{color:{VERDE};}}
</style>
<div class="banner">
  <h1>Auscultación de Pavimentos · ASTM D6433</h1>
  <p>Detección, medición y clasificación de severidad de fallas mediante YOLOv8-seg y soporte fotogramétrico 3D</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Calibración & Configuración")
    modo = st.radio("Modo de calibración",
                    ["Soporte 3D Paramétrico (Altura y Ángulo variables)", "Calibración Manual con Testigo Métrico"])
    modo3d = modo.startswith("Soporte")
    if modo3d:
        altura = st.slider("Altura del soporte (cm)", 20, 150, 40)
        angulo = st.slider("Inclinación respecto a la vertical (°) — 0° = cenital", 0, 45, 10)
        with st.expander("Parámetros de la cámara"):
            hfov = st.number_input("Campo de visión horizontal (°)", min_value=30.0, max_value=120.0,
                                   value=65.0, step=1.0)
        if angulo > LIMITE_ORTOGONAL:
            st.warning(f"La inclinación supera {LIMITE_ORTOGONAL:.0f}°: habrá distorsión de perspectiva.")
    else:
        ref_mm = st.number_input("Referencia real (mm)", min_value=1.0, value=300.0)
        ref_px = st.number_input("Referencia en la foto (píxeles)", min_value=1.0, value=1200.0)
    conf = st.slider("Umbral de confianza", 0.05, 0.90, 0.25, 0.05)

    st.header("Datos del reporte")
    tramo = st.text_input("Tramo / Vía evaluada", value="Av. Principal — Tramo Muestra 01")
    distrito = st.text_input("Distrito / Ubicación", value="Lima Metropolitana")
    responsable = st.text_input("Responsable técnico", value="Área de Gestión Vial")

archivo = st.file_uploader("Imagen del pavimento", type=["jpg", "jpeg", "png"])

if archivo is not None and st.button("Analizar", type="primary"):
    img = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("No se pudo leer la imagen.")
    else:
        f = min(1.0, MAX_LADO / max(img.shape[:2]))
        if f < 1.0:                       # al reducir la foto, la referencia en px se reduce igual
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if modo3d:
            escala, distorsion = escala_soporte_3d(altura, angulo, img.shape[1], hfov)
        else:
            escala, distorsion = ref_mm / (ref_px * f), None
        with st.spinner("Analizando..."):
            anotada, tabla = medir_defectos(img, escala, conf)
            clasico = deteccion_clasica(img)
        ahora = datetime.now(TZ_LIMA)
        st.session_state["res"] = {
            "tabla": tabla, "img_yolo": anotada, "img_cv": clasico, "escala": escala,
            "modo3d": modo3d, "altura": altura if modo3d else None, "angulo": angulo if modo3d else None,
            "hfov": hfov if modo3d else None, "distorsion": distorsion,
            "ref_mm": None if modo3d else ref_mm, "ref_px": None if modo3d else ref_px,
            "fecha": ahora, "codigo": f"INF-D6433-{ahora:%Y%m%d-%H%M}",
        }

r = st.session_state.get("res")
if r:
    k = calcular_kpis(r["tabla"])
    st.subheader("Resultados")
    c = st.columns(5)
    c[0].metric("Total de fallas", k["n"])
    c[1].metric("Ancho máx. crítico", f"{k['ancho_max']:.1f} mm")
    c[2].metric("Área dañada", f"{k['area_m2']:.4f} m²")
    c[3].metric("Severidad global", k["sev"] or "—")
    c[4].metric("Escala", f"{r['escala']:.4f} mm/px")
    st.info(f"**Prioridad de atención:** {k['prioridad']}")
    if r["modo3d"] and r["angulo"] > LIMITE_ORTOGONAL:
        st.warning(f"Inclinación de {r['angulo']:.0f}° mayor a {LIMITE_ORTOGONAL:.0f}°: "
                   "las mediciones son solo referenciales.")

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(r["img_yolo"], cv2.COLOR_BGR2RGB), caption="YOLOv8n-seg (color = severidad)")
    c2.image(cv2.cvtColor(r["img_cv"], cv2.COLOR_BGR2RGB), caption="OpenCV clásico")

    st.subheader("Medidas y severidad")
    if r["tabla"].empty:
        st.info("No se detectaron defectos con esta confianza.")
    else:
        st.dataframe(r["tabla"])

    try:
        pdf = generar_pdf(r, {"tramo": tramo, "distrito": distrito, "responsable": responsable})
        st.download_button("Descargar Informe Técnico (PDF)", data=pdf,
                           file_name=f"{r['codigo']}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"No se pudo generar el PDF: {e}")
