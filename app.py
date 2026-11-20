"""Interface web (Streamlit) que simula o momento da postagem em uma rede social.

O usuário envia uma foto (ou tira pela câmera), o sistema analisa e, se encontrar
um gesto de risco, mostra o alerta antes da "publicação". A decisão final é do usuário.

Executar:  streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st

from src import GESTOS, DetectorGestos

st.set_page_config(page_title="Detector de gestos de risco", page_icon="✋", layout="wide")


@st.cache_resource
def carregar_detector(usar_yolo: bool) -> DetectorGestos:
    return DetectorGestos(usar_yolo=usar_yolo)


def para_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("Sobre")
    st.write(
        "Prototipo que analisa a foto **antes da publicacao** e alerta quando ha "
        "gestos com as maos que podem ser interpretados como sinais de faccoes criminosas."
    )
    st.subheader("Gestos monitorados")
    for g in GESTOS.values():
        if g.risco:
            st.markdown(f"**{g.nome}** — {g.descricao}")
    st.divider()
    usar_yolo = st.toggle("Usar YOLO para localizar pessoas", value=True,
                          help="Melhora a deteccao de maos pequenas em fotos de grupo.")
    mostrar_pessoas = st.toggle("Mostrar caixas das pessoas", value=False)

st.title("✋ Verificacao antes de publicar")
st.caption("YOLO (pessoas) + MediaPipe (pontos da mao) + OpenCV (processamento e alerta)")

aba_upload, aba_camera = st.tabs(["Enviar foto", "Tirar foto"])
with aba_upload:
    arquivo = st.file_uploader("Escolha a foto que voce vai postar", type=["jpg", "jpeg", "png", "webp"])
with aba_camera:
    foto_cam = st.camera_input("Tire uma foto") if st.toggle("Ativar camera") else None

entrada = foto_cam or arquivo
if entrada is None:
    st.info("Envie ou tire uma foto para comecar.")
    st.stop()

img = cv2.imdecode(np.frombuffer(entrada.getvalue(), np.uint8), cv2.IMREAD_COLOR)
det = carregar_detector(usar_yolo)
with st.spinner("Analisando a foto..."):
    res = det.analisar(img)

col1, col2 = st.columns(2)
col1.image(para_rgb(img), caption="Foto original", width="stretch")
col2.image(para_rgb(det.desenhar(img, res, mostrar_pessoas)), caption="Analise", width="stretch")

st.write(f"Pessoas encontradas: **{len(res.pessoas)}** · Maos analisadas: **{len(res.maos)}**")

if not res.alerta:
    st.success("Nenhum gesto de risco encontrado. A foto pode ser publicada.")
    if st.button("Publicar", type="primary"):
        st.balloons()
        st.success("Foto publicada (simulacao).")
    st.stop()

st.error(f"⚠️ Atencao: encontramos {len(res.maos_de_risco)} gesto(s) que podem ser associados a faccoes criminosas.")
for m in res.maos_de_risco:
    st.markdown(f"- **{m.gesto.nome}** (dedos esticados: {', '.join(m.dedos_esticados)}): {m.gesto.descricao}")
st.warning(
    "Pessoas ja foram confundidas com membros de faccoes por causa de gestos em fotos. "
    "Voce pode publicar mesmo assim, publicar uma versao com o gesto borrado ou cancelar."
)

segura = det.borrar_gestos(img, res)
with st.expander("Ver versao com o gesto borrado", expanded=True):
    st.image(para_rgb(segura), width="stretch")

c1, c2, c3 = st.columns(3)
if c1.button("Publicar versao borrada", type="primary"):
    st.success("Versao borrada publicada (simulacao).")
c2.download_button("Baixar versao borrada", cv2.imencode(".jpg", segura)[1].tobytes(),
                   file_name="foto_segura.jpg", mime="image/jpeg")
if c3.button("Publicar original mesmo assim"):
    st.info("Foto original publicada por decisao do usuario (simulacao).")
