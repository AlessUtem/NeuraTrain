import streamlit as st

# Configurar la página principal
st.set_page_config(page_title="NeuraTrain", page_icon="🤖", layout="centered")

# Título y subtítulo
st.markdown("<h1 style='text-align: center; '>⚡NeuraTrain🧠</h1>", unsafe_allow_html=True)
st.markdown("NeuraTrain es una appweb que te permite visualizar y entrenar una red neuronal a través de su interfaz.", unsafe_allow_html=True)
st.divider()

# Botones de redirección con HTML
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <a href="/Modelo_Categorico" target="_self">
            <button style="width:100%; padding:10px; background-color:#4CAF50; color:white; border:none; border-radius:5px;">
                Modelo Clasificación de Imágenes
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <a href="/Modelo_Regresion" target="_self">
            <button style="width:100%; padding:10px; background-color:#2196F3; color:white; border:none; border-radius:5px;">
                Modelo Regresión/Clasificación CSV
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

# Pie de página
st.markdown("---")
st.caption("Desarrollado por Alessandro Robles para la Universidad Tecnológica Metropolitana")
