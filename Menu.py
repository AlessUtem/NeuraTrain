import streamlit as st

# Configurar la página principal
st.set_page_config(page_title="NeuraTrain", page_icon="🤖", layout="centered")

# Título y subtítulo
st.title("NeuraTrain")
st.subheader("Interfaz de entrenamiento de Redes Neuronales Artificiales")

# Espacio entre secciones
st.markdown("---")

# Botones para navegar a otras páginas
st.write("Seleccione el tipo de red neuronal que desea entrenar:")

# Crear columnas para botones
col1, col2 = st.columns(2)

with col1:
    st.markdown('<a href="/Modelo_Categorico" target="_self"><button style="width:100%; padding:10px;">Modelo Categórico </button></a>', unsafe_allow_html=True)
    st.markdown('<a href="/Modelo_Regresion" target="_self"><button style="width:100%; padding:10px;">Modelo Regresión</button></a>', unsafe_allow_html=True)

with col2:
    st.markdown('<a href="/categorical" target="_self"><button style="width:100%; padding:10px;">Clustering</button></a>', unsafe_allow_html=True)
    st.markdown('<a href="/Dataset" target="_self"><button style="width:100%; padding:10px;">Entrena con tu dataset</button></a>', unsafe_allow_html=True)


# Pie de página
st.markdown("---")
st.caption("Desarrollado por NeuraTrainCloud")
