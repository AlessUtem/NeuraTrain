import streamlit as st

# Configurar la página principal
st.set_page_config(page_title="NeuraTrain", page_icon="🤖", layout="centered")

# Título y subtítulo
st.markdown("<h1 style='text-align: center;'>⚡NeuraTrain🧠</h1>", unsafe_allow_html=True)
st.markdown(
    "**Explora y entrena redes neuronales de manera intuitiva y accesible, diseñado para todos los niveles de experiencia.**",
    unsafe_allow_html=True
)

with st.expander("Más sobre NeuraTrain"):
    st.markdown("""
    **¿Qué es NeuraTrain?**  
    NeuraTrain es una aplicación web interactiva creada para democratizar el aprendizaje y la experimentación con redes neuronales. Con un enfoque en la simplicidad y la accesibilidad, permite a los usuarios explorar datasets preconfigurados, configurar modelos de aprendizaje profundo y entrenar redes neuronales sin necesidad de conocimientos técnicos avanzados.
    
    **Características principales**:
    - Interfaz intuitiva para guiar a usuarios novatos.
    - Datasets precargados como MNIST, CIFAR-10, Iris, Boston Housing, entre otros.
    - Configuración flexible de modelos con capas, hiperparámetros y opciones avanzadas como Early Stopping.
    - Visualización gráfica de resultados para facilitar el análisis y aprendizaje.
    - Ideal para educación y experimentos rápidos.

    **Objetivo:**  
    Brindar una experiencia de usuario accesible y enriquecedora que fomente la curiosidad y el aprendizaje en inteligencia artificial, empoderando a usuarios de todos los niveles para experimentar y comprender redes neuronales.
    """)
# Botones de redirección con HTML
st.markdown("### Navega por los modelos disponibles:")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <a href="/Modelo_Clasificacion_de_imagenes" target="_self">
            <button style="width:100%; padding:20px 30px; background-color:#4CAF50; color:white; border:none; border-radius:10px; margin:10px;">
                Modelo Clasificación de Imágenes
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <a href="/Modelo_Regresion_y_Clasificacion_CSV" target="_self">
            <button style="width:100%; padding:20px 30px; background-color:#2196F3; color:white; border:none; border-radius:10px; margin:10px;">
                Modelo Regresión/Clasificación CSV
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )


# Menú de introducción y conceptos clave
with st.expander("Introducción y conceptos clave"):
    st.markdown("""
    ### Introducción
    - **¿Qué es una red neuronal?**
      Las redes neuronales son sistemas computacionales inspirados en las redes de neuronas humanas. Se utilizan en tareas como clasificación de imágenes, predicción de datos y más.
    - **¿Qué ofrece NeuraTrain?**
      NeuraTrain permite explorar dos modelos principales:
        1. **Modelo de Clasificación de Imágenes**: Para categorizar imágenes en diferentes clases.
        2. **Modelo de Regresión/Clasificación CSV**: Para realizar predicciones numéricas o categóricas basadas en datos tabulares.

    ### Instrucciones para cada modelo
    **Modelo de Clasificación de Imágenes**
    - **Datasets disponibles**:
      - MNIST: Dígitos escritos a mano (0-9).
      - Fashion MNIST: Ropa y accesorios.
      - CIFAR-10: Imágenes a color clasificadas en 10 categorías.
    - **Pasos**:
      1. Selecciona el dataset deseado.
      2. Configura las capas de la red neuronal (número de capas, tamaño, etc.).
      3. Ajusta los hiperparámetros: tasa de aprendizaje, batch size, épocas.
      4. Activa Early Stopping para optimizar el entrenamiento.
      5. Entrena y evalúa el modelo.

    **Modelo de Regresión/Clasificación CSV**
    - **Datasets disponibles**:
      - **Clasificación**:
        - Iris: Datos de especies de flores.
        - Wine: Clasificación de vinos.
        - Breast Cancer: Diagnóstico de cáncer de mama.
      - **Regresión**:
        - Boston Housing: Predicción de precios de casas.
        - Diabetes: Predicción de progresión de la diabetes.
    - **Pasos**:
      1. Selecciona el dataset tabular deseado.
      2. Configura las capas de la red neuronal.
      3. Ajusta los hiperparámetros: tasa de aprendizaje, batch size, épocas.
      4. Activa Early Stopping para optimizar el entrenamiento.
      5. Entrena y evalúa el modelo.

    ### Consejos prácticos
    - **Explora los datos**: Siempre revisa las muestras iniciales del dataset antes de entrenar.
    - **Empieza simple**: Utiliza configuraciones básicas para entender el comportamiento inicial del modelo.
    - **Monitorea las métricas**: Asegúrate de revisar la pérdida y precisión para detectar problemas.

    ### Flujo general del entrenamiento
    1. **Selecciona el dataset**: Elige entre los datasets precargados.
    2. **Configura las capas**: Diseña la estructura de tu red neuronal.
    3. **Ajusta hiperparámetros**: Define los valores que controlarán el entrenamiento.
    4. **Activa Early Stopping**: Optimiza el tiempo y evita el sobreajuste.
    5. **Entrena y evalúa**: Observa el desempeño de tu modelo con los datos de prueba.
    """)
# Pie de página
st.markdown("---")
st.caption("Desarrollado por Alessandro Robles para la Universidad Tecnológica Metropolitana")
