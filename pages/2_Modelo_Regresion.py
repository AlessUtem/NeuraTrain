import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import math
import numpy as np
import time
import pandas as pd

from threading import Thread
from matplotlib.animation import FuncAnimation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Config

def new_line():
    st.write("\n")
 
# with st.sidebar:
#    st.image("./assets/sb-quick.png",  use_container_width=True)


st.markdown("<h1 style='text-align: center; '>⚡NeuraTrain🧠</h1>", unsafe_allow_html=True)
st.markdown("🧬Modelo de regresión/clasificación para datos tabulares(csv)", unsafe_allow_html=True)
st.markdown("👉🏻Utiliza el panel lateral para comenzar a crear tu red neuronal artifical!", unsafe_allow_html=True)

st.divider()


# Inicializar estado de la sesión
if 'graph_positions' not in st.session_state:
    st.session_state['graph_positions'] = None
if 'graph' not in st.session_state:
    st.session_state['graph'] = None
if 'layer_config' not in st.session_state:
    st.session_state['layer_config'] = []
if 'hyperparams' not in st.session_state:
    st.session_state['hyperparams'] = {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'epochs': 5,
        'batch_size': 32,
        'loss_function': 'categorical_crossentropy'
    }
if 'logs' not in st.session_state:
    st.session_state['logs'] = []
if 'training_in_progress' not in st.session_state:
    st.session_state['training_in_progress'] = False
if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = 'Iris'
if 'training_finished' not in st.session_state:
    st.session_state['training_finished'] = False
if 'modelDownload' not in st.session_state:
    st.session_state['modelDownload'] = None
# Inicializar estado de la sesión para el botón de métricas
if 'show_metrics' not in st.session_state:
    st.session_state['show_metrics'] = False    
if 'train_ratio' not in st.session_state:
    st.session_state['train_ratio'] = 80  # Valor predeterminado para el entrenamiento
if 'val_ratio' not in st.session_state:
    st.session_state['val_ratio'] = 10  # Valor predeterminado para la validación
if 'test_ratio' not in st.session_state:
    st.session_state['test_ratio'] = 10  # Valor predeterminado para la prueba

if "dataset_loaded" not in st.session_state:
    st.session_state["dataset_loaded"] = False
if "full_dataset" not in st.session_state:
    st.session_state["full_dataset"] = False
if "X_original" not in st.session_state:
    st.session_state["X_original"] = None
if "y_original" not in st.session_state:
    st.session_state["y_original"] = None
if "columns_removed" not in st.session_state:
    st.session_state["columns_removed"] = False
if "missing_handled" not in st.session_state:
    st.session_state["missing_handled"] = False
if "categorical_encoded" not in st.session_state:
    st.session_state["categorical_encoded"] = False
if "scaling_done" not in st.session_state:
    st.session_state["scaling_done"] = False
if "dataset_split" not in st.session_state:
    st.session_state["dataset_split"] = False

if 'loss_values' not in st.session_state:
    st.session_state['loss_values'] = []
if 'accuracy_values' not in st.session_state:
    st.session_state['accuracy_values'] = []
if 'val_loss_values' not in st.session_state:
    st.session_state['val_loss_values'] = []
if 'val_accuracy_values' not in st.session_state:
    st.session_state['val_accuracy_values'] = []
if 'problem_type' not in st.session_state:
    st.session_state['problem_type'] = "Clasificación"  # Valor predeterminado
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'split_type' not in st.session_state:
    st.session_state['split_type'] = "Entrenamiento y Prueba"  # Valor predeterminado1

# Inyectar CSS para capturar el color de fondo
st.markdown(
    """
    <style>
    .stApp {
        background-color: var(--background-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Capturar el color de fondo actual
def get_background_color():
    # Verificar si es modo claro o oscuro
    return st.session_state.get("background_color", "#ffffff")




# Mostrar gráfico y botón para entrenar
preview_placeholder = st.empty()
dynamic_placeholder = st.empty()

# Función para inicializar configuración de capas según el dataset
def initialize_layer_config(dataset):
    # Verificar si el dataset ya fue inicializado
    if "initialized_dataset" not in st.session_state or st.session_state["initialized_dataset"] != dataset:
        st.session_state["initialized_dataset"] = dataset  # Marcar el dataset como inicializado

        if dataset == 'CIFAR-10':
            # Capas base específicas de CIFAR-10
            st.session_state['layer_config'] = [
                {"name": "Conv2D1", "type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu", "base": True},
                {"name": "Conv2D2", "type": "Conv2D", "filters": 64, "kernel_size": 3, "activation": "relu", "base": True},
                {"name": "MaxPooling2D1", "type": "MaxPooling2D", "pool_size": 2, "base": True},
                {"name": "Flatten1", "type": "Flatten", "base": True},
                {"name": "Dense1", "type": "Dense", "neurons": 128, "activation": "relu", "base": True},
                {"name": "Dropout1", "type": "Dropout", "dropout_rate": 0.5, "base": True},
            ]
        # Reiniciar capas intermedias al cambiar dataset
        st.session_state['num_intermediate_layers'] = 0


# Función para manejar actualización de capas intermedias
def update_intermediate_layers():
    num_layers = st.session_state['num_intermediate_layers']
    current_intermediate_layers = len([
        layer for layer in st.session_state['layer_config']
        if not layer.get("base", False)
    ])

    # Actualizar solo si hay un cambio
    if num_layers != current_intermediate_layers:
        # Eliminar capas intermedias actuales
        st.session_state['layer_config'] = [
            layer for layer in st.session_state['layer_config'] if layer.get("base", False)
        ]

        # Agregar nuevas capas intermedias
        for i in range(num_layers):
            st.session_state['layer_config'].insert(
                -2,  # Insertar antes de las capas finales (Flatten y Output)
                {
                    "name": f"IntermediateLayer{i + 1}",
                    "type": "Dense",
                    "neurons": 64,
                    "activation": "relu",
                    "base": False,
                }
            )


from sklearn.preprocessing import StandardScaler

# Cargar y dividir dataset
from sklearn.preprocessing import StandardScaler
def load_dataset(name, problem_type):
    # Cargar dataset completo con nombres de columnas correctos
    if name == "Iris":
        data = load_iris(as_frame=True)
        df = data.data
        df["species"] = pd.Series(data.target).map(dict(enumerate(data.target_names)))
    elif name == "Wine":
        data = load_wine(as_frame=True)
        df = data.data
        df["wine_class"] = pd.Series(data.target).map(dict(enumerate(data.target_names)))
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.data
        df["diagnosis"] = data.target
    elif name == "Digits":
        data = load_digits(as_frame=False)
        df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
        df["digit"] = data.target
    elif name == "Boston Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.data
        df["median_value"] = data.target
    elif name == "Diabetes":
        data = load_diabetes(as_frame=True)
        df = data.data
        df["disease_progression"] = data.target
    else:
        raise ValueError("Dataset no soportado")

    return df



def initialize_graph(layers):
    """
    Inicializa una representación de la red neuronal con Input y Output únicos.
    """
    if not layers:
        st.session_state['graph'] = []
        return

    # Redefinir gráfico con Input, capas intermedias y Output
    st.session_state['graph'] = [
        {'name': 'Input', 'num_neurons': 1, 'type': 'Input', 'layer_index': 0}
    ]

    for i, layer in enumerate(layers):
        st.session_state['graph'].append({
            'name': layer['name'],
            'num_neurons': layer.get('neurons', 4),  # Valor predeterminado
            'type': layer['type'],
            'layer_index': i + 1
        })

    st.session_state['graph'].append({
        'name': 'Output', 'num_neurons': 1, 'type': 'Output', 'layer_index': len(layers) + 1
    })



def generate_graph_fig(layers, active_layer=None, epoch=None, neurons_per_point=12, animated_layer=None, progress_step=0):
    """
    Genera el gráfico unificado para preview y dinámico con animación.
    """
    fig = go.Figure()
    layer_positions = {}

    background_color = get_background_color()

    # Determinar tamaño de texto basado en la cantidad de capas
    num_layers = len(layers)
    text_size = 12 if num_layers <= 10 else max(8, 12 - (num_layers - 10) // 2)

    # Calcular posiciones y conexiones primero
    for i, layer in enumerate(st.session_state['graph']):
        x_position = i * 300
        if layer['type'] in ["Dense", "Input", "Output"]:
            total_neurons = layer['num_neurons']
            num_points = math.ceil(total_neurons / neurons_per_point)
            y_positions = list(range(-num_points // 2, num_points // 2 + 1))[:num_points]
        else:
            y_positions = [0]  # Una sola posición para capas simbólicas

        # Guardar posiciones
        for j, y in enumerate(y_positions):
            layer_positions[(i, j)] = (x_position, y)

    # Dibujar conexiones primero (en el fondo)
    for i in range(len(st.session_state['graph']) - 1):
        for j1 in layer_positions:
            if j1[0] == i:
                for j2 in layer_positions:
                    if j2[0] == i + 1:
                        x0, y0 = layer_positions[j1]
                        x1, y1 = layer_positions[j2]
                        is_current_connection = i == animated_layer
                        line_color = "lightblue" if not is_current_connection else f"rgba(0, 200, 200, {0.5 + progress_step / 2})"
                        line_width = 2 if not is_current_connection else 3
                        fig.add_trace(go.Scatter(
                            x=[x0, x1], y=[y0, y1],
                            mode="lines",
                            line=dict(width=line_width, color=line_color),
                            hoverinfo="none"
                        ))

    # Dibujar formas (capas) encima
    for i, layer in enumerate(st.session_state['graph']):
        x_position = i * 300
        if layer['type'] in ["Dense", "Input", "Output"]:
            total_neurons = layer['num_neurons']
            num_points = math.ceil(total_neurons / neurons_per_point)
            y_positions = list(range(-num_points // 2, num_points // 2 + 1))[:num_points]
            if layer['type'] != "Dense":   
                text1=[f"{layer['name']}"]
                # Dibujar neuronas como puntos
                node_color = "gray" if i != animated_layer else f"rgba(255, 165, {255 - int(progress_step * 200)}, 1)"
                for j, y in enumerate(y_positions):
                    fig.add_trace(go.Scatter(
                        x=[x_position], y=[y],
                        mode="markers",
                        marker=dict(size=20, color=node_color),
                        hoverinfo="none",
                        text=text1
                    ))
            else:
                text1=[f"{layer['type']}<br>{layer['num_neurons']} Neuronas"]
                # Dibujar neuronas como puntos
                node_color = "blue" if i != animated_layer else f"rgba(255, 165, {255 - int(progress_step * 200)}, 1)"
                for j, y in enumerate(y_positions):
                    fig.add_trace(go.Scatter(
                        x=[x_position], y=[y],
                        mode="markers",
                        marker=dict(size=10, color=node_color),
                        hoverinfo="none",
                        text=text1
                    ))



        else:
            # Capas sin neuronas: formas simbólicas
            if layer['type'] == "Conv2D":
                color = "green"
            elif layer['type'] == "MaxPooling2D":
                color = "purple"
            elif layer['type'] == "Flatten":
                color = "gray"
            elif layer['type'] == "Dropout":
                color = "orange"
            elif layer['type'] == "BatchNormalization":
                color = "pink"
            else:
                color = "black"

            fig.add_trace(go.Scatter(
                x=[x_position], y=[0],
                mode="markers",
                marker=dict(size=20, color=color),
                hoverinfo="none",
                text=[f"{layer['name']}<br>{layer['type']}"]
            ))

        # Etiquetas para Input/Output y capas
        label_text = (
            f"{layer['type']}"
            if layer['type'] in ["Dense", "Input", "Output"]
            else layer['type']
        )

        # Ajustar posición de etiquetas alternadas
        if i % 2 == 0:
            label_y_offset = max(y_positions) + 1.5  # Posición arriba
            label_position = "top center"
        else:
            label_y_offset = min(y_positions) - 2.5  # Posición abajo (más alejado)

        fig.add_trace(go.Scatter(
            x=[x_position], 
            y=[label_y_offset],
            mode="text",
            text=label_text,
            textposition=label_position,
            textfont=dict(size=text_size),
            hoverinfo="none"
        ))

    # Configuración del gráfico
    title = f"Training Progress - Epoch {epoch + 1}" if epoch is not None else "Network Architecture Preview"
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor=background_color,  # Fondo del gráfico
        paper_bgcolor=background_color,  # Fondo del área alrededor del gráfico
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig



def update_graph_with_smooth_color_transition(layers, epoch, placeholder, neurons_per_point=12, animation_steps=30):
    """
    Muestra una animación visual fluida de flujo entre capas mediante cambios progresivos de color.
    """
    total_layers = len(layers)
    background_color = get_background_color()

    for layer_index in range(total_layers - 1):  # No incluye la capa final
        for step in range(animation_steps):
            progress_step = step / animation_steps  # Progreso gradual del color
            
            fig = go.Figure()
            layer_positions = {}

            # Calcular posiciones de capas
            for i, layer in enumerate(layers):
                x_position = i * 300
                total_neurons = layer['num_neurons'] if layer['type'] in ["Dense", "Input", "Output"] else 1

                # Calcular cuántos puntos representar
                num_points = math.ceil(total_neurons / neurons_per_point)
                y_positions = list(range(-num_points // 2, num_points // 2 + 1))[:num_points]

                # Guardar posiciones de la capa
                for j, y in enumerate(y_positions):
                    layer_positions[(i, j)] = (x_position, y)

            # Dibujar conexiones primero (fondo)
            for i in range(total_layers - 1):  # No conecta después de Output
                for j1 in layer_positions:
                    if j1[0] == i:
                        for j2 in layer_positions:
                            if j2[0] == i + 1:
                                x0, y0 = layer_positions[j1]
                                x1, y1 = layer_positions[j2]

                                # Color de las conexiones
                                if i == layer_index:
                                    line_color = f"rgba(255, {int(progress_step * 255)}, 0, 1)"  # De naranja a azul
                                else:
                                    line_color = "lightblue"

                                fig.add_trace(go.Scatter(
                                    x=[x0, x1], y=[y0, y1],
                                    mode="lines",
                                    line=dict(width=2, color=line_color),
                                    hoverinfo="none"
                                ))

            # Dibujar puntos o formas de las capas (frente)
            for i, layer in enumerate(layers):
                x_position = i * 300
                total_neurons = layer['num_neurons'] if layer['type'] in ["Dense", "Input", "Output"] else 1
                num_points = math.ceil(total_neurons / neurons_per_point)
                y_positions = list(range(-num_points // 2, num_points // 2 + 1))[:num_points]

                # Determinar color y forma de la capa
                node_color = "blue" if i != layer_index and i != layer_index + 1 else f"rgba(255, 165, {255 - int(progress_step * 200)}, 1)"
                if layer['type'] in ["Dense"]:
                    # Dibujar nodos (puntos)
                    for j, y in enumerate(y_positions):
                        fig.add_trace(go.Scatter(
                            x=[x_position], y=[y],
                            mode="markers",
                            marker=dict(size=10, color=node_color),
                            hoverinfo="none"
                        ))
                else:
                    # Dibujar formas simbólicas para otras capas
                    color = {
                        "Conv2D": ( "green"),
                        "MaxPooling2D": ( "purple"),
                        "Flatten": ("gray"),
                        "Dropout": ("orange"),
                        "BatchNormalization": ("pink")
                    }.get(layer['type'], ("black"))
                    
                    fig.add_trace(go.Scatter(
                        x=[x_position], y=[-1],
                        mode="markers",
                        marker=dict(size=20, color=color),
                        hoverinfo="none"
                    ))

                # Añadir etiquetas
                label_text = (
                    f"{layer['type']}"
                    if layer['type'] in ["Dense", "Input", "Output"]
                    else layer['type']
                )

                # Ajustar posición de etiquetas alternadas
                if i % 2 == 0:
                    label_y_offset = max(y_positions) + 1.5  # Posición arriba
                    label_position = "top center"
                else:
                    label_y_offset = min(y_positions) - 2.5  # Posición abajo (más alejado)

                fig.add_trace(go.Scatter(
                    x=[x_position], 
                    y=[label_y_offset],
                    mode="text",
                    text=label_text,
                    textposition=label_position,
                    hoverinfo="none"
                ))

            # Configuración del gráfico
            fig.update_layout(
                title=f"Progreso del entrenamiento - Época {epoch + 1}" if epoch is not None else "Arquitectura del modelo",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor=background_color,  # Fondo del gráfico
                paper_bgcolor=background_color,  # Fondo del área alrededor del gráfico
                margin=dict(l=10, r=10, t=30, b=10)
            )

            # Renderizar el gráfico
            placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"training_{epoch}_{layer_index}_{step}_{time.time()}"
            )
            time.sleep(0.03)  # Ajusta para controlar la fluidez




def preview_graph(layers, placeholder, neurons_per_point=12):
    """
    Genera un gráfico estático con Input, capas y Output.
    """
    initialize_graph(layers)
    if not st.session_state['graph']:
        placeholder.empty()
        st.warning("No hay capas configuradas. Agregue una capa para visualizar el gráfico.")
        return

    fig = generate_graph_fig(layers, neurons_per_point=neurons_per_point)
    placeholder.plotly_chart(fig, use_container_width=True, key=f"preview_{time.time()}")





    
# Función para registrar logs
def log_event(log_placeholder, message):
    st.session_state['logs'].append(message)
    log_placeholder.text_area("Registro del entrenamiento", "\n".join(st.session_state['logs']), height=300)









# Función para entrenar el modelo con gráficos dinámicos
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
import pandas as pd

def train_model(layers, hyperparams, preview_placeholder, dynamic_placeholder):
    # Limpiar logs y placeholders
    st.session_state['logs'] = []
    st.session_state['loss_values'] = []
    st.session_state['accuracy_values'] = []
    st.session_state['val_loss_values'] = []
    st.session_state['val_accuracy_values'] = []


    preview_placeholder.empty()
    

    # Configuración de contenedores para evitar desplazamiento
    col_dynamic, col_metrics = st.columns([6, 1])  # Ajusta proporciones

    # Placeholder para gráfico dinámico (mantiene la posición)
    dynamic_graph_placeholder = col_dynamic.empty()

    # Placeholder para métricas y logs (separado del gráfico)
    metrics_placeholder = col_metrics.empty()
    log_placeholder = col_metrics.empty()
    
    loss_chart_placeholder = col_metrics.empty()
    accuracy_chart_placeholder = col_metrics.empty()

    # Recuperar datos configurados previamente
    if not st.session_state.get("dataset_split", False):
        st.error("El dataset no está configurado correctamente. Configúralo antes de entrenar.")
        return

    splits = st.session_state['splits']
    if len(splits) == 4:  # Entrenamiento y prueba
        X_train, X_test, y_train, y_test = splits
        X_val, y_val = None, None
    elif len(splits) == 6:  # Entrenamiento, validación y prueba
        X_train, X_val, X_test, y_train, y_val, y_test = splits
    else:
        st.error("La división del dataset no es válida. Reconfigura el dataset.")
        return

    input_shape = (X_train.shape[1],)
    problem_type = st.session_state['problem_type']

    # Preparar etiquetas para clasificación
    if problem_type == "Clasificación":
        num_classes = len(np.unique(y_train))
        if np.any(y_train >= num_classes) or np.any(y_train < 0):
            st.error("Error: Hay etiquetas fuera del rango válido para la clasificación.")
            return
        y_train = to_categorical(y_train, num_classes)
        if y_val is not None:
            y_val = to_categorical(y_val, num_classes)
        y_test_original = y_test  # Guardar etiquetas originales
        y_test = to_categorical(y_test, num_classes)

    # Crear el modelo
    model = Sequential()
    for i, layer in enumerate(layers):
        if layer['type'] == "Dense":
            # Añadir regularización L2
            regularizer = None
            if 'regularization' in layer and layer['regularization'] == 'l2':
                regularizer = tf.keras.regularizers.l2(layer['regularization_rate'])

            model.add(Dense(
                layer['neurons'],
                activation=layer['activation'],
                input_shape=input_shape if i == 0 else None
            ))
        elif layer['type'] == "Dropout":
            model.add(Dropout(layer['dropout_rate']))

    # Capa de salida
    #if problem_type == "Clasificación":
    #    model.add(Dense(num_classes, activation="softmax"))
    #    loss_function = "categorical_crossentropy"
    #else:  # Regresión
    #    model.add(Dense(1, activation="linear"))
    #    loss_function = "mean_squared_error"

    # Configuración del optimizador
    optimizers = {
        "Adam": Adam(learning_rate=hyperparams['learning_rate']),
        "SGD": SGD(learning_rate=hyperparams['learning_rate']),
        "RMSprop": RMSprop(learning_rate=hyperparams['learning_rate'])
    }
    optimizer = optimizers[hyperparams['optimizer']]
    loss_function = hyperparams['loss_function']


    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=["accuracy"] if problem_type == "Clasificación" else ["mae"]  # Precisión para clasificación, MAE para regresión
    )

    # Inicialización de métricas
    loss_values = []
    accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    # Entrenamiento por épocas
    for epoch in range(hyperparams['epochs']):
        start_time = time.time()
        log_event(log_placeholder, f"Época {epoch + 1}/{hyperparams['epochs']} iniciada.")

        # Animación del gráfico dinámico
        update_graph_with_smooth_color_transition(
            st.session_state['graph'],
            epoch,
            dynamic_graph_placeholder,
            neurons_per_point=12,
            animation_steps=15
        )

        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=hyperparams['batch_size'],
            epochs=1,
            verbose=0
        )

        # Actualizar métricas
        loss = history.history['loss'][0]
        accuracy = history.history.get('accuracy', [None])[0]
        val_loss = history.history.get('val_loss', [None])[0]
        val_accuracy = history.history.get('val_accuracy', [None])[0]

        st.session_state['loss_values'].append(loss)
        if accuracy is not None:
            st.session_state['accuracy_values'].append(accuracy)
        if val_loss is not None:
            st.session_state['val_loss_values'].append(val_loss)
        if val_accuracy is not None:
            st.session_state['val_accuracy_values'].append(val_accuracy)


        # Gráfico de pérdida (aplica a ambos tipos de problemas)
        with loss_chart_placeholder:
            fig_loss, ax_loss = plt.subplots(figsize=(4, 2))
            ax_loss.plot(range(1, len(st.session_state['loss_values']) + 1),
                        st.session_state['loss_values'], marker='o', color='blue', label="Pérdida")
            if st.session_state['val_loss_values']:
                ax_loss.plot(range(1, len(st.session_state['val_loss_values']) + 1),
                            st.session_state['val_loss_values'], linestyle="--", color='red', label="Validación")
            ax_loss.set_title("Pérdida")
            ax_loss.grid(True)
            ax_loss.legend()
            loss_chart_placeholder.pyplot(fig_loss, clear_figure=True)

        # Gráfico de precisión (solo para clasificación)
        if problem_type == "Clasificación":
            with accuracy_chart_placeholder:
                if st.session_state['accuracy_values']:
                    fig_accuracy, ax_accuracy = plt.subplots(figsize=(4, 2))
                    ax_accuracy.plot(range(1, len(st.session_state['accuracy_values']) + 1),
                                    st.session_state['accuracy_values'], marker='o', color='green', label="Precisión")
                    if st.session_state['val_accuracy_values']:
                        ax_accuracy.plot(range(1, len(st.session_state['val_accuracy_values']) + 1),
                                        st.session_state['val_accuracy_values'], linestyle="--", color='orange', label="Validación")
                    ax_accuracy.set_title("Precisión")
                    ax_accuracy.grid(True)
                    ax_accuracy.legend()
                    accuracy_chart_placeholder.pyplot(fig_accuracy, clear_figure=True)
        else:
            with accuracy_chart_placeholder:
                if accuracy is not None:
                    fig_accuracy, ax_accuracy = plt.subplots(figsize=(4, 2))
                    ax_accuracy.plot(range(1, len(accuracy_values) + 1), accuracy_values, marker='o', color='green', label="Precisión")
                    if val_accuracy is not None:
                        ax_accuracy.plot(range(1, len(val_accuracy_values) + 1), val_accuracy_values, linestyle="--", color='orange', label="Validación")
                    ax_accuracy.set_title("Precisión")
                    ax_accuracy.grid(True)
                    ax_accuracy.legend()
                    accuracy_chart_placeholder.pyplot(fig_accuracy, clear_figure=True)

        # Verificar valores de las métricas y asignar 'N/A' si son None
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        val_accuracy_str = f"{val_accuracy:.4f}" if val_accuracy is not None else "N/A"

        # Log de fin de época
        elapsed_time = time.time() - start_time
        log_event(
            log_placeholder,
            f"Época {epoch + 1} completada en {elapsed_time:.2f} segundos. "
            f"Pérdida: {loss_str}, Precisión: {accuracy_str}, "
            f"Pérdida Validación: {val_loss_str}, Precisión Validación: {val_accuracy_str}"
        )

    # Calcular métricas finales
    # Mostrar métricas finales según el tipo de problema
    if problem_type == "Clasificación":
        y_pred = np.argmax(model.predict(X_test), axis=1)
        f1 = f1_score(y_test_original, y_pred, average='weighted')
        precision = precision_score(y_test_original, y_pred, average='weighted')
        recall = recall_score(y_test_original, y_pred, average='weighted')

        st.write("### Métricas Finales - Clasificación")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**Precisión (Precision):** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.text(classification_report(y_test_original, y_pred))
    else:
        st.write("### Métricas Finales - Regresión")
        final_loss = st.session_state['loss_values'][-1]
        st.write(f"**Pérdida Final (MAE):** {final_loss:.4f}")
        if st.session_state['val_loss_values']:
            final_val_loss = st.session_state['val_loss_values'][-1]
            st.write(f"**Pérdida Validación Final (MAE):** {final_val_loss:.4f}")



    # Guardar modelo entrenado
    st.session_state["modelDownload"] = model
    st.success("Entrenamiento finalizado con éxito.")





def train_model_classification(layers, hyperparams, preview_placeholder, dynamic_placeholder):
    # Limpiar logs y placeholders
    st.session_state['logs'] = []
    st.session_state['loss_values'] = []
    st.session_state['accuracy_values'] = []
    st.session_state['val_loss_values'] = []
    st.session_state['val_accuracy_values'] = []

    preview_placeholder.empty()

    # Configuración de contenedores
    col_dynamic, col_metrics = st.columns([6, 1])
    dynamic_graph_placeholder = col_dynamic.empty()
    log_placeholder = st.empty()

    # Validar splits (ya existente)
    if not st.session_state.get("dataset_split", False):
        st.error("El dataset no está configurado correctamente.")
        return

    splits = st.session_state['splits']
    if len(splits) == 4:
        X_train, X_test, y_train, y_test = splits
        X_val, y_val = None, None
    elif len(splits) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = splits
    else:
        st.error("La división del dataset no es válida. Reconfigura el dataset.")
        return



    st.write("Validación inmediata de dimensiones después de la división:")
    st.write(f"Dimensiones de X_train: {X_train.shape}")
    st.write(f"Dimensiones de X_val: {X_val.shape}")
    st.write(f"Dimensiones de X_test: {X_test.shape}")
    st.write(f"Dimensiones de y_train: {y_train.shape}")
    st.write(f"Dimensiones de y_val: {y_val.shape}")
    st.write(f"Dimensiones de y_test: {y_test.shape}")
    st.write(f"Dimensiones de y_test_original: {st.session_state["y_test_original"].shape}")

    # Validar dimensiones antes del entrenamiento
    if len(st.session_state["splits"][3]) != len(st.session_state["splits"][3]):
        st.error("Dimensiones inconsistentes entre X_test y y_test_original antes del entrenamiento.")
        st.stop()

    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    if y_val is not None:
        y_val = to_categorical(y_val, num_classes)
    y_test_original = y_test
    y_test = to_categorical(y_test, num_classes)

    if X_test.isnull().values.any():
        st.error("X_test contiene valores NaN. Revisa el preprocesamiento.")
        st.stop()
    if X_test.shape[1] != 4:
        st.error(f"X_test tiene {X_test.shape[1]} columnas, pero el modelo espera 4 características.")
        st.stop()



    # Crear modelo
    input_shape = (X_train.shape[1],)
    model = Sequential()
    for i, layer in enumerate(layers):
        if layer['type'] == "Dense":
            model.add(Dense(layer['neurons'], activation=layer['activation'], input_shape=input_shape if i == 0 else None,kernel_regularizer=l2(layer['l2']) if layer.get('enable_l2', False) else None))
        elif layer['type'] == "Dropout":
            model.add(Dropout(layer['dropout_rate']))

    st.write(f"Configuración de capas utilizada: {st.session_state['layer_config']}")
    # Configurar optimizador y compilar modelo
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Entrenar modelo
    for epoch in range(hyperparams['epochs']):
        log_event(log_placeholder, f"Época {epoch + 1}/{hyperparams['epochs']} iniciada.")

        if X_val is not None and y_val is not None:
            # Con conjunto de validación
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                batch_size=hyperparams['batch_size'], epochs=1, verbose=0)
        else:
            # Sin conjunto de validación
            history = model.fit(X_train, y_train, batch_size=hyperparams['batch_size'], 
                                epochs=1, verbose=0)

        # Registrar métricas
        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        log_event(log_placeholder, f"Época {epoch + 1}: Pérdida en entrenamiento: {train_loss:.4f}")
        log_event(log_placeholder, f"Época {epoch + 1}: Precisión en entrenamiento: {train_accuracy:.4f}")

        if X_val is not None and y_val is not None:
            val_loss = history.history['val_loss'][0]
            val_accuracy = history.history['val_accuracy'][0]
            log_event(log_placeholder, f"Época {epoch + 1}: Pérdida en validación: {val_loss:.4f}")
            log_event(log_placeholder, f"Época {epoch + 1}: Precisión en validación: {val_accuracy:.4f}")

        st.session_state['loss_values'].append(train_loss)
        st.session_state['accuracy_values'].append(train_accuracy)
        if X_val is not None and y_val is not None:
            st.session_state['val_loss_values'].append(val_loss)
            st.session_state['val_accuracy_values'].append(val_accuracy)

        # Actualizar visualización dinámica
        update_graph_with_smooth_color_transition(
            st.session_state['graph'], epoch, dynamic_graph_placeholder, neurons_per_point=10, animation_steps=30
        )

    # Métricas finales
    st.write(f"Dimensiones de X_test: {X_test.shape}")
    st.write(f"Dimensiones de y_test_original: {y_test_original.shape}")


    if np.isnan(X_test.values).any():
        st.error("X_test contiene valores NaN. Revisa el preprocesamiento.")
        st.stop()

    st.session_state["modelDownload"] = model
    st.success("Entrenamiento finalizado con éxito.")


def train_model_regression(layers, hyperparams, preview_placeholder, dynamic_placeholder):
    # Limpiar logs y placeholders
    st.session_state['logs'] = []
    st.session_state['loss_values'] = []
    st.session_state['val_loss_values'] = []

    preview_placeholder.empty()

    # Configuración de contenedores
    col_dynamic, col_metrics = st.columns([6, 1])
    dynamic_graph_placeholder = col_dynamic.empty()
    log_placeholder = st.empty()

    # Validar configuración del dataset
    if not st.session_state.get("dataset_split", False):
        st.error("El dataset no está configurado correctamente. Configúralo antes de entrenar.")
        return

    # Recuperar splits
    splits = st.session_state['splits']
    if len(splits) == 4:
        X_train, X_test, y_train, y_test = splits
        X_val, y_val = None, None
    elif len(splits) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = splits
    else:
        st.error("La división del dataset no es válida. Reconfigura el dataset.")
        return

    # Crear modelo
    input_shape = (X_train.shape[1],)
    model = Sequential()
    for i, layer in enumerate(layers):
        if layer['type'] == "Dense":
            model.add(Dense(layer['neurons'], activation=layer['activation'], input_shape=input_shape if i == 0 else None,kernel_regularizer=l2(layer['l2']) if layer.get('enable_l2', False) else None))
        elif layer['type'] == "Dropout":
            model.add(Dropout(layer['dropout_rate']))
    model.add(Dense(1, activation="linear"))

    # Configurar optimizador y compilar modelo
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

    # Entrenar modelo
    for epoch in range(hyperparams['epochs']):
        log_event(log_placeholder, f"Época {epoch + 1}/{hyperparams['epochs']} iniciada.")

        if X_val is not None and y_val is not None:
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                batch_size=hyperparams['batch_size'], epochs=1, verbose=0)
        else:
            history = model.fit(X_train, y_train, batch_size=hyperparams['batch_size'], 
                                epochs=1, verbose=0)

        # Registrar métricas
        train_loss = history.history['loss'][0]
        train_mae = history.history['mae'][0]
        log_event(log_placeholder, f"Época {epoch + 1}: Pérdida (MSE) en entrenamiento: {train_loss:.4f}")
        log_event(log_placeholder, f"Época {epoch + 1}: Error absoluto medio (MAE) en entrenamiento: {train_mae:.4f}")

        if X_val is not None:
            val_loss = history.history['val_loss'][0]
            val_mae = history.history['val_mae'][0]
            log_event(log_placeholder, f"Época {epoch + 1}: Pérdida (MSE) en validación: {val_loss:.4f}")
            log_event(log_placeholder, f"Época {epoch + 1}: Error absoluto medio (MAE) en validación: {val_mae:.4f}")

        # Actualizar métricas para gráficos
        st.session_state['loss_values'].append(train_loss)
        if X_val is not None:
            st.session_state['val_loss_values'].append(val_loss)

        # Actualizar visualización dinámica
        update_graph_with_smooth_color_transition(
            st.session_state['graph'], epoch, dynamic_graph_placeholder, neurons_per_point=10, animation_steps=30
        )

    # Métricas finales
    final_loss = st.session_state['loss_values'][-1]
    st.write(f"Pérdida Final (MSE): {final_loss:.4f}")
    st.session_state["modelDownload"] = model
    st.success("Entrenamiento finalizado con éxito.")








current_intermediate_layers = max(0, len(st.session_state['layer_config']) - 6)

# Inicializar estado para el número de capas
if 'num_layers_selected' not in st.session_state:
    st.session_state['num_layers_selected'] = len(st.session_state['layer_config'])
if 'previous_layer_config' not in st.session_state:
    st.session_state['previous_layer_config'] = []
# Inicializar el estado para capas intermedias si no existe
if 'num_intermediate_layers' not in st.session_state:
    st.session_state['num_intermediate_layers'] = current_intermediate_layers

# Función para manejar la actualización de capas
def update_layer_config():
    num_layers = st.session_state['num_layers_selected']
    current_num_layers = len(st.session_state['layer_config'])

    if num_layers > current_num_layers:
        # Añadir capas
        for i in range(current_num_layers, num_layers):
            st.session_state['layer_config'].append({
                "name": f"Layer{i + 1}",
                "type": "Dense",
                "neurons": 64,
                "dropout_rate": 0.2
            })
    elif num_layers < current_num_layers:
        # Reducir capas
        st.session_state['layer_config'] = st.session_state['layer_config'][:num_layers]

    # Guardar una copia del estado actual para evitar rebotes
    st.session_state['previous_layer_config'] = st.session_state['layer_config'][:]



# Diccionario de arquitecturas predefinidas
architectures = {
    "Iris": {
        "Simple MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 16, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense3", "type": "Dense", "neurons": 3, "activation": "softmax", "base": True, "l2": 0.0, "enable_l2": False}
        ],
        "Deep MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense3", "type": "Dense", "neurons": 16, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense4", "type": "Dense", "neurons": 3, "activation": "softmax", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    },
    "Wine": {
        "Simple MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense3", "type": "Dense", "neurons": 3, "activation": "softmax", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    },
    "Breast Cancer": {
        "Simple MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 16, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense3", "type": "Dense", "neurons": 2, "activation": "softmax", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    },
    "Digits": {
        "Simple MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense3", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    },
    "Boston Housing": {
        "Simple Regression": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Output", "type": "Dense", "neurons": 1, "activation": "linear", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    },
    "Diabetes": {
        "Simple Regression": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Dense2", "type": "Dense", "neurons": 32, "activation": "relu", "base": True, "l2": 0.01, "enable_l2": True},
            {"name": "Output", "type": "Dense", "neurons": 1, "activation": "linear", "base": True, "l2": 0.0, "enable_l2": False}
        ]
    }
}


# Función para manejar el cambio de dataset
if "selected_dataset_previous" not in st.session_state or st.session_state["selected_dataset_previous"] != st.session_state["selected_dataset"]:
    st.session_state["selected_dataset_previous"] = st.session_state["selected_dataset"]
    
    # Reinicia configuración de capas y arquitectura
    initialize_layer_config(st.session_state["selected_dataset"])



from sklearn.metrics import mean_squared_error




# Configuración con pestañas
tabs = st.sidebar.radio("Configuración:", ["Dataset","Capas", "Hiperparámetros", "Early Stopping"])


# Configuración del Dataset
from sklearn.impute import SimpleImputer
import numpy as np



# Aplicar ajustes avanzados
def apply_advanced_settings(X_train, X_test, numeric_null_option, categorical_null_option, normalize_data, scaling_option, encoding_option):
    # Manejo de valores nulos - Numéricos
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    if numeric_null_option == "Eliminar filas":
        X_train.dropna(subset=numeric_cols, inplace=True)
        X_test.dropna(subset=numeric_cols, inplace=True)
    elif numeric_null_option == "Reemplazar con 0":
        X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(0)
    elif numeric_null_option == "Reemplazar con la media":
        imputer = SimpleImputer(strategy='mean')
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
    elif numeric_null_option == "Reemplazar con la mediana":
        imputer = SimpleImputer(strategy='median')
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
    elif numeric_null_option == "Propagar adelante/atrás":
        X_train[numeric_cols] = X_train[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        X_test[numeric_cols] = X_test[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    # Manejo de valores nulos - Categóricos
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    if categorical_null_option == "Eliminar filas":
        X_train.dropna(subset=categorical_cols, inplace=True)
        X_test.dropna(subset=categorical_cols, inplace=True)
    elif categorical_null_option == "Reemplazar con 'Ninguno'":
        X_train[categorical_cols] = X_train[categorical_cols].fillna('Ninguno')
        X_test[categorical_cols] = X_test[categorical_cols].fillna('Ninguno')
    elif categorical_null_option == "Reemplazar con la moda":
        imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = imputer.transform(X_test[categorical_cols])

    # Normalización y escalamiento
    if normalize_data:
        scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Codificación de variables categóricas
    if encoding_option == "One-Hot Encoding":
        X_train = pd.get_dummies(X_train, columns=categorical_cols)
        X_test = pd.get_dummies(X_test, columns=categorical_cols)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    elif encoding_option == "Label Encoding":
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            X_train[col] = label_encoder.fit_transform(X_train[col].astype(str))
            X_test[col] = label_encoder.transform(X_test[col].astype(str))

    return X_train, X_test





# Función principal para configurar el dataset
def configure_dataset():
    st.sidebar.header("Configuración del Dataset")

    # Paso 1: Selección del dataset
    st.sidebar.subheader("Paso 1: Selección del Dataset")
    problem_type = st.sidebar.selectbox(
        "Seleccione el tipo de problema",
        ["Clasificación", "Regresión"],
        key="problem_type_selectbox"
    )

    st.session_state['problem_type'] = problem_type

    dataset_name = st.sidebar.selectbox(
        "Seleccione el dataset",
        ["Iris", "Wine", "Breast Cancer", "Digits"] if problem_type == "Clasificación" else ["Boston Housing", "Diabetes"],
        key="dataset_name_selectbox"
    )

    st.session_state['selected_dataset'] = dataset_name

    if st.sidebar.button("Cargar Dataset", key="load_dataset_button"):
    # Reiniciar el estado relacionado con el dataset al cambiarlo
        if st.session_state.get("dataset_loaded", False):
            st.session_state["dataset_loaded"] = False
            st.session_state.pop("full_dataset", None)
            st.session_state.pop("target_variable", None)
            st.session_state.pop("X_original", None)
            st.session_state.pop("y_original", None)
            st.session_state.pop("splits", None)
            st.session_state["dataset_split"] = False

    # Cargar el nuevo dataset
    df = load_dataset(dataset_name, problem_type)
    if df.empty:
        st.sidebar.error("El dataset cargado está vacío. Seleccione un dataset válido.")
        st.stop()

    # Guardar el dataset en el estado de la sesión
    st.session_state["dataset_loaded"] = True
    st.session_state["full_dataset"] = df
    st.success("Dataset cargado con éxito.")
    st.sidebar.write("Vista previa del dataset:", df.head())

    
    # Paso 2: Selección de la variable objetivo
    if st.session_state.get("dataset_loaded", False):
        st.sidebar.subheader("Paso 2: Selección de la Variable Objetivo")
        target_variable = st.sidebar.selectbox(
            "Seleccione la variable objetivo:",
            st.session_state["full_dataset"].columns,
            key="target_variable_selectbox"
        )
        if target_variable != st.session_state.get("target_variable", None):
            st.session_state["target_variable"] = target_variable
            st.session_state["X_original"] = st.session_state["full_dataset"].drop(columns=[target_variable]).copy()
            st.session_state["y_original"] = st.session_state["full_dataset"][target_variable].copy()

        st.sidebar.success(f"Variable objetivo seleccionada: {st.session_state['target_variable']}")
        st.sidebar.write("Características (X):", st.session_state["X_original"].head())
        # Verificar si y_original es un numpy.ndarray y convertirlo en un DataFrame/Series antes de usar .head()
        if isinstance(st.session_state["y_original"], np.ndarray):
            y_display = pd.Series(st.session_state["y_original"])
        else:
            y_display = st.session_state["y_original"]

        # Mostrar la vista previa de la variable objetivo
        st.sidebar.write("Variable objetivo (y):", y_display.head())
                # Paso 3: Eliminación de columnas

    if st.session_state.get("target_variable"):
        st.sidebar.subheader("Paso 3: Eliminación de Columnas")
        selected_columns = st.sidebar.multiselect(
            "Seleccione las columnas a eliminar",
            st.session_state["X_original"].columns,
            key="columns_to_drop_multiselect"
        )
        if st.sidebar.button("Aplicar Eliminación", key="apply_column_removal_button"):
            if target_variable in selected_columns:
                st.sidebar.error("La variable objetivo no puede ser eliminada.")
                st.stop()
            st.session_state["X_original"] = st.session_state["X_original"].drop(columns=selected_columns).copy()
            if st.session_state["X_original"].shape[1] == 0:
                st.sidebar.error("No quedan columnas en el dataset después de la eliminación.")
                st.stop()
            st.session_state["columns_removed"] = True
            st.success("Columnas eliminadas con éxito.")
            st.sidebar.write("Vista previa del dataset actualizado:", st.session_state["X_original"].head())

        # Paso 4: Manejo de valores nulos
        st.sidebar.subheader("Paso 4: Manejo de Valores Nulos")
        X = st.session_state["X_original"]
        nulos_totales = X.isnull().sum().sum()

        if nulos_totales == 0:
            st.sidebar.success("No existen valores nulos en el dataset.")
            st.session_state["missing_handled"] = True
        else:
            numeric_null_option = st.sidebar.selectbox(
                "Valores Nulos - Numéricos",
                ["Eliminar filas", "Reemplazar con 0", "Reemplazar con la media", "Reemplazar con la mediana"],
                key="numeric_null_option_selectbox"
            )
            categorical_null_option = st.sidebar.selectbox(
                "Valores Nulos - Categóricos",
                ["Eliminar filas", "Reemplazar con 'Ninguno'", "Reemplazar con la moda"],
                key="categorical_null_option_selectbox"
            )
            if st.sidebar.button("Aplicar Manejo de Nulos", key="apply_null_handling_button"):
                X_cleaned = handle_nulls(st.session_state["X_original"], numeric_null_option, categorical_null_option)
                st.session_state["X_original"] = X_cleaned
                st.session_state["missing_handled"] = True
                st.sidebar.success("Manejo de valores nulos aplicado con éxito.")
                st.sidebar.write("Vista previa del dataset actualizado:", X_cleaned.head())

        # Paso 5: Codificación de variables categóricas
        if st.session_state.get("missing_handled", False):
            st.sidebar.subheader("Paso 5: Codificación de Variables Categóricas")
            categorical_cols = st.session_state["X_original"].select_dtypes(include=["object", "category"]).columns

            # Codificar columnas categóricas en X
            if len(categorical_cols) == 0:
                st.sidebar.success("No existen columnas categóricas en las características (X).")
                st.session_state["categorical_encoded"] = True
            else:
                encoding_option = st.sidebar.selectbox(
                    "Método de Codificación para características (X):",
                    ["One-Hot Encoding", "Label Encoding"],
                    key="encoding_option_selectbox"
                )
                if st.sidebar.button("Aplicar Codificación en X", key="apply_encoding_button"):
                    X_encoded = encode_categorical(st.session_state["X_original"], encoding_option)
                    st.session_state["X_original"] = X_encoded
                    st.session_state["categorical_encoded"] = True
                    st.sidebar.success("Codificación aplicada con éxito a las características (X).")
                    st.sidebar.write("Vista previa del dataset actualizado (X):", X_encoded.head())

            # Codificar la variable objetivo (y) si es categórica y el problema es de clasificación
            if st.session_state["problem_type"] == "Clasificación" and st.session_state["y_original"].dtype == "object":
                st.sidebar.subheader("Codificación de la Variable Objetivo (y)")
                if st.sidebar.button("Aplicar Codificación en y", key="apply_y_encoding_button"):
                    label_encoder = LabelEncoder()
                    st.session_state["y_original"] = label_encoder.fit_transform(st.session_state["y_original"])
                    st.session_state["label_encoder"] = label_encoder
                    st.sidebar.success("Codificación aplicada con éxito a la variable objetivo (y).")
                    st.sidebar.write("Vista previa de la variable objetivo codificada (y):", st.session_state["y_original"][:5])

        # Paso 6: Normalización y escalamiento
        if st.session_state.get("categorical_encoded", False):
            st.sidebar.subheader("Paso 6: Normalización y Escalamiento")
            numeric_columns = st.session_state["X_original"].select_dtypes(include=["float64", "int64"]).columns
            scaling_option = st.sidebar.selectbox(
                "Método de Escalamiento:",
                ["StandardScaler", "MinMaxScaler"],
                key="scaling_option_selectbox"
            )
            if st.sidebar.button("Aplicar Escalamiento", key="apply_scaling_button"):
                numeric_columns = st.session_state["X_original"].select_dtypes(include=["float64", "int64"]).columns
                scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
                scaled_data = scaler.fit_transform(st.session_state["X_original"][numeric_columns])
                st.session_state["X_original"].loc[:, numeric_columns] = scaled_data
                st.session_state["scaling_done"] = True
                st.sidebar.success("Escalamiento aplicado con éxito.")
                st.sidebar.write("Vista previa del dataset escalado:", st.session_state["X_original"].head())

        # Validar que y_original no haya sido alterado
        if st.session_state["problem_type"] == "Clasificación" and st.session_state["y_original"].dtype not in [np.int32, np.int64, np.float32, np.float64]:
            st.write(st.session_state["y_original"].dtype)
            st.sidebar.error("La variable objetivo debe seguir siendo numérica después del escalamiento. Verifica el flujo.")
            st.stop()

        # Paso 7: División del dataset
        if st.session_state.get("scaling_done", False):
            st.sidebar.subheader("Paso 7: División del Dataset")
            
            split_type = st.sidebar.selectbox(
                "Tipo de División:",
                ["Entrenamiento y Prueba", "Entrenamiento, Validación y Prueba"],
                key="split_type_selectbox"
            )
            st.session_state["split_type"] = split_type
            train_ratio = st.sidebar.slider("Entrenamiento (%)", 10, 90, 70, key="train_ratio_slider")

            # Calcular proporciones para validación y prueba
            if split_type == "Entrenamiento, Validación y Prueba":
                val_ratio = st.sidebar.slider("Validación (%)", 5, 50, 15, key="val_ratio_slider")
                test_ratio = 100 - train_ratio - val_ratio
            else:
                val_ratio = 0
                test_ratio = 100 - train_ratio

            st.sidebar.text(f"Prueba (%): {test_ratio}")
            
            # Botón para aplicar división
            if st.sidebar.button("Aplicar División", key="apply_split_button") and not st.session_state.get("dataset_split", False):
                if split_type == "Entrenamiento y Prueba":
                    # División simple: Entrenamiento y Prueba
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state["X_original"], st.session_state["y_original"],
                        test_size=test_ratio / 100, random_state=42
                    )
                    # Guardar y_test_original antes de cualquier transformación
                    st.session_state['y_test_original'] = y_test.copy()
                    st.session_state["splits"] = (X_train, X_test, y_train, y_test)

                elif split_type == "Entrenamiento, Validación y Prueba":
                    # División avanzada: Entrenamiento, Validación y Prueba
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        st.session_state["X_original"], st.session_state["y_original"],
                        test_size=(val_ratio + test_ratio) / 100, random_state=42
                    )
                    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=1 - val_ratio_adjusted, random_state=42
                    )
                    # Después de la división

                    st.session_state['y_test_original'] = y_test.copy()
                    
                    st.session_state["splits"] = (X_train, X_val, X_test, y_train, y_val, y_test)
                    # Verificar dimensiones después de la división
                    if len(X_test) != len(y_test):
                        st.error("Dimensiones inconsistentes entre X_test y y_test después de la división.")



                # Verificación de dimensiones
                splits = st.session_state["splits"]
                
                if len(splits) == 4:
                    X_train, X_test, y_train, y_test = splits
                elif len(splits) == 6:
                    X_train, X_val, X_test, y_train, y_val, y_test = splits

                if len(X_test) != len(y_test):
                    st.error("Dimensiones inconsistentes entre X_test y y_test después de la división.")
                    st.stop()

                # Guardar copia de y_test_original para referencia
                st.session_state["y_test_original"] = y_test.copy()

                st.session_state["dataset_split"] = True
                st.sidebar.success("División del dataset realizada con éxito.")


    # Mensaje final
    if st.session_state.get("dataset_split", False):
        st.sidebar.header("¡Configuración Completada!")
        st.sidebar.success("El dataset está listo. Ahora puedes proceder a entrenar el modelo.")




# Función para manejar valores nulos
def handle_nulls(X, numeric_null_option, categorical_null_option):
    """
    Maneja los valores nulos en los datos para columnas numéricas y categóricas.

    Args:
        X (pd.DataFrame): Datos de entrada.
        numeric_null_option (str): Opción seleccionada para manejar valores nulos numéricos.
        categorical_null_option (str): Opción seleccionada para manejar valores nulos categóricos.

    Returns:
        pd.DataFrame: DataFrame con valores nulos gestionados.
    """
    from sklearn.impute import SimpleImputer

    # Copiar el DataFrame para no modificar el original
    X_cleaned = X.copy()

    # Manejo de columnas numéricas
    numeric_cols = X_cleaned.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_cols.empty:
        if numeric_null_option == "Eliminar filas":
            X_cleaned.dropna(subset=numeric_cols, inplace=True)
        elif numeric_null_option == "Reemplazar con 0":
            X_cleaned[numeric_cols] = X_cleaned[numeric_cols].fillna(0)
        elif numeric_null_option == "Reemplazar con la media":
            imputer = SimpleImputer(strategy='mean')
            X_cleaned[numeric_cols] = imputer.fit_transform(X_cleaned[numeric_cols])
        elif numeric_null_option == "Reemplazar con la mediana":
            imputer = SimpleImputer(strategy='median')
            X_cleaned[numeric_cols] = imputer.fit_transform(X_cleaned[numeric_cols])
    else:
        st.sidebar.success("No hay valores nulos en las columnas numéricas.")

    # Manejo de columnas categóricas
    categorical_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        if categorical_null_option == "Eliminar filas":
            X_cleaned.dropna(subset=categorical_cols, inplace=True)
        elif categorical_null_option == "Reemplazar con 'Ninguno'":
            X_cleaned[categorical_cols] = X_cleaned[categorical_cols].fillna('Ninguno')
        elif categorical_null_option == "Reemplazar con la moda":
            imputer = SimpleImputer(strategy='most_frequent')
            X_cleaned[categorical_cols] = imputer.fit_transform(X_cleaned[categorical_cols])
    else:
        st.sidebar.success("No hay valores nulos en las columnas categóricas.")

    return X_cleaned




# Función para normalizar y escalar datos numéricos
def normalize_and_scale(X, scaling_option):
    """
    Normaliza y escala las columnas numéricas de un DataFrame.
    """
    # Selecciona solo columnas numéricas
    numeric_cols = X.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        raise ValueError("No hay columnas numéricas para escalar.")

    # Selecciona el tipo de escalador
    scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
    
    # Aplica el escalamiento
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X_scaled



# Función para codificar variables categóricas
def encode_categorical(X, encoding_option):
    """
    Codifica las variables categóricas según la opción seleccionada: One-Hot Encoding o Label Encoding.

    Args:
        X (pd.DataFrame): Datos de entrada.
        encoding_option (str): Método de codificación seleccionado ("One-Hot Encoding" o "Label Encoding").

    Returns:
        pd.DataFrame: DataFrame transformado con la codificación aplicada.
    """
    # Identificar columnas categóricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if not categorical_cols.empty:
        if encoding_option == "One-Hot Encoding":
            # Aplicar One-Hot Encoding
            X = pd.get_dummies(X, columns=categorical_cols)
        elif encoding_option == "Label Encoding":
            # Aplicar Label Encoding
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                X[col] = label_encoder.fit_transform(X[col].astype(str))
        else:
            raise ValueError(f"Método de codificación no soportado: {encoding_option}")
    else:
        st.sidebar.success("No hay variables categóricas en el dataset.")

    return X

# Configuración del Dataset
if tabs == "Dataset":
   # Configuración del Dataset
    configure_dataset()



# Visualizar ejemplos del dataset
elif tabs == "Capas":
    st.sidebar.subheader("Configuración de Capas")

    # Seleccionar arquitectura base
    st.sidebar.subheader("Arquitectura")
    selected_architecture = st.sidebar.selectbox(
        "Seleccione una Arquitectura",
        options=list(architectures[st.session_state['selected_dataset']].keys()),
        help="Elija la arquitectura del modelo basada en el dataset seleccionado."
    )

    # Aplicar la arquitectura seleccionada
    if st.sidebar.button("Aplicar Arquitectura"):
        # Obtener las capas de la arquitectura seleccionada
        selected_architecture_layers = architectures[st.session_state['selected_dataset']][selected_architecture]

        # Reiniciar las configuraciones de capas
        st.session_state['layer_config'] = selected_architecture_layers[:]
        st.session_state['num_layers_selected'] = len(selected_architecture_layers)

        # Validar tipos de capas según el dataset
        valid_types = ["Dense", "Dropout"] if st.session_state['selected_dataset'] in ['MNIST'] else [
            "Dense", "Dropout", "Conv2D", "MaxPooling2D", "Flatten", "BatchNormalization"
        ]

        for layer in st.session_state['layer_config']:
            if layer['type'] not in valid_types:
                st.warning(f"Tipo de capa no válido detectado: {layer['type']}.")
                st.session_state['layer_config'] = []
                break

        st.sidebar.success(f"Arquitectura '{selected_architecture}' aplicada correctamente.")

    # Validar cambios externos en configuraciones
    if st.session_state['layer_config'] != st.session_state.get('previous_layer_config', []):
        st.session_state['previous_layer_config'] = st.session_state['layer_config'][:]

    # Mostrar configuración personalizada de capas
    st.sidebar.subheader("Capas Personalizadas")
    st.sidebar.number_input(
        "Número de Capas",
        min_value=0,
        max_value=30,
        value=st.session_state['num_layers_selected'],
        step=1,
        key='num_layers_selected',
        help="Número total de capas que desea incluir en su modelo.",
        on_change=update_layer_config
    )

    # Validar si la configuración actual de capas es válida
    valid_types = ["Dense", "Dropout", "Conv2D", "MaxPooling2D", "Flatten", "BatchNormalization"]
    invalid_layers = any(layer.get("type") not in valid_types for layer in st.session_state['layer_config'])

    if invalid_layers:
        st.session_state['layer_config'] = []
        st.warning("La configuración previa de capas no es válida para el dataset seleccionado. Las capas se han reiniciado.")

    # Configuración de cada capa
    for i, layer in enumerate(st.session_state['layer_config']):
        with st.sidebar.expander(f"Configuración de la Capa {i + 1}", expanded=True):
            layer['type'] = st.selectbox(
                "Tipo de Capa",
                valid_types,
                index=valid_types.index(layer.get("type", "Dense")),
                key=f"layer_type_{i}",
                help="Seleccione el tipo de capa que desea añadir."
            )

            # Configuración específica para cada tipo de capa
            if layer['type'] == "Dense":
                layer['neurons'] = st.number_input(
                    "Número de Neuronas",
                    min_value=1,
                    value=layer.get('neurons', 64),
                    key=f"neurons_{i}",
                    help="Número de neuronas en la capa densa."
                )
                layer['activation'] = st.selectbox(
                    "Activación",
                    ["relu", "sigmoid", "tanh", "softmax","linear"],
                    index=["relu", "sigmoid", "tanh", "softmax","linear"].index(layer.get('activation', "relu")),
                    key=f"activation_{i}",
                    help="Función de activación para la capa densa."
                )
                # Checkbox para habilitar regularización L2
                layer['enable_l2'] = st.checkbox(
                    "Habilitar Regularización L2",
                    value=layer.get('enable_l2', False),
                    key=f"enable_l2_{i}",
                    help="Activa la regularización L2 para esta capa."
                )
                if layer['enable_l2']:
                    # Slider para definir el coeficiente L2 si está habilitado
                    layer['l2'] = st.slider(
                        "Regularización L2",
                        min_value=0.0,
                        max_value=0.1,
                        value=layer.get('l2', 0.01),
                        step=0.01,
                        key=f"l2_{i}",
                        help="Coeficiente de regularización L2 para los pesos de la capa densa."
                    )
            elif layer['type'] == "Dropout":
                layer['dropout_rate'] = st.slider(
                    "Tasa de Dropout",
                    0.0,
                    0.9,
                    value=layer.get('dropout_rate', 0.2),
                    step=0.1,
                    key=f"dropout_{i}",
                    help="Proporción de unidades que se desactivarán aleatoriamente en esta capa."
                )
            elif layer['type'] == "Conv2D":
                layer['filters'] = st.number_input(
                    "Número de Filtros",
                    min_value=1,
                    value=layer.get('filters', 32),
                    key=f"filters_{i}",
                    help="Define cuántos filtros utilizará esta capa de convolución."
                )
                layer['kernel_size'] = st.slider(
                    "Tamaño del Kernel",
                    min_value=1,
                    max_value=5,
                    value=layer.get('kernel_size', 3),
                    step=1,
                    key=f"kernel_{i}",
                    help="Tamaño del filtro o kernel que se moverá sobre la entrada."
                )
                layer['activation'] = st.selectbox(
                    "Activación",
                    ["relu", "sigmoid", "tanh", "softmax","linear"],
                    index=["relu", "sigmoid", "tanh", "softmax","linear"].index(layer.get('activation', "relu")),
                    key=f"activation_conv_{i}",
                    help="Función de activación para la capa de convolución."
                )
            elif layer['type'] == "MaxPooling2D":
                layer['pool_size'] = st.slider(
                    "Tamaño del Pool",
                    min_value=1,
                    max_value=5,
                    value=layer.get('pool_size', 2),
                    step=1,
                    key=f"pool_size_{i}",
                    help="Tamaño de la ventana para la operación de pooling."
                )
            elif layer['type'] == "Flatten":
                st.info("Capa que aplana la entrada para conectarla con capas densas.")
            elif layer['type'] == "BatchNormalization":
                st.info("Capa para normalizar las salidas de la capa anterior y estabilizar el entrenamiento.")

            # Actualizar el nombre de la capa
            layer['name'] = f"{layer['type']}{i + 1}"




# Configurar hiperparámetros
elif tabs == "Hiperparámetros":
    st.sidebar.subheader("Hiperparámetros")
    st.session_state['hyperparams']['optimizer'] = st.sidebar.selectbox(
        "Optimizador",
        ["Adam", "SGD", "RMSprop"],
        help="Elija el optimizador que se utilizará para ajustar los pesos del modelo."
    )
    st.session_state['hyperparams']['learning_rate'] = st.sidebar.slider(
        "Tasa de Aprendizaje",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4g",  # Mostrar hasta 4 decimales
        help="Velocidad con la que el modelo ajusta sus pesos durante el entrenamiento."
    )
    st.session_state['hyperparams']['epochs'] = st.sidebar.number_input(
        "Número de Épocas",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Número de veces que el modelo verá todo el conjunto de datos durante el entrenamiento."
    )
    st.session_state['hyperparams']['batch_size'] = st.sidebar.number_input(
        "Tamaño de Batch",
        min_value=8,
        max_value=128,
        value=32,
        step=8,
        help="Cantidad de muestras procesadas antes de actualizar los pesos del modelo."
    )
    st.session_state['hyperparams']['loss_function'] = st.sidebar.selectbox(
        "Función de Pérdida",
        ["categorical_crossentropy", "mean_squared_error"],
        help="Métrica utilizada para evaluar qué tan bien se está entrenando el modelo."
    )


# Configuración de Early Stopping
elif tabs == "Early Stopping":
    st.sidebar.subheader("Early Stopping")
    enable_early_stopping = st.sidebar.checkbox(
        "Habilitar Early Stopping",
        value=False,
        help="Detiene el entrenamiento si no hay mejora en la métrica monitoreada después de un número determinado de épocas."
    )
    if enable_early_stopping:
        patience = st.sidebar.number_input(
            "Patience (Número de épocas sin mejora)",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Número de épocas sin mejora antes de detener el entrenamiento."
        )
        monitor_metric = st.sidebar.selectbox(
            "Métrica a Monitorear",
            ["val_loss", "val_accuracy"],
            index=0,
            help="Métrica que se monitoreará para decidir si detener el entrenamiento."
        )



# Checkbox para mostrar/ocultar métricas (siempre disponible)
st.session_state['show_metrics'] = st.checkbox(
    "Mostrar Gráficos de Métricas",
    value=st.session_state.get('show_metrics', False),
    disabled=st.session_state['training_in_progress']  # Deshabilitar si está entrenando
)


# Función para resetear el estado de entrenamiento
def reset_training_state():
    st.session_state['training_in_progress'] = True
    st.session_state['training_finished'] = False
    st.session_state['logs'] = []  # Limpiar logs solo al comenzar nuevo entrenamiento
    st.session_state['modelDownload'] = None  # Limpiar modelo previo
    dynamic_placeholder.empty()  # Limpiar placeholders
    preview_placeholder.empty()


# Mostrar preview solo si no está entrenando
if not st.session_state['training_in_progress']:
    preview_graph(st.session_state['layer_config'], preview_placeholder)

# Botón para iniciar el entrenamiento
if not st.session_state['training_in_progress'] and not st.session_state['training_finished']:
    if st.button("Comenzar entrenamiento"):
        reset_training_state()  # Resetear estados antes de iniciar
        st.rerun()

# Botón para cancelar el entrenamiento
if st.session_state['training_in_progress']:
    col_cancel, col_info = st.columns([1, 3])
    with col_cancel:
        if st.button("Cancelar Entrenamiento"):
            st.session_state['training_in_progress'] = False
            st.session_state['training_finished'] = False
            st.session_state['logs'].append("Entrenamiento cancelado por el usuario.")
            dynamic_placeholder.text("Entrenamiento cancelado.")
            st.rerun()

# Mostrar progreso del entrenamiento
if st.session_state['training_in_progress']:
    st.write("Tipo de problema seleccionado:", st.session_state.get('problem_type'))
    if st.session_state.get('problem_type') == "Clasificación":
        train_model_classification(
            st.session_state['layer_config'],
            st.session_state['hyperparams'],
            preview_placeholder,
            dynamic_placeholder
        )
    elif st.session_state.get('problem_type') == "Regresión":
        train_model_regression(
            st.session_state['layer_config'],
            st.session_state['hyperparams'],
            preview_placeholder,
            dynamic_placeholder
        )
    else:
        st.error("El tipo de problema no está configurado correctamente.")
        st.stop()

    st.session_state['training_in_progress'] = False
    st.session_state['training_finished'] = True
    st.rerun()


# Mostrar el botón de guardar modelo una vez terminado el entrenamiento
if st.session_state['training_finished']:
    st.text_area(
        "Logs del Entrenamiento",
        "\n".join(st.session_state['logs']),
        height=300,
        key="final_logs"
    )

    # Botón para guardar el modelo
    if st.button("Guardar Modelo"):
        st.session_state['modelDownload'].save("trained_model.h5")
        with open("trained_model.h5", "rb") as file:
            st.download_button("Descargar Modelo", file, file_name="trained_model.h5")

    # Mostrar gráficos detallados de métricas bajo un checkbox
    if st.checkbox("Mostrar Gráficos de Métricas finales", key="show_final_metrics"):
        col1, col2 = st.columns(2)  # Dividir gráficos en columnas para mejor organización

        # Gráfico de pérdida
        with col1:
            st.write("#### Pérdida por Época")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state['loss_values']) + 1)),
                y=st.session_state['loss_values'],
                mode='lines+markers',
                name="Pérdida de Entrenamiento",
                line=dict(color='blue')
            ))
            if st.session_state['val_loss_values']:
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state['val_loss_values']) + 1)),
                    y=st.session_state['val_loss_values'],
                    mode='lines+markers',
                    name="Pérdida de Validación",
                    line=dict(color='red', dash='dash')
                ))
            fig_loss.update_layout(
                title="Pérdida por Época",
                xaxis_title="Época",
                yaxis_title="Pérdida",
                legend_title="Tipo"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # Gráfico de precisión
        with col2:
            if st.session_state['accuracy_values']:
                st.write("#### Precisión por Época")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state['accuracy_values']) + 1)),
                    y=st.session_state['accuracy_values'],
                    mode='lines+markers',
                    name="Precisión de Entrenamiento",
                    line=dict(color='green')
                ))
                if st.session_state['val_accuracy_values']:
                    fig_acc.add_trace(go.Scatter(
                        x=list(range(1, len(st.session_state['val_accuracy_values']) + 1)),
                        y=st.session_state['val_accuracy_values'],
                        mode='lines+markers',
                        name="Precisión de Validación",
                        line=dict(color='orange', dash='dash')
                    ))
                fig_acc.update_layout(
                    title="Precisión por Época",
                    xaxis_title="Época",
                    yaxis_title="Precisión",
                    legend_title="Tipo"
                )
                st.plotly_chart(fig_acc, use_container_width=True)

        # Métricas finales según el tipo de problema
        st.write("#### Métricas Finales")
        if st.session_state['problem_type'] == "Clasificación":
            # Verificar que el modelo exista
            if "modelDownload" not in st.session_state or st.session_state["modelDownload"] is None:
                st.error("El modelo no está entrenado. Por favor, entrena el modelo antes de intentar predecir.")
                st.stop()

            # Cargar el modelo desde session_state
            model = st.session_state["modelDownload"]

            # Datos de prueba
            split_type = st.session_state["split_type"]

            if split_type == "Entrenamiento y Prueba":
                X_test = st.session_state["splits"][1]
                y_test_original = st.session_state["splits"][3]
            elif split_type == "Entrenamiento, Validación y Prueba":
                X_test = st.session_state["splits"][2]
                y_test_original = st.session_state["splits"][5]
            else:
                st.error("Tipo de división desconocido. Revisa la configuración.")
                st.stop()

            # Predicción
            predictions = model.predict(X_test)
            y_pred = np.argmax(predictions, axis=1)

            # Validar dimensiones después de predecir
            if len(y_pred) != len(y_test_original):
                st.error(f"Dimensiones inconsistentes: y_pred tiene {len(y_pred)} muestras, pero y_test_original tiene {len(y_test_original)}.")
                st.stop()

            # Calcular métricas
            f1 = f1_score(y_test_original, y_pred, average='weighted')
            precision = precision_score(y_test_original, y_pred, average='weighted')
            recall = recall_score(y_test_original, y_pred, average='weighted')

            # Visualización de la matriz de confusión
            cm = confusion_matrix(y_test_original, y_pred)
            st.write("Matriz de confusión:")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)

            # Comparar predicciones con etiquetas reales
            st.write("Comparación de predicciones individuales:")
            for i in range(min(5, len(y_test_original))):  # Mostrar máximo 5 ejemplos
                st.write(f"Ejemplo {i+1}: Real: {y_test_original[i]}, Predicción: {y_pred[i]}")

            # Mostrar métricas
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**Precisión (Precision):** {precision:.4f}")
            st.write(f"**Recall:** {recall:.4f}")

        else:  # Para regresión
            final_loss = st.session_state['loss_values'][-1]
            st.write(f"**Pérdida Final (MAE):** {final_loss:.4f}")
            if st.session_state['val_loss_values']:
                final_val_loss = st.session_state['val_loss_values'][-1]
                st.write(f"**Pérdida Validación Final (MAE):** {final_val_loss:.4f}")

            # Gráfico de métricas finales (regresión)
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(name="Pérdida Final", x=["Pérdida"], y=[final_loss], marker_color='blue'))
            if 'val_loss_values' in st.session_state and st.session_state['val_loss_values']:
                fig_metrics.add_trace(go.Bar(name="Pérdida Validación", x=["Pérdida Validación"], y=[final_val_loss], marker_color='red'))
            fig_metrics.update_layout(title="Pérdidas Finales", barmode="group")
            st.plotly_chart(fig_metrics, use_container_width=True)

    # Botón para reiniciar el entrenamiento
    if st.button("Comenzar Entrenamiento"):
        reset_training_state()
        st.rerun()
