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
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# Ajustar el ancho del sidebar con CSS personalizado
st.markdown("""
    <style>
    /* Ajustar el ancho del sidebar */
    [data-testid="stSidebar"] {
        width: 300px; /* Cambia el valor según lo desees */
    }
    /* Ajustar el contenido del sidebar para que se ajuste al ancho */
    [data-testid="stSidebar"] .css-1d391kg {
        width: 300px; /* Cambia el valor para que coincida con el ancho del sidebar */
    }
    </style>
    """, unsafe_allow_html=True)



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

if 'scaler_y' not in st.session_state:
    st.session_state['scaler_y'] = None

if 'scaler_X' not in st.session_state:
    st.session_state['scaler_X'] = None

if 'early_stopping' not in st.session_state:
    st.session_state['early_stopping'] = {
        "enabled": False,
        "patience": 0,
        "monitor": "val_loss"
    }


if 'num_classes' not in st.session_state:
    st.session_state['num_classes'] = 0

if 'splits' not in st.session_state:
    st.session_state['splits'] = []

if 'disabled_button_train' not in st.session_state:
    st.session_state['disabled_button_train'] = False

if 'f1_score' not in st.session_state:
    st.session_state['f1_score'] = 0
if  'precision_score' not in st.session_state:
    st.session_state['precision_score'] = 0
if 'recall_score' not in st.session_state:
    st.session_state['recall_score'] = 0

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


    # Configurar Early Stopping si está habilitado
    callbacks = []
    early_stopping_config = st.session_state.get('early_stopping', {})
    if early_stopping_config.get("enabled", False):
        patience = early_stopping_config.get("patience", 0)
        monitor_metric = early_stopping_config.get("monitor", "val_loss")
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        log_event(log_placeholder, f"Early Stopping habilitado: Monitoreando '{monitor_metric}' con paciencia de {patience} épocas.")



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



    # Crear modelo
    input_shape = (X_train.shape[1],)
    model = Sequential()
    for i, layer in enumerate(layers):
        if layer['type'] == "Dense":
            model.add(Dense(layer['neurons'], activation=layer['activation'], input_shape=input_shape if i == 0 else None,kernel_regularizer=l2(layer['l2']) if layer.get('enable_l2', False) else None))
        elif layer['type'] == "Dropout":
            model.add(Dropout(layer['dropout_rate']))

    # Configurar optimizador y compilar modelo
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Entrenar modelo
    for epoch in range(hyperparams['epochs']):
        log_event(log_placeholder, f"Época {epoch + 1}/{hyperparams['epochs']} iniciada.")

        if X_val is not None and y_val is not None:
            # Con conjunto de validación  
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                batch_size=hyperparams['batch_size'], epochs=1, verbose=0, callbacks=callbacks)
        else:
            # Sin conjunto de validación
            history = model.fit(X_train, y_train, batch_size=hyperparams['batch_size'],
                                epochs=1, verbose=0, callbacks=callbacks)

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
    st.session_state['mae_values'] = []
    st.session_state['val_loss_values'] = []
    st.session_state['val_mae_values'] = []

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


    # Configurar Early Stopping si está habilitado
    callbacks = []
    early_stopping_config = st.session_state.get('early_stopping', {})
    if early_stopping_config.get("enabled", False):
        patience = early_stopping_config.get("patience", 0)
        monitor_metric = early_stopping_config.get("monitor", "val_loss")
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        log_event(log_placeholder, f"Early Stopping habilitado: Monitoreando '{monitor_metric}' con paciencia de {patience} épocas.")


    # Crear modelo
    input_shape = (X_train.shape[1],)
    model = Sequential()
    for i, layer in enumerate(layers):
        if layer['type'] == "Dense":
            model.add(Dense(layer['neurons'], activation=layer['activation'],
                            input_shape=input_shape if i == 0 else None,
                            kernel_regularizer=l2(layer['l2']) if layer.get('enable_l2', False) else None))
        elif layer['type'] == "Dropout":
            model.add(Dropout(layer['dropout_rate']))

    # Configurar optimizador y compilar modelo
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

    # Entrenar modelo
    for epoch in range(hyperparams['epochs']):
        log_event(log_placeholder, f"Época {epoch + 1}/{hyperparams['epochs']} iniciada.")

        if X_val is not None and y_val is not None:
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                batch_size=hyperparams['batch_size'], epochs=1, verbose=0, callbacks=callbacks)
        else:
            history = model.fit(X_train, y_train, batch_size=hyperparams['batch_size'],
                                epochs=1, verbose=0, callbacks=callbacks)

        # Registrar métricas de entrenamiento
        train_loss = history.history['loss'][0]
        train_mae = history.history['mae'][0]
        log_event(log_placeholder, f"Entrenamiento - Pérdida (MSE): {train_loss:.4f}, Precisión (MAE): {train_mae:.4f}")

        # Registrar métricas de validación, si aplican
        if X_val is not None:
            val_loss = history.history['val_loss'][0]
            val_mae = history.history['val_mae'][0]
            log_event(log_placeholder, f"Validación - Pérdida (MSE): {val_loss:.4f}, Precisión (MAE): {val_mae:.4f}")

        # Actualizar métricas para gráficos
        st.session_state['loss_values'].append(train_loss)
        st.session_state['mae_values'].append(train_mae)
        if X_val is not None:
            st.session_state['val_loss_values'].append(val_loss)
            st.session_state['val_mae_values'].append(val_mae)

        # Actualizar visualización dinámica
        update_graph_with_smooth_color_transition(
            st.session_state['graph'], epoch, dynamic_graph_placeholder, neurons_per_point=10, animation_steps=30
        )

    # Guardar modelo
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





# Configuración del Dataset
from sklearn.impute import SimpleImputer
import numpy as np




# Función principal para configurar el dataset
def configure_dataset():
    st.sidebar.header("Configuración del Dataset")

    # Paso 1: Selección del dataset
    st.sidebar.subheader("Paso 1: Selección del Dataset")
    problem_type = st.sidebar.selectbox(
        "Seleccione el tipo de problema",
        ["Clasificación", "Regresión"],
        key="problem_type_selectbox",
        help="Clasificación es para predecir categorías, como tipos de flores. Regresión es para predecir valores continuos, como precios de casas."
    )
    st.sidebar.info("""
        **Definición**: El dataset es el conjunto de datos que será utilizado para entrenar y evaluar el modelo. 

        **Sugerencia**: Selecciona un tipo de problema (clasificación o regresión) y el dataset apropiado para el tipo de análisis que deseas realizar. 
        Por ejemplo:
        - **Clasificación**: Usarás el modelo para predecir categorías, como tipos de flores.
        - **Regresión**: Usarás el modelo para predecir valores continuos, como precios de casas.
        """)

    st.session_state['problem_type'] = problem_type

    dataset_name = st.sidebar.selectbox(
        "Seleccione el dataset",
        ["Iris", "Wine", "Breast Cancer", "Digits"] if problem_type == "Clasificación" else ["Boston Housing", "Diabetes"],
        key="dataset_name_selectbox",
        help="Elija el dataset apropiado según el tipo de problema que seleccionó anteriormente."
    )

    st.session_state['selected_dataset'] = dataset_name

    if st.sidebar.button("Cargar Dataset", key="load_dataset_button",):
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
            key="target_variable_selectbox",
            help="Seleccione la columna que desea predecir. Por ejemplo, en un dataset de casas, puede ser el precio."
        )
        if target_variable != st.session_state.get("target_variable", None):
            st.session_state["target_variable"] = target_variable
            st.session_state["X_original"] = st.session_state["full_dataset"].drop(columns=[target_variable]).copy()
            st.session_state["y_original"] = st.session_state["full_dataset"][target_variable].copy()

        st.sidebar.success(f"Variable objetivo seleccionada: {st.session_state['target_variable']}")
        st.sidebar.write("Características (X):", st.session_state["X_original"].head())
        # Verificar si y_original es un numpy.ndarray y convertirlo en un DataFrame/Series antes de usar .head()
        if isinstance(st.session_state["y_original"], np.ndarray):
            y_display = pd.Series(st.session_state["y_original"].ravel())
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
            key="columns_to_drop_multiselect",
            help="Elimine columnas irrelevantes o redundantes, como IDs o nombres que no aporten al análisis."
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
                key="numeric_null_option_selectbox",
                help="Elija cómo manejar valores nulos en columnas numéricas. Por ejemplo, reemplazar con la media es útil en la mayoría de los casos."
            )
            categorical_null_option = st.sidebar.selectbox(
                "Valores Nulos - Categóricos",
                ["Eliminar filas", "Reemplazar con 'Ninguno'", "Reemplazar con la moda"],
                key="categorical_null_option_selectbox",
                help="Elija cómo manejar valores nulos en columnas categóricas. Por ejemplo, reemplazar con la moda es común en problemas de clasificación."
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
                    key="encoding_option_selectbox",
                    help="Use One-Hot Encoding para categorías sin orden (como colores). Use Label Encoding para categorías ordenadas (como niveles de educación)."
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
                key="scaling_option_selectbox",
                help="StandardScaler es útil para datos con distribución normal. MinMaxScaler es útil para datos en rangos específicos."
            )
            if st.sidebar.button("Aplicar Escalamiento", key="apply_scaling_button",help="Aplica un escalamiento a la variable objetivo para mejorar el rendimiento del modelo en problemas de regresión."):
                numeric_columns = st.session_state["X_original"].select_dtypes(include=["float64", "int64"]).columns
                scaler_x = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
                scaled_data = scaler_x.fit_transform(st.session_state["X_original"][numeric_columns])
                st.session_state["X_original"].loc[:, numeric_columns] = scaled_data
                st.session_state["scaler_x"] = scaler_x
                # Escalamiento de la variable objetivo (y) en caso de regresión
                if st.session_state["problem_type"] == "Regresión":
                    st.sidebar.subheader("Escalamiento de la Variable Objetivo (y)")
                    scaler_y = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
                    y_scaled = scaler_y.fit_transform(st.session_state["y_original"].values.reshape(-1, 1))
                    st.session_state["y_original"] = y_scaled
                    st.session_state["scaler_y"] = scaler_y  # Guardar el escalador para revertir predicciones
                    st.sidebar.success("Escalamiento aplicado a la variable objetivo (y).")
                    st.sidebar.write("Vista previa de la variable objetivo escalada:", st.session_state["y_original"][:5])                 
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
                key="split_type_selectbox",
                help="Entrenamiento y Prueba es una división básica. Agregar Validación ayuda a ajustar el modelo y prevenir sobreajuste."
            )
            st.session_state["split_type"] = split_type
            train_ratio = st.sidebar.slider("Entrenamiento (%)", 10, 90, 70, key="train_ratio_slider",help="Seleccione el porcentaje de datos que se usarán para entrenar el modelo. Usualmente es entre 70-80%.")

            # Calcular proporciones para validación y prueba
            if split_type == "Entrenamiento, Validación y Prueba":
                val_ratio = st.sidebar.slider("Validación (%)", 5, 50, 15, key="val_ratio_slider",help="El porcentaje restante se divide entre validación y prueba. Asegúrate de dejar suficientes datos para cada conjunto.")
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

                st.session_state['train_ratio'] = train_ratio
                st.session_state['val_ratio'] = val_ratio
                st.session_state['test_ratio'] = test_ratio


                # Verificación de dimensiones
                splits = st.session_state["splits"]
                
                if len(splits) == 4:
                    X_train, X_test, y_train, y_test = splits
                elif len(splits) == 6:
                    X_train, X_val, X_test, y_train, y_val, y_test = splits
                
                if len(X_test) != len(y_test):
                    st.error("Dimensiones inconsistentes entre X_test y y_test después de la división.")
                    st.stop()
                if problem_type == "Clasificación":
                    num_classes = len(np.unique(y_train))
                    st.session_state['num_classes'] = num_classes
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



def validate_layer_config(layer_config, problem_type):
    warnings = []
    output_layer = layer_config[-1]  # Última capa
    num_classes = st.session_state['num_classes']
    # Validar que la última capa sea adecuada para el problema
        
    if st.session_state['splits'] == []:  
        warnings.append("Primero debes dividir el dataset antes de entrenar el modelo.")
        st.session_state['disabled_button_train'] = True
        return warnings
    if problem_type == "Clasificación":
        if num_classes == 2:  # Clasificación Binaria
            if output_layer['neurons'] != 1 or output_layer['activation'] != "sigmoid":
                warnings.append("Para clasificación binaria, la capa de salida debe tener 1 neurona con activación 'sigmoid'.")
        elif num_classes > 2:  # Clasificación Multiclase
            if output_layer['neurons'] != num_classes or output_layer['activation'] != "softmax":
                warnings.append(f"Para clasificación multiclase, la capa de salida debe tener {num_classes} neuronas con activación 'softmax'.")
    elif problem_type == "Regresión":
        if output_layer['neurons'] != 1 or output_layer['activation'] != "linear":
            warnings.append("Para problemas de regresión, la capa de salida debe tener 1 neurona con activación 'linear'.")
    elif problem_type == "Clasificación Multietiqueta":
        if output_layer['neurons'] != num_classes or output_layer['activation'] != "sigmoid":
            warnings.append(f"Para clasificación multietiqueta, la capa de salida debe tener {num_classes} neuronas con activación 'sigmoid'.")

    # Validar uso adecuado de capas Flatten
    flatten_count = sum(1 for layer in layer_config if layer['type'] == "Flatten")
    if flatten_count > 1:
        warnings.append("Se detectaron múltiples capas 'Flatten'. Generalmente sólo se necesita una.")
    elif flatten_count == 0 and any(layer['type'] in ["Conv2D", "MaxPooling2D"] for layer in layer_config):
        warnings.append("Se requiere al menos una capa 'Flatten' después de 'Conv2D' o 'MaxPooling2D' para conectar con capas densas.")

    # Validar uso de Dropout en redes profundas
    if len(layer_config) > 5 and not any(layer['type'] == "Dropout" for layer in layer_config):
        warnings.append("Se recomienda incluir capas 'Dropout' para prevenir el sobreajuste en redes profundas.")

    # Validar configuración de Conv2D y MaxPooling2D
    if any(layer['type'] == "Conv2D" for layer in layer_config) and not any(layer['type'] == "MaxPooling2D" for layer in layer_config):
        warnings.append("Se recomienda agregar una capa 'MaxPooling2D' después de 'Conv2D' para reducir dimensiones.")

    # Validar que Conv2D tenga un número razonable de filtros
    for layer in layer_config:
        if layer['type'] == "Conv2D" and layer.get('filters', 0) < 8:
            warnings.append("Las capas 'Conv2D' deberían tener al menos 8 filtros para ser efectivas.")

    # Validar que Dropout tenga valores razonables
    for layer in layer_config:
        if layer['type'] == "Dropout" and not (0.0 < layer.get('dropout_rate', 0.0) <= 0.9):
            warnings.append("La capa 'Dropout' debe tener una tasa entre 0.1 y 0.9.")

    # Validar que las capas Dense tengan un número positivo de neuronas
    for layer in layer_config:
        if layer['type'] == "Dense" and layer.get('neurons', 0) <= 0:
            warnings.append("Las capas 'Dense' deben tener un número positivo de neuronas.")

    # Validar que la primera capa sea adecuada para el dataset
    if problem_type == "Clasificación" and layer_config[0]['type'] != "Dense":
        warnings.append("La primera capa debe ser de tipo 'Dense' para manejar correctamente los datos de entrada en problemas de clasificación.")
    elif problem_type == "Regresión" and layer_config[0]['type'] != "Dense":
        warnings.append("La primera capa debe ser de tipo 'Dense' para manejar correctamente los datos de entrada en problemas de regresión.")

    # Validar que BatchNormalization no sea la última capa
    if layer_config[-1]['type'] == "BatchNormalization":
        warnings.append("La última capa no debe ser 'BatchNormalization'. Considera moverla antes de la capa de salida.")

    # Validar que las dimensiones de salida sean compatibles con el problema
    if problem_type == "Clasificación":
        num_classes = st.session_state.get('num_classes', 0)
        if num_classes > 2 and layer_config[-1].get('neurons', 1) != num_classes:
            warnings.append(f"La última capa debe tener {num_classes} neuronas para coincidir con el número de clases en el problema de clasificación.")
    elif problem_type == "Regresión":
        if layer_config[-1].get('neurons', 1) != 1:
            warnings.append("La última capa debe tener exactamente 1 neurona para problemas de regresión.")

    # Validar que MaxPooling2D tenga un tamaño de pool razonable
    for layer in layer_config:
        if layer['type'] == "MaxPooling2D" and layer.get('pool_size', 1) <= 0:
            warnings.append("La capa 'MaxPooling2D' debe tener un tamaño de pool mayor a 0.")

    return warnings


# Configuración con pestañas
tabs = st.sidebar.radio("Configuración:", ["Dataset","Capas", "Hiperparámetros", "Early Stopping"],disabled=st.session_state['disable_other_tabs'] if 'disable_other_tabs' in st.session_state else False)

# Validar configuración de capas
if st.session_state['layer_config'] != []: 
    warnings = validate_layer_config(st.session_state['layer_config'], st.session_state['problem_type'])
    # Mostrar advertencias al usuario
    if warnings:
        st.sidebar.warning("⚠️ Se detectaron posibles problemas en la configuración de las capas:")
        for warning in warnings:
            st.sidebar.write(f"- {warning}")
    else:
        st.sidebar.success("✅ La configuración de capas parece adecuada para el problema seleccionado.")

# Configuración del Dataset
if tabs == "Dataset":
   # Configuración del Dataset
    configure_dataset()



# Visualizar ejemplos del dataset
elif tabs == "Capas":
    st.sidebar.subheader("Configuración de Capas",help="Definición: La configuración de capas es un paso clave para diseñar una red neuronal, definiendo cómo procesará los datos y aprenderá durante el entrenamiento. Sugerencia: Si no estás seguro de cómo diseñar la arquitectura, selecciona una de las arquitecturas predefinidas para tu dataset y ajusta las capas según sea necesario.")
    st.sidebar.info("""
    **Definición**: Cada capa en una red neuronal tiene una función específica para procesar los datos. Las configuraciones permiten personalizar cómo se comporta cada capa.

    **Sugerencia**:
    - Comienza con pocas capas si no estás seguro de cómo diseñar la arquitectura.
    - Usa **Dropout** si notas que el modelo se ajusta demasiado a los datos de entrenamiento.
    - Agrega **BatchNormalization** para estabilizar el entrenamiento si utilizas redes profundas.
    """)

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
                help="""Seleccione el tipo de capa que desea añadir. Cada tipo tiene un propósito específico en la red neuronal:
                - **Dense**: Totalmente conectada, útil para patrones complejos.
                - **Dropout**: Reduce el sobreajuste al desactivar neuronas aleatoriamente.
                - **Conv2D**: Extrae patrones espaciales en imágenes.
                - **MaxPooling2D**: Reduce dimensiones seleccionando características clave.
                - **Flatten**: Convierte datos multidimensionales a un solo vector.
                - **BatchNormalization**: Normaliza las salidas para estabilizar y acelerar el entrenamiento.""",
                on_change=update_layer_config
            )

            # Configuración específica para cada tipo de capa
            if layer['type'] == "Dense":
                layer['neurons'] = st.number_input(
                    "Número de Neuronas",
                    min_value=1,
                    value=layer.get('neurons', 64),
                    key=f"neurons_{i}",
                    help="Define cuántas unidades tendrá esta capa para aprender patrones complejos. Un mayor número de neuronas puede capturar más información, pero podría incrementar el tiempo de entrenamiento.",
                    on_change=update_layer_config
                )
                layer['activation'] = st.selectbox(
                    "Activación",
                    ["relu", "sigmoid", "tanh", "softmax","linear"],
                    index=["relu", "sigmoid", "tanh", "softmax","linear"].index(layer.get('activation', "relu")),
                    key=f"activation_{i}",
                    help="""Función que determina cómo se transforman las salidas de esta capa.
                    - **ReLU**: Ideal para muchas capas, evita valores negativos.
                    - **Sigmoid**: Convierte salidas a un rango entre 0 y 1, útil para clasificación binaria.
                    - **Tanh**: Rango entre -1 y 1, útil para datos centrados.
                    - **Softmax**: Convierte salidas a probabilidades, ideal para clasificación multicategoría.
                    - **Linear**: Mantiene las salidas sin cambios, útil en regresión.""",
                    on_change=update_layer_config
                )
                # Checkbox para habilitar regularización L2
                layer['enable_l2'] = st.checkbox(
                    "Habilitar Regularización L2",
                    value=layer.get('enable_l2', False),
                    key=f"enable_l2_{i}",
                    help="Penaliza grandes valores de los pesos para evitar el sobreajuste. Utilízalo si notas que el modelo se ajusta demasiado a los datos de entrenamiento.",
                    on_change=update_layer_config
                )
                if layer['enable_l2']:
                    # Slider para definir el coeficiente L2 si está habilitado
                    layer['l2'] = st.slider(
                        "Regularización L2",
                        min_value=0.001,
                        max_value=0.1,
                        value=layer.get('l2', 0.01),
                        step=0.001,
                        key=f"l3_{i}",
                        help="Controla la intensidad de la penalización en los pesos. Valores más altos aplican más restricción.",
                        on_change=update_layer_config
                    )
            elif layer['type'] == "Dropout":
                layer['dropout_rate'] = st.slider(
                    "Tasa de Dropout",
                    0.0,
                    0.9,
                    value=layer.get('dropout_rate', 0.2),
                    step=0.1,
                    key=f"dropout_{i}",
                    help="Porcentaje de neuronas que se desactivarán aleatoriamente durante el entrenamiento. Ayuda a prevenir el sobreajuste y mejora la generalización del modelo.",
                    on_change=update_layer_config
                )
            elif layer['type'] == "Conv2D":
                layer['filters'] = st.number_input(
                    "Número de Filtros",
                    min_value=1,
                    value=layer.get('filters', 32),
                    key=f"filters_{i}",
                    help="Número de detectores de características que analizarán las entradas. Más filtros pueden detectar patrones más variados, pero incrementan el tiempo de entrenamiento.",
                    on_change=update_layer_config
                )
                layer['kernel_size'] = st.slider(
                    "Tamaño del Kernel",
                    min_value=1,
                    max_value=5,
                    on_change=update_layer_config,
                    value=layer.get('kernel_size', 3),
                    step=1,
                    key=f"kernel_{i}",
                    help="Tamaño del filtro que se mueve sobre la entrada. Por ejemplo, un kernel de 3x3 analiza pequeños fragmentos de la entrada a la vez."
                )
                layer['activation'] = st.selectbox(
                    "Activación",
                    ["relu", "sigmoid", "tanh", "softmax","linear"],
                    index=["relu", "sigmoid", "tanh", "softmax","linear"].index(layer.get('activation', "relu")),
                    key=f"activation_conv_{i}",
                    on_change=update_layer_config,
                    help="""Función que transforma las salidas de la convolución.
                        - **ReLU**: Comúnmente usada en capas convolucionales.
                        - **Sigmoid** y **Tanh**: Útiles en casos específicos como segmentación de imágenes."""
                    )
            elif layer['type'] == "MaxPooling2D":
                layer['pool_size'] = st.slider(
                    "Tamaño del Pool",
                    min_value=1,
                    max_value=5,
                    on_change=update_layer_config,
                    value=layer.get('pool_size', 2),
                    step=1,
                    key=f"pool_size_{i}",
                    help="Tamaño de la ventana utilizada para reducir las dimensiones. Por ejemplo, un tamaño de pool de 2x2 seleccionará el valor máximo en cada área de 2x2 píxeles."
                )
            elif layer['type'] == "Flatten":
                on_change=update_layer_config
                st.info("Capa que convierte entradas multidimensionales en un único vector. Esto es necesario para conectar capas convolucionales con capas densas.")
            elif layer['type'] == "BatchNormalization":
                on_change=update_layer_config
                st.info("Capa que normaliza las salidas de la capa anterior para estabilizar el entrenamiento. Es especialmente útil en redes profundas para acelerar la convergencia.")

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


elif tabs == "Early Stopping":
    st.sidebar.subheader("Early Stopping")

    # Verificar si el conjunto de validación está disponible
    has_validation_data = len(st.session_state.get('splits', [])) == 6

    enable_early_stopping = st.sidebar.checkbox(
        "Habilitar Early Stopping",
        value=False,
        help="Detiene el entrenamiento si no hay mejora en la métrica monitoreada después de un número determinado de épocas.",
        disabled=not has_validation_data  # Deshabilitar si no hay validación
    )

    if not has_validation_data:
        st.sidebar.warning("El conjunto de validación no está configurado. Early Stopping requiere un conjunto de validación.")

    if enable_early_stopping and has_validation_data:
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
        st.session_state['early_stopping'] = {
            "enabled": enable_early_stopping,
            "patience": patience,
            "monitor": monitor_metric
        }
    else:
        st.session_state['early_stopping'] = {
            "enabled": False
        }



# Checkbox para mostrar/ocultar métricas (siempre disponible)
st.session_state['show_metrics'] = st.checkbox(
    "Mostrar Gráficos en tiempo de entrenamiento",
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
    if st.button("Comenzar entrenamiento", disabled = st.session_state['disabled_button_train']):
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
    # Botón para reiniciar el entrenamiento
    if st.button("Comenzar Entrenamiento"):
        reset_training_state()
        st.rerun()
# Mostrar gráficos detallados de métricas bajo un checkbox
    if st.checkbox("Mostrar Gráficos de Métricas finales", key="show_final_metrics"):
        st.header("📊 Métricas finales")

        col1, col2 = st.columns(2)  # Dividir gráficos en columnas para mejor organización

        if st.session_state['problem_type'] == "Clasificación":
            # Gráfico de pérdida
            with col1:
                st.write("#### Pérdida por Época")
                with st.expander("¿Qué significa este gráfico?"):
                    st.write("Este gráfico muestra cómo la pérdida (error) cambia en cada época. Una tendencia descendente indica que el modelo está aprendiendo y ajustándose mejor a los datos.")
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
                    st.write(
                        "#### Precisión por Época",
                    )
                    with st.expander("¿Qué significa este gráfico?"):
                        st.write("Este gráfico muestra cómo la precisión del modelo cambia en cada época. Una tendencia ascendente indica que el modelo está mejorando en sus predicciones.")
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


        elif st.session_state['problem_type'] == "Regresión":
            # Gráfico de Pérdida (MSE)
            with col1:
                st.write("#### Pérdida por Época (MSE)")
                with st.expander("¿Qué significa este gráfico?"):
                    st.write("Este gráfico muestra cómo cambia el error cuadrático medio (MSE) durante el entrenamiento. Una pérdida menor indica que el modelo se está ajustando mejor a los datos.")
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state['loss_values']) + 1)),
                    y=st.session_state['loss_values'],
                    mode='lines+markers',
                    name="Pérdida de Entrenamiento (MSE)",
                    line=dict(color='blue')
                ))
                if st.session_state['val_loss_values']:
                    fig_loss.add_trace(go.Scatter(
                        x=list(range(1, len(st.session_state['val_loss_values']) + 1)),
                        y=st.session_state['val_loss_values'],
                        mode='lines+markers',
                        name="Pérdida de Validación (MSE)",
                        line=dict(color='red', dash='dash')
                    ))
                fig_loss.update_layout(
                    title="Pérdida por Época (MSE)",
                    xaxis_title="Época",
                    yaxis_title="Pérdida (MSE)",
                    legend_title="Tipo"
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            # Gráfico de Precisión (MAE)
            with col2:
                st.write("#### Precisión por Época (MAE)")
                with st.expander("¿Qué significa este gráfico?"):
                    st.write("Este gráfico muestra cómo cambia el error absoluto medio (MAE) durante el entrenamiento. Un MAE menor indica que las predicciones del modelo están más cerca de los valores reales.")
        
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state['mae_values']) + 1)),
                    y=st.session_state['mae_values'],
                    mode='lines+markers',
                    name="Precisión de Entrenamiento (MAE)",
                    line=dict(color='green')
                ))
                if st.session_state['val_mae_values']:
                    fig_mae.add_trace(go.Scatter(
                        x=list(range(1, len(st.session_state['val_mae_values']) + 1)),
                        y=st.session_state['val_mae_values'],
                        mode='lines+markers',
                        name="Precisión de Validación (MAE)",
                        line=dict(color='orange', dash='dash')
                    ))
                fig_mae.update_layout(
                    title="Precisión por Época (MAE)",
                    xaxis_title="Época",
                    yaxis_title="Precisión (MAE)",
                    legend_title="Tipo"
                )
                st.plotly_chart(fig_mae, use_container_width=True)

        # Métricas finales según el tipo de problema
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
            st.session_state['f1_score'] = f1
            precision = precision_score(y_test_original, y_pred, average='weighted')
            st.session_state['precision_score'] = precision
            recall = recall_score(y_test_original, y_pred, average='weighted')
            st.session_state['recall_score'] = recall

            # Visualización de la matriz de confusión
            cm = confusion_matrix(y_test_original, y_pred)
            st.session_state['confusion_matrix'] = cm
            st.write("#### Matriz de confusión:")
            with st.expander("¿Qué significa esta matriz?"):
                st.write("""
                Una matriz de confusión compara las predicciones del modelo con los valores reales. Cada celda representa:
                
                - **Fila**: La clase real.
                - **Columna**: La clase predicha.
                
                Interpretación:
                - **Diagonal principal**: Cantidad de predicciones correctas.
                - **Fuera de la diagonal**: Errores de predicción (predicciones incorrectas).
                
                Por ejemplo:
                - Un valor alto en la diagonal principal indica que el modelo predijo correctamente para esa clase.
                - Un valor alto fuera de la diagonal indica errores específicos entre dos clases.
                """)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)

            # Comparar predicciones con etiquetas reales
            st.write("Comparación de predicciones individuales:")
            for i in range(min(5, len(y_test_original))):  # Mostrar máximo 5 ejemplos
                if isinstance(y_test_original, np.ndarray):  # Si es un array de NumPy
                    real_value = y_test_original[i]
                else:  # Si es un DataFrame o Serie de Pandas
                    real_value = y_test_original.iloc[i]
                
                st.write(f"Ejemplo {i+1}: Real: {real_value}, Predicción: {y_pred[i]}")

            # Mostrar métricas
            # Mostrar métricas finales de clasificación
            st.write("#### Métricas Finales")
            st.text_input(
                "**F1 Score:**",
                f"{f1:.4f}",
                help="El F1 Score es una medida de equilibrio entre precisión y recall, especialmente útil en datasets desbalanceados. Valores cercanos a 1 indican un buen rendimiento."
            )
            st.text_input(
                "**Precisión (Precision):**",
                f"{precision:.4f}",
                help="La precisión indica qué proporción de las predicciones positivas es correcta. Es clave en problemas donde los falsos positivos son costosos."
            )
            st.text_input(
                "**Recall:**",
                f"{recall:.4f}",
                help="El recall indica qué proporción de los casos positivos reales fue correctamente identificada. Es clave cuando los falsos negativos son costosos."
            )

        else:  # Para regresión
            final_loss = st.session_state['loss_values'][-1]
            final_mae = st.session_state['mae_values'][-1]  # Precisión final (MAE)
            # Métricas finales de regresión
            st.text_input(
                "**Pérdida Final (MSE):**",
                f"{final_loss:.4f}",
                help="El Mean Squared Error (MSE) mide el error promedio cuadrático entre los valores reales y las predicciones. Valores más bajos indican mayor precisión."
            )
            st.text_input(
                "**Precisión Final (MAE):**",
                f"{final_mae:.4f}",
                help="El Mean Absolute Error (MAE) mide el error promedio absoluto entre los valores reales y las predicciones. Es más robusto frente a valores atípicos en comparación con el MSE."
            )
            if st.session_state['val_loss_values']:
                final_val_loss = st.session_state['val_loss_values'][-1]
                final_val_mae = st.session_state['val_mae_values'][-1]  # Precisión validación (MAE)
                st.write(f"**Pérdida Validación Final (MSE):** {final_val_loss:.4f}")
                st.write(f"**Precisión Validación Final (MAE):** {final_val_mae:.4f}")

            # Gráfico de métricas finales (regresión)
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(
                name="Pérdida Final (MSE)",
                x=["Pérdida"],
                y=[final_loss],
                marker_color='blue'
            ))
            fig_metrics.add_trace(go.Bar(
                name="Precisión Final (MAE)",
                x=["Precisión"],
                y=[final_mae],
                marker_color='green'
            ))
            if 'val_loss_values' in st.session_state and st.session_state['val_loss_values']:
                fig_metrics.add_trace(go.Bar(
                    name="Pérdida Validación (MSE)",
                    x=["Pérdida Validación"],
                    y=[final_val_loss],
                    marker_color='red'
                ))
                fig_metrics.add_trace(go.Bar(
                    name="Precisión Validación (MAE)",
                    x=["Precisión Validación"],
                    y=[final_val_mae],
                    marker_color='orange'
                ))
            fig_metrics.update_layout(
                title="Métricas Finales (Regresión)",
                barmode="group",
                xaxis_title="Métrica",
                yaxis_title="Valor"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

            # Comparar predicciones con valores reales (prueba)
            model = st.session_state["modelDownload"]
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

            # Revertir escalado si se aplicó
            if "scaler_y" in st.session_state and st.session_state["scaler_y"] is not None:
                scaler = st.session_state["scaler_y"]
                y_test_original = scaler.inverse_transform(y_test_original.reshape(-1, 1)).flatten()
                y_pred = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
            else:
                y_pred = model.predict(X_test).flatten()
                y_test_original = y_test_original.flatten()

            # Comparación de predicciones con valores reales
            st.write(
                "#### Comparación de Predicciones con Valores Reales",
            )
            with st.expander("¿Qué significa este gráfico?"):
                st.write("""
                Este gráfico compara los valores reales (lo que deberíamos obtener) con las predicciones realizadas por el modelo.
                
                Interpretación:
                - **Línea Azul (Valores Reales)**: Representa los valores verdaderos del dataset de prueba.
                - **Línea Verde (Predicciones)**: Representa los valores que el modelo ha predicho.
                
                Un buen modelo debería tener las líneas muy cercanas, lo que indica que las predicciones se aproximan bien a los valores reales.
                
                ¿Qué buscar?
                - Si las líneas están muy separadas, podría ser una señal de que el modelo no está generalizando bien.
                - Si las líneas se cruzan constantemente, indica un modelo que sigue las tendencias correctamente.
                """)
            for i in range(min(5, len(y_test_original))):  # Mostrar máximo 5 ejemplos
                st.write(f"Ejemplo {i+1}: Real: {float(y_test_original[i]):.4f}, Predicción: {float(y_pred[i]):.4f}")

            # Gráfico de comparación
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=list(range(len(y_test_original))),
                y=y_test_original,
                mode='lines+markers',
                name="Real",
                line=dict(color='blue')
            ))
            fig_comparison.add_trace(go.Scatter(
                x=list(range(len(y_pred))),
                y=y_pred,
                mode='lines+markers',
                name="Predicción",
                line=dict(color='green', dash='dash')
            ))
            fig_comparison.update_layout(
                title="Comparación de Predicciones y Valores Reales",
                xaxis_title="Índice de Prueba",
                yaxis_title="Valor",
                legend_title="Tipo"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

# Probar el modelo guardado
if st.session_state["training_finished"] and st.session_state["modelDownload"]:
    
    if st.checkbox("Probar modelo", key="Test_model"):

        st.header("🔍 Probar el Modelo")

        # Mostrar instrucciones para el usuario
        st.write("Modifique los valores de las características para probar el modelo y observe cómo cambian las predicciones.")

        # Recuperar el modelo entrenado
        model = st.session_state["modelDownload"]

        # Determinar las características del dataset
        if "splits" in st.session_state:
            if st.session_state["split_type"] == "Entrenamiento y Prueba":
                X_test = st.session_state["splits"][1]  # Tomar X_test para pruebas
                y_test = st.session_state["splits"][3]  # Tomar y_test para pruebas
            elif st.session_state["split_type"] == "Entrenamiento, Validación y Prueba":
                X_test = st.session_state["splits"][2]  # Tomar X_test para pruebas      
                y_test = st.session_state["splits"][5]  # Tomar y_test para pruebas      
        else:
            st.error("No se encontró el conjunto de datos de prueba.")
            st.stop()

        # Crear sliders para cada característica (en su forma original)
        input_data = {}
        original_X_test = X_test.copy()  # Copia del conjunto de prueba original

        # Revertir escalamiento de X_test si se aplicó escalamiento
        if "scaler_x" in st.session_state and st.session_state["scaler_x"] is not None:
            scaler = st.session_state["scaler_x"]
              # Validar si las dimensiones son compatibles
            if scaler.scale_.shape[0] == X_test.shape[1]:
                original_X_test[:] = scaler.inverse_transform(X_test)
            else:
                st.error("Dimensiones incompatibles entre el escalador y el conjunto de prueba.")
                st.stop()

        # Configurar sliders basados en los valores originales
        for col in X_test.columns:
            min_val = original_X_test[col].min()
            max_val = original_X_test[col].max()
            default_val = original_X_test[col].iloc[0]

            input_data[col] = st.slider(
                f"Modificar {col}",
                float(min_val),
                float(max_val),
                float(default_val),
                step=(max_val - min_val) / 100  # Paso del slider
            )

        # Convertir los datos de entrada a un DataFrame
        input_df = pd.DataFrame([input_data])

        # Escalar los datos de entrada nuevamente para la predicción si se aplicó escalamiento
        if "scaler_x" in st.session_state and st.session_state["scaler_x"] is not None:
            input_df = st.session_state["scaler_x"].transform(input_df)

        # Generar predicción
        prediction = model.predict(input_df)

        # Mostrar resultados según el tipo de problema
        if st.session_state["problem_type"] == "Clasificación":
            # Si es clasificación, revertir las etiquetas codificadas si aplica
            if "label_encoder" in st.session_state:
                predicted_class = st.session_state["label_encoder"].inverse_transform([np.argmax(prediction, axis=1)[0]])[0]
            else:
                predicted_class = np.argmax(prediction, axis=1)[0]

            # Obtener el nombre de la columna objetivo
            target_column_name = st.session_state.get("target_variable", "Variable Objetivo")

            # Mostrar la predicción con el nombre de la columna objetivo
            st.subheader(f"Predicción de la variable '{target_column_name}'")
            st.write(f"Clase predicha: {predicted_class}")

            # Si el modelo permite probabilidades, mostrarlas
            if hasattr(model, "predict_proba"):
                st.write("Probabilidades:")
                for idx, prob in enumerate(prediction[0]):
                    st.write(f"Clase {idx}: {prob:.4f}")

        elif st.session_state["problem_type"] == "Regresión":
            # Revertir el escalamiento del valor predicho si aplica
            if "scaler_y" in st.session_state:
                prediction = st.session_state["scaler_y"].inverse_transform(prediction.reshape(-1, 1)).flatten()

            # Obtener el nombre de la columna objetivo
            target_column_name = st.session_state.get("target_variable", "Variable Objetivo")

            # Mostrar la predicción con el nombre de la columna objetivo
            st.subheader(f"Predicción de la variable '{target_column_name}'")
            st.write(f"Valor predicho: {prediction[0]:.4f}")

        if st.session_state["problem_type"] == "Clasificación":
            if "label_encoder" in st.session_state:
                y_test_original = st.session_state["label_encoder"].inverse_transform(y_test)
            else:
                y_test_original = y_test

            st.write("### Valores reales disponibles:")

            if isinstance(y_test_original, np.ndarray):  # Si es un array de NumPy
                unique_values = np.unique(y_test_original)  # Obtener valores únicos
            else:  # Si es un DataFrame o Serie de Pandas
                unique_values = y_test_original.unique()  # Obtener valores únicos

            # Mostrar valores únicos en una lista ordenada
            st.write(f"#### Valores únicos encontrados ({len(unique_values)}):")
            st.markdown("  \n".join([f"- **{val}**" for val in unique_values]))
        elif st.session_state["problem_type"] == "Regresión":
            y_test_original = y_test

            # Revertir el escalamiento de los valores reales si aplica
            if "scaler_y" in st.session_state:
                y_test_original = st.session_state["scaler_y"].inverse_transform(y_test_original.reshape(-1, 1)).flatten()

            st.write("Primeros 5 valores reales disponibles:")
            st.write(y_test_original[:5])



if st.session_state["training_finished"] and st.session_state["modelDownload"]:
    import json
    # Función para convertir objetos no serializables a JSON
    def serialize_object(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")  # Convertir DataFrame a diccionario
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convertir ndarray a lista
        elif isinstance(obj, list):
            return [serialize_object(item) for item in obj]  # Procesar listas recursivamente
        elif isinstance(obj, dict):
            return {key: serialize_object(value) for key, value in obj.items()}  # Procesar diccionarios recursivamente
        return obj  # Devolver el objeto original si es compatible

    # Botón para exportar configuración
    if st.button("Exportar Configuración del Modelo"):
        # Determinar tipo de división
        splits = st.session_state.get("splits", [])
        split_type = "Entrenamiento y Prueba" if len(splits) == 4 else "Entrenamiento, Validación y Prueba"

        # Recopilar métricas según el tipo de problema
        problem_type = st.session_state.get("problem_type", "Clasificación")
        metrics = {
            "Clasificación": {
                "loss_values": st.session_state.get("loss_values"),
                "accuracy_values": st.session_state.get("accuracy_values"),
                "val_loss_values": st.session_state.get("val_loss_values"),
                "val_accuracy_values": st.session_state.get("val_accuracy_values"),
                "confusion_matrix": st.session_state.get("confusion_matrix"),
                "f1_score": st.session_state.get("f1_score"),
                "precision": st.session_state.get("precision_score"),
                "recall": st.session_state.get("recall_score"),
            },
            "Regresión": {
                "loss_values": st.session_state.get("loss_values"),
                "mae_values": st.session_state.get("mae_values"),
                "val_loss_values": st.session_state.get("val_loss_values"),
                "val_mae_values": st.session_state.get("val_mae_values"),
                "mse": st.session_state.get("mse"),
                "rmse": st.session_state.get("rmse"),
            }
        }.get(problem_type, {})

        # Recopilar datos relevantes
        config_data = {
            "dataset": {
                "selected_dataset": st.session_state.get("selected_dataset"),
                "train_ratio": st.session_state.get("train_ratio"),
                "val_ratio": st.session_state.get("val_ratio"),
                "test_ratio": st.session_state.get("test_ratio"),
                "split_type": split_type,
                "target_variable": st.session_state.get("target_variable"),
                "columns_removed": st.session_state.get("columns_removed"),
                "missing_handled": st.session_state.get("missing_handled"),
                "categorical_encoded": st.session_state.get("categorical_encoded"),
                "scaling_done": st.session_state.get("scaling_done"),
            },
            "layer_config": st.session_state.get("layer_config"),
            "hyperparams": st.session_state.get("hyperparams"),
            "early_stopping": st.session_state.get("early_stopping"),
            "metrics": serialize_object(metrics),
        }

        # Serializar el objeto de configuración
        serialized_data = serialize_object(config_data)

        # Convertir a JSON y mostrar
        config_json = json.dumps(serialized_data, indent=4)
        st.json(json.loads(config_json))  # Mostrar configuración en la app

        serialized_data = serialize_object(config_data)
        st.session_state['serialized_results'] = serialized_data
        # Descargar archivo JSON
        st.download_button(
            label="Descargar Configuración como JSON",
            data=config_json,
            file_name="model_config.json",
            mime="application/json"
        )