import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import math
import numpy as np
import time

from threading import Thread
from matplotlib.animation import FuncAnimation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator



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
    st.session_state['selected_dataset'] = 'MNIST'
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
if 'shuffle_data' not in st.session_state:
    st.session_state['shuffle_data'] = True  # Valor predeterminado para aleatorizar datos




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


def load_dataset(name, train_ratio=0.8, val_ratio=0.1, shuffle=True, normalize=True):
    if name == 'MNIST':
        (x, y), (x_test, y_test) = mnist.load_data()
        if normalize:
            x = x.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
        x = x.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y = to_categorical(y, 10)
        y_test = to_categorical(y_test, 10)
        input_shape = (28, 28, 1)
        num_classes = 10
    elif name == 'Fashion MNIST':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()
        if normalize:
            x = x.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
        x = x.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y = to_categorical(y, 10)
        y_test = to_categorical(y_test, 10)
        input_shape = (28, 28, 1)
        num_classes = 10
    elif name == 'CIFAR-10':
        (x, y), (x_test, y_test) = cifar10.load_data()
        if normalize:
            x = x.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
        y = to_categorical(y, 10)
        y_test = to_categorical(y_test, 10)
        input_shape = (32, 32, 3)
        num_classes = 10
    else:
        raise ValueError("Dataset no soportado")

    # Dividir el conjunto de datos según las proporciones
    indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(len(x) * train_ratio)
    val_end = int(len(x) * (train_ratio + val_ratio))
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test, input_shape, num_classes



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

# Función para entrenar el modelo
def train_model(layers, hyperparams, preview_placeholder, dynamic_placeholder):
    st.session_state['logs'] = []
    preview_placeholder.empty()
    loss_chart_placeholder = st.empty()
    accuracy_chart_placeholder = st.empty()

    initialize_graph(layers)

    dataset_name = st.session_state['selected_dataset']
    
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape, num_classes = load_dataset(
        st.session_state['selected_dataset'],
        st.session_state['train_ratio'] / 100,
        st.session_state['val_ratio'] / 100,
        shuffle=st.session_state['shuffle_data'],
        normalize=st.session_state['normalize_data']
    )



    # Configuración de Data Augmentation (si está activado)
    if st.session_state['selected_dataset'] == 'CIFAR-10' and st.session_state.get('data_augmentation', False):
        datagen = ImageDataGenerator(
            rotation_range=st.session_state['rotation_range'],
            width_shift_range=st.session_state['width_shift_range'],
            height_shift_range=st.session_state['height_shift_range'],
            horizontal_flip=st.session_state['horizontal_flip']
        )
        datagen.fit(x_train)  # Ajustar el generador al conjunto de entrenamiento
    else:
        datagen = None  # No usar data augmentation si no está activado

    optimizers = {
        "Adam": Adam(learning_rate=hyperparams['learning_rate']),
        "SGD": SGD(learning_rate=hyperparams['learning_rate']),
        "RMSprop": RMSprop(learning_rate=hyperparams['learning_rate'])
    }
    optimizer = optimizers[hyperparams['optimizer']]

    model = Sequential()

    # Early Stopping Configuration
    callbacks = []
    if 'enable_early_stopping' in hyperparams and hyperparams['enable_early_stopping']:
        early_stopping = EarlyStopping(
            monitor=hyperparams['monitor_metric'],
            patience=hyperparams['patience'],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    if st.session_state['selected_dataset'] == 'CIFAR-10':
        # Capas base iniciales para CIFAR-10
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='Conv2D_1'))
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='Conv2D_2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1'))

        # Capas intermedias (configurables por el usuario)
        for layer in layers:
            if not layer.get('base', False):
                if layer['type'] == "Conv2D":
                    model.add(Conv2D(
                        layer['filters'], 
                        layer['kernel_size'], 
                        activation=layer['activation'], 
                        name=layer['name']
                    ))
                    if layer.get('batch_norm', False):
                        model.add(BatchNormalization(name=f"BatchNorm_{layer['name']}"))
                elif layer['type'] == "MaxPooling2D":
                    model.add(MaxPooling2D(pool_size=layer['pool_size'], name=layer['name']))
                elif layer['type'] == "Dropout":
                    model.add(Dropout(layer['rate'], name=layer['name']))
                elif layer['type'] == "Dense":
                    model.add(Dense(layer['neurons'], activation=layer['activation'], name=layer['name']))

        # Capas base finales para CIFAR-10
        model.add(Flatten(name='Flatten'))
        model.add(Dense(128, activation='relu', name='Dense'))
        model.add(Dropout(0.2, name='Dropout'))

    else:
        # Modelo para MNIST y Fashion MNIST
        model.add(Flatten(input_shape=input_shape, name='Flatten_Base'))
        for layer in layers:
            if layer['type'] == "Dense":
                model.add(Dense(layer['neurons'], activation=layer['activation'], name=layer['name']))
                if layer.get('batch_norm', False):
                    model.add(BatchNormalization(name=f"BatchNorm_{layer['name']}"))
            elif layer['type'] == "Dropout":
                model.add(Dropout(layer['dropout_rate'], name=layer['name']))

    model.add(Dense(num_classes, activation='softmax', name="Output"))

    model.compile(optimizer=optimizer, loss=hyperparams['loss_function'], metrics=['accuracy'])

    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']

    # Crear la disposición en una fila
    col_dynamic, col_metrics = st.columns([6, 1])  # Anchos personalizados: Log, Dinámico, Métricas


    log_placeholder = st.empty()

    # Placeholder para el gráfico dinámico
    with col_dynamic:
        dynamic_placeholder = st.empty()

    # Placeholders para métricas
    loss_chart_placeholder = col_metrics.empty()
    accuracy_chart_placeholder = col_metrics.empty()

    # Inicializar listas para almacenar las métricas
    loss_values = []
    accuracy_values = []

    for epoch in range(epochs):
        
        if not st.session_state['training_in_progress']:
            break



        epoch_start_time = time.time()
        log_event(log_placeholder, f"Época {epoch + 1}/{epochs} - Iniciando entrenamiento... ({dataset_name})")


        
        # Lanzar animación sincronizada por capas
          # Llamar a la animación sincronizada
        update_graph_with_smooth_color_transition(
            st.session_state['graph'], 
            epoch, 
            dynamic_placeholder, 
            neurons_per_point=12, 
            animation_steps=15
        )

        # Entrenamiento del modelo
        if datagen:
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
        else:
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=0
            )


        
        # Almacenar las métricas
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        loss_values.append(loss)
        accuracy_values.append(accuracy)

        # Actualizar gráficos de métricas
        if st.session_state['show_metrics']: 
            with col_metrics:
                # Gráfico de Pérdida
                fig_loss, ax_loss = plt.subplots(figsize=(4, 2))
                ax_loss.plot(range(1, len(loss_values) + 1), loss_values, label="Pérdida", marker='o', color='blue')
                ax_loss.set_title("Pérdida")
                ax_loss.set_xlabel("Épocas")
                ax_loss.set_ylabel("Pérdida")
                ax_loss.grid(True)
                loss_chart_placeholder.pyplot(fig_loss, clear_figure=True)

                # Gráfico de Precisión
                fig_accuracy, ax_accuracy = plt.subplots(figsize=(4, 2))
                ax_accuracy.plot(range(1, len(accuracy_values) + 1), accuracy_values, label="Precisión", marker='o', color='green')
                ax_accuracy.set_title("Precisión")
                ax_accuracy.set_xlabel("Épocas")
                ax_accuracy.set_ylabel("Precisión")
                ax_accuracy.grid(True)
                accuracy_chart_placeholder.pyplot(fig_accuracy, clear_figure=True)

        # Log de la época
        epoch_end_time = time.time()

        with log_placeholder:
            log_event(
                log_placeholder,
                f"Época {epoch + 1}/{epochs} - Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}, "
                f"Pérdida Validación: {history.history['val_loss'][0]:.4f}, Precisión Validación: {history.history['val_accuracy'][0]:.4f}\n"
                f"Tiempo por Época: {epoch_end_time - epoch_start_time:.2f}s"
            )

        # Early stopping check
        if 'enable_early_stopping' in hyperparams and hyperparams['enable_early_stopping']:
            if early_stopping.stopped_epoch > 0:
                log_event(log_placeholder, "Entrenamiento detenido por Early Stopping.")
                break 

    # Guardar el modelo después del entrenamiento
    st.session_state['modelDownload'] = model


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
    "MNIST": {
        "Simple MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True},
            {"name": "Dropout1", "type": "Dropout", "dropout_rate": 0.2, "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ],
        "Deep MLP": [
            {"name": "Dense1", "type": "Dense", "neurons": 128, "activation": "relu", "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 64, "activation": "relu", "base": True},
            {"name": "Dropout1", "type": "Dropout", "dropout_rate": 0.2, "base": True},
            {"name": "Dense3", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ]
    },
    "Fashion MNIST": {
        "Simple CNN": [
            {"name": "Conv2D1", "type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "MaxPooling2D1", "type": "MaxPooling2D", "pool_size": 2, "base": True},
            {"name": "Flatten1", "type": "Flatten", "base": True},
            {"name": "Dense1", "type": "Dense", "neurons": 64, "activation": "relu", "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ],
        "Deep CNN": [
            {"name": "Conv2D1", "type": "Conv2D", "filters": 64, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "Conv2D2", "type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "MaxPooling2D1", "type": "MaxPooling2D", "pool_size": 2, "base": True},
            {"name": "Flatten1", "type": "Flatten", "base": True},
            {"name": "Dense1", "type": "Dense", "neurons": 128, "activation": "relu", "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ]
    },
    "CIFAR-10": {
        "VGG-like": [
            {"name": "Conv2D1", "type": "Conv2D", "filters": 64, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "Conv2D2", "type": "Conv2D", "filters": 64, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "MaxPooling2D1", "type": "MaxPooling2D", "pool_size": 2, "base": True},
            {"name": "Conv2D3", "type": "Conv2D", "filters": 128, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "MaxPooling2D2", "type": "MaxPooling2D", "pool_size": 2, "base": True},
            {"name": "Flatten1", "type": "Flatten", "base": True},
            {"name": "Dense1", "type": "Dense", "neurons": 256, "activation": "relu", "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ],
        "Custom CNN": [
            {"name": "Conv2D1", "type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu", "base": True},
            {"name": "MaxPooling2D1", "type": "MaxPooling2D", "pool_size": 2, "base": True},
            {"name": "Flatten1", "type": "Flatten", "base": True},
            {"name": "Dense1", "type": "Dense", "neurons": 128, "activation": "relu", "base": True},
            {"name": "Dense2", "type": "Dense", "neurons": 10, "activation": "softmax", "base": True}
        ]
    }
}


# Función para manejar el cambio de dataset
if "selected_dataset_previous" not in st.session_state or st.session_state["selected_dataset_previous"] != st.session_state["selected_dataset"]:
    st.session_state["selected_dataset_previous"] = st.session_state["selected_dataset"]
    
    # Reinicia configuración de capas y arquitectura
    initialize_layer_config(st.session_state["selected_dataset"])




# Configuración con pestañas
tabs = st.sidebar.radio("Configuración:", ["Dataset","Capas", "Hiperparámetros", "Early Stopping"])


# Configuración del Dataset
if tabs == "Dataset":
    st.sidebar.subheader("Configuración del Dataset")
    # Mostrar select box para dataset
    st.sidebar.subheader("Dataset")
    st.session_state['selected_dataset'] = st.sidebar.selectbox(
        "Seleccione un Dataset para Entrenar",
        options=['MNIST', 'Fashion MNIST', 'CIFAR-10']
    )

    # Inicializar configuración de capas según el dataset seleccionado
    initialize_layer_config(st.session_state['selected_dataset'])

    # Proporción de datos: activación de configuración avanzada
    advanced_split = st.sidebar.checkbox(
        "Configurar proporciones específicas",
        value=False,
        help="Permite ajustar individualmente las proporciones de entrenamiento, validación y prueba."
    )

    if advanced_split:
        # Ajustes de proporciones: Entrenamiento, Validación y Prueba
        train_ratio = st.sidebar.slider(
            "Entrenamiento (%)",
            min_value=10,
            max_value=90,
            value=st.session_state.get('train_ratio', 80),
            step=1,
            help="Porcentaje de datos para entrenamiento. La suma debe ser 100%."
        )
        val_ratio = st.sidebar.slider(
            "Validación (%)",
            min_value=5,
            max_value=50,
            value=st.session_state.get('val_ratio', 10),
            step=1,
            help="Porcentaje de datos para validación. La suma debe ser 100%."
        )
        test_ratio = 100 - train_ratio - val_ratio
        st.sidebar.text(f"Prueba (%): {test_ratio}")

        if train_ratio + val_ratio > 100:
            st.warning("Las proporciones de entrenamiento y validación exceden el 100%. Ajusta los valores.")
            st.stop()
    else:
        train_ratio = 80
        val_ratio = 10
        test_ratio = 10

    # Guardar las proporciones en `st.session_state`
    st.session_state['train_ratio'] = train_ratio
    st.session_state['val_ratio'] = val_ratio
    st.session_state['test_ratio'] = test_ratio

    # Aleatorización de datos
    st.session_state['shuffle_data'] = st.sidebar.checkbox(
        "Aleatorizar Datos",
        value=st.session_state.get('shuffle_data', True),
        help="Determina si los datos deben ser aleatorizados antes de dividirlos."
    )

    # Normalización
    st.session_state['normalize_data'] = st.sidebar.checkbox(
        "Normalizar Datos",
        value=st.session_state.get('normalize_data', True),
        help="Si está activado, escala los datos entre 0 y 1."
    )

    # Data Augmentation (Solo para CIFAR-10)
    if st.session_state['selected_dataset'] == 'CIFAR-10':
        st.session_state['data_augmentation'] = st.sidebar.checkbox(
            "Aplicar Data Augmentation",
            value=st.session_state.get('data_augmentation', False),
            help="Realiza aumentos de datos como rotaciones o traslaciones para mejorar la robustez del modelo."
        )
        if st.session_state['data_augmentation']:
            st.session_state['rotation_range'] = st.sidebar.slider(
                "Rango de Rotación (°)",
                min_value=0,
                max_value=45,
                value=st.session_state.get('rotation_range', 10),
                step=1,
                help="Rango de rotaciones aleatorias aplicadas a las imágenes."
            )
            st.session_state['width_shift_range'] = st.sidebar.slider(
                "Desplazamiento Horizontal (%)",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.get('width_shift_range', 0.1),
                step=0.01,
                help="Desplazamiento horizontal máximo como porcentaje del ancho de la imagen."
            )
            st.session_state['height_shift_range'] = st.sidebar.slider(
                "Desplazamiento Vertical (%)",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.get('height_shift_range', 0.1),
                step=0.01,
                help="Desplazamiento vertical máximo como porcentaje de la altura de la imagen."
            )
            st.session_state['horizontal_flip'] = st.sidebar.checkbox(
                "Volteo Horizontal",
                value=st.session_state.get('horizontal_flip', True),
                help="Permite voltear horizontalmente las imágenes de forma aleatoria."
            )

    # Tamaño de batch
    st.session_state['batch_size'] = st.sidebar.number_input(
        "Tamaño de Batch",
        min_value=8,
        max_value=256,
        value=st.session_state.get('batch_size', 32),
        step=8,
        help="Tamaño de los lotes que se utilizarán durante el entrenamiento."
    )

    # Mostrar ejemplos del dataset
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape, num_classes = load_dataset(
        st.session_state['selected_dataset'],
        st.session_state['train_ratio'] / 100,
        st.session_state['val_ratio'] / 100,
        shuffle=st.session_state['shuffle_data'],
        normalize=st.session_state['normalize_data']
    )
    st.sidebar.subheader("Ejemplo de Dataset")
    for i in range(5):
        st.sidebar.image(
            x_train[i].reshape(input_shape[:2]) if st.session_state['selected_dataset'] != 'CIFAR-10' else x_train[i],
            caption=f"Etiqueta: {y_train[i].argmax()}",
            width=100
        )



# Visualizar ejemplos del dataset
elif tabs == "Capas":
    st.sidebar.subheader("Configuración de Capas")
    # Mostrar select box para arquitectura

    
    st.sidebar.subheader("Arquitectura")
    selected_architecture = st.sidebar.selectbox(
        "Seleccione una Arquitectura",
        options=list(architectures[st.session_state['selected_dataset']].keys()),
        help="Elija la arquitectura del modelo basada en el dataset seleccionado."
    )

    # Actualizar configuración de capas según la arquitectura seleccionada
    if st.sidebar.button("Aplicar Arquitectura"):
        # Obtener las capas de la arquitectura seleccionada
        selected_architecture_layers = architectures[st.session_state['selected_dataset']][selected_architecture]

        # Reiniciar las configuraciones de capas
        st.session_state['layer_config'] = selected_architecture_layers[:]
        st.session_state['num_layers_selected'] = len(selected_architecture_layers)

        # Validar tipos de capas según el dataset
        if st.session_state['selected_dataset'] in ['MNIST']:
            valid_types = ["Dense", "Dropout"]
        else:
            valid_types = ["Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout", "BatchNormalization"]

        # Asegurarse de que las capas sean válidas
        for layer in st.session_state['layer_config']:
            if layer['type'] not in valid_types:
                st.warning(f"Tipo de capa no válido detectado: {layer['type']}.")
                st.session_state['layer_config'] = []  # Limpia configuraciones inválidas
                break

        st.sidebar.success(f"Arquitectura '{selected_architecture}' aplicada correctamente.")

    # Validar si las configuraciones cambian en otros lugares
    if st.session_state['layer_config'] != st.session_state.get('previous_layer_config', []):
        st.session_state['previous_layer_config'] = st.session_state['layer_config'][:]

    if st.session_state['selected_dataset'] == 'CIFAR-10':
        # Interfaz de usuario para número de capas intermedias
        st.sidebar.subheader("Capas Intermedias")
        st.sidebar.number_input(
            "Número de Capas Intermedias",
            min_value=0,
            max_value=15,
            value=st.session_state['num_intermediate_layers'],
            step=1,
            key='num_intermediate_layers',
            help="Define cuántas capas intermedias desea incluir entre las capas iniciales y finales. Ajuste según las necesidades de su modelo.",
            on_change=update_intermediate_layers
        )

        # Mostrar configuración de las capas
        for i, layer in enumerate(st.session_state['layer_config']):
            if layer.get('base', False):
                st.sidebar.text_input(
                    f"{layer['name']} (Base, No editable)",
                    value=layer['type'],
                    disabled=True,
                    key=f"text_input_{i}",
                    help="Esta es una capa base predefinida y no puede ser modificada."
                )
            else:
                with st.sidebar.expander(f"Configuración de la Capa {i + 1}", expanded=True):
                    layer['type'] = st.selectbox(
                        "Tipo de Capa",
                        ["Conv2D", "MaxPooling2D", "Dense", "Dropout", "BatchNormalization"],
                        key=f"layer_type_{i}",
                        help="Seleccione el tipo de capa que desea añadir al modelo."
                    )
                    if layer['type'] == "Conv2D":
                        layer['filters'] = st.number_input(
                            "Número de Filtros",
                            min_value=1,
                            value=layer.get('filters', 32),
                            key=f"filters_{i}",
                            help="Define cuántos filtros utilizará esta capa de convolución."
                        )
                        layer['kernel_size'] = st.slider(
                            "Tamaño del Kernel",
                            1,
                            5,
                            value=3,
                            step=1,
                            key=f"kernel_{i}",
                            help="Tamaño del filtro o kernel que se moverá sobre la imagen en la capa de convolución."
                        )
                        layer['activation'] = st.selectbox(
                            "Activación",
                            ["relu", "sigmoid", "tanh","softmax"],
                            index=0,
                            key=f"activation_{i}",
                            help="Función de activación que transforma la salida de la capa."
                        )
                        layer['batch_normalization'] = st.checkbox(
                            "Agregar Batch Normalization",
                            value=layer.get('batch_normalization', False),
                            key=f"batch_norm_{i}",
                            help="Si está activado, agrega una normalización de lotes después de la capa."
                        )
                    elif layer['type'] == "MaxPooling2D":
                        layer['pool_size'] = st.slider(
                            "Tamaño del Pool",
                            1,
                            3,
                            value=2,
                            step=1,
                            key=f"pool_{i}",
                            help="Tamaño de la ventana para realizar la operación de pooling."
                        )
                    elif layer['type'] == "Dense":
                        layer['neurons'] = st.number_input(
                            "Número de Neuronas",
                            min_value=1,
                            value=layer.get('neurons', 64),
                            key=f"neurons_{i}",
                            help="Número de neuronas en la capa densa."
                        )
                        layer['activation'] = st.selectbox(
                            "Activación",
                            ["relu", "sigmoid", "tanh","softmax"],
                            index=0,
                            key=f"activation_dense_{i}",
                            help="Función de activación para la capa densa."
                        )
                        layer['dropout_rate'] = st.slider(
                            "Tasa de Dropout",
                            0.0,
                            0.9,
                            value=layer.get('dropout_rate', 0.2),
                            step=0.1,
                            key=f"dropout_dense_{i}",
                            help="Proporción de unidades que se desactivarán aleatoriamente en la capa."
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
                    elif layer['type'] == "BatchNormalization":
                        layer['batch_normalization'] = True
                    layer['name'] = f"{layer['type']}{i + 1}"


    # Configuración para otros datasets
    else: 
        # UI para número de capas
        st.sidebar.subheader("Capas")
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

        # Validar si la configuración de capas cambió externamente
        if st.session_state['layer_config'] != st.session_state['previous_layer_config']:
            st.session_state['layer_config'] = st.session_state['previous_layer_config'][:]

        if st.session_state['selected_dataset'] == 'MNIST':
            valid_types = ["Dense", "Dropout", "Flatten", "BatchNormalization"]
        else:
            valid_types = ["Dense", "Dropout", "Conv2D", "MaxPooling2D", "Flatten", "BatchNormalization"]

        # Validar si la configuración actual de capas es válida para el dataset seleccionado
        invalid_layers = any(layer.get("type") not in valid_types for layer in st.session_state['layer_config'])

        
        if invalid_layers:
        # Reiniciar las capas si hay configuraciones inválidas
            st.session_state['layer_config'] = []
            st.warning("La configuración previa de capas no es válida para el dataset seleccionado. Las capas se han reiniciado.")

        # Mostrar configuración de cada capa
        for i, layer in enumerate(st.session_state['layer_config']):
            with st.sidebar.expander(f"Configuración de la Capa {i + 1}", expanded=True):
                layer['type'] = st.selectbox(
                    "Tipo de Capa",
                    valid_types,
                    index=valid_types.index(layer.get("type", "Dense")),
                    key=f"layer_type_{i}",
                    help="Seleccione el tipo de capa que desea añadir."
                )
                # Configuración específica de cada tipo de capa
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
                        ["relu", "sigmoid", "tanh", "softmax"],
                        index=["relu", "sigmoid", "tanh", "softmax"].index(layer.get('activation', "relu")),
                        key=f"activation_{i}",
                        help="Función de activación para la capa densa."
                    )
                elif layer['type'] == "Dropout":
                    layer['dropout_rate'] = st.slider(
                        "Tasa de Dropout",
                        0.0,
                        0.9,
                        value=layer.get('dropout_rate', 0.2),
                        step=0.1,
                        key=f"dropout_{i}",
                        help="Proporción de unidades que se desactivarán aleatoriamente."
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
                        help="Tamaño del filtro o kernel que se moverá sobre la imagen."
                    )
                    layer['activation'] = st.selectbox(
                        "Activación",
                        ["relu", "sigmoid", "tanh", "softmax"],
                        index=["relu", "sigmoid", "tanh", "softmax"].index(layer.get('activation', "relu")),
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
                elif layer['type'] == "GlobalAveragePooling2D":
                    st.info("Reduce cada canal de características a un único valor promedio.")

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
    train_model(st.session_state['layer_config'], st.session_state['hyperparams'], preview_placeholder, dynamic_placeholder)
    st.session_state['training_in_progress'] = False
    st.session_state['training_finished'] = True
    st.rerun()


# Mostrar el botón de guardar modelo una vez terminado el entrenamiento
if st.session_state['training_finished']:
    st.success("Entrenamiento finalizado con éxito.")
    st.text_area(
        "Logs del Entrenamiento",
        "\n".join(st.session_state['logs']),
        height=300,
        key="final_logs"
    )    
    if st.button("Guardar Modelo"):
        st.session_state['modelDownload'].save("trained_model.h5")
        with open("trained_model.h5", "rb") as file:
            st.download_button("Descargar Modelo", file, file_name="trained_model.h5")
    
    # Mostrar el preview del gráfico después de finalizar
    preview_graph(st.session_state['layer_config'], preview_placeholder)
    
    # Botón para reiniciar el entrenamiento
    if st.button("Comenzar Entrenamiento"):
        reset_training_state()
        st.rerun()
