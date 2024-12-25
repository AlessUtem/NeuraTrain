import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image as Image

# Config
page_icon = Image.open("./assets/icon.png")
st.set_page_config(layout="centered", page_title="Click ML", page_icon=page_icon)

if 'df' not in st.session_state:
    st.session_state.df = None

if "delete_features" not in st.session_state:
    st.session_state.delete_features = None

if "missing_done" not in st.session_state:
    st.session_state.missing_done = False

if "cat_enc_done" not in st.session_state:
    st.session_state.cat_enc_done = False

if "num_scale_done" not in st.session_state:
    st.session_state.num_scale_done = False

if "split_done" not in st.session_state:
    st.session_state.split_done = False

if "X_train" not in st.session_state:
    st.session_state.X_train = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_train" not in st.session_state:
    st.session_state.y_train = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "X_val" not in st.session_state:
    st.session_state.X_val = None

if "y_val" not in st.session_state:
    st.session_state.y_val = None

if "split_type" not in st.session_state:
    st.session_state.split_type = None

if "build_model_done" not in st.session_state:
    st.session_state.build_model_done = False

if "no_svm" not in st.session_state:
    st.session_state.no_svm = False

def new_line():
    st.write("\n")

with st.sidebar:
    st.image("./assets/sb-quick.png",  use_container_width=True)


st.markdown("<h1 style='text-align: center; '>🚀 QuickML</h1>", unsafe_allow_html=True)
st.markdown("QuickML is a tool that helps you to build a Machine Learning model in just a few clicks.", unsafe_allow_html=True)
st.divider()


st.header("Sube tu archivo CSV", anchor=False)
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
st.divider()

if uploaded_file:

    if st.session_state.df is None:
        df = pd.read_csv(uploaded_file)
    else:
        df = st.session_state.df 
        # st.dataframe(df)
        new_line()



    














    # Borrar caractersiticas del dataset
    st.subheader("🗑️ Elimina columnas", anchor=False)
    new_line()
    if not st.session_state.delete_features:
        delete_features = st.multiselect("Selecciona las columnas que quieres eliminar", df.columns.tolist())
        no_delete_features = st.button("No eliminar", key="reset_delete")
        new_line()
        if delete_features:
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Eliminar", key="delete"):
                st.session_state.delete_features = True
                st.session_state.df = df.drop(delete_features, axis=1)
                st.success("Columnas eliminadas satisfactoriamente. Ahora puedes continuar manejando los valores faltantes.")
        if no_delete_features:
            st.session_state.delete_features = True
            st.success("No se eliminaron columnas. Ahora puedes continuar manejando los valores faltantes.")

        
    
    # Mostrar el dataset actualizado
    if st.session_state.delete_features:
        st.markdown("### 📊 Dataset Actualizado")
        if "df" in st.session_state and st.session_state.df is not None:
            st.dataframe(st.session_state.df, use_container_width=True)
            st.markdown(f"**Filas:** {st.session_state.df.shape[0]}  |  **Columnas:** {st.session_state.df.shape[1]}")
        else:
            st.warning("El dataset aún no está disponible. Asegúrate de haber cargado y modificado un archivo.")

    if st.session_state.delete_features:

        # Missing Values
        st.subheader("⚠️ Valores faltantes", anchor=False)

        # Verificar si hay valores faltantes
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            new_line()
            st.warning(f"Hay {total_missing} valores faltantes en el dataset. Por favor, maneja estos valores antes de continuar.")
            new_line()

            # Métodos generales
            col1, col2 = st.columns(2)

            col1.markdown("<h6 style='text-align: center;'>Manejo general de características numéricas</h6>", unsafe_allow_html=True)
            general_num_method = col1.selectbox(
                "Selecciona el método general para manejar valores faltantes en características numéricas",
                ["Media", "Mediana", "Moda", "Rellenar Adelante/Atrás", "Eliminar Filas", "Reemplazar con 0"],
                key="general_num_method"
            )

            col2.markdown("<h6 style='text-align: center;'>Manejo general de características categóricas</h6>", unsafe_allow_html=True)
            general_cat_method = col2.selectbox(
                "Selecciona el método general para manejar valores faltantes en características categóricas",
                ["Moda", "Eliminar Filas", "Reemplazar con Ninguno"],
                key="general_cat_method"
            )

            # Checkbox para habilitar imputación específica por columna
            use_specific_imputation = st.checkbox("Habilitar imputación específica por columna", value=False)

            # Configuración para imputación específica
            specific_imputations = []
            if use_specific_imputation:
                st.markdown("### Imputación específica por columna")
                
                # Inicializar configuraciones si no existen
                if "specific_imputation_config" not in st.session_state:
                    st.session_state.specific_imputation_config = [{"col": None, "method": None}]

                # Mostrar configuraciones existentes
                for i, config in enumerate(st.session_state.specific_imputation_config):
                    col1, col2, col3 = st.columns([3, 3, 1])

                    # Selección de columna
                    config["col"] = col1.selectbox(
                        "Columna",
                        [None] + df.columns[df.isnull().any()].tolist(),
                        index=(df.columns[df.isnull().any()].tolist().index(config["col"]) + 1) if config["col"] else 0,
                        key=f"col_{i}",
                    )

                    # Selección de método
                    if config["col"]:
                        col_type = "Numérica" if config["col"] in df.select_dtypes(include=np.number).columns else "Categórica"
                        config["method"] = col2.selectbox(
                            "Método de imputación",
                            ["Media", "Mediana", "Moda", "Rellenar Adelante/Atrás", "Reemplazar con 0"]
                            if col_type == "Numérica"
                            else ["Moda", "Reemplazar con Ninguno", "Eliminar Filas"],
                            key=f"method_{i}",
                        )

                    # Botón para eliminar configuración
                    if col3.button("❌", key=f"remove_{i}"):
                        st.session_state.specific_imputation_config.pop(i)

                # Botón para añadir nueva configuración
                if st.button("+ Añadir columna específica"):
                    st.session_state.specific_imputation_config.append({"col": None, "method": None})

            # Botón para aplicar imputaciones
            if st.button("Aplicar imputaciones"):
                # Aplicar imputaciones específicas
                if use_specific_imputation:
                    for config in st.session_state.specific_imputation_config:
                        if config["col"] and config["method"]:
                            col = config["col"]
                            method = config["method"]
                            col_type = (
                                "Numérica" if col in df.select_dtypes(include=np.number).columns else "Categórica"
                            )

                            # Aplicar imputación específica
                            if col_type == "Numérica":
                                if method == "Media":
                                    df[col] = df[col].fillna(df[col].mean())
                                elif method == "Mediana":
                                    df[col] = df[col].fillna(df[col].median())
                                elif method == "Moda":
                                    df[col] = df[col].fillna(df[col].mode()[0])
                                elif method == "Rellenar Adelante/Atrás":
                                    df[col] = df[col].ffill().bfill()
                                elif method == "Reemplazar con 0":
                                    df[col] = df[col].fillna(0)
                                elif method == "Eliminar Filas":
                                    df[col] = df.dropna(subset=[col]).reset_index(drop=True)

                            elif col_type == "Categórica":
                                if method == "Moda":
                                    df[col] = df[col].fillna(df[col].mode()[0])
                                elif method == "Reemplazar con Ninguno":
                                    df[col] = df[col].fillna("Ninguno")
                                elif method == "Eliminar Filas":
                                    df = df.dropna(subset=[col]).reset_index(drop=True)

                            # Evitar que el método general se aplique a esta columna
                            specific_imputations.append(col)

                # Aplicar métodos generales al resto de las columnas
                remaining_num_cols = [
                    col for col in df.select_dtypes(include=np.number).columns if col not in specific_imputations
                ]
                remaining_cat_cols = [
                    col for col in df.select_dtypes(include=object).columns if col not in specific_imputations
                ]

                # Imputación general para características numéricas
                if remaining_num_cols:
                    if general_num_method == "Media":
                        df[remaining_num_cols] = df[remaining_num_cols].fillna(df[remaining_num_cols].mean())
                    elif general_num_method == "Mediana":
                        df[remaining_num_cols] = df[remaining_num_cols].fillna(df[remaining_num_cols].median())
                    elif general_num_method == "Moda":
                        for col in remaining_num_cols:
                            df[col] = df[col].fillna(df[col].mode()[0])
                    elif general_num_method == "Rellenar Adelante/Atrás":
                        df[remaining_num_cols] = df[remaining_num_cols].ffill().bfill()
                    elif general_num_method == "Eliminar Filas":
                        df = df.dropna(subset=remaining_num_cols).reset_index(drop=True)
                    elif general_num_method == "Reemplazar con 0":
                        df[remaining_num_cols] = df[remaining_num_cols].fillna(0)

                # Imputación general para características categóricas
                if remaining_cat_cols:
                    if general_cat_method == "Moda":
                        for col in remaining_cat_cols:
                            df[col] = df[col].fillna(df[col].mode()[0])
                    elif general_cat_method == "Eliminar Filas":
                        df = df.dropna(subset=remaining_cat_cols).reset_index(drop=True)
                    elif general_cat_method == "Reemplazar con Ninguno":
                        df[remaining_cat_cols] = df[remaining_cat_cols].fillna("Ninguno")

                # Actualizar el DataFrame en la sesión
                st.session_state.df = df
                st.success("Imputaciones aplicadas correctamente.")
                st.session_state.missing_done = True

        else:
            st.success("No se encontraron valores faltantes en el dataset.")





    # Informe posterior a la imputación
    if st.session_state.missing_done:
        st.markdown("### Resumen de la Imputación")
        imputed_columns = df.columns[df.isnull().sum() == 0].tolist()
        st.write(f"Columnas imputadas correctamente: {imputed_columns}")
        st.dataframe(df.head(), use_container_width=True)




    # Codificación de características categóricas
    if st.session_state.missing_done:
        new_line()
        st.subheader("☢️ Codificación de Características Categóricas", anchor=False)
        new_line()

        if len(df.select_dtypes(include=object).columns.tolist()) > 0:
            new_line()

            st.markdown("<h6 style='text-align: center;'>Selecciona el método para codificar características categóricas</h6>", unsafe_allow_html=True)
            new_line()
            cat_enc_meth = st.selectbox(
                "Selecciona el método para codificar características categóricas",
                ["Codificación Ordinal", "Codificación One Hot", "Codificación por Frecuencia"]
            )
            new_line()

            if cat_enc_meth:
                col1, col2, col3 = st.columns([1, 0.5, 1])
                if col2.button("Aplicar", key="cat_enc"):
                    cat_cols = df.select_dtypes(include=object).columns.tolist()

                    if cat_enc_meth == "Codificación Ordinal":
                        from sklearn.preprocessing import OrdinalEncoder
                        oe = OrdinalEncoder()
                        df[cat_cols] = oe.fit_transform(df[cat_cols])
                        st.session_state.df = df

                    elif cat_enc_meth == "Codificación One Hot":
                        df = pd.get_dummies(df, columns=cat_cols)
                        st.session_state.df = df

                    elif cat_enc_meth == "Codificación por Frecuencia":
                        for col in cat_cols:
                            df[col] = df[col].map(df[col].value_counts() / len(df))
                        st.session_state.df = df
                        
                    st.session_state.cat_enc_done = True
                    st.success("Características categóricas codificadas exitosamente. Ahora puedes proceder a Escalamiento y Transformación.")

        else:
            st.session_state.cat_enc_done = True
            st.success("No se encontraron características categóricas en el dataset.")

    # Escalamiento y Transformación de Características Numéricas
    if st.session_state.cat_enc_done and st.session_state.missing_done:
        new_line()
        st.subheader("🧬 Escalamiento y Transformación", anchor=False)
        new_line()

        if not st.session_state.num_scale_done:
            if len(df.select_dtypes(include=np.number).columns.tolist()) > 0:
                new_line()

                st.markdown("<h6 style='text-align: left;'>Selecciona el método para escalar y transformar características numéricas</h6>", unsafe_allow_html=True)
                new_line()
                col1, col2 = st.columns(2)
                not_scale = col1.multiselect(
                    "Selecciona las características que **no** quieres escalar ni transformar **__Incluye la variable objetivo si es un problema de Clasificación__**",
                    df.select_dtypes(include=np.number).columns.tolist()
                )
                num_scale_meth = col2.selectbox(
                    "Selecciona el método para escalar y transformar características numéricas",
                    ["Estandarización", "Escalador MinMax", "Escalador Robusto", "Transformación Logarítmica", "Transformación Raíz Cuadrada"]
                )
                new_line()

                if num_scale_meth:
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    if col2.button("Aplicar", key="num_scale"):
                        st.session_state.num_scale_done = True
                        if not_scale:
                            num_cols = df.select_dtypes(include=np.number).columns.tolist()
                            # Elimina las características que no se quieren escalar
                            for not_scale_feat in not_scale:
                                num_cols.remove(not_scale_feat)
                        else:
                            num_cols = df.select_dtypes(include=np.number).columns.tolist()

                        if num_scale_meth == "Estandarización":
                            from sklearn.preprocessing import StandardScaler
                            ss = StandardScaler()
                            df[num_cols] = ss.fit_transform(df[num_cols])
                            st.session_state.df = df

                        elif num_scale_meth == "Escalador MinMax":
                            from sklearn.preprocessing import MinMaxScaler
                            mms = MinMaxScaler()
                            df[num_cols] = mms.fit_transform(df[num_cols])
                            st.session_state.df = df

                        elif num_scale_meth == "Escalador Robusto":
                            from sklearn.preprocessing import RobustScaler
                            rs = RobustScaler()
                            df[num_cols] = rs.fit_transform(df[num_cols])
                            st.session_state.df = df

                        elif num_scale_meth == "Transformación Logarítmica":
                            df[num_cols] = np.log(df[num_cols])
                            st.session_state.df = df

                        elif num_scale_meth == "Transformación Raíz Cuadrada":
                            df[num_cols] = np.sqrt(df[num_cols])
                            st.session_state.df = df

                        st.success("Las características numéricas se han escalado y transformado exitosamente. Ahora puedes proceder a dividir el dataset.")
            else:
                st.warning("No se encontraron características numéricas en el dataset. Por favor verifica el dataset nuevamente.")
        else:
            st.session_state.num_scale_done = True
            st.success("Las características numéricas se han escalado y transformado exitosamente. Ahora puedes proceder a dividir el dataset.")

    

   # División del dataset
    if st.session_state.cat_enc_done and st.session_state.missing_done:
        new_line()
        st.subheader("✂️ División del dataset", anchor=False)
        new_line()

        if not st.session_state.split_done:
            new_line()

            col1, col2 = st.columns(2)
            objetivo = col1.selectbox("Selecciona la variable objetivo", df.columns.tolist())
            conjuntos = col2.selectbox("Selecciona el tipo de división", ["Entrenamiento y Prueba", "Entrenamiento, Validación y Prueba"])
            st.session_state.split_type = conjuntos
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Aplicar", key="split"):
                st.session_state.split_done = True

                if conjuntos == "Entrenamiento y Prueba":
                    from sklearn.model_selection import train_test_split
                    X = df.drop(objetivo, axis=1)
                    y = df[objetivo]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.success("Dataset dividido exitosamente. Ahora puedes proceder a construir el modelo.")

                elif conjuntos == "Entrenamiento, Validación y Prueba":
                    from sklearn.model_selection import train_test_split
                    X = df.drop(objetivo, axis=1)
                    y = df[objetivo]
                    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)
                    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.X_val = X_val
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_val = y_val
                    st.success("Dataset dividido exitosamente. Ahora puedes proceder a construir el modelo.")

        else:
            if len(str(st.session_state.split_type).split()) == 4:
                st.success("El dataset se dividió exitosamente en Entrenamiento, Validación y Prueba. Ahora puedes proceder a construir el modelo.")

            elif len(st.session_state.split_type.split()) == 3:
                st.success("El dataset se dividió exitosamente en Entrenamiento y Prueba. Ahora puedes proceder a construir el modelo.")


    # Construcción del modelo
    if st.session_state.split_done:
        new_line()
        st.subheader("🧠 Construcción del modelo", anchor=False)
        target, problem_type, model = None, None, None
        new_line()

        col1, col2, col3 = st.columns(3)
        target = col1.selectbox("Selecciona la variable objetivo", df.columns.tolist(), key="target_model")
        problem_type = col2.selectbox("Selecciona el tipo de problema", ["Clasificación", "Regresión"])
        if problem_type == "Clasificación":
            model = col3.selectbox("Selecciona el modelo", [
                "Logistic Regression", "K Nearest Neighbors", "Support Vector Machine",
                "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"
            ])
        elif problem_type == "Regresión":
            model = col3.selectbox("Selecciona el modelo", [
                "Linear Regression", "K Nearest Neighbors", "Support Vector Machine",
                "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"
            ])

        new_line()
        if target and problem_type and model:
            col1, col2, col3 = st.columns([1, 0.8, 1])
            if col2.button("Aplicar", key="build_model", use_container_width=True):
                st.session_state.build_model_done = True
                if problem_type == "Clasificación":
                    if model == "Logistic Regression":
                        from sklearn.linear_model import LogisticRegression
                        import pickle
                        lr = LogisticRegression()
                        lr.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(lr, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key='class_log_reg', use_container_width=True)

                    elif model == "K Nearest Neighbors":
                        from sklearn.neighbors import KNeighborsClassifier
                        import pickle
                        knn = KNeighborsClassifier()
                        knn.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(knn, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key='class_knn', use_container_width=True)

                    elif model == "Support Vector Machine":
                        from sklearn.svm import SVC
                        import pickle
                        svm = SVC()
                        svm.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(svm, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key='class_svm', use_container_width=True)


                elif problem_type == "Regresión":
                    if model == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        import pickle
                        lr = LinearRegression()
                        lr.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(lr, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key="reg_lin_reg", use_container_width=True)

                    elif model == "K Nearest Neighbors":
                        from sklearn.neighbors import KNeighborsRegressor
                        import pickle
                        knn = KNeighborsRegressor()
                        knn.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(knn, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key="reg_knn", use_container_width=True)

                    elif model == "Support Vector Machine":
                        from sklearn.svm import SVR
                        import pickle
                        svr = SVR()
                        svr.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(svr, open('model.pkl', 'wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key="reg_svm", use_container_width=True)

                    elif model == "Decision Tree":
                        from sklearn.tree import DecisionTreeRegressor
                        import pickle

                        dt = DecisionTreeRegressor()
                        dt.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(dt, open('model.pkl','wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, "model.pkl", key="reg_dt", use_container_width=True)

                    elif model == "Random Forest":
                        from sklearn.ensemble import RandomForestRegressor
                        import pickle

                        rf = RandomForestRegressor()
                        rf.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(rf, open('model.pkl','wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open("model.pkl", "rb")
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, 'model.pkl', key="reg_rf", use_container_width=True)



                    elif model == "XGBoost":
                        from xgboost import XGBRegressor
                        import pickle

                        xgb = XGBRegressor()
                        xgb.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(xgb, open('model.pkl','wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open('model.pkl', 'rb')
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, 'model.pkl', key="reg_xgb", use_container_width=True)


                    elif model == "LightGBM":
                        from lightgbm import LGBMRegressor
                        import pickle

                        lgbm = LGBMRegressor()
                        lgbm.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(lgbm, open('model.pkl','wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open('model.pkl', 'rb')
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, 'model.pkl', key="reg_lgbm", use_container_width=True)


                    elif model == "CatBoost":
                        from catboost import CatBoostRegressor
                        import pickle

                        cb = CatBoostRegressor()
                        cb.fit(st.session_state.X_train, st.session_state.y_train)

                        pickle.dump(cb, open('model.pkl','wb'))
                        st.success("Modelo construido exitosamente. Ahora puedes proceder a la evaluación.")

                        model_file = open('model.pkl', 'rb')
                        model_bytes = model_file.read()
                        col2.download_button("Descargar Modelo", model_bytes, 'model.pkl', key="reg_cb", use_container_width=True)

    # # Evaluación
    if st.session_state.build_model_done:
            new_line()
            st.subheader("Evaluación", anchor=False)
            new_line()
            if st.session_state.split_type == "Entrenamiento y Prueba":
                
                if problem_type == "Clasificación":
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
                    import pickle

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)
                    if not st.session_state.no_svm:
                        y_prob = model.predict_proba(st.session_state.X_test)[:,1]


                    # DataFrame para almacenar los valores de las métricas para cada conjunto
                    metrics_df = pd.DataFrame(columns=["Exactitud", "Precisión", "Recall", "F1", "ROC AUC"], index=["Entrenamiento", "Prueba"])
                    metrics_df.loc["Entrenamiento", "Exactitud"] = accuracy_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Precisión"] = precision_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Recall"] = recall_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "F1"] = f1_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    if not st.session_state.no_svm:
                        metrics_df.loc["Entrenamiento", "ROC AUC"] = roc_auc_score(st.session_state.y_train, model.predict_proba(st.session_state.X_train)[:, 1])
                    metrics_df.loc["Prueba", "Exactitud"] = accuracy_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Precisión"] = precision_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Recall"] = recall_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "F1"] = f1_score(st.session_state.y_test, y_pred)
                    if not st.session_state.no_svm:
                        metrics_df.loc["Prueba", "ROC AUC"] = roc_auc_score(st.session_state.y_test, y_prob)

                    new_line()

                    # Gráfico de métricas usando plotly
                    st.markdown("#### Gráfico de Métricas")
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(name='Train', x=metrics_df.columns.tolist(), y=metrics_df.loc["Train", :].values.tolist()),
                        go.Bar(name='Test', x=metrics_df.columns.tolist(), y=metrics_df.loc["Test", :].values.tolist())
                    ])
                    st.plotly_chart(fig)


                    # Curva ROC usando px
                    import plotly.express as px
                    from sklearn.metrics import roc_curve

                    fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_prob)
                    fig = px.area(
                        x=fpr, y=tpr,
                        title=f'Curva ROC (AUC={metrics_df.loc["Prueba", "ROC AUC"]:.4f})',
                        labels=dict(x='Tasa de Falsos Positivos', y='Tasa de Verdaderos Positivos'),
                        width=400, height=500
                    )
                    fig.add_shape(
                        type='line', line=dict(dash='dash'),
                        x0=0, x1=1, y0=0, y1=1
                    )

                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    fig.update_xaxes(constrain='domain')
                    st.plotly_chart(fig)

                    # Mostrar los valores de las métricas
                    new_line()
                    st.markdown("##### Valores de las Métricas")
                    st.write(metrics_df)

                    # Matriz de confusión
                    # from sklearn.metrics import plot_confusion_matrix
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                    st.markdown("#### Matriz de Confusión")
                    new_line()

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)

                    # cm = confusion_matrix(y_test, y_pred_test)
                    fig, ax = plt.subplots(figsize=(6,6))
                    ConfusionMatrixDisplay.from_predictions(st.session_state.y_test, y_pred, ax=ax)
                    st.pyplot(fig)
                    


                elif problem_type == "Regresión":
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    import pickle

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)

                    # DataFrame para almacenar los valores de las métricas para cada conjunto con RMSE
                    metrics_df = pd.DataFrame(columns=["Error Cuadrático Medio", "Error Absoluto Medio", "Puntaje R2", "RMSE"], index=["Entrenamiento", "Prueba"])
                    metrics_df.loc["Entrenamiento", "Error Cuadrático Medio"] = mean_squared_error(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Error Absoluto Medio"] = mean_absolute_error(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Puntaje R2"] = r2_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "RMSE"] = np.sqrt(metrics_df.loc["Entrenamiento", "Error Cuadrático Medio"])
                    metrics_df.loc["Prueba", "Error Cuadrático Medio"] = mean_squared_error(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Error Absoluto Medio"] = mean_absolute_error(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Puntaje R2"] = r2_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "RMSE"] = np.sqrt(metrics_df.loc["Prueba", "Error Cuadrático Medio"])

                    new_line()

                    # Gráfico de métricas usando plotly
                    st.markdown("#### Gráfico de Métricas")
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(name='Entrenamiento', x=metrics_df.columns.tolist(), y=metrics_df.loc["Entrenamiento", :].values.tolist()),
                        go.Bar(name='Prueba', x=metrics_df.columns.tolist(), y=metrics_df.loc["Prueba", :].values.tolist())
                    ])
                    st.plotly_chart(fig)

                    # Mostrar los valores de las métricas
                    new_line()
                    st.markdown("##### Valores de las Métricas")
                    st.write(metrics_df)


            elif st.session_state.split_type == "Entrenamiento, Validación y Prueba":
                
                if problem_type == "Clasificación":

                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
                    import pickle

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)
                    if not st.session_state.no_svm:
                        y_prob = model.predict_proba(st.session_state.X_test)[:,1]

                    # DataFrame para almacenar los valores de las métricas
                    metrics_df = pd.DataFrame(columns=["Exactitud", "Precisión", "Recall", "F1", "ROC AUC"], index=["Entrenamiento", "Validación", "Prueba"])
                    metrics_df.loc["Entrenamiento", "Exactitud"] = accuracy_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Precisión"] = precision_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Recall"] = recall_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "F1"] = f1_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    if not st.session_state.no_svm:
                        metrics_df.loc["Entrenamiento", "ROC AUC"] = roc_auc_score(st.session_state.y_train, model.predict_proba(st.session_state.X_train)[:, 1])
                    metrics_df.loc["Validación", "Exactitud"] = accuracy_score(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "Precisión"] = precision_score(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "Recall"] = recall_score(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "F1"] = f1_score(st.session_state.y_val, model.predict(st.session_state.X_val))
                    if not st.session_state.no_svm:
                        metrics_df.loc["Validación", "ROC AUC"] = roc_auc_score(st.session_state.y_val, model.predict_proba(st.session_state.X_val)[:, 1])
                    metrics_df.loc["Prueba", "Exactitud"] = accuracy_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Precisión"] = precision_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Recall"] = recall_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "F1"] = f1_score(st.session_state.y_test, y_pred)
                    if not st.session_state.no_svm:
                        metrics_df.loc["Prueba", "ROC AUC"] = roc_auc_score(st.session_state.y_test, y_prob)


                    new_line()

                    # Gráfico de Métricas
                    st.markdown("### Gráfico de Métricas")
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                                go.Bar(name='Entrenamiento', x=metrics_df.columns.tolist(), y=metrics_df.loc["Entrenamiento", :].values.tolist()),
                                go.Bar(name='Validación', x=metrics_df.columns.tolist(), y=metrics_df.loc["Validación", :].values.tolist()),
                                go.Bar(name='Prueba', x=metrics_df.columns.tolist(), y=metrics_df.loc["Prueba", :].values.tolist())
                            ])
                    st.plotly_chart(fig)


                    # Curva ROC
                    if not st.session_state.no_svm:
                        import plotly.express as px
                        from sklearn.metrics import roc_curve

                        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_prob)
                        fig = px.area(
                            x=fpr, y=tpr,
                            title=f'Curva ROC (AUC={metrics_df.loc["Prueba", "ROC AUC"]:.4f})',
                            labels=dict(x='Tasa de Falsos Positivos', y='Tasa de Verdaderos Positivos'),
                            width=400, height=500
                        )
                        fig.add_shape(
                            type='line', line=dict(dash='dash'),
                            x0=0, x1=1, y0=0, y1=1
                        )

                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                        fig.update_xaxes(constrain='domain')
                        st.plotly_chart(fig)

                    # Mostrar las métricas
                    new_line()
                    st.markdown("### Valores de las Métricas")
                    st.write(metrics_df)

                    # Matriz de confusión
                    # from sklearn.metrics import plot_confusion_matrix
                    import matplotlib.pyplot as plt

                    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                    st.markdown("#### Matriz de Confusión")
                    new_line()

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)

                    # cm = confusion_matrix(y_test, y_pred_test)
                    fig, ax = plt.subplots(figsize=(6,6))
                    ConfusionMatrixDisplay.from_predictions(st.session_state.y_test, y_pred, ax=ax)
                    st.pyplot(fig)


                


                elif problem_type == "Regresión":

                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    import pickle

                    model = pickle.load(open('model.pkl','rb'))
                    y_pred = model.predict(st.session_state.X_test)

                    # DataFrame para las métricas
                    metrics_df = pd.DataFrame(columns=["Error Cuadrático Medio", "Error Absoluto Medio", "Puntaje R2", "RMSE"], index=["Entrenamiento", "Validación", "Prueba"])
                    metrics_df.loc["Entrenamiento", "Error Cuadrático Medio"] = mean_squared_error(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Error Absoluto Medio"] = mean_absolute_error(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "Puntaje R2"] = r2_score(st.session_state.y_train, model.predict(st.session_state.X_train))
                    metrics_df.loc["Entrenamiento", "RMSE"] = np.sqrt(metrics_df.loc["Entrenamiento", "Error Cuadrático Medio"])
                    metrics_df.loc["Validación", "Error Cuadrático Medio"] = mean_squared_error(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "Error Absoluto Medio"] = mean_absolute_error(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "Puntaje R2"] = r2_score(st.session_state.y_val, model.predict(st.session_state.X_val))
                    metrics_df.loc["Validación", "RMSE"] = np.sqrt(metrics_df.loc["Validación", "Error Cuadrático Medio"])
                    metrics_df.loc["Prueba", "Error Cuadrático Medio"] = mean_squared_error(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Error Absoluto Medio"] = mean_absolute_error(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "Puntaje R2"] = r2_score(st.session_state.y_test, y_pred)
                    metrics_df.loc["Prueba", "RMSE"] = np.sqrt(metrics_df.loc["Prueba", "Error Cuadrático Medio"])

                    new_line()

                    # Gráfico de Métricas
                    st.markdown("### Gráfico de Métricas")
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                                go.Bar(name='Entrenamiento', x=metrics_df.columns.tolist(), y=metrics_df.loc["Entrenamiento", :].values.tolist()),
                                go.Bar(name='Validación', x=metrics_df.columns.tolist(), y=metrics_df.loc["Validación", :].values.tolist()),
                                go.Bar(name='Prueba', x=metrics_df.columns.tolist(), y=metrics_df.loc["Prueba", :].values.tolist())
                    ])
                    st.plotly_chart(fig)

                    # Mostrar métricas
                    new_line()
                    st.markdown("### Valores de las Métricas")
                    st.write(metrics_df)

else:
    st.info("Por favor, sube un archivo CSV para continuar.")