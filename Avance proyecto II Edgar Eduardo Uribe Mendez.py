from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import openpyxl as xlsx
import unicodedata
import re


@dataclass
class ConfiguracionLumina:
    nombre_archivo: str = "Lumina_Datos_Operativos.xlsx"
    hoja: str = "Datos"
    target: str = "demanda_unidades"
    random_state: int = 42
    test_size: float = 0.20
    margen_seguridad: float = 0.10
    carpeta_resultados: str = "resultados_lumina"
    


#print(ConfiguracionLumina.__doc__)

class CargadorDatos:
    def __init__(self, config):
        self.config = config
        self.columnas_requeridas = {
        "Producto"
        }
        self.variables_numericas = ["Precio", "Inventario inicial", "Visitas tienda"]
        self.variables_categoricas = ["Categoría"]  
        self.ruta_guardado = Path(__file__).parent / self.config.carpeta_resultados

    @staticmethod
    def normalizar_texto(texto):
        texto = unicodedata.normalize("NFKD", str(texto))
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        texto = texto.lower().strip().replace("%", "pct").replace("°", "")
        return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")

    def generar_carpeta_lumina(self):
        self.ruta_guardado.mkdir(parents=False, exist_ok=True)

    def cargar(self, ruta=None):
        ruta = Path(ruta) 
        print(f"Cargando datos desde: {ruta}")
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        if ruta.suffix.lower() not in {".xlsx", ".xls"}:
            raise ValueError("La fuente de datos debe ser un archivo Excel.")
        datos = pd.read_excel(ruta, sheet_name=self.config.hoja) 
        #print(datos)  
        return datos
        

    def validar_columnas(self, datos):
        faltantes = self.columnas_requeridas.difference(datos.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
        return pd.DataFrame({
            "faltantes": datos.isna().sum(),
            "porcentaje_faltante": (datos.isna().mean() * 100).round(2),
            "valores_unicos": datos.nunique(dropna=True),
            "datos_duplicados": datos.duplicated().sum()
            })
#Limpia datos    
class LimpiezaDatos:

    def __init__(self, datos):
        self.datos = datos
    def limpiar_datos(self, datos):
        antes = len(datos)
        datos = datos.drop_duplicates()
        #print(datos["Categoría"].value_counts())
        duplicados_eliminados = antes - len(datos)

      
        datos["Categoría"] = datos["Categoría"].replace({"Electronica": "Electrónica"})
        datos["Mes"] = datos["Fecha"].dt.month
        datos["dia_semana"] = datos["Fecha"].dt.dayofweek

        datos.columns = (
            datos.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        print("Datos2")
        print(datos)
        #print(datos[config.target])
        
        q1, q3 = datos[config.target].quantile([0.25, 0.75])
        iqr = q3 - q1
        mascara = datos[config.target].between(
            q1 - 1.5 * iqr,
            q3 + 1.5 * iqr
            )
        datos = datos.loc[mascara].copy()
        #print(mascara
        
        filas_sin_target = antes - len(datos)
        resumen = {
            'duplicados_eliminados': duplicados_eliminados,
            'filas_eliminadas_sin_target': filas_sin_target,
            'filas_finales': len(datos)
        }
        #print("resumen")
        #print(resumen)
        plt.figure(figsize=(8,4))
        sns.histplot(data=datos, x=config.target, kde=True)
        plt.title('Distribución de la demanda')
        plt.xlabel('Demanda en unidades')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / config.carpeta_resultados / 'distribucion_demanda.png', dpi=150)
        #plt.show()
        return datos

    
#Explorador datos
class ExploradorDatos:
    def __init__(self, config):
        self.config = config
        self.ruta_guardado = Path(__file__).parent / config.carpeta_resultados

    def graficar_distribucion(self, datos):
        fig, ax = plt.subplots()
        sns.histplot(data=datos, x=self.config.target, kde=True, ax=ax)
        sns.boxplot(data=datos, x=self.config.target, ax=ax)
        demanda_producto = (
            datos.groupby("producto", as_index=False)[self.config.target]
            .mean()
            .sort_values(self.config.target, ascending=False)
            )
        numericas = datos.select_dtypes(include=np.number)
        sns.heatmap(numericas.corr(), cmap="vlag", center=0)
        plt.savefig(self.ruta_guardado / 'distribucion_datos.png')
        #plt.show()

        

#Preparar datos
class PreparadorDatos:
    def __init__(self):
        self.variables_numericas = ["precio", "inventario_inicial", "visitas_tienda"]
        self.variables_categoricas = ["categoría"]

    def preparar_datos(self, X_train, X_test):
        pipeline_numerico = Pipeline([
            ("imputar_mediana", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler())
            ])
        pipeline_categorico = Pipeline([
            ("imputar_moda", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"))
            ])
        self.preprocesador = ColumnTransformer([
            ("numericas", pipeline_numerico, self.variables_numericas),
            ("categoricas", pipeline_categorico, self.variables_categoricas)
            ])
       
        X_train_transformado = self.preprocesador.fit_transform(X_train)
        X_test_transformado = self.preprocesador.transform(X_test)
        return X_train_transformado, X_test_transformado

#Entrenar modelo
class EntrenadorModelo:
    def __init__(self, config):
        self.config = config
        self.ruta_guardado = Path(__file__).parent / config.carpeta_resultados

    def separar_entrenamiento(self, datos):
        from sklearn.model_selection import train_test_split
        X = datos.drop(columns=[self.config.target])
        y = datos[self.config.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
        return X_train, X_test, y_train, y_test


    def entrenar_modelo(self, X_train, y_train, feature_names=None):
        from sklearn.ensemble import RandomForestRegressor
        linea_base_media = DummyRegressor(strategy="mean")
        linea_base_media.fit(X_train, y_train)
        #print("x_train")
        #print(X_train)
        #print("y_train")
        modelo = RandomForestRegressor(n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=self.config.random_state,
            n_jobs=-1)
        modelo.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(22, 12))
        plot_tree(
            modelo.estimators_[0],
            feature_names=feature_names,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=8,
            ax=ax
        )
        fig.tight_layout()
        fig.savefig(
            Path(__file__).parent / config.carpeta_resultados / "arbol_random_forest.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close(fig)
        fig_dummy, ax_dummy = plt.subplots(figsize=(12, 6))

        # Obtener predicciones
        y_pred_m = modelo.predict(X_train)
        indices = range(len(y_train))

        # Trazar valores reales
        ax_dummy.scatter(
            indices,
            y_train,
            color="gray",
            alpha=0.5,
            label="Valores Reales (y_train)",
            s=20,
        )

        # Trazar línea constante del DummyRegressor
        media_val = linea_base_media.constant_[0][0]
        ax_dummy.axhline(
            y=media_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Línea Base Media ({media_val:.2f})",
        )

        # Trazar predicciones de Random Forest
        ax_dummy.plot(
            indices,
            y_pred_m,
            color="green",
            alpha=0.7,
            linewidth=1,
            label="Predicción Random Forest",
        )

        ax_dummy.set_title(
            "Comparación: Línea Base (Dummy) vs Random Forest vs Datos Reales"
        )
        ax_dummy.set_xlabel("Índice de Muestra")
        ax_dummy.set_ylabel("Valor Objetivo (y)")
        ax_dummy.legend()
        ax_dummy.grid(True, linestyle=":", alpha=0.6)

        fig_dummy.tight_layout()
        fig_dummy.savefig(
            Path(__file__).parent / config.carpeta_resultados / "grafica_linea_base.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close(fig_dummy)


        return modelo, linea_base_media

    def generar_matriz_confusion_demanda(self, y_real, y_pred, limites):
   

        etiquetas = ["Baja", "Media", "Alta"]
        bins = [-np.inf, *limites, np.inf]

        real_categoria = pd.cut(
            np.asarray(y_real), bins=bins, labels=etiquetas, right=False, 
        )
        pred_categoria = pd.cut(
            np.asarray(y_pred), bins=bins, labels=etiquetas, right=False
        )

        matriz = confusion_matrix(
            real_categoria,
            pred_categoria,
            labels=etiquetas
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=etiquetas
        ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusión de demanda")
      
        fig.tight_layout()
        fig.savefig(self.ruta_guardado / "matriz_confusion_demanda.png", dpi=150)
        plt.close(fig)

        return pd.DataFrame(
            matriz,
            index=[f"Real {etiqueta}" for etiqueta in etiquetas],
            columns=[f"Predicción {etiqueta}" for etiqueta in etiquetas]
        )
#Calcular mae
    def calcular_mae(self, modelo, X_test, y_test):
        from sklearn.metrics import mean_absolute_error
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\n MAE: {mae:.4f} unidades")
        print(f"Error absoluto promedio {mae:.2f} \n")
        return mae
#Calcular rmse
    def calcular_rmse(self, modelo, X_test, y_test):
        from sklearn.metrics import root_mean_squared_error
        y_pred = modelo.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"\nRMSE: {rmse:.4f} unidades")
        print(f"Errores grandes {rmse:.2f} \n")
        return rmse
#Calcular R"
    def calcular_r2(self, modelo, X_test, y_test):
        from sklearn.metrics import r2_score
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"\nR2: {r2:.4f}")
        print(f"Variabilidad de los datos {r2*100:.2f}% \n")
        return r2

#Genera metricas de modelos

    def generar_csv_calidad_inicial(self,datos):
        datos.to_csv(self.ruta_guardado / 'calidad_inicial.csv', index=False)
        
    def generar_csv_bitacora_limpieza(self,datos):
            datos.to_csv(self.ruta_guardado / 'bitacora_limpieza.csv', index=False)

    def generar_csv_predicciones(self, datos):
            datos.to_csv(self.ruta_guardado / 'predicciones_prueba.csv', index=False)
    
    def generar_reporte_txt(self, mae, rmse, r2, datos_limpios, y_test):
        """Genera reporte ejecutivo en archivo .txt"""
        from datetime import datetime
        print('GENERANDO REPORTE...')
        reporte = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    REPORTE EJECUTIVO - LUMINA                      ║
║                  SISTEMA DE PREDICCIÓN DE DEMANDA                  ║
╚════════════════════════════════════════════════════════════════════╝

📅 FECHA: {datetime.now().strftime('%d de %B de %Y - %H:%M')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJETIVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predecir la demanda de unidades de productos basándose en variables
operacionales (precio, inventario, visitas, categoría).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Registros totales: {len(datos_limpios)}
✓ Registros de prueba: {len(y_test)}
✓ Variables: Precio, Inventario Inicial, Visitas, Categoría

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algoritmo: Random Forest Regressor
├─ Árboles: 300
├─ Profundidad máxima: 8
├─ Muestras mínimas: 2
└─ Split: 80% entrenamiento, 20% prueba

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULTADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAE (Error Absoluto Medio):       {mae:.4f} unidades
RMSE (Raíz Error Cuadrático):     {rmse:.4f} unidades
R² (Coeficiente Determinación):   {r2:.4f}

Demanda Promedio:                 {y_test.mean():.2f} unidades
Error Relativo:                   {(mae/y_test.mean()*100):.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El modelo se equivoca en promedio {mae:.2f} unidades. El error relativo
es del {(mae/y_test.mean()*100):.2f}%, lo que indica desempeño satisfactorio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Modelo con desempeño satisfactorio
✅ Listo para predicciones operacionales
✅ Errores aceptables para toma de decisiones

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMENDACIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Implementar modelo en sistema de predicción
• Monitorear desempeño mensualmente
• Reentrenar cada trimestre con datos nuevos

╔════════════════════════════════════════════════════════════════════╗
║                     FIN DEL REPORTE                                ║
╚════════════════════════════════════════════════════════════════════╝
"""
        
        # Mostrar en consola
        #print(reporte)
        
        # Guardar en archivo
        with open(self.ruta_guardado / "reporte_ejecutivo.txt", "w", encoding="utf-8") as f:
            f.write(reporte)
        
        #print("Reporte guardado en: reporte_ejecutivo.txt")

    #Recomendar inventario
    def recomendar_inventario(
        self, X_test, y_test, y_pred, margen_seguridad=0.10
    ):
      
        df_recomendaciones = X_test.copy()
        df_recomendaciones["demanda_real"] = y_test
        df_recomendaciones["demanda_predicha"] = np.round(y_pred, 2)

        df_recomendaciones["stock_seguridad"] = np.ceil(
            df_recomendaciones["demanda_predicha"] * margen_seguridad
        )
        df_recomendaciones["inventario_objetivo"] = (
            df_recomendaciones["demanda_predicha"]
            + df_recomendaciones["stock_seguridad"]
        )

        df_recomendaciones["unidades_a_reabastecer"] = (
            df_recomendaciones["inventario_objetivo"]
            - df_recomendaciones["inventario_inicial"]
        ).apply(lambda x: max(0, int(np.ceil(x))))

        def evaluar_estado(row):
            if row["inventario_inicial"] < row["demanda_predicha"]:
                return "🚨 Riesgo de Desabasto"
            elif row["inventario_inicial"] > row["inventario_objetivo"] * 1.5:
                return "⚠️️ Sobreinventario"
            else:
                return "✅ Inventario Saludable"

        df_recomendaciones["estado_inventario"] = df_recomendaciones.apply(
            evaluar_estado, axis=1
        )

        df_recomendaciones.to_csv(
            self.ruta_guardado / "recomendaciones_inventario.csv", index=False
        )
        print("Recomendaciones de inventario guardadas exitosamente.")

        return df_recomendaciones

    def comparacion_modelos(self, X_test_transformado, modelo, y_test):
        
        y_pred = modelo.predict(X_test_transformado)
        mae = entrenador.calcular_mae(modelo, X_test_transformado, y_test)
        rmse = entrenador.calcular_rmse(modelo, X_test_transformado, y_test)
        r2 = entrenador.calcular_r2(modelo, X_test_transformado, y_test)

        mae_linea_base = entrenador.calcular_mae(linea_base_media, X_test_transformado, y_test)
        rmse_linea_base = entrenador.calcular_rmse(linea_base_media, X_test_transformado, y_test)
        r2_linea_base = entrenador.calcular_r2(linea_base_media, X_test_transformado, y_test)

        comparacion_modelos = pd.DataFrame([
            {"Modelo": "Random Forest", "MAE": mae, "RMSE": rmse, "R2": r2},
            {"Modelo": "Línea base (promedio)", "MAE": mae_linea_base,
             "RMSE": rmse_linea_base, "R2": r2_linea_base}
        ])

        comparacion = f""" 
RANDOM FOREST:
    MAE: {mae:.4f} unidades
    Error absoluto promedio: {mae:.2f} unidades
    RMSE: {rmse:.4f} unidades
    Errores grandes: {rmse:.2f} unidades
    R2: {r2:.4f}
    Variabilidad de los datos: {r2*100:.2f}%
    -------------------------------------------\n

LINEA BASE (PROMEDIO):
    MAE: {mae_linea_base:.4f} unidades
    Error absoluto promedio: {mae_linea_base:.2f} unidades
    RMSE: {rmse_linea_base:.4f} unidades
    Errores grandes: {rmse_linea_base:.2f} unidades
    R2: {r2_linea_base:.4f}
    Variabilidad de los datos: {r2_linea_base*100:.2f}%
    -------------------------------------------"""

        with open(self.ruta_guardado / "comparacion_modelos.txt", "w", encoding="utf-8") as f:
            f.write(comparacion)
        return y_pred, mae, rmse, r2

    def generar_zip_resultados(self):
        print(self.ruta_guardado)
        ruta_zip = self.ruta_guardado.parent / "Resultados_Lumina"
        shutil.make_archive(str(ruta_zip), "zip", root_dir=str(self.ruta_guardado))
        
if __name__ == "__main__":
        print('Python:', sys.version)
        print('Entorno preparado correctamente')
        config = ConfiguracionLumina()
        cargador = CargadorDatos(config)
        limpieza = LimpiezaDatos(cargador)
        explorador = ExploradorDatos(config)
        entrenador = EntrenadorModelo(config)
        preparador = PreparadorDatos()
        datos = cargador.cargar("C:\\" + config.nombre_archivo)
        cargador.generar_carpeta_lumina()
        datos_limpiar = cargador.validar_columnas(datos)
        datos_limpios = limpieza.limpiar_datos(datos)
        explorador.graficar_distribucion(datos_limpios)

        X_train, X_test, y_train, y_test = entrenador.separar_entrenamiento(datos_limpios)
        X_train_transformado, X_test_transformado = preparador.preparar_datos(X_train, X_test)
        nombres = preparador.preprocesador.get_feature_names_out()
        modelo, linea_base_media = entrenador.entrenar_modelo(X_train_transformado, y_train, nombres)

        entrenador.generar_csv_calidad_inicial(datos_limpiar)
        entrenador.generar_csv_bitacora_limpieza(datos_limpios)
        y_pred, mae, rmse, r2 = entrenador.comparacion_modelos(X_test_transformado, modelo, y_test)
        predicciones = pd.DataFrame({"real": y_test.to_numpy(), "prediccion": y_pred})
        entrenador.generar_csv_predicciones(predicciones)
        entrenador.generar_reporte_txt(mae, rmse, r2, datos_limpios, y_test)
        entrenador.recomendar_inventario(X_test, y_test, y_pred, config.margen_seguridad)
        entrenador.generar_matriz_confusion_demanda(y_test, y_pred, limites=[25, 70])
        entrenador.generar_zip_resultados()