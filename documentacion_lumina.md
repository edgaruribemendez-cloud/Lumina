# Documentación breve del sistema Lúmina

## Configuración de funcionamiento

La clase `ConfiguracionLumina` concentra los parámetros principales: archivo Excel `Lumina_Datos_Operativos.xlsx`, hoja `Datos`, variable objetivo `demanda_unidades`, semilla aleatoria `42`, división de prueba del `20 %`, margen de seguridad de inventario del `10 %` y carpeta de resultados `resultados_lumina`.

## Módulo de carga

`CargadorDatos` verifica que el archivo exista y tenga extensión `.xlsx` o `.xls`, y carga la hoja configurada con pandas. También comprueba que exista la columna `Producto` y genera un resumen de valores faltantes, valores únicos y duplicados.

## Limpieza de datos

`LimpiezaDatos` elimina filas duplicadas, normaliza el nombre de la categoría electrónica, extrae el mes y el día de la semana desde `Fecha`, y convierte los nombres de columnas a minúsculas y formato con guiones bajos. Después filtra valores atípicos de la demanda usando el rango intercuartílico (IQR).

## Exploración de datos

Se generan visualizaciones de la distribución y caja de la demanda, el promedio de demanda por producto y la correlación entre variables numéricas. Las gráficas se guardan en la carpeta de resultados.

## Entrenamiento de datos

La variable objetivo se separa de las características. Los datos se dividen en entrenamiento y prueba (80/20), usando la semilla configurada para que la división sea reproducible. El modelo se entrena con las características transformadas por el preprocesador.

## Funcionamiento del preprocesador

`ColumnTransformer` aplica dos flujos:

- Variables numéricas (`precio`, `inventario_inicial`, `visitas_tienda`): reemplaza faltantes con la mediana y estandariza los valores.
- Variable categórica (`categoría`): reemplaza faltantes con el valor más frecuente y aplica codificación One-Hot. Las categorías desconocidas se ignoran.

El preprocesador se ajusta con entrenamiento y luego transforma los datos de prueba, evitando ajustarlo con información de prueba.

## Modelos utilizados

- **Random Forest Regressor:** modelo principal, con 300 árboles, profundidad máxima de 8, mínimo de 2 muestras por hoja y semilla aleatoria 42.
- **Dummy Regressor:** línea base que predice la demanda promedio; permite comparar el desempeño del modelo principal.

## Cálculo del MAE, RMSE y R²

Las métricas se calculan comparando las predicciones con la demanda real del conjunto de prueba:

- **MAE:** promedio del error absoluto; expresa el error típico en unidades.
- **RMSE:** raíz del promedio de los errores al cuadrado; penaliza más los errores grandes.
- **R²:** proporción de variabilidad explicada por el modelo; cuanto más cercano a 1, mejor ajuste.

Se reportan para Random Forest y para la línea base.

## Recomendación de inventario

Para cada registro de prueba, el sistema calcula:

- **Stock de seguridad:** demanda predicha × margen de seguridad.
- **Inventario objetivo:** demanda predicha + stock de seguridad.
- **Unidades a reabastecer:** máximo entre cero e inventario objetivo menos inventario inicial, redondeado hacia arriba.

También clasifica el estado como riesgo de desabasto, sobreinventario o inventario saludable. Las recomendaciones se guardan en `recomendaciones_inventario.csv`.
