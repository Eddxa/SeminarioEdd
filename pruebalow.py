"""
Script para hacer fine-tuning (adaptación de dominio) de un modelo de lenguaje enmascarado (MLM)
usando un corpus local de archivos Markdown.

Este script adapta el proceso descrito en el tutorial de Hugging Face sobre
fine-tuning de MLM (https://huggingface.co/learn/nlp-course/chapter7/3?framework=tf)
para trabajar con archivos .md locales en lugar de un dataset del Hub.
"""

# --- 0. Importaciones Necesarias ---
import os                     # Para interactuar con el sistema operativo (rutas de archivos)
import glob                   # Para encontrar archivos que coincidan con un patrón (ej. *.md)
import math                   # Para calcular la perplejidad (exponencial de la loss)
import collections            # Para usar defaultdict en el collator WWM (opcional)
import numpy as np            # Para operaciones numéricas (usado en collator WWM)
import tensorflow as tf       # El framework de deep learning que usaremos
try:
    # Esta línea es importante para la compatibilidad con Transformers y Keras 3
    import tf_keras as keras
except ImportError:
    print("Advertencia: No se pudo importar tf_keras. Usando keras de TensorFlow.")
    from tensorflow import keras

from datasets import Dataset, DatasetDict # Para manejar y estructurar los datos
from transformers import (
    AutoTokenizer,            # Para cargar el tokenizador automáticamente según el checkpoint
    TFAutoModelForMaskedLM,   # Para cargar el modelo pre-entrenado para Masked Language Modeling en TF
    DataCollatorForLanguageModeling, # Collator estándar que crea lotes y máscaras para MLM
    create_optimizer,         # Función de ayuda para crear el optimizador AdamW con schedule
    PushToHubCallback,        # Callback para subir el modelo entrenado al Hugging Face Hub
    pipeline,                 # Para usar fácilmente el modelo entrenado para tareas como fill-mask
)
from transformers.data.data_collator import tf_default_data_collator 
from huggingface_hub import notebook_login, login # Para iniciar sesión en el Hub






#PASO 0 - CONFIGURACIÓN PRINCIPAL DEL MODELO Y RUTA DE ARCHIVOS 







# --- 1. Configuración Principal ---

MARKDOWN_FOLDER_PATH = "./testfiles/" # Ruta a los datos 

# --- Modelo y Tokenización ---

MODEL_CHECKPOINT = "distilbert-base-uncased" # Identificador del modelo pre-entrenado en Hugging Face Hub.
CHUNK_SIZE = 128  # Tamaño de los fragmentos (chunks) en los que se dividirá el texto concatenado.
MLM_PROBABILITY = 0.15 # Probabilidad de que un token sea enmascarado durante el entrenamiento.
WWM_PROBABILITY = 0.20 # Probabilidad de enmascarar una palabra completa.

# --- Configuración del Entrenamiento ---

TRAIN_TEST_SPLIT_SEED = 42 # Semilla para la aleatoriedad en la división de datos (reproducibilidad).
VALIDATION_SPLIT_PERCENTAGE = 0.1 # Porcentaje del dataset de entrenamiento que se usará para validación.
BATCH_SIZE = 4 # Número de ejemplos (chunks) procesados en cada paso.
LEARNING_RATE = 2e-5 # Tasa de aprendizaje inicial para el optimizador AdamW. 
WEIGHT_DECAY = 0.01 # Parámetro de regularización para el optimizador.
NUM_WARMUP_STEPS = 50 # Número de pasos iniciales donde el learning rate aumenta linealmente desde 0 hasta LEARNING_RATE.
NUM_EPOCHS = 1 # Número de veces que el modelo verá el dataset de entrenamiento completo.

# --- Configuración para Guardar el Modelo ---

PUSH_TO_HUB = False # Poner en True si quieres guardar tu modelo fine-tuned en el Hugging Face Hub. 

HF_HUB_MODEL_NAME = f"{MODEL_CHECKPOINT.split('/')[-1]}-finetuned-custom-markdown" # Nombre para tu modelo en el Hub.
HF_HUB_OUTPUT_DIR = f"{MODEL_CHECKPOINT.split('/')[-1]}-finetuned-custom-markdown" # Nombre de la carpeta local donde se guardará permanentemente.
#================================================================================================================================================================
#================================================================================================================================================================








#PASO 1 CARGAR LOS DATOS Y CONVERTIRLOS A FORMATO DATASET PARA 
#       TRABAJARLOS EN HUGGINFACE 







# --- 2. Cargar Datos Locales ---
# Esta función lee todos los archivos .md de la carpeta especificada.
def load_markdown_dataset(folder_path):
    """
    Carga todos los archivos .md de una carpeta y los convierte en un Dataset de Hugging Face.
    Cada archivo .md se considera un documento/ejemplo inicial.
    """
    # Construye el patrón para encontrar todos los archivos .md en la carpeta
    path_pattern = os.path.join(folder_path, "*.md")
    filepaths = glob.glob(path_pattern) # Obtiene la lista de rutas a los archivos .md

    # Verifica si se encontraron archivos
    if not filepaths:
        raise FileNotFoundError(f"No se encontraron archivos .md en la carpeta: {folder_path}. "
                              "Asegúrate de que la ruta es correcta y contiene archivos Markdown.")

    all_texts = [] # Lista para almacenar el contenido de cada archivo
    print(f"Encontrados {len(filepaths)} archivos Markdown. Leyendo contenido...")
    # Itera sobre cada ruta de archivo encontrada
    for filepath in filepaths:
        try:
            # Abre el archivo en modo lectura ('r') con codificación UTF-8 (importante para textos diversos)
            with open(filepath, 'r', encoding='utf-8') as f:
                all_texts.append(f.read()) # Lee todo el contenido y lo añade a la lista
        except Exception as e:
            # Informa si hay un error leyendo un archivo específico
            print(f"Error leyendo el archivo {filepath}: {e}")

    # Crea una lista de diccionarios. Cada diccionario es un ejemplo con la clave "text".
    # Se filtran los textos vacíos o que solo contienen espacios en blanco.
    data = [{"text": text} for text in all_texts if text and text.strip()]

    # Verifica si se pudo leer contenido válido
    if not data:
        raise ValueError(f"No se pudo leer contenido válido de los archivos .md en: {folder_path}")

    print(f"Se cargaron {len(data)} documentos Markdown con contenido.")

    # Convierte la lista de diccionarios al formato Dataset de Hugging Face
    # Se crea un diccionario donde la clave es el nombre de la columna ("text")
    # y el valor es una lista con todos los textos.
    hf_dataset = Dataset.from_dict({"text": [item["text"] for item in data]})
    return hf_dataset





#PASO 2 CARGAR LOS DATOS Y SEPARARLOS EN 3 DIVISIONES (TRAIN - VALIDATION - TEST)





print(f"--- Paso 2: Cargando datos desde: {MARKDOWN_FOLDER_PATH} ---")
try:
    # Llama a la función para cargar los datos
    raw_dataset = load_markdown_dataset(MARKDOWN_FOLDER_PATH)
    print("Dataset cargado exitosamente:")
    print(raw_dataset) # Muestra información básica del dataset cargado

    # --- División del Dataset ---
    
    # Verifica si hay suficientes datos para dividir
    if len(raw_dataset) < 3:
         print("\nAdvertencia: No hay suficientes datos para crear divisiones train/validation/test significativas.")
         # Decide cómo manejar esto: usar todo para entrenar, o detener.
         # Opción: Usar todo para entrenar (sin validación/test formal)
         split_datasets = DatasetDict({
             'train': raw_dataset,
             'validation': raw_dataset.select([]), # Dataset vacío
             'test': raw_dataset.select([])        # Dataset vacío
         })
         print("Usando todo el dataset para entrenamiento debido a la escasez de datos.")
    else:
        # Dividimos el dataset cargado en conjuntos de entrenamiento, validación y prueba.
        
        test_size_fraction = 0.1
        # Calcular tamaño absoluto asegurando al menos 1 si es posible
        num_test_samples = max(1, int(len(raw_dataset) * test_size_fraction)) if len(raw_dataset) > 1 else 0

        if num_test_samples > 0 and len(raw_dataset) > num_test_samples:
             train_val_test_split = raw_dataset.train_test_split(test_size=num_test_samples, seed=TRAIN_TEST_SPLIT_SEED)
             test_set = train_val_test_split['test']
             train_val_set = train_val_test_split['train']
        else: # No hay suficientes datos para test o no se pueden separar
             test_set = raw_dataset.select([])
             train_val_set = raw_dataset

        # Asegurarse de que validation_size no sea 0
        validation_size_fraction = VALIDATION_SPLIT_PERCENTAGE
        num_validation_samples = 0
        if len(train_val_set) > 1: # Solo se puede sacar validación si quedan al menos 2 ejemplos
             num_validation_samples = max(1, int(len(train_val_set) * validation_size_fraction))

        if num_validation_samples > 0 and len(train_val_set) > num_validation_samples:
             final_train_valid_split = train_val_set.train_test_split(test_size=num_validation_samples, seed=TRAIN_TEST_SPLIT_SEED)
             train_set = final_train_valid_split['train']
             validation_set = final_train_valid_split['test']
        else: # No hay suficientes datos para validación o solo queda 1
             train_set = train_val_set
             validation_set = train_val_set.select([]) # Dataset vacío

        # Crear el DatasetDict final con las divisiones estándar
        split_datasets = DatasetDict({
            'train': train_set,
            'validation': validation_set,
            'test': test_set
        })

    print("\nDataset dividido en train, validation y test:")
    print(split_datasets) # Muestra la estructura final del dataset dividido

except (FileNotFoundError, ValueError) as e:
    # Captura errores comunes de carga de datos y termina el script
    print(f"\nError Crítico: {e}")
    print("Por favor, verifica la variable MARKDOWN_FOLDER_PATH y asegúrate de que la carpeta existe y contiene archivos .md válidos.")
    exit() # Termina la ejecución si no se pueden cargar los datos








#PASO 3 SE CARGA EL MODELO PRE-ENTRENADO Y SE CARGA EL TOKENIZADOR






# --- 3. Cargar Modelo y Tokenizador ---
# Cargamos el modelo pre-entrenado y su tokenizador correspondiente desde Hugging Face Hub.

print(f"\n--- Paso 3: Cargando Tokenizador y Modelo ({MODEL_CHECKPOINT}) ---")
# Carga el tokenizador asociado al checkpoint.
# `use_fast=True` intenta cargar la implementación rápida (Rust), que es más eficiente
# y necesaria para obtener `word_ids` fácilmente (útil para Whole Word Masking).
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

# Carga el modelo pre-entrenado específico para Masked Language Modeling (MLM) en TensorFlow.
model = TFAutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

print("\nResumen de la arquitectura del modelo:")
model.summary() # Muestra las capas y el número de parámetros del modelo.

# Verifica la longitud máxima de secuencia que el modelo puede manejar.
print(f"\nLongitud máxima de contexto del tokenizador/modelo: {tokenizer.model_max_length}")


# Advierte si el CHUNK_SIZE elegido es mayor que el máximo soportado.
if CHUNK_SIZE > tokenizer.model_max_length:
    print(f"¡Advertencia!: CHUNK_SIZE ({CHUNK_SIZE}) es mayor que model_max_length ({tokenizer.model_max_length}). "
          f"Esto puede causar errores o truncamiento inesperado. Se recomienda reducir CHUNK_SIZE.")







#PASO 4 # Preparamos los datos para el entrenamiento de MLM. Esto implica:
     # 1. Tokenizar: Convertir el texto en secuencias de IDs numéricos que el modelo entiende.
     # 2. Agrupar y Fragmentar: Concatenar todos los textos tokenizados 
     # y dividirlos en fragmentos (chunks) de tamaño fijo.








# --- 4. Preprocesamiento de Datos ---
print("\n--- Paso 4: Preprocesando los Datos ---")

def tokenize_function(examples):
    """
    Tokeniza los textos de entrada usando el tokenizador cargado.
    """
    # Aplica el tokenizador a los textos en el lote (`examples["text"]`).
    # `truncation=False` es importante aquí porque queremos concatenar *antes* de truncar/dividir en chunks.
    result = tokenizer(examples["text"], truncation=False)

    # Si el tokenizador es rápido, podemos obtener fácilmente los 'word_ids'.
    # `word_ids` mapea cada token a la palabra original a la que pertenece. Es útil para WWM.
    if tokenizer.is_fast:
        # `result.word_ids(i)` devuelve los word_ids para el i-ésimo ejemplo en el lote.
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

print("Tokenizando los datasets (puede tardar un poco)...")
# Aplicamos la función `tokenize_function` a todos los splits del dataset.
# `batched=True` procesa múltiples ejemplos a la vez (más rápido).
# `remove_columns=["text"]` elimina la columna de texto original, ya no la necesitamos.
tokenized_datasets = split_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
print("Datasets tokenizados:")
print(tokenized_datasets) # Muestra la estructura después de tokenizar


def group_texts(examples):
    """
    Concatena todos los textos tokenizados de un lote y los divide en chunks de tamaño `CHUNK_SIZE`.
    Descarta el último chunk si es más pequeño que `CHUNK_SIZE`.
    Crea la columna 'labels' necesaria para MLM.
    """
    # `examples` es un diccionario donde las claves son los nombres de las columnas ('input_ids', 'attention_mask', 'word_ids')
    # y los valores son listas de listas (una lista por cada ejemplo original en el lote).

    # 1. Concatenar todas las listas de cada característica en una sola lista grande.
    #    `sum(examples[k], [])` concatena las listas internas para la clave `k`.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # 2. Calcular la longitud total del texto concatenado (basado en 'input_ids').
    total_length = len(concatenated_examples[list(examples.keys())[0]]) # Usa la primera clave como referencia

    # 3. Ajustar la longitud total para que sea un múltiplo exacto de CHUNK_SIZE.
    #    Se descarta el final si no completa un chunk. `//` es división entera.
    if total_length < CHUNK_SIZE:
         # Si el total es menor que un chunk, no se puede hacer nada, devolver vacío
         # o manejar de otra forma. Aquí devolvemos vacío para evitar errores.
         print(f"Advertencia: Longitud total concatenada ({total_length}) es menor que CHUNK_SIZE ({CHUNK_SIZE}). No se generarán chunks.")
         # Devolver un diccionario con listas vacías para mantener la estructura
         return {k: [] for k in examples.keys() if k in concatenated_examples}


    total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE

    # 4. Dividir las listas concatenadas en chunks de tamaño `CHUNK_SIZE`.
    #    Se usa una comprensión de listas para crear las nuevas listas de chunks.
    result = {
        k: [t[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
        for k, t in concatenated_examples.items()
    }

    # 5. Crear la columna 'labels'. En MLM, el modelo debe predecir los tokens originales
    #    (antes de enmascarar). Por lo tanto, las 'labels' son inicialmente una copia de 'input_ids'.
    #    El Data Collator se encargará de enmascarar los 'input_ids' y poner -100 en las 'labels'
    #    correspondientes a los tokens no enmascarados.
    # Asegurarse de que 'input_ids' existe en el resultado antes de copiar
    if "input_ids" in result:
        result["labels"] = result["input_ids"].copy()
    else:
        # Si no hay input_ids (porque total_length < CHUNK_SIZE), crear 'labels' vacía
        result["labels"] = []

    return result

print(f"\nAgrupando textos en chunks de tamaño {CHUNK_SIZE} (puede tardar)...")
# Aplicamos la función `group_texts` a los datasets tokenizados.
# `batched=True` es crucial aquí para que la concatenación funcione correctamente sobre lotes grandes.
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print("Datasets listos para el entrenamiento MLM (agrupados y fragmentados):")
print(lm_datasets) # Muestra la estructura final lista para el Data Collator








#PASO 5 Toma los datos ya procesados, los agrupa eficientemente y aplica la estrategia 
    # de enmascaramiento aleatorio justo antes de que cada lote entre al modelo, asegurando 
    # que el modelo reciba los datos en el formato correcto y con los objetivos de predicción 
    # adecuados para la tarea de Masked Language Modeling









# --- 5. Data Collator ---
# El Data Collator es una función que toma una lista de ejemplos del dataset
# y los agrupa en un lote (batch). Para MLM, también realiza el enmascaramiento aleatorio.

print("\n--- Paso 5: Configurando el Data Collator ---")

# --- Opción 1: Data Collator Estándar (Recomendado para empezar) ---
# Este collator enmascara tokens individuales aleatoriamente.
print(f"Usando DataCollatorForLanguageModeling con probabilidad de máscara {MLM_PROBABILITY}")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,            # El tokenizador para saber qué ID corresponde a [MASK]
    mlm_probability=MLM_PROBABILITY,# La probabilidad de enmascarar un token
    return_tensors="tf"             # Especifica que debe devolver tensores de TensorFlow
)







#PASO 6 Se preparan los datos para empezar con el entrenamiento 










print("\n--- Paso 6: Preparando tf.data Datasets ---")

tf_train_dataset = None
tf_eval_dataset = None
tf_test_dataset = None

# Solo preparar si el dataset tiene filas
if "train" in lm_datasets and len(lm_datasets["train"]) > 0:
    tf_train_dataset = model.prepare_tf_dataset(
        lm_datasets["train"],
        collate_fn=data_collator,
        shuffle=True,             # Barajar el dataset de entrenamiento en cada época
        batch_size=BATCH_SIZE
    )
else:
     print("Advertencia: El dataset de entrenamiento está vacío después del preprocesamiento.")


if "validation" in lm_datasets and len(lm_datasets["validation"]) > 0:
    tf_eval_dataset = model.prepare_tf_dataset(
        lm_datasets["validation"], # Usar el set de validación para evaluar durante el entrenamiento
        collate_fn=data_collator,
        shuffle=False,            # No barajar los sets de evaluación/prueba
        batch_size=BATCH_SIZE
    )
else:
    # tf_eval_dataset ya es None por defecto
    print("Advertencia: No hay dataset de validación para evaluar durante el entrenamiento.")


if "test" in lm_datasets and len(lm_datasets["test"]) > 0:
    tf_test_dataset = model.prepare_tf_dataset(
        lm_datasets["test"],      # Usar el set de prueba para la evaluación final
        collate_fn=data_collator,
        shuffle=False,
        batch_size=BATCH_SIZE
    )
else:
    # tf_test_dataset ya es None por defecto
    print("Advertencia: No hay dataset de test para la evaluación final.")


print("Datasets tf.data listos para el entrenamiento y evaluación.")










#PASO 7 Se configura el entrenamiento









# --- 7. Configuración del Entrenamiento ---
# Definimos el optimizador, compilamos el modelo y configuramos callbacks.

print("\n--- Paso 7: Configurando el Entrenamiento ---")

# --- Inicio de sesión en Hugging Face Hub (Opcional) ---
if PUSH_TO_HUB:
    print("Intentando iniciar sesión en Hugging Face Hub...")
    try:
        # `login()` funciona tanto en scripts como en notebooks.
        # Buscará un token guardado localmente o pedirá uno si no lo encuentra.
        login()
        print("Inicio de sesión en Hugging Face Hub exitoso.")
    except Exception as e:
        print(f"No se pudo iniciar sesión automáticamente en Hugging Face Hub: {e}")
        print("Si quieres subir el modelo, asegúrate de haber ejecutado 'huggingface-cli login' en tu terminal "
              "o proporciona un token válido.")
       
        # PUSH_TO_HUB = False # Desactivar subida si falla el login
        # exit() # Detener el script

# --- Optimizador y Learning Rate Scheduler ---
# Calculamos el número total de pasos de entrenamiento para el scheduler.
# Necesitamos asegurarnos de que tf_train_dataset no sea None o vacío
if tf_train_dataset is None:
     print("Error: El dataset de entrenamiento está vacío o no se pudo preparar. No se puede continuar.")
     exit()

# Estimar pasos por época si es posible
num_train_steps = 0
try:
    # tf.data.experimental.cardinality es más robusto
    steps_per_epoch = tf.data.experimental.cardinality(tf_train_dataset).numpy()
    if steps_per_epoch == tf.data.UNKNOWN_CARDINALITY or steps_per_epoch == tf.data.INFINITE_CARDINALITY:
         # Si la cardinalidad no se puede determinar, usar una estimación basada en el tamaño del dataset original
         if "train" in lm_datasets and len(lm_datasets["train"]) > 0:
              steps_per_epoch = math.ceil(len(lm_datasets["train"]) / BATCH_SIZE)
              print(f"Advertencia: Cardinalidad del dataset desconocida. Estimando steps_per_epoch = {steps_per_epoch}")
         else:
              print("Error: No se puede estimar steps_per_epoch porque lm_datasets['train'] está vacío.")
              exit()

    elif steps_per_epoch == 0:
         print("Error: El dataset de entrenamiento tiene cardinalidad 0 después de la preparación.")
         exit()

    num_train_steps = steps_per_epoch * NUM_EPOCHS

except Exception as e: # Captura errores más generales al obtener cardinalidad
     print(f"Error al obtener la cardinalidad del dataset de entrenamiento: {e}")
     # Intentar estimar como fallback
     if "train" in lm_datasets and len(lm_datasets["train"]) > 0:
          steps_per_epoch = math.ceil(len(lm_datasets["train"]) / BATCH_SIZE)
          print(f"Estimando steps_per_epoch como fallback = {steps_per_epoch}")
          num_train_steps = steps_per_epoch * NUM_EPOCHS
     else:
          print("Error: No se puede estimar steps_per_epoch porque lm_datasets['train'] está vacío.")
          exit()


print(f"Número total de pasos de entrenamiento estimado: {num_train_steps}")
if num_train_steps <= 0:
     print("Error: Número de pasos de entrenamiento es 0 o negativo. Verifica el tamaño del dataset y BATCH_SIZE.")
     exit()


# Asegurarse de que num_warmup_steps no sea negativo o mayor que num_train_steps
actual_num_warmup_steps = min(NUM_WARMUP_STEPS, num_train_steps // 10)
actual_num_warmup_steps = max(0, actual_num_warmup_steps) # Asegurar que no sea negativo

optimizer, schedule = create_optimizer(
    init_lr=LEARNING_RATE,
    num_warmup_steps=actual_num_warmup_steps,
    num_train_steps=num_train_steps,
    weight_decay_rate=WEIGHT_DECAY,
)

# --- Compilación del Modelo ---
# Compilamos el modelo Keras. TensorFlow necesita esto antes de entrenar.
# Al no especificar una `loss` aquí, el modelo usará su loss interna por defecto,
# que para `TFAutoModelForMaskedLM` es la cross-entropy adecuada para MLM,
# calculada solo sobre los tokens enmascarados.
model.compile(optimizer=optimizer)
print("Modelo compilado con el optimizador AdamW.")


# --- Callbacks ---

# Los callbacks son funciones que se ejecutan en diferentes puntos del entrenamiento (ej. al final de cada época).
callbacks = [] # Lista para almacenar los callbacks activos

# Callback para subir el modelo al Hub
if PUSH_TO_HUB:
    print("Configurando PushToHubCallback para guardar en el repositorio: '{HF_HUB_MODEL_NAME}'")

    os.makedirs(HF_HUB_OUTPUT_DIR, exist_ok=True)
    hub_callback = PushToHubCallback(
        output_dir=HF_HUB_OUTPUT_DIR,    # Directorio local temporal/final
        tokenizer=tokenizer,             # El tokenizador a guardar junto con el modelo
        hub_model_id=HF_HUB_MODEL_NAME   # El nombre completo del repositorio en el Hub (ej. "tu_usuario/nombre_modelo")
    )
    callbacks.append(hub_callback)
else:
    print("El modelo se guardará localmente al final del entrenamiento (PUSH_TO_HUB=False).")










#PASO 8 Se evalúa la perfplejidad inicial antes de hacer el fine tuning, para luego comparar










print("\n--- Paso 8: Evaluando Perplejidad Inicial (Antes del Fine-tuning) ---")
perplexity_before = float('inf') # Valor inicial por defecto
if tf_eval_dataset:
    try:
        # `model.evaluate` calcula la loss en el dataset proporcionado.
        print("Evaluando en el dataset de validación...")
        eval_loss_before = model.evaluate(tf_eval_dataset, verbose=0) # verbose=0 para menos salida
        # Calculamos la perplejidad
        if eval_loss_before is not None and eval_loss_before > 0: # Asegurar que la loss es válida
             perplexity_before = math.exp(eval_loss_before)
             print(f"Perplejidad en el set de validación (antes): {perplexity_before:.2f}")
        else:
             print(f"Loss de evaluación inicial inválida o no positiva ({eval_loss_before}). No se puede calcular perplejidad.")

    except tf.errors.ResourceExhaustedError as e:
         print(f"Error de memoria (OOM) durante la evaluación inicial: {e}")
         print("Intenta reducir BATCH_SIZE en la configuración.")
    except Exception as e:
         # Otros posibles errores
         print(f"Error inesperado durante la evaluación inicial: {e}")
else:
     print("No hay dataset de validación para la evaluación inicial.")











#PASO 9 Se inicia el proceso de fine tuning












# --- 9. Entrenamiento ---
# Iniciamos el proceso de fine-tuning.

print("\n--- Paso 9: Iniciando Fine-tuning ---")
print(f"Entrenando por {NUM_EPOCHS} épocas...")

# `model.fit` es la función principal de Keras para entrenar modelos.
# Necesita validation_data si queremos monitorear la loss de validación
try:
    history = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset if tf_eval_dataset else None, # Pasar None si no hay validación
        epochs=NUM_EPOCHS,
        callbacks=callbacks       # Lista de callbacks a usar (ej. PushToHubCallback)
    )
    print("--- Fine-tuning completado ---")

except tf.errors.ResourceExhaustedError as e:
    print(f"\nError de memoria (OOM) durante el entrenamiento: {e}")
    print(f"El PC se congeló probablemente debido a falta de RAM.")
    print(f"Intenta reducir aún más BATCH_SIZE (actual: {BATCH_SIZE}). Prueba con 2 o 1.")
    print("También puedes intentar reducir CHUNK_SIZE si BATCH_SIZE ya es muy bajo.")
    exit() # Salir del script si falla el entrenamiento por OOM
except Exception as e:
    print(f"\nError inesperado durante el entrenamiento: {e}")
    exit()














#PASO 10 Se evalúa nuevamente la perplejidad una vez hecho el fine tuning










# --- 10. Evaluación Final (Perplejidad) ---
# Evaluamos la perplejidad nuevamente en los sets de validación y prueba
# para ver la mejora y el rendimiento final del modelo.

print("\n--- Paso 10: Evaluando Perplejidad Final (Después del Fine-tuning) ---")
perplexity_after = float('inf')
test_perplexity = float('inf')

if tf_eval_dataset:
    try:
        # Evaluar en el set de validación
        print("Evaluando en el dataset de validación (final)...")
        eval_loss_after = model.evaluate(tf_eval_dataset, verbose=0)
        if eval_loss_after is not None and eval_loss_after > 0:
            perplexity_after = math.exp(eval_loss_after)
            print(f"Perplejidad en el set de validación (después): {perplexity_after:.2f}")
            # Comparación
            if perplexity_before != float('inf'):
                 print(f"Mejora de perplejidad en validación: {perplexity_before:.2f} -> {perplexity_after:.2f}")
            else:
                 print("(No se pudo calcular la perplejidad inicial para comparación)")
        else:
            print(f"Loss de evaluación final inválida o no positiva ({eval_loss_after}). No se puede calcular perplejidad.")


    except tf.errors.ResourceExhaustedError as e:
         print(f"Error de memoria (OOM) durante la evaluación final de validación: {e}")
         print("Intenta reducir BATCH_SIZE.")
    except Exception as e:
         print(f"Error inesperado durante la evaluación final de validación: {e}")
else:
     print("No hay dataset de validación para la evaluación final.")


if tf_test_dataset:
    try:
        # Evaluar en el set de prueba (la medida final del rendimiento en datos no vistos)
        print("Evaluando en el dataset de test (final)...")
        test_loss = model.evaluate(tf_test_dataset, verbose=0)
        if test_loss is not None and test_loss > 0:
            test_perplexity = math.exp(test_loss)
            print(f"Perplejidad en el set de test (final): {test_perplexity:.2f}")
        else:
             print(f"Loss de evaluación de test inválida o no positiva ({test_loss}). No se puede calcular perplejidad.")

    except tf.errors.ResourceExhaustedError as e:
         print(f"Error de memoria (OOM) durante la evaluación final de test: {e}")
         print("Intenta reducir BATCH_SIZE.")
    except Exception as e:
         print(f"Error inesperado durante la evaluación final de test: {e}")
else:
     print("No hay dataset de test para la evaluación final.")











#PASO 11 Se guardan el modelo localemente 











# --- 11. Guardar Modelo Localmente (si no se usó PushToHub) ---
# Si no se configuró para subir al Hub, guardamos el modelo y el tokenizador
# en la carpeta especificada por `HF_HUB_OUTPUT_DIR`.

if not PUSH_TO_HUB:
    save_directory = f"./{HF_HUB_OUTPUT_DIR}" # Añade './' para indicar el directorio actual
    print(f"\n--- Paso 11: Guardando modelo y tokenizador localmente en: {save_directory} ---")
    try:
        os.makedirs(save_directory, exist_ok=True) # Crea el directorio si no existe
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("Modelo y tokenizador guardados localmente con éxito.")
    except Exception as e:
        print(f"Error al guardar el modelo localmente: {e}")

