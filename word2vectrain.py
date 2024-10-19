from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences

warnings.filterwarnings(action='ignore')

# Cargar el dataset con las reseñas preprocesadas
data = pd.read_csv(r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_procesados_con_sentimiento_manual.csv') 
output_file_path = r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_vectorizados_word2vec_manual.csv'

# Convertir los tokens en una lista de listas de palabras (para entrenar Word2Vec)
sentences = data['tokens'].apply(eval).tolist()  # Si la columna tokens está en formato string

# Entrenar el modelo de Word2Vec con las secuencias de palabras
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Guardar el modelo entrenado
word2vec_model.save("word2vec_model.model")

# Función para vectorizar cada reseña manteniendo las secuencias de palabras
def vectorize_sequence(tokens, model, vector_size=100):
    # Obtener los vectores de Word2Vec para las palabras de la reseña
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    # Si la reseña tiene palabras reconocidas por Word2Vec, devolver la secuencia de vectores
    if len(word_vectors) > 0:
        return word_vectors  # Devuelve la secuencia de vectores
    else:
        # Si no hay palabras reconocidas, devolver una secuencia de ceros
        return [np.zeros(vector_size)]  # Mantener la estructura de secuencia

# Aplicar la vectorización a cada reseña, manteniendo la secuencia de palabras
data['vector_sequence'] = data['tokens'].apply(lambda x: vectorize_sequence(eval(x), word2vec_model))

# Convertir la columna de secuencias vectorizadas en una lista de listas
sequences = data['vector_sequence'].tolist()

# Aplicar padding para que todas las secuencias tengan la misma longitud
# Usa `maxlen` para definir la longitud máxima de las secuencias (esto es importante para la RNN)
X_padded = pad_sequences(sequences, dtype='float32', padding='post', maxlen=100)  # maxlen es configurable

# Guardar el dataset con las secuencias vectorizadas y aplicadas con padding
np.save(output_file_path.replace('.csv', '_padded.npy'), X_padded)

print("Vectorización y padding completados y guardados.")

# Revisar el tamaño de X_padded
print(X_padded.shape)

# Asegurarte de que las representaciones secuenciales están correctamente guardadas
print(X_padded[0])  # Ver la primera secuencia después del padding

# Asegúrate de que las etiquetas están en la columna "sentimiento"
# Convertir 'sentimiento' a valores numéricos
# Negativo es 2 para que sea compatible con SparseCategoricalCrossEntropy
data['sentimiento'] = data['sentimiento'].map({'positivo': 1, 'neutral': 0, 'negativo': 2})

# Extraer las etiquetas
y = data['sentimiento'].values

# Guardar las etiquetas en un archivo .npy
np.save(r'C:\Universidad\2024-1\Seminario 1\RNN\sentimientos.npy', y)

print("Etiquetas guardadas exitosamente en 'sentimientos.npy'.")

# Asegurarse de que la columna 'category' está presente
if 'category' in data.columns:
    # Extraer las categorías
    categories = data['category'].values

    # Guardar las categorías en un archivo .npy
    np.save(r'C:\Universidad\2024-1\Seminario 1\RNN\categories.npy', categories)
    print("Categorías guardadas exitosamente en 'categories.npy'.")
else:
    print("La columna 'category' no se encuentra en el dataset.")