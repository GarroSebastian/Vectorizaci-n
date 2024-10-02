import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_procesados_con_sentimiento_manual.csv'
output_file_path = r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_vectorizados_tfidf_manual.csv'

# Cargar el dataset
data = pd.read_csv(input_file_path)

# Asegurarse de que no haya valores nulos en 'texto_limpio'
data['texto_limpio'] = data['texto_limpio'].fillna('')

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Vectorizar el texto limpio
tfidf_matrix = tfidf_vectorizer.fit_transform(data['texto_limpio'])

# Convertir la matriz TF-IDF a un DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenar el DataFrame TF-IDF con las columnas originales
data_tfidf = pd.concat([data.reset_index(drop=True), tfidf_df], axis=1)

# Guardar el dataset vectorizado
data_tfidf.to_csv(output_file_path, index=False)

print("Dataset vectorizado y guardado con éxito en la ruta especificada.")
