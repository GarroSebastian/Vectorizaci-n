from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

#vectorizar dataset con word2vec en vez de TF-IDF para datos secuenciales que necesita la RNN

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_procesados_con_sentimiento_manual.csv'
output_file_path = r'C:\Universidad\2024-1\Seminario 1\Vectorización\textos_vectorizados_word2vec_manual.csv'

