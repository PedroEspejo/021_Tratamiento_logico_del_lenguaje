import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargar recursos necesarios (ejecutar una vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Ejemplo 1: Tokenización de texto
texto = "El procesamiento de lenguaje natural (NLP) es una rama de la inteligencia artificial que se enfoca en la interacción entre las computadoras y el lenguaje humano."
palabras = word_tokenize(texto)
print(palabras)

# Ejemplo 2: Tokenización de oraciones
texto = "El NLP tiene muchas aplicaciones. Algunas de ellas incluyen chatbots, análisis de sentimiento y resumen automático de texto."
oraciones = sent_tokenize(texto)
print(oraciones)

# Ejemplo 3: Eliminación de stopwords
texto = "El análisis de sentimiento es importante en NLP. Ayuda a determinar la actitud de un texto."
palabras = word_tokenize(texto)
stop_words = set(stopwords.words('spanish'))
palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stop_words]
print(palabras_filtradas)

# Ejemplo 4: Stemming (derivación)
ps = PorterStemmer()
palabras = ["corriendo", "corre", "corrió", "corredor"]
palabras_stem = [ps.stem(palabra) for palabra in palabras]
print(palabras_stem)

# Ejemplo 5: Análisis de frecuencia de palabras
texto = "El análisis de frecuencia de palabras es útil para determinar las palabras clave en un texto. Cuantas más veces aparezca una palabra, más importante es."
palabras = word_tokenize(texto)
frecuencia = FreqDist(palabras)
print(frecuencia.most_common(5))  # Muestra las 5 palabras más comunes

# Ejemplo 6: Análisis de sentimiento
analyzer = SentimentIntensityAnalyzer()
texto = "Estoy muy feliz de aprender NLP. Es un tema fascinante."
sentimiento = analyzer.polarity_scores(texto)
print(sentimiento)