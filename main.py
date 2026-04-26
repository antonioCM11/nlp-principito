import spacy
import pandas as pd
from nltk.stem import SnowballStemmer

# ==============================
# CARGAR MODELO DE SPACY
# ==============================
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo...")
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# ==============================
# LEER ARCHIVO
# ==============================
with open("cap1_principito.txt", "r", encoding="utf-8") as f:
    texto_principito = f.read()

print(f"Texto cargado. Longitud: {len(texto_principito)} caracteres.\n")

# ==============================
# 1. TOKENIZACIÓN
# ==============================
doc = nlp(texto_principito)

print("--- 1. TOKENIZACIÓN ---")
print(f"Total tokens: {len(doc)}")
print([token.text for token in doc][:20])

# ==============================
# 2. FILTRADO DE STOP WORDS
# ==============================
tokens_relevantes = []
tokens_ruido = []

for token in doc:
    if not token.is_stop and not token.is_punct and token.text.strip():
        tokens_relevantes.append(token.text)
    elif token.is_stop:
        tokens_ruido.append(token.text)

print("\n--- 2. FILTRADO ---")
print(f"Ruido: {tokens_ruido[:10]}")
print(f"Contenido: {tokens_relevantes[:10]}")
print(f"Reducción: {len(doc)} -> {len(tokens_relevantes)}")

# ==============================
# 3. LEMATIZACIÓN
# ==============================
tokens_normalizados = []
cambios = []

for token in doc:
    if not token.is_stop and not token.is_punct and token.text.strip():
        lema = token.lemma_.lower()
        tokens_normalizados.append(lema)

        if token.text.lower() != lema:
            cambios.append(f"{token.text} -> {lema}")

print("\n--- 3. LEMATIZACIÓN ---")
print(f"Total tokens finales: {len(tokens_normalizados)}")
print("Ejemplos de cambios:")
print(cambios[:10])

print("\nPrimeros tokens finales:")
print(tokens_normalizados[:10])

# ==============================
# 4. STEMMING vs LEMATIZACIÓN
# ==============================
stemmer = SnowballStemmer("spanish")

data = []

for token in doc:
    if not token.is_punct and not token.is_space:
        stem = stemmer.stem(token.text)
        lema = token.lemma_

        data.append({
            "Original": token.text,
            "Stemming": stem,
            "Lematización": lema,
            "Coinciden": stem == lema
        })

df = pd.DataFrame(data)

print("\n--- 4. COMPARACIÓN ---")

palabras_interesantes = ["hombres", "olvidado", "eres", "domesticado", "invisible", "ojos"]
filtro = df[df["Original"].isin(palabras_interesantes)]

print("\nCasos interesantes:")
print(filtro.to_string(index=False))

print("\nPrimeros 10 tokens:")
print(df.head(10).to_string(index=False))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ==============================
# CREAR CORPUS POR ORACIONES
# ==============================
corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if not token.is_punct and not token.is_space and not token.is_stop
    ]
    
    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"\nTotal de oraciones: {len(corpus_lematizado)}")

# ==============================
# BAG OF WORDS
# ==============================
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

print("\n--- BAG OF WORDS ---")
print(f"Dimensiones: {X_bow.shape}")
print("Vocabulario (primeras 10):")
print(bow_vectorizer.get_feature_names_out()[:10])

# ==============================
# TF-IDF
# ==============================
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

print("\n--- TF-IDF ---")
print(f"Dimensiones: {X_tfidf.shape}")
print("Vocabulario (primeras 10):")
print(tfidf_vectorizer.get_feature_names_out()[:10])

from gensim.models import Word2Vec
import multiprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# PREPARAR DATOS PARA WORD2VEC
# ==============================
sentences = []

for sent in doc.sents:
    tokens = [
        token.lemma_.lower()
        for token in sent
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    if len(tokens) > 1:
        sentences.append(tokens)

print(f"\nOraciones para Word2Vec: {len(sentences)}")

# ==============================
# ENTRENAR MODELO
# ==============================
model_w2v = Word2Vec(
    sentences,
    vector_size=10,
    window=5,
    min_count=1,
    workers=multiprocessing.cpu_count(),
    seed=42
)

# ==============================
# VISUALIZACIÓN 3D
# ==============================
vocabulario = list(model_w2v.wv.index_to_key)
vectores = model_w2v.wv[vocabulario]

pca = PCA(n_components=3)
vectores_3d = pca.fit_transform(vectores)

df = pd.DataFrame(vectores_3d, columns=['x', 'y', 'z'])
df['palabra'] = vocabulario

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['x'], df['y'], df['z'])

for i, row in df.iterrows():
    ax.text(row['x'], row['y'], row['z'], row['palabra'], size=8)

ax.set_title("Word2Vec - Espacio Semántico 3D")

plt.savefig("word2vec_3d.png")
plt.show()