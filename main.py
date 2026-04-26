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