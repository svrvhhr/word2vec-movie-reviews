# Word2Vec avec Gensim

## Introduction
Dans le cadre d'un travail pratique (TP) sur le traitement automatique du langage naturel (NLP). Word2Vec est un modèle d'apprentissage non supervisé utilisé pour apprendre des représentations vectorielles des mots à partir d'un corpus de texte.

## Objectifs
L'objectif principal de ce TP est de former un modèle Word2Vec sur un corpus de critiques de films et d'explorer les relations sémantiques entre les mots.

## 1. Configuration et Importation des bibliothèques
Avant d'entraîner Word2Vec, nous avons importé les bibliothèques essentielles :

```python
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
```

## 2. Préparation des données

### Chargement et prétraitement du corpus
- Le corpus est constitué de critiques de films.
- Tokenisation des phrases en mots avec `nltk.word_tokenize`.
- Suppression des stopwords et ponctuations.

```python
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Minuscule
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Nettoyage
    return tokens
```

## 3. Entraînement du modèle Word2Vec
Le modèle est entraîné en utilisant la méthode `Word2Vec` de Gensim :

```python
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, sg=1 , negative =15 )
```

### Paramètres utilisés
- `vector_size=100` : Chaque mot est représenté par un vecteur de 100 dimensions.
- `window=5` : La fenêtre contextuelle est de 5 mots autour du mot cible.
- `min_count=2` : Seuls les mots apparaissant au moins 2 fois sont pris en compte.
- `workers=4` : Utilisation de 4 cœurs pour le traitement parallèle.
- `epochs=10` : Nombre de passages sur l'ensemble du corpus (epochs) pour un meilleur apprentissage des représentations.

### Lancement de l'entraînement
L'entraînement du modèle est réalisé avec :

```python
model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
```

Cette étape permet au modèle d'ajuster les vecteurs de mots en fonction du contexte dans lequel ils apparaissent.

## 4. Analyse des résultats

### Similarité entre les mots
Une fois le modèle entraîné, nous pouvons explorer les relations entre les mots :

```python
print(model.wv.most_similar('good'))
```

Ce code retourne les mots les plus similaires à "good" selon le modèle entraîné.

### Visualisation des vecteurs de mots
Nous utilisons la réduction de dimension avec t-SNE pour représenter les vecteurs dans un espace bidimensionnel :

```python
from sklearn.manifold import TSNE
import numpy as np

def plot_words(model, words):
    word_vectors = np.array([model.wv[word] for word in words])
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
    plt.show()
```

## 5. Conclusion
Ce TP nous a permis de :
- Comprendre l'utilisation de Word2Vec pour l'apprentissage de représentations de mots.
- Observer les relations sémantiques entre les mots.
- Visualiser les représentations vectorielles avec t-SNE.
- Expérimenter l'impact du nombre d'epochs sur la qualité des embeddings appris.

L'utilisation de Word2Vec est particulièrement utile pour des tâches telles que l'analyse des sentiments, la classification de documents et le résumé automatique.