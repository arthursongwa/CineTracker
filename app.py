import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement de ta base de données
with open("movies.json", "r", encoding="utf-8") as f:
  data = json.load(f)

df = pd.DataFrame(data["library"])

# Nettoyage de base
df["userRating"] = df["userRating"].fillna(0)
df["rating"] = df["rating"].fillna(5.0)
df["userNotes"] = df["userNotes"].fillna("")
df["overview"] = df["overview"].fillna("")
df["genres_str"] = df["genres"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
)
df["director"] = df["director"].fillna("")

rated_items = df[df["userRating"] > 0]
if rated_items.empty:
  rated_items = df[df["rating"] >= 7.5]


# ==========================================
# MÉTHODE 1 : L'APPROCHE MANUELLE (TF-IDF)
# ==========================================
print("Exécution de l'algorithme Manuel (Mots-clés / Genres / Réalisateurs)...")
df["combined_features"] = (
    df["genres_str"] + " " + df["genres_str"] + " " + df["director"]
)
tfidf = TfidfVectorizer(stop_words="english")
count_matrix = tfidf.fit_transform(df["combined_features"])
cosine_sim_manual = cosine_similarity(count_matrix, count_matrix)

manual_results = []
for idx, row in df.iterrows():
  if row["status"] != "À voir":
    continue
  total_score = 0
  best_match, max_sim = "", -1

  for _, rated_row in rated_items.iterrows():
    sim = cosine_sim_manual[idx][rated_row.name]
    note = rated_row["userRating"]
    if sim > max_sim:
      max_sim = sim
      best_match = rated_row["title"]

    if note == 5:
      total_score += sim * 4.0
    elif note == 4:
      total_score += sim * 2.0
    elif note == 3:
      total_score += sim * 0.5
    elif note == 2:
      total_score -= sim * 2.0
    elif note == 1:
      total_score -= sim * 4.0

  final_score = total_score + ((row["rating"] / 10.0) * 0.2)
  manual_results.append({
      "title": row["title"],
      "mediaType": row["mediaType"],
      "year": row["year"],
      "rating": row["rating"],
      "score": final_score,
      "match": best_match,
  })

df_manual = (
    pd.DataFrame(manual_results).sort_values(by="score", ascending=False).head(5)
)


# ==========================================
# MÉTHODE 2 : L'APPROCHE IA LOCALE (SÉMANTIQUE)
# ==========================================
print("Chargement du modèle d'IA pour l'approche Sémantique...")
df["rich_text"] = (
    df["title"]
    + " - Genres: "
    + df["genres_str"]
    + " - Réalisateur: "
    + df["director"]
    + " - Résumé: "
    + df["overview"]
    + " - Mes notes perso: "
    + df["userNotes"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["rich_text"].tolist(), show_progress_bar=False)
cosine_sim_ai = cosine_similarity(embeddings, embeddings)

ai_results = []
for idx, row in df.iterrows():
  if row["status"] != "À voir":
    continue
  total_score = 0
  best_match, max_sim = "", -1

  for _, rated_row in rated_items.iterrows():
    sim = cosine_sim_ai[idx][rated_row.name]
    note = rated_row["userRating"]
    if sim > max_sim:
      max_sim = sim
      best_match = rated_row["title"]

    if note == 5:
      total_score += sim * 4.0
    elif note == 4:
      total_score += sim * 2.0
    elif note == 3:
      total_score += sim * 0.5
    elif note == 2:
      total_score -= sim * 2.0
    elif note == 1:
      total_score -= sim * 4.0

  final_score = total_score + ((row["rating"] / 10.0) * 0.2)
  ai_results.append({
      "title": row["title"],
      "mediaType": row["mediaType"],
      "year": row["year"],
      "rating": row["rating"],
      "score": final_score,
      "match": best_match,
  })

df_ai = pd.DataFrame(ai_results).sort_values(by="score", ascending=False).head(5)


# ==========================================
# AFFICHAGE COMPARATIF
# ==========================================
print("\n" + "=" * 70)
print(" 🔬 COMPARATIF DES RECOMMANDATIONS : MÉTHODE MANUELLE VS IA")
print("=" * 70)

print("\n--- 🛠️ TOP 5 : MÉTHODE MANUELLE (Mots-clés, Genres, Réalisateurs) ---")
for _, row in df_manual.iterrows():
  m_type = "Série" if row["mediaType"] != "movie" else "Film"
  note_est = min(9.9, max(4.0, round(row["rating"] + (row["score"] * 0.1), 1)))
  print(
      f"• [{m_type}] {row['title']} ({row['year']}) | Note estimée : {note_est}/10"
      f" (Lié à '{row['match']}')"
  )

print("\n--- 🤖 TOP 5 : MÉTHODE IA (Sémantique, Résumés & Notes persos) ---")
for _, row in df_ai.iterrows():
  m_type = "Série" if row["mediaType"] != "movie" else "Film"
  note_est = min(9.9, max(4.0, round(row["rating"] + (row["score"] * 0.1), 1)))
  print(
      f"• [{m_type}] {row['title']} ({row['year']}) | Note estimée : {note_est}/10"
      f" (Proche de '{row['match']}')"
  )

print("\n" + "=" * 70)