import json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

print(
    "⏳ Démarrage du serveur... Chargement du modèle d'IA sémantique en Python"
    " (patientez ~20s)..."
)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Modèle d'IA et serveur prêts sur http://127.0.0.1:5000 !")


@app.route("/api/recommendations/<mode>", methods=["POST"])
def api_recommendations(mode):
  req_data = request.get_json()
  library = req_data.get("library", [])

  if not library:
    return jsonify([])

  df = pd.DataFrame(library)

  # Nettoyage
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

  # Sélection du moteur selon le bouton cliqué sur le site
  if mode == "ai":
    print("🤖 Exécution du calcul IA lourd (Sémantique)...")
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
    embeddings = model.encode(df["rich_text"].tolist(), show_progress_bar=False)
    cosine_sim = cosine_similarity(embeddings, embeddings)
  else:
    print("🛠️ Exécution du calcul Manuel (TF-IDF)...")
    df["combined_features"] = (
        df["genres_str"] + " " + df["genres_str"] + " " + df["director"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    count_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

  results = []
  for idx, row in df.iterrows():
    if row["status"] != "À voir":
      continue
    total_score = 0
    best_match, max_sim = "", -1

    for _, rated_row in rated_items.iterrows():
      sim = cosine_sim[idx][rated_row.name]
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
    note_est = min(9.9, max(4.0, round(row["rating"] + (final_score * 0.1), 1)))

    results.append({
        "tmdbId": row["tmdbId"],
        "mediaType": row["mediaType"],
        "title": row["title"],
        "year": row["year"],
        "rating": row["rating"],
        "posterPath": row["posterPath"],
        "score": final_score,
        "estimatedRating": note_est,
        "match": best_match,
    })

  top_recos = (
      pd.DataFrame(results)
      .sort_values(by="score", ascending=False)
      .head(5)
      .to_dict(orient="records")
  )
  return jsonify(top_recos)


if __name__ == "__main__":
  app.run(port=5000)