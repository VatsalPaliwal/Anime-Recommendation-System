import streamlit as st
import pickle
import pandas as pd
from rapidfuzz import process
import ast
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title="AniList Recommendation System", layout="wide")
# Load saved files
with open(r'C:\Users\LENOVO\Desktop\code\project ideas\anime recommendation system\Notebook\model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(r'C:\Users\LENOVO\Desktop\code\project ideas\anime recommendation system\Notebook\similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

rating = pd.read_csv(r"C:\Users\LENOVO\Desktop\code\project ideas\anime recommendation system\Notebook\user_data.csv")
name = pd.read_csv(r"C:\Users\LENOVO\Desktop\code\project ideas\anime recommendation system\Notebook\anime_data.csv")

name['genre_list'] = name['genre_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def get_recommendations(anime_title, top_n=10):
    anime_index = name[(name['name'].str.lower()) == anime_title.lower()].index[0]
    top_anime = name['members'].quantile(0.90)
    similar = list(enumerate(similarity_matrix[anime_index]))
    similar_sorted = sorted(similar, key=lambda x: x[1], reverse=True)
    penalized = []
    for idx, score in similar_sorted:
        members = name['members'].iloc[idx]
        if members > top_anime:
            score -= 0.3
        penalized.append((idx, score))
    penalized = sorted(penalized, key=lambda x: x[1], reverse=True)
    selected = []
    genre_count = {}
    for idx, score in penalized:
        if idx == anime_index:
            continue
        if name['baysian_rating'].iloc[idx] < 6.5:
            continue
        genres = name['genre_list'].iloc[idx]
        allow = True
        for g in genres:
            if genre_count.get(g, 0) >= 4:
                allow = False
                break
        if not allow:
            continue
        selected.append(idx)
        for g in genres:
            genre_count[g] = genre_count.get(g, 0) + 1
        if len(selected) == top_n:
            break
    if len(selected) < top_n:
        for idx, score in penalized:
            if idx == anime_index or idx in selected:
                continue
            if name['baysian_rating'].iloc[idx] >= 6.5:
                selected.append(idx)
            if len(selected) == top_n:
                break
    return name['name'].iloc[selected].tolist()


def get_collaborative_recommendations(user_id, top_n=10):
    watched = rating[rating['user_id'] == user_id]['anime_id'].tolist()
    not_watched = name[~name['anime_id'].isin(watched)]['anime_id'].tolist()
    top_anime = name['members'].quantile(0.90)
    scores = []
    for i in not_watched:
        pred = model.predict(user_id, i)
        scores.append(pred)
    penalized = []
    for pred in scores:
        est = pred.est
        if name[name['anime_id'] == int(pred.iid)]['members'].values[0] > top_anime:
            est -= 0.5
        penalized.append((pred.iid, est))
    similar_sorted = sorted(penalized, key=lambda x: x[1], reverse=True)
    top_scores = similar_sorted[:top_n]
    recommended_anime_ids = [int(iid) for iid, est in top_scores]
    genre_freq = {}
    for anime_id in recommended_anime_ids:
        genre = name[name['anime_id'] == anime_id]['genre_list'].values[0][0]
        genre_freq[genre] = genre_freq.get(genre, 0) + 1
    dominant_genres = {g: c for g, c in genre_freq.items() if c > 4}
    if dominant_genres:
        to_remove = []
        genre_seen = {}
        for anime_id in recommended_anime_ids:
            genre = name[name['anime_id'] == anime_id]['genre_list'].values[0][0]
            if genre in dominant_genres:
                genre_seen[genre] = genre_seen.get(genre, 0) + 1
                if genre_seen[genre] > 4:
                    to_remove.append(anime_id)
        for anime_id in to_remove:
            recommended_anime_ids.remove(anime_id)
        substitutes = name[
            (~name['anime_id'].isin(recommended_anime_ids)) &
            (~name['anime_id'].isin(watched)) &
            (name['baysian_rating'] >= 6.5) &
            (~name['genre_list'].apply(lambda x: x[0] if x else None).isin(dominant_genres))
        ].sort_values('baysian_rating', ascending=False)
        substitute_ids = substitutes['anime_id'].tolist()[:len(to_remove)]
        recommended_anime_ids.extend(substitute_ids)
    result = name.set_index('anime_id').loc[recommended_anime_ids]['name'].tolist()
    return result


def get_hybrid_recommendations(user_id):
    user_rating_count = rating[rating['user_id'] == user_id].shape[0]
    top_anime_title = name[name['anime_id'].isin(
        rating[rating['user_id'] == user_id].nlargest(1, 'rating')['anime_id']
    )]['name'].values[0]
    if user_rating_count < 20:
        content_results = get_recommendations(top_anime_title)
        final_scores = {anime: (10 - i) for i, anime in enumerate(content_results)}
    elif 20 <= user_rating_count <= 57:
        collab_results = get_collaborative_recommendations(user_id)
        content_results = get_recommendations(top_anime_title)
        collab_scores = {anime: (10 - i) for i, anime in enumerate(collab_results)}
        content_scores = {anime: (10 - i) for i, anime in enumerate(content_results)}
        all_anime = set(collab_scores.keys()) | set(content_scores.keys())
        final_scores = {anime: 0.7 * collab_scores.get(anime, 0) +
                       0.3 * content_scores.get(anime, 0) for anime in all_anime}
    else:
        collab_results = get_collaborative_recommendations(user_id)
        content_results = get_recommendations(top_anime_title)
        collab_scores = {anime: (10 - i) for i, anime in enumerate(collab_results)}
        content_scores = {anime: (10 - i) for i, anime in enumerate(content_results)}
        all_anime = set(collab_scores.keys()) | set(content_scores.keys())
        final_scores = {anime: 0.9 * collab_scores.get(anime, 0) +
                       0.1 * content_scores.get(anime, 0) for anime in all_anime}
    final_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return [anime for anime, score in final_recommendations]
def display_cards(anime_list):

    anime_df = name.set_index('name').loc[anime_list].reset_index()

    html = """
    <div style="
        display: flex;
        gap: 20px;
        padding: 10px;
        flex-wrap: wrap;
    ">
    """

    for _, row in anime_df.iterrows():

        img_url = get_anime_image(row['name'])

        html += f"""
        <div style="
            width: 180px;
            text-align: center;
            font-family: sans-serif;
        ">
            <img src="{img_url}"
                 width="180"
                 style="border-radius:10px;">
            
            <p style="margin: 6px 0; font-weight: bold; font-size:14px; color:white;">
                {row['name'][:30]}
            </p>
            
            <p style="margin: 0; font-size:12px; color:#facc15;">
                ⭐ {round(row.get('baysian_rating', 0), 2)}
            </p>
        
        </div>
        """

    html += "</div>"

    components.html(html, height=600, scrolling=True)
@st.cache_data
def get_anime_image(anime_name):
    
    url = "https://graphql.anilist.co"

    query = """
    query ($search: String) {
      Media(search: $search, type: ANIME) {
        coverImage {
          large
        }
      }
    }
    """

    variables = {"search": anime_name}

    try:
        response = requests.post(url, json={
            "query": query,
            "variables": variables
        })

        data = response.json()
        return data['data']['Media']['coverImage']['large']

    except:
        return "https://via.placeholder.com/150"
# App header
st.title("AniList Recommendation System")
st.markdown("*Correcting popularity bias to surface anime you'll actually love*")

tab1, tab2 = st.tabs(["Find Similar Anime", "Personalized Recommendations"])

with tab1:
    st.header("Find Similar Anime")
    anime_input = st.text_input("Enter an anime you liked")
    if anime_input:
        matched = process.extractOne(anime_input, name['name'].tolist())
        matched_title = matched[0]
        st.write(f"Showing results for: **{matched_title}**")
        results = get_recommendations(matched_title)
        display_cards(results)

with tab2:
    st.header("Personalized Recommendations")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    if st.button("Get Recommendations"):
        user_rating_count = rating[rating['user_id'] == user_id].shape[0]
        if user_rating_count == 0:
            st.error("User ID not found in dataset")
        else:
            if user_rating_count < 20:
                tier = "New User — Content Based"
            elif user_rating_count <= 57:
                tier = "Regular User — Hybrid"
            else:
                tier = "Power User — Collaborative"
            st.info(f"User tier: {tier} ({user_rating_count} ratings)")
            results = get_hybrid_recommendations(user_id)
            display_cards(results)