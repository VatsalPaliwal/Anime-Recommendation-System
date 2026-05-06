# Anime Recommendation System
### Correcting Popularity Bias to Surface Anime You'll Actually Love

---

## Overview

Standard recommendation systems are popularity machines. They surface the same titles repeatedly — not because they're the best fit for a user, but because accumulated attention generates stronger signals. This project investigates that structural bias in anime recommendation data and builds a hybrid system designed to correct it.

Built on the MyAnimeList dataset, this system combines content-based filtering, collaborative filtering, and a custom bias correction layer to deliver recommendations that are both personalized and genuinely diverse.

---

## The Problem

The data tells a clear story:

- The **top 1% of anime** (123 titles) absorb **23.4% of all rating activity**
- **43% of the catalog** (5,297 anime) fall below the minimum member threshold for statistical reliability — effectively invisible to standard recommenders
- Average ratings increase progressively across popularity tiers — niche anime average **5.69**, mid-tier **6.14**, popular anime **7.38** — a 1.7 point gap confirmed by both mean and median
- A **Pearson correlation of 0.39** between member count and average rating confirms this relationship is systematic, not incidental
- Within MyAnimeList's compressed rating culture where scores below 6 are considered poor by platform standards, this gap places niche anime in below-average territory by default — not because of quality, but because of exposure

Genres like **Josei, Drama, Police, Historical, and Mystery** show average ratings above 6.1 with median member counts under 120 — passionate audiences that a standard recommender would never surface.

---

## Key Findings from EDA

**User Behavior**
- 75% of users have rated at least 18 anime, median user has rated 57 — sufficient signal for collaborative filtering across most of the platform
- Heavy raters average 5.9 vs 6.7 for casual raters — a calibration gap requiring per-user mean centering before modeling
- Rating variance is identical across both segments (13.77) — the difference is purely a mean shift, not behavioral
- Top 1% of users contribute 9.5% of total ratings — no power user distortion risk

**Catalog**
- Top 1% of anime absorb 23.4% of all rating activity — more than double the user-side concentration
- 43% of the catalog falls below the 1,000 member reliability threshold
- Popular anime converge into a stable 7.5–9.5 rating band while niche anime show extreme score variance — large audiences create an illusion of consistent quality

**Popularity Bias**
- Niche 5.69 → Mid 6.14 → Popular 7.38 — progressive score inflation confirmed by median
- Pearson correlation of 0.39 between member count and average rating
- Within the platform's compressed scale, this is the difference between a good anime and a poor one

**Genre**
- Catalog is genuinely diverse across 43 genres — no sharp concentration cliff
- Common genres are not the same as popular ones — Comedy dominates by count (12.8%) but has a median of only 2,745 members per title
- Thriller appears rarely but commands 55,566 median members — a consistently engaged audience
- Josei, Drama, Police, Historical, Mystery: high ratings, low visibility, underserved audiences

---

## System Architecture

```
User Input
    │
    ├── Anime Title → Content Based Filtering
    │       │
    │       ├── One-hot encoded genre similarity (cosine)
    │       ├── Bayesian adjusted rating filter (≥ 6.5)
    │       ├── Popularity penalty (top 10% by members)
    │       └── Genre diversity enforcement (max 40% per genre)
    │
    └── User ID → Hybrid Recommendations
            │
            ├── < 20 ratings   → Pure Content Based (α = 0)
            ├── 20–57 ratings  → Weighted Hybrid (α = 0.7 collaborative)
            └── 57+ ratings    → Collaborative Dominant (α = 0.9)
                    │
                    └── SVD Matrix Factorization (Surprise library)
                            ├── Popularity penalty
                            └── Genre diversity enforcement
```

---

## Bias Correction Layer

Two mechanisms applied as post-processing on all recommendation outputs:

**Popularity Penalty**
Anime in the top 10% by member count receive a score penalty before final ranking. This prevents SVD's learned patterns — which are biased toward popular titles — from dominating outputs.

**Genre Diversity Enforcement**
If any single genre exceeds 40% of the recommendation list, the lowest-ranked excess entries are replaced with highly-rated alternatives from underrepresented genres. Ensures recommendations surface the full spectrum of a user's potential interests.

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn
- **Surprise** — SVD collaborative filtering
- **Streamlit** — Interactive web interface
- **RapidFuzz** — Fuzzy anime title matching
- **Matplotlib / Seaborn** — EDA visualizations

---

## Project Structure

```
anime-recommendation-system/
├── Notebook/
│   ├── Phase1_Data_Cleaning.ipynb
│   ├── Phase2_EDA.ipynb
│   ├── Phase3_Content_Based_Filtering.ipynb
│   ├── Phase4_Collaborative_Filtering.ipynb
│   └── Phase5_6_Hybrid_System.ipynb
├── app.py
├── model.pkl
├── similarity_matrix.pkl
├── anime_data.csv
├── user_data.csv
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn scikit-surprise streamlit rapidfuzz

# Run the app
streamlit run app.py
```

---

## Evaluation

| Metric | Value |
|--------|-------|
| RMSE (SVD baseline) | 1.13 |
| Effective rating scale | 7–10 (compressed) |
| Catalog visibility threshold | 1,000 members |
| User rating threshold | 20 ratings |

**Note on RMSE:** RMSE measures prediction accuracy, not recommendation usefulness. A system can achieve low RMSE while still recommending obvious, popular titles. This project additionally optimizes for catalog coverage and intra-list genre diversity — metrics that better reflect real recommendation quality.

---

## Limitations

- Dataset does not contain release dates — recency analysis was not possible
- User IDs are tied to the MyAnimeList dataset — live AniList API integration is a planned enhancement
- Collaborative filtering requires a minimum of 20 ratings per user — new users receive content-based recommendations only
- Genre ordering within the genre_list column is undocumented — primary genre inference uses first listed genre as a simplification

---

## Future Enhancements

- AniList API integration for live user watch history
- Anime cover poster display via AniList GraphQL API
- Session-based feedback loop — thumbs up/down to refine recommendations
- Temporal weighting — recent ratings weighted more heavily than older ones
- Neural collaborative filtering to replace SVD

---

*Dataset: MyAnimeList Dataset — Kaggle*
