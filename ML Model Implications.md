[[Anime Recommendation System Project Idea]]
**Model Implications — Compiled**

1. Rating threshold of 20 as minimum user rating count for collaborative filtering — justified by 25th percentile sitting at 18
2. Per-user mean centering required before collaborative filtering to correct for systematic mean shift between heavy and casual raters
3. High end rating differences (7 vs 8 vs 9) carry more signal than low end differences given compressed effective scale — similarity calculations should account for this
4. No genre based segmentation needed between heavy and casual raters — a single content model with mean centered ratings is sufficient across both segments
5. No concentration based weighting needed in collaborative filtering — rating distribution is sufficiently broad across userbase
6. Anime below 1,000 members should be excluded from collaborative filtering but remain eligible for content based recommendations — metadata signal is still valid even when rating signal is not
7. Popularity penalty required — 1.7 point systematic score inflation exists between niche and popular tiers, bias correction layer must discount popular anime scores
8. Bayesian average or damped mean scoring should be considered for anime with low member counts to correct for small sample noise
9. Genre recommendation weighting should incorporate both median and total member count as a combined parameter
10. Bias correction layer should apply genre specific boosting for Josei, Drama, Police, Historical and Mystery whenever user history shows any affinity signal