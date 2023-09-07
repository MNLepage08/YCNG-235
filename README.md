# YCNG-235: Recommender Systems

## :mortar_board: Courses

| # | Sessions | 
| ------------- | ------------- |
| 1 | Introduction to recommender systems. Popularity-based recommender systems.|
| 2 | Content-based filtering: How to leverage the state-of-the-arts NLP models to build recommender systems? |
| 3 | Nearest neighbor collaborative filtering: Distance-based methods for fun and profit. |
| 4 | Matrix factorization in collaborative-filtering - Part 1 |
| 5 | Matrix factorization in collaborative-filtering - Part 2 |
| 6 | Deep learning for collaborative-filtering. |
| 7 | Visual similarity-based recommendation system; Contrastive Language-Image Pretraining (CLIP) for recom- mendation systems. |
| 8 | Serverless recommendation engine on AWS. |
| 9 | Deploying a recommendation model using Docker and Streamlit. |
| 10 | Review lectures and projects presentation. |


## :pencil2: Notes

<details close>
<summary>Introduction to recommender systems. Popularity-based recommender systems.<p></summary>

* **Recommendation systems:** Algorithms designed to suggest relevant items (articles, clothes, songs, videos, etc.) to users based on many different factors. In many industries, such as e-commerce, the usage of recommendation systems can generate a huge amount of revenue.<p>
  **1. Content-based filtering:** It is mainly based on the principle of similar contents. It creates a profile for a user or an item to characterize its nature. Content: Text, Image. Movie recommendation:
  1. Item profile: genre, actors, director, creation date, description, etc.
  2. User profile: age, gender, occupation, zip code, etc.
  
  **2. Collaborative filtering:** Nowadays, many of recommendation systems use collaborative filtering in some form. It uses similarities between users and items to make recommendations. This is done by collecting preferences from many users without requiring the creation of explicit profiles. Key assumption: If Tom and Bob share similar preference over a set of items, Bob is more likely to share Tom’s taste on a novel item than any randomly chosen user.
  1. Use-Case 1: Recommend most relevant items per user.<br><img width="495" alt="Capture d’écran, le 2023-09-07 à 11 14 43" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/ff6c1e8c-ec65-4616-9424-4badc80e9508">
  2. Use-Case 2: item-to-item recommendation.<br><img width="530" alt="Capture d’écran, le 2023-09-07 à 11 14 51" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/92d31e7e-0acb-43e5-81c6-390688b4e338">
  3. Use-Case 3: relevant users to a specific item.<br><img width="402" alt="Capture d’écran, le 2023-09-07 à 11 15 01" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/7653da2e-c529-42e9-902b-24791642df95">

  **3. Hybrid**<br><img width="554" alt="Capture d’écran, le 2023-09-07 à 11 21 37" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/cdfba381-4dd0-45f0-8a36-4c2d0e84bac5">

* **Popularity-based:** If a product is usually purchased or a video is frequently viewed by Canadian-based users, it can be suggested to any new user from Canada. It is a generic recommendation algorithm. It can be used to address user cold-start problem in collaborative filtering. [Last.fm dataset](https://www.kaggle.com/datasets/neferfufi/lastfm)

* **Main steps of Content-Based Filtering:**
  1. Feature extraction: convert text/image into numerical vectors.
  2. Compute the cosine similarity of a given item and any other items in the dataset.
  3. Pick items with the greatest cosine similarity (top-N).
 
* **Information Retrieval:** Term Frequency-Inverse Document Frequency (TF-IDF). It is often used in various natural language processing tasks, including text classification, information retrieval, and document similarity analysis. It helps in identifying important terms and reducing the impact of common and less informative words in text data. It calculates how relevant a word (t) is to a document (d) in a corpus.<br>
  **Term frequency:** frequency (count) of words in a document.<br>
  **Inverse document frequency:** how common is a word in a document.<br>

  **Example:** Review: this song is great, and it is sad. Number of words: 8. Suppose that we have 3 phrases (documents). IDF: log(# documents d / 3 documents with term t)<br>
  <img width="314" alt="Capture d’écran, le 2023-09-07 à 13 15 09" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/c1251975-d7bf-430e-8a79-52442e6ddad4"><br>
  Higher TF-IDF scores indicate that a term is both frequent in a particular document (TF) and rare across the entire collection (IDF), making it a potentially important and distinctive term for that document.
  

</details> 


## :books: Bibliography

| <img width="249" alt="Capture d’écran, le 2023-09-06 à 21 58 08" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/53368a31-8a49-4df3-a7f0-1d8b3b806cd3"> | 
| :-------------: | 
| Hands-On Recommendation Systems with Python | 
