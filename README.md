# YCNG-235: Recommender Systems

## :mortar_board: Courses

| # | Sessions | 
| ------------- | ------------- |
| 1 | Introduction to recommender systems. Popularity-based recommender systems.|
| 2 | Content-based filtering: How to leverage the state-of-the-arts NLP models to build recommender systems?|
| 3 | Nearest neighbor collaborative filtering: Distance-based methods for fun and profit. |
| 4 | Matrix factorization in collaborative-filtering - Part 1 |
| 5 | Matrix factorization in collaborative-filtering - Part 2 |
| 6 | Deep learning for collaborative-filtering. |
| 7 | Visual similarity-based recommendation system; Contrastive Language-Image Pretraining (CLIP) for recommendation systems. |
| 8 | Serverless recommendation engine on AWS. |
| 9 | Deploying a recommendation model using Docker and Streamlit. |
| 10 | Review lectures and projects presentation. |


## :pencil2: Notes

<details close>
<summary>1. Introduction to recommender systems. Popularity-based recommender systems.<p></summary>

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

</details> 


<details close>
<summary>2. Content-based filtering: How to leverage the state-of-the-arts NLP models to build recommender systems?<p></summary>

* **Content-Based Filtering:** It is mainly based on the principle of similar contents. It creates a profile for a user or an item to characterize its nature. 

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

* [Transformer Architecture:](https://arxiv.org/abs/1706.03762) has had a profound influence on the field of natural language processing (NLP) and various sequence-to-sequence tasks. It serves as the fundamental building block for numerous cutting-edge NLP models such as BERT, GPT, and more.<br>

  **Self-Attention Mechanism:** The core innovation of the Transformer is the self-attention mechanism. It allows the model to weigh the importance of different words in a sentence when encoding or decoding it. This enables the model to consider the context of each word in relation to all other words, regardless of their position in the sequence.<br>

  **Multi-Head Attention:** This means they perform self-attention multiple times in parallel, each time with different learned weights. This allows the model to focus on different parts of the input sequence for different tasks.<br>

  **Positional Encoding:** Since Transformers do not have a built-in understanding of word order or sentence structure like RRNs and CNNs, positional encoding is necessary to inject this sequential information into the model.

  **Encoder-Decoder Architecture:** In sequence-to- sequence tasks like machine translation, the Transformer uses an encoder-decoder architecture. The encoder processes the input sequence, while the decoder generates the output sequence. Both encoder and decoder consist of stacks of layers, each containing multi-head self-attention mechanisms and feedforward neural networks.

  **Masking**: In tasks like language modeling, a masking mechanism is used to ensure that the model attends only to previous positions and not future positions in the input sequence (Cheating proof masking).

* [Sentence Transformers:](https://www.sbert.net) Collection of several state-of-the-art pre-trained NLP models. It offers an easy approach to generate dense vector representations for sentences, paragraphs, and images. These models are built upon transformer networks. They are fine-tuned for various use-cases including semantic search, clustering, translated sentence mining, etc. 
  [GitHub](https://github.com/UKPLab/sentence-transformers), 
  [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)

* [Universal Sentence Encoder:](https://tfhub.dev) One of Google models for sentence encoding. It summarizes any given sentence to 512-dimensional sentence embedding resulting in a generic sentence embedding that transfers universally to wide variety of NLP tasks. Encoding architectures: Deep Average Network(DAN), Transformer Encoder. [Paper](https://arxiv.org/abs/1803.11175)<br>

  **Cosine Similarity:** <br>
  <img width="297" alt="Capture d’écran, le 2023-09-07 à 14 18 48" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/ef54155c-1b9e-4d21-adc3-776db3aca378">

</details> 


<details close>
<summary>3. Nearest neighbor collaborative filtering: Distance-based methods for fun and profit.<p></summary>

<img width="300" align="right" alt="Capture d’écran, le 2023-09-22 à 10 50 34" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/d5b3a0a0-c8f7-486b-b42d-b14527e638cd">

* **Collaborative Filtering:** It uses similarities between users and items to make recommendations. The information collected from other users is used to recommend new items to the current user. It works for any kind of problem. Howeverm it suffers from cold-start problem (new useres have noo historym new items have no ratings). Solve cold-start problem by if then statement with example content-based & collaborative filtering or with hybrid system.
  
* **Formal model:** U: set of users. I: set of items. R: set or ratings (explicit or implicit). Utility finction: U x I -> R. It is potentially a large-scale matrix where users and items are rows and columns of the matrix. It is a super sparce matrix.
  
* **Scoring in collaborative filtering:**
    * Most convenient is to use high quality explicit feedback (like / dislike buttons in YouTube videos).
    * In many cases explicit feedback is not available, which requires using implicit feedback (purchase history, browsing history, search patterns).
    * Create a common currency of liked items (explicit / implicit feedback).
 
* **Collaborative filtering variants:**
    * Item-based: similarities between items in the training dataset is calculated.
    * User-based: similarities between users in the training dataset is calculated.
    * Item-based variant is prefered over user-based (more scalable, stable over time -> keep embeddings).
    * User-oriented is not usually easy to scale given the dynamic preference of users.

* **Nearest neighbors:** The main steps
    * **Step 1:** Create nearest neighbors for items using a similarity measure (e.g. cosine similarity).
       1. Compute similarity ($S_{ij}$) between item i.<br>
          <img width="458" alt="Capture d’écran, le 2023-09-22 à 11 30 04" src="https://github.com/MNLepage08/YCNG-288-DevOps/assets/113123425/c038c525-18fb-4724-b65a-9b2922b6b6c7">
       2. Identify k items (neighbors) rated by user u taht are most similar to item i by using the similarity measure. Let denote them by $S^k(i, u)$.
    * **Step 2:** Compute the weighted average for each item.<br>
      <img width="200" alt="Capture d’écran, le 2023-09-22 à 11 39 08" src="https://github.com/MNLepage08/YCNG-288-DevOps/assets/113123425/cc7dde0c-472b-435e-967a-355db0a5edef">
 

</details> 


<details close>
<summary>4. Matrix factorization in collaborative-filtering - Part 1<p></summary>

* **Matrix factorization:** We can represent user-item interactions with a low dimensional latent space of features. The model predicts users preferences of unseen items. We can guess what people like, but don't know what they don't like (implicit feedback). Prediction: <img width="82" alt="image" src="https://github.com/MNLepage08/YCNG-235/assets/113123425/4495adcd-05a7-4a9d-b1ca-1c35149b8877"><br>
  <img width="524" alt="Capture d’écran, le 2023-09-28 à 11 37 53" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/7888f1f6-2830-45fb-b158-0b83cdcaea08"><br>
  where n = number of users, m = number of items, k = is our embedding dimension (latent features). The objective is to estimate the matrix R.<br>
  <img width="766" alt="Capture d’écran, le 2023-09-28 à 11 46 30" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/9cbc4af9-3ea2-41f7-9235-45b62efd8191">

* **Use-case:** (see in note 1)<br>
  * Recommend most relevant items per user: personalization items for users.
  * Item-to-item recommendation: non-personalization recommendation (list of similar item).
  * Suggest relevant users to a specific item: Create retention marketing campaign and target the users that we think they would be interested into those 500 new arrivals.
 
* [Alternating Least Square (ALS):](http://yifanhu.net/PUB/cf.pdf) Collaborative Filtering for Implicit Feedback Datasets. The cost function contains m by n terms, where m is the number of users and n is the number of items.<br>
<img width="541" alt="Capture d’écran, le 2023-09-28 à 12 45 03" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/61442a30-20c7-4481-a1ec-95a9abe82eed"><p>
  <img width="138" align="left" alt="Capture d’écran, le 2023-09-28 à 13 00 51" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/b0aabfbd-e9c3-47ee-9786-2cbf4e1fc42d">
  In toher words, if a user u consumed item i ($r_{ui}$ > 0), then we have an indication that u likes i ($p_{ui}$ = 1). On the other hand, if u never consumed i, we believe no preference.<br>
<img width="235" align="left" alt="Capture d’écran, le 2023-09-28 à 13 01 00" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/65ce9ebd-276b-42ac-aba3-0133867f7e0c"> c is the level of confidence under the user preference. We can use linear or logarithmic. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img width="423" align="right" alt="Capture d’écran, le 2023-09-28 à 13 01 08" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/bdef5f4f-c71e-429b-b839-83f244f3d3f8"><br><br><br><br>
 ALS minimizes two loss functions alternatively:<br>
   * It first holds user-factors fixed and runs gradient descent with item-factors;
   * Then it hods item-factors fixed and runs gradient descent with user-factors.<br>
Prediction (recommendation): <img width="105" alt="Capture d’écran, le 2023-09-28 à 14 49 02" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/59412344-a3e1-416b-b0d5-1e71782f9f11">

* **Evaluation: MAP@k or NDCG**
  * Precision: fraction of relevant recommended items: <br><img width="200" alt="Capture d’écran, le 2023-09-28 à 14 59 53" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/113338da-e174-406e-9303-474dd2ad2339">Precision: 60%
  * Precision at k (P@k): fraction of relevant items in top k recommendations: P@1:0, P@2:0.5, P@3:0.33, P@4:0.5, P@5:0.6
  * Average Precision at k (AP@k): sum of P@k for different values of k divided by the total number of relevant recommendations in top k results. (For one user)
  * Mean Average Precision at k (MAP@k): the average P@k which average over the entire dataset. (For all users)

* **Approximate ALS:** Naïve approach: ranking every single item for every single user. Speed up generating. Recommendations using approximate nearest neighbor libraries (NMSLIB, Annoy, Faiss). Risk: potential missing of relevant results. [Approximate Nearest Neighbours for Recommender Systems](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/).

</details>


## :books: Bibliography

| <img width="249" alt="Capture d’écran, le 2023-09-06 à 21 58 08" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/53368a31-8a49-4df3-a7f0-1d8b3b806cd3"> | 
| :-------------: | 
| Hands-On Recommendation Systems with Python | 
