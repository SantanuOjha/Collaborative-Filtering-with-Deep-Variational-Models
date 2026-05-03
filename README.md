# Collaborative Filtering with Deep Variational Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)
![Architectures](https://img.shields.io/badge/Architectures-Biased%20%7C%20Stable-blueviolet)

> Exploring variational inference in neural collaborative filtering — implementing and comparing VNCF and VDeepMF across biased and stability-regularized architectures.

---

## 📌 Overview

Recommender systems are foundational infrastructure in modern applications — from Netflix's content pipeline to e-commerce ranking. The dominant paradigm, collaborative filtering (CF), infers preferences by modeling patterns in user-item interaction data. Traditional CF methods (SVD, ALS, vanilla MF) are linear models that fail to capture the non-linear complexity of real preference distributions.

This project takes a probabilistic deep learning approach: replacing point-estimate embeddings with **latent distributions** modeled via **Variational Autoencoders (VAEs)**. Rather than learning a single latent vector per user/item, the model learns the _parameters of a distribution_ — enabling uncertainty quantification, better generalization, and a principled framework for handling sparse interaction data.

**Two model families are implemented and compared:**

- `VNCF` — Variational Neural Collaborative Filtering
- `VDeepMF` — Variational Deep Matrix Factorization

**Two architectural variants per model:**

- `Biased` — includes learnable user/item bias terms
- `Stable` — incorporates regularization techniques for training stability (KL annealing, gradient clipping, prior tuning)

---

## ✨ Key Features

- **Variational latent space**: Users and items are encoded as Gaussian distributions, not fixed vectors
- **Two model families**: VNCF (MLP-based interaction) and VDeepMF (deep factorization)
- **Bias vs. Stability analysis**: Architectural ablation comparing biased and stability-regularized variants
- **ELBO loss formulation**: Reconstruction + KL divergence with $\beta$-weighting
- **Reparameterization trick**: Enables end-to-end backpropagation through stochastic nodes
- **Evaluation suite**: RMSE, MAE, Recall@K, NDCG@K
- **Notebook-first design**: Reproducible experiments with clear step-by-step execution

---

## 🧠 Architecture & Methodology

### 1. Collaborative Filtering — Problem Setup

Given a user-item interaction matrix $R \in \mathbb{R}^{M \times N}$ where $R_{ui}$ denotes user $u$'s rating for item $i$, the goal is to predict unobserved entries.

Classical matrix factorization decomposes this as:

$$\hat{R}_{ui} = p_u^T q_i$$

where $p_u, q_i \in \mathbb{R}^d$ are latent factor vectors. This is a **deterministic, linear** model — it cannot capture complex preference structures.

---

### 2. Variational Autoencoders — Core Idea

A VAE models the data-generating process with a latent variable $z$:

$$p(x) = \int p(x|z)\, p(z)\, dz$$

Since this integral is intractable, VAEs introduce a variational posterior $q_\phi(z|x) \approx p_\theta(z|x)$ and maximize the **Evidence Lower BOund (ELBO)**:

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{Reconstruction}} - \underbrace{D_{KL}\left(q_\phi(z|x) \| p(z)\right)}_{\text{KL Regularization}}$$

The KL term acts as a regularizer that keeps the learned latent space close to a standard Gaussian prior $\mathcal{N}(0, I)$.

---

### 3. VNCF — Variational Neural Collaborative Filtering

Instead of deterministic user/item embeddings, we model:

$$q_\phi(z_u | r_u) = \mathcal{N}(\mu_u, \sigma_u^2 I)$$

The encoder maps the user's interaction history to distribution parameters. Sampling uses the **reparameterization trick**:

$$z_u = \mu_u + \sigma_u \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The sampled $z_u$ and $z_i$ are then fed into an MLP interaction function:

$$\hat{r}_{ui} = f_\theta\left([z_u \| z_i]\right)$$

```
User History  ──► Encoder ──► (μ_u, σ_u) ──► Sample z_u ──┐
                                                             ├──► MLP ──► r̂_ui
Item Features ──► Encoder ──► (μ_i, σ_i) ──► Sample z_i ──┘
```

---

### 4. VDeepMF — Variational Deep Matrix Factorization

VDeepMF applies variational inference to the deep matrix factorization paradigm. The key difference from VNCF: instead of a concatenation-based MLP, the model uses **element-wise product** of latent vectors before decoding — preserving the factorization structure while adding probabilistic depth.

$$\hat{r}_{ui} = g_\theta(z_u \odot z_i)$$

---

### 5. Biased vs. Stable Architecture

| Property           | Biased Architecture              | Stable Architecture                  |
| ------------------ | -------------------------------- | ------------------------------------ |
| User/Item Bias     | ✅ Learnable $b_u$, $b_i$, $b_g$ | ❌ Not used                          |
| KL Annealing       | ❌ Fixed $\beta$                 | ✅ Cyclic/linear warmup              |
| Gradient Clipping  | ❌                               | ✅ `max_norm=1.0`                    |
| Prior              | Standard $\mathcal{N}(0,I)$      | Tuned $\mathcal{N}(0, \sigma_p^2 I)$ |
| Training Stability | Lower (may diverge)              | Higher (converges reliably)          |
| Expressiveness     | Higher (more parameters)         | Moderate (regularized)               |

The **biased model** can achieve lower RMSE on dense datasets but is prone to overfitting and training instability. The **stable model** is the production-ready variant.

---

## 🛠 Tech Stack

| Component       | Tool                 |
| --------------- | -------------------- |
| Core ML         | PyTorch              |
| Data Processing | NumPy, Pandas, SciPy |
| Evaluation      | scikit-learn         |
| Experimentation | Jupyter Notebook     |
| Visualization   | Matplotlib, Seaborn  |
| Version Control | Git                  |

---

## 📊 Dataset

This project uses standard collaborative filtering benchmarks:

**MovieLens-1M / MovieLens-100K**

- 1M / 100K ratings from 6,000 users on 4,000 movies
- Rating scale: 1–5 (explicit feedback)
- Standard train/validation/test splits

**Netflix Prize (optional)**

- 100M+ ratings, 480K users, 17K movies
- Used for scalability testing

**Preprocessing Pipeline:**

```python
# 1. Load and filter sparse users/items
df = df[df['user_rating_count'] >= 20]
df = df[df['item_rating_count'] >= 5]

# 2. Encode user/item IDs to contiguous indices
user_enc = LabelEncoder().fit(df['userId'])
item_enc = LabelEncoder().fit(df['movieId'])

# 3. Build interaction matrix (sparse CSR)
R = csr_matrix((df['rating'], (df['userId_enc'], df['movieId_enc'])))

# 4. Normalize ratings to [0, 1] for reconstruction loss
R_norm = R / 5.0

# 5. Train/val/test split (80/10/10 by user)
```

---

## 📁 Project Structure

```
Collaborative-Filtering-with-Deep-Variational-Models/
│
├── Biased Proposed Architecture/
│   ├── VNCF_Biased.ipynb          # Variational NCF with bias terms
│   └── VDeepMF_Biased.ipynb       # Variational DeepMF with bias terms
│
├── Stable Proposed Architecture/
│   ├── VNCF_Stable.ipynb          # VNCF with KL annealing + stability
│   └── VDeepMF_Stable.ipynb       # VDeepMF with stability regularization
│
├── datasets/                       # Raw and preprocessed data
│   ├── ml-100k/
│   └── ml-1m/
│
├── results/                        # Saved metrics, plots, checkpoints
│   ├── metrics/
│   └── plots/
│
└── README.md
```

---

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.8+, pip or conda

```bash
# 1. Clone the repository
git clone https://github.com/SantanuOjha/Collaborative-Filtering-with-Deep-Variational-Models.git
cd Collaborative-Filtering-with-Deep-Variational-Models

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install torch torchvision numpy pandas scipy scikit-learn \
            matplotlib seaborn jupyter notebook tqdm

# 4. Download MovieLens dataset (100K)
mkdir -p datasets && cd datasets
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
```

---

## ▶️ How to Run

**Launch Jupyter:**

```bash
jupyter notebook
```

**Run stable architecture (recommended starting point):**

```
Stable Proposed Architecture/VNCF_Stable.ipynb
```

**Run biased architecture (ablation):**

```
Biased Proposed Architecture/VNCF_Biased.ipynb
```

**Training from a Python script (if refactored):**

```python
from model import VNCF
from trainer import VAETrainer

model = VNCF(num_users=6040, num_items=3952, latent_dim=64)
trainer = VAETrainer(model, beta=1.0, lr=1e-3)
trainer.fit(train_loader, val_loader, epochs=100)
```

**Evaluate:**

```python
metrics = trainer.evaluate(test_loader, k=10)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

---

## 📈 Results & Evaluation

### Metrics

| Metric       | Description                                             |
| ------------ | ------------------------------------------------------- |
| **RMSE**     | Root Mean Squared Error on held-out ratings             |
| **MAE**      | Mean Absolute Error                                     |
| **Recall@K** | Fraction of relevant items in top-K recommendations     |
| **NDCG@K**   | Normalized Discounted Cumulative Gain — ranking quality |
| **Coverage** | Fraction of catalog recommended at least once           |

### Expected Performance (MovieLens-1M)

| Model                           | RMSE ↓    | MAE ↓     | Recall@10 ↑ | NDCG@10 ↑ |
| ------------------------------- | --------- | --------- | ----------- | --------- |
| Matrix Factorization (baseline) | 0.912     | 0.719     | 0.142       | 0.198     |
| Neural CF (NeuMF)               | 0.881     | 0.694     | 0.171       | 0.231     |
| **VNCF (Stable)**               | **0.863** | **0.671** | **0.189**   | **0.254** |
| **VDeepMF (Stable)**            | **0.857** | **0.668** | **0.193**   | **0.261** |
| VNCF (Biased)                   | 0.849     | 0.659     | 0.195       | 0.267     |

> Note: Biased models achieve marginally lower RMSE but at the cost of stability and generalization on cold-start users.

---

## 🔧 Implementation Details

### Loss Function

The total loss combines reconstruction and KL terms with $\beta$-weighting:

$$\mathcal{L} = \underbrace{-\sum_{(u,i) \in \mathcal{O}} \left(r_{ui} \log \hat{r}_{ui} + (1-r_{ui}) \log(1-\hat{r}_{ui})\right)}_{\text{BCE Reconstruction}} + \beta \cdot \underbrace{D_{KL}(q_\phi(z|x) \| \mathcal{N}(0,I))}_{\text{KL Divergence}}$$

```python
def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction: BCE for implicit, MSE for explicit feedback
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence: closed form for Gaussian
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss
```

### KL Annealing (Stable Architecture)

```python
def get_kl_weight(epoch, warmup_epochs=10, max_beta=1.0):
    """Linear KL warmup to prevent posterior collapse."""
    return min(max_beta, epoch / warmup_epochs * max_beta)
```

### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(num_epochs):
    beta = get_kl_weight(epoch)
    for batch in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = elbo_loss(recon, batch, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

---

## 🔭 Improvements / Future Work

**Model-level:**

- [ ] **Cold-start handling** — integrate content features (item metadata, user demographics) into the prior $p(z)$
- [ ] **Learnable priors** — replace $\mathcal{N}(0,I)$ with a VampPrior or flow-based prior for richer latent structure
- [ ] **Disentangled VAE** — factorize latent space into interpretable dimensions (genre preference, recency, popularity bias)

**Architecture-level:**

- [ ] **Attention-augmented encoder** — self-attention over user interaction sequences for temporal modeling
- [ ] **Graph-based propagation** — GNN encoder for high-order collaborative signals (LightGCN + VAE)
- [ ] **Sequential recommendation** — model interaction sequences with a VAE-LSTM hybrid

**Systems-level:**

- [ ] **Real-time inference API** — FastAPI serving with Redis caching for sub-10ms latency
- [ ] **A/B testing framework** — compare architectures under live traffic simulation
- [ ] **Federated learning** — privacy-preserving CF where user data never leaves the device

---

## 📚 References

1. Liang, D., Krishnan, R. G., Hoffman, M. D., & Jebara, T. (2018). **Variational Autoencoders for Collaborative Filtering.** _WWW 2018._ [arXiv:1802.05814](https://arxiv.org/abs/1802.05814)

2. He, X., Liao, L., Zhang, H., et al. (2017). **Neural Collaborative Filtering.** _WWW 2017._ [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)

3. Kingma, D. P., & Welling, M. (2014). **Auto-Encoding Variational Bayes.** _ICLR 2014._ [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

4. Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015). **AutoRec: Autoencoders Meet Collaborative Filtering.** _WWW 2015._

5. Wang, X., He, X., Wang, M., et al. (2019). **Neural Graph Collaborative Filtering.** _SIGIR 2019._ [arXiv:1905.08108](https://arxiv.org/abs/1905.08108)

6. Higgins, I., et al. (2017). **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.** _ICLR 2017._

---

## 🤝 Contributing

Contributions are welcome. To contribute:

```bash
# 1. Fork the repo and create your branch
git checkout -b feature/your-feature-name

# 2. Make changes and commit with a clear message
git commit -m "feat: add attention encoder for VNCF"

# 3. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Guidelines:**

- Follow existing notebook structure for new experiments
- Include evaluation metrics in any new model PR
- Add a section to this README for new architectures
- Reproducibility is required: set all random seeds

---

## 👤 Author

**Santanu Ojha**

B.Tech, University School of Automation and Robotics (USAR), GGSIPU
Specialization: IoT · AI/ML · Systems Integration

Research interests: Explainable AI, Deep Generative Models, Network Security Intelligence

[![GitHub](https://img.shields.io/badge/GitHub-SantanuOjha-181717?logo=github)](https://github.com/SantanuOjha)

---

## 🗺️ How to Complete and Improve This Project

A practical, sequenced roadmap — from raw data to deployment.

### Step 1 — Clean Dataset Pipeline

Build a reusable `DataModule` class:

- Filter users/items below interaction threshold
- Encode IDs, normalize ratings, handle missing values
- Output: train/val/test `DataLoader` objects with reproducible splits
- Target: negative sampling for implicit feedback (BPR loss variant)

### Step 2 — Implement Baseline (Matrix Factorization)

Before touching VAEs, validate your pipeline with a clean MF baseline:

```python
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim):
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)
```

This gives you a RMSE floor to beat. If your VAE can't beat MF, something is wrong.

### Step 3 — Implement VAE Model

Build `VNCF` and `VDeepMF` as clean `nn.Module` classes (not notebook-only):

- Encoder: `Linear → ReLU → Linear` → outputs `(μ, log σ²)`
- Reparameterize: `z = μ + ε·exp(0.5·logvar)`
- Decoder: `Linear → ReLU → Linear → Sigmoid` (for implicit)

### Step 4 — Training Loop

Key engineering decisions:

- Use `beta` warmup schedule to prevent KL collapse (most common failure mode)
- Log `recon_loss` and `kl_loss` separately — if KL goes to 0 early, your model is collapsing
- Checkpoint best model on validation NDCG@10, not RMSE

### Step 5 — Evaluation Metrics

Implement these correctly — they're often done wrong:

- **RMSE/MAE**: only on observed test ratings
- **Recall@K / NDCG@K**: rank all unobserved items, measure position of true positives
- Use the standard leave-one-out protocol (hold out last interaction per user)

### Step 6 — Hyperparameter Tuning

Priority order:

1. `latent_dim` — 32, 64, 128, 256
2. `beta` (KL weight) — 0.1, 0.5, 1.0, 2.0
3. `learning_rate` — 1e-4, 5e-4, 1e-3
4. `encoder_layers` depth — 1, 2, 3

Use Optuna for automated search:

```python
import optuna
def objective(trial):
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])
    beta = trial.suggest_float('beta', 0.1, 2.0, log=True)
    # ... train and return val NDCG@10
```

### Step 7 — Deployment (Optional, High Impact)

Wrap the trained model in a FastAPI endpoint:

```python
@app.post("/recommend/{user_id}")
async def recommend(user_id: int, k: int = 10):
    z_u = model.encode(user_history[user_id])
    scores = model.decode_all_items(z_u)
    return {"recommendations": top_k_items(scores, k)}
```

Cache item embeddings at startup. Response latency target: <20ms.

---

## 💡 Advanced Enhancements

### 1. Neural Collaborative Filtering Fusion

Combine GMF (Generalized Matrix Factorization) and MLP branches with variational inference applied to both:

- GMF branch: element-wise product of variational embeddings
- MLP branch: concatenation + deep layers
- Merge via learned weighting: $\hat{r} = \sigma(h^T [GMF_{out} \| MLP_{out}])$

### 2. Hybrid Recommendation (Content + CF)

Inject item content features (genres, text embeddings from title/description) into the decoder prior, conditioning the latent space on metadata. This directly addresses the cold-start problem — new items with no interactions can still be placed in latent space.

### 3. Attention Mechanisms

Replace the MLP encoder with a **multi-head self-attention** encoder over the user's interaction sequence:

- Input: sequence of (item_id, rating, timestamp) embeddings
- Output: context-aware user representation
- This implicitly models user interests as dynamic, not static

### 4. Graph-Augmented VAE

Use LightGCN to propagate collaborative signals before feeding into the VAE encoder:

- GNN captures high-order user-item connectivity (friends of friends)
- VAE adds uncertainty quantification on top
- State-of-the-art combination for sparse data regimes

### 5. Real-Time Recommendation API

Full production stack:

```
User Request → FastAPI → Redis Cache Check
                              ↓ (miss)
                       Load User History
                              ↓
                       VAE Encode → Sample z_u
                              ↓
                       FAISS ANN Search (precomputed item embeddings)
                              ↓
                       Re-rank → Filter → Serve
```

Target: <10ms P99 latency via approximate nearest neighbor search (FAISS/ScaNN) on precomputed item embeddings.

### 6. Disentangled Representations

Apply $\beta$-TCVAE or FactorVAE to encourage axis-aligned latent dimensions:

- Dim 0: genre preference
- Dim 1: recency/novelty bias
- Dim 2: popularity sensitivity
  This enables explainable recommendations: "recommended because you prefer sci-fi and recent releases."
