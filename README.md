# Cross-Lingual Semantic Alignment in Multilingual Transformers

## Overview
This project investigates how multilingual transformer models build semantic alignment between Chinese and English sentences across hidden layers.

Using a synthetic bilingual dataset and the multilingual transformer model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the notebook analyzes whether semantically equivalent Chinese-English sentence pairs become more aligned in deeper transformer layers.

The project also explores:
- Whether cross-lingual alignment behaves similarly to translation.
- Whether culturally loaded sentences align differently from neutral sentences.

---

## Research Questions

### RQ1 — General Cross-Lingual Alignment
Do semantically equivalent Chinese-English sentence pairs become more similar across transformer layers?

### Translation Baseline
Is multilingual alignment equivalent to translating Chinese into English internally?

### RQ2 — Neutral vs Culturally Loaded Meaning
Do culturally loaded sentence pairs align differently compared to neutral sentence pairs?

---

## Dataset
The project uses a synthetic Chinese-English sentence-pair dataset.

### Dataset Structure
Each row contains:
- English sentence
- Chinese sentence
- Category label
- Optional subcategory labels
- English translation baseline

### Categories
#### Neutral Sentences
Examples include:
- Daily actions
- School/work activities
- Weather
- Objects
- Emotions
- Locations

#### Culturally Loaded Sentences
Examples include:
- Family values
- Social norms
- Festivals
- Traditional customs
- Cultural symbols
- Idiomatic meanings

The dataset contains:
- 50 neutral sentence pairs
- 50 culturally loaded sentence pairs

---

## Methodology

### Model
The notebook uses:

```python
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This multilingual transformer model produces shared multilingual semantic representations.

### Layer-Wise Representation Extraction
Instead of using only final sentence embeddings, the project extracts hidden states from all transformer layers.

For each layer:
1. Hidden states are extracted.
2. Mean pooling is applied.
3. Sentence embeddings are generated.
4. Cosine similarity is computed between English and Chinese representations.

### Similarity Analysis
The project measures:
- Layer-wise cosine similarity
- Alignment gain
- Alignment onset
- Layer-wise variation

### Statistical Testing
The notebook includes:
- Paired t-tests
- Wilcoxon signed-rank tests
- Independent t-tests
- Mann–Whitney U tests

---

## Key Findings

### 1. Cross-Lingual Alignment Improves Across Layers
Chinese-English sentence pairs become significantly more similar in deeper transformer layers.

Main observation:
- Initial-layer similarity ≈ 0.41
- Final-layer similarity ≈ 0.87

This suggests multilingual transformers gradually build a shared semantic space.

### 2. Alignment Is Not Simply Translation
The project compares:
- English ↔ Chinese
- English ↔ translated English

Results show:
- Early layers strongly favor same-language similarity.
- Deeper layers reduce the translation gap.

This suggests multilingual alignment is more than simple internal translation.

### 3. Cultural Meaning Aligns Differently
Neutral sentence pairs generally:
- Align earlier
- Reach higher similarity
- Show more stable alignment

Culturally loaded sentence pairs often:
- Require deeper layers for alignment
- Show lower overall similarity
- Exhibit greater variability across layers

This indicates culturally grounded meanings remain harder for multilingual models to fully align.

---

## Visualizations
The notebook includes:
- Layer-wise similarity curves
- Translation baseline comparisons
- Neutral vs cultural alignment plots
- Boxplots for similarity metrics
- Statistical comparison summaries

---


## Future Work
Potential extensions include:
- Using larger multilingual models
- Testing additional language pairs
- Comparing encoder vs decoder architectures
- Evaluating real-world bilingual datasets
- Investigating token-level alignment
- Studying alignment for idioms and metaphorical language



