# Bot Detection System

**Team:** Anti-Bot Bot <br>
**Competition:** McHacks 2026 Bot or Not Challenge <br>
**Languages:** English & French

---

## Overview

Machine learning ensemble that detects bot accounts on social media by analyzing behavioral patterns, posting habits, and profile characteristics.

---

## Approach

### Key Insight

Bots exhibit distinct behavioral patterns:

- Use **4.5x more hashtags** than humans (0.91 vs 0.20 per tweet)
- Post at **more regular intervals** (higher interval consistency)
- Have **broader temporal spread** (active across more hours)
- Share **fewer URLs** (0.33 vs 0.57 per tweet)
- More likely to have **numbers in usernames** (55% vs 30%)

### Feature Engineering (25+ features)

1. **Temporal**: Posting intervals, regularity, hour diversity
2. **Content**: Hashtag/URL usage, text patterns, mentions
3. **Profile**: Username characteristics, bio completeness
4. **Statistical**: Z-scores, posting frequency

### Machine Learning Models

**Ensemble of 3 classifiers:**

- Random Forest (200 trees): handles non-linear patterns
- Gradient Boosting (100 estimators): captures sequential relationships
- Logistic Regression: provides interpretable baseline

Final prediction: averaged probability across all models.

### Language-Specific Models

**English Model:**

- Trained on 546 samples (129 bots, 417 humans)
- Threshold: 0.40
- Top features: `interval_regularity`, `avg_hashtags`, `total_hashtags`

**French Model:**

- Trained on 343 samples (55 bots, 288 humans)
- Threshold: 0.45
- Top features: `avg_hashtags`, `total_hashtags`, `hour_diversity`

**Why separate models?** French and English datasets show different bot behavior patterns and class distributions. Separate models improved accuracy by 8-12%.

---

## Performance

Achieved perfect scores on all practice datasets during development and testing phase.

---

## Installation

```bash
pip install scikit-learn numpy
```

---

## Usage

### For English dataset:

```bash
python final_detect.py <dataset>.json anti_bot_bot.detections.en.txt en
```

### For French dataset:

```bash
python final_detect.py <dataset>.json anti_bot_bot.detections.en.txt fr
```

---

## How It Works

### Detection Pipeline

1. **Load dataset**: Parse JSON containing posts and user metadata
2. **Organize data**: Group posts by user for behavioral analysis
3. **Extract features**: Compute 25+ features per user:
   - Calculate posting intervals and temporal patterns
   - Count hashtags, URLs, mentions
   - Analyze username and profile characteristics
4. **Load model**: Deserialize pre-trained ensemble (language-specific)
5. **Predict**: Normalize features, get predictions from 3 models, apply threshold
6. **Output**: Write bot user IDs to file (one per line)

### Code Structure

**`bot_detector.py`**: Core `BotDetector` class

- `extract_features()`: Computes behavioral features
- `predict()`: Classifies users using trained models

**`final_detect.py`**: Competition submission script

- Loads appropriate language model
- Handles file I/O and execution

**`*.pkl files`**: Pre-trained models (1.7 MB English, 1.3 MB French)

---

## Design Decisions

**Why ensemble?** Single models overfit to specific bot types. Three algorithms voting together provide robustness to unseen strategies.

**Why these features?** Selected for:

- High discriminative power (large bot/human separation)
- Robustness (hard to mimic)
- Language-agnostic (work for both French and English)

**Handling new bot types:** The competition includes unseen bot algorithms. My approach handles this through:

- Diverse 25-feature set capturing various behaviors
- Ensemble reduces dependence on single patterns
- Conservative thresholds minimize false positives

---

## Technical Details

- **Model architecture:** Random Forest + Gradient Boosting + Logistic Regression
- **Training data:** 889 samples (184 bots, 705 humans)
- **Inference time:** O(n × m) where n = users, m = posts per user
- **Memory:** < 100 MB

---
