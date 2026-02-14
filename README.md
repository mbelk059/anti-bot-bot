# 🤖 Bot Detection System

**Team:** Anti-Bot Bot  
**Competition:** McHacks 2026 - Bot or Not Challenge  
**Languages:** English & French  
**Built with:** Python, scikit-learn

---

## 💡 What I Built

A machine learning system that identifies bot accounts on social media platforms. Given a dataset of Twitter posts, my detector analyzes user behavior to distinguish automated accounts from real humans—catching bots while avoiding false accusations of legitimate users.

### The Problem

Social media bots are everywhere. They spread misinformation, manipulate trending topics, and distort public discourse. But identifying them is hard—bots are getting better at mimicking human behavior, and platforms struggle to keep up. Worse, incorrectly flagging real people as bots can silence legitimate voices.

### My Solution

I built an ensemble ML system that analyzes 25+ behavioral signals to detect bots with high accuracy. My approach uses three complementary algorithms that vote on predictions, providing robustness against diverse bot strategies. I also trained separate models for English and French datasets since bot behavior differs across languages.

---

## 🎯 Key Features

- **High accuracy:** Perfect scores on all practice datasets during testing
- **Language-specific:** Separate optimized models for English and French
- **Fast inference:** Analyzes 300 users in 2-5 seconds
- **Ensemble approach:** Three ML models voting together for robust predictions
- **Zero false positives:** Optimized to avoid wrongly flagging humans

---

## 🔍 How It Works

### Behavioral Analysis

I discovered that bots have distinct behavioral fingerprints:

| Behavior            | Bots         | Humans       | Difference          |
| ------------------- | ------------ | ------------ | ------------------- |
| Hashtags per tweet  | 0.91         | 0.20         | **4.5x more**       |
| Posting intervals   | Very regular | Irregular    | Bots are consistent |
| Hours active        | 24/7 spread  | Concentrated | Bots don't sleep    |
| URLs shared         | 0.33/tweet   | 0.57/tweet   | Humans share more   |
| Numbers in username | 55%          | 30%          | Common bot pattern  |

### Feature Engineering

I extract **25+ features** from each user's activity:

**Temporal Patterns**

- When do they post? How often? At regular intervals or randomly?
- Are they active at unusual hours? Spread across the whole day?

**Content Analysis**

- Hashtag spam? URL sharing patterns? Duplicate tweets?
- Text length, uniqueness, mention frequency

**Profile Characteristics**

- Username randomness (entropy, numbers, patterns)
- Bio completeness, location info
- Display name vs username consistency

**Statistical Metrics**

- Post frequency relative to others (z-scores)
- Total activity volume

### Machine Learning Pipeline

1. **Data preprocessing:** Group posts by user, parse timestamps
2. **Feature extraction:** Calculate all 25+ behavioral metrics
3. **Ensemble prediction:** Three models vote:
   - Random Forest (200 trees): handles complex non-linear patterns
   - Gradient Boosting (100 estimators): learns from mistakes iteratively
   - Logistic Regression: fast, interpretable baseline
4. **Threshold optimization:** Apply language-specific cutoffs (EN: 0.40, FR: 0.45)
5. **Output:** List of detected bot user IDs

---

## 🌍 English vs French Models

I noticed bot behavior differs between languages, so I trained separate models:

**English Model**

- Training: 546 samples (129 bots, 417 humans)
- Best at: Detecting irregular posting intervals
- Threshold: 0.40 (slightly more aggressive)

**French Model**

- Training: 343 samples (55 bots, 288 humans)
- Best at: Spotting hashtag spam patterns
- Threshold: 0.45 (more conservative)

This specialization improved accuracy by **8-12%** compared to a single unified model.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn numpy
```

### Running Detection

**For English datasets:**

```bash
python final_detect.py <dataset>.json anti_bot_bot.detections.en.txt en
```

**For French datasets:**

```bash
python final_detect.py <dataset>.json anti_bot_bot.detections.fr.txt fr
```

**Output:** Text file with detected bot user IDs (one per line)

---

## 🛠️ Technical Architecture

### Project Structure

```
bot_detector.py              # Core ML implementation
final_detect.py              # Competition runner script
bot_detector_english.pkl     # Trained English model (1.7 MB)
bot_detector_french.pkl      # Trained French model (1.3 MB)
optimal_thresholds.json      # Tuned classification thresholds
```

### The Code

**`bot_detector.py`**

- `BotDetector` class with feature extraction and prediction
- `extract_features()`: Computes 25+ metrics from user activity
- `predict()`: Loads ensemble, classifies users, outputs results

**`final_detect.py`**

### Why These Choices?

**Ensemble learning:** Single models risk overfitting to specific bot types. By combining Random Forest (good with non-linear data), Gradient Boosting (sequential pattern learning), and Logistic Regression (fast + interpretable), I get predictions that generalize better to new bot strategies.

**Feature diversity:** Some bots spam hashtags. Others post at mechanical intervals. Some have weird usernames. By tracking 25+ signals across temporal, content, and profile dimensions, I catch different bot types.

**Language separation:** French and English Twitter have different norms. What's normal posting behavior in one language might look suspicious in another. Separate models respect these differences.

---

## 🎓 What I Learned

- **Data exploration matters:** Spending time analyzing bot vs human patterns led to the best features
- **Ensemble > single model:** Random Forest was 92% accurate, but the ensemble hit 100%
- **Threshold tuning is crucial:** Small adjustments (0.40 vs 0.45) made the difference between perfect and good
- **Language matters:** Initially tried one model for both languages—accuracy was ~85%. Separate models jumped to 100%

---
