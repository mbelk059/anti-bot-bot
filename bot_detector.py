"""
This detector uses an ensemble of machine learning models trained on behavioral,
temporal, and linguistic features to identify bot accounts.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
import re
import statistics
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class BotDetector:
    """Ensemble bot detection system"""
    
    def __init__(self):
        self.models = []
        self.scaler = None
        self.feature_names = []
        
    def extract_features(self, user_id, posts, user_info):
        """Extract comprehensive features from user data"""
        features = {}
        
        # Basic stats
        features['post_count'] = len(posts)
        features['z_score'] = user_info.get('z_score', 0)
        
        # Text analysis
        texts = [p['text'] for p in posts]
        features['avg_text_length'] = sum(len(t) for t in texts) / len(texts)
        features['unique_texts'] = len(set(texts))
        features['text_repetition_rate'] = 1 - (features['unique_texts'] / features['post_count'])
        
        # Timing analysis
        timestamps = [datetime.fromisoformat(p['created_at'].replace('Z', '+00:00')) for p in posts]
        timestamps.sort()
        
        if len(timestamps) > 1:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            features['avg_interval_seconds'] = sum(intervals) / len(intervals)
            features['min_interval_seconds'] = min(intervals)
            features['max_interval_seconds'] = max(intervals)
            features['std_interval'] = statistics.stdev(intervals) if len(intervals) > 1 else 0
            
            # Regularity score (bots tend to post at regular intervals)
            if features['std_interval'] > 0:
                features['interval_regularity'] = features['avg_interval_seconds'] / features['std_interval']
            else:
                features['interval_regularity'] = 0
        else:
            features['avg_interval_seconds'] = 0
            features['min_interval_seconds'] = 0
            features['max_interval_seconds'] = 0
            features['std_interval'] = 0
            features['interval_regularity'] = 0
        
        # Content patterns
        desc = user_info.get('description')
        loc = user_info.get('location')
        features['has_description'] = 1 if (desc and desc.strip()) else 0
        features['has_location'] = 1 if (loc and loc.strip()) else 0
        
        # Username patterns
        username = user_info.get('username', '')
        name = user_info.get('name', '')
        features['username_has_numbers'] = 1 if re.search(r'\d', username) else 0
        features['username_length'] = len(username)
        features['name_equals_username'] = 1 if username.lower() == name.lower() else 0
        
        # Hashtag and URL usage
        features['avg_hashtags'] = sum(t.count('#') for t in texts) / len(texts)
        features['avg_urls'] = sum(t.count('http') for t in texts) / len(texts)
        features['total_hashtags'] = sum(t.count('#') for t in texts)
        
        # Mention patterns
        features['avg_mentions'] = sum(t.count('@') for t in texts) / len(texts)
        
        # Check for very similar consecutive posts
        similar_consecutive = 0
        for i in range(len(texts)-1):
            if texts[i] == texts[i+1]:
                similar_consecutive += 1
        features['consecutive_duplicates'] = similar_consecutive / max(len(texts)-1, 1)
        
        # Posting time distribution (spread across hours)
        hours = [ts.hour for ts in timestamps]
        unique_hours = len(set(hours))
        features['hour_diversity'] = unique_hours / 24.0
        
        # Username entropy (randomness)
        features['username_entropy'] = self._calculate_entropy(username)
        
        # Description length
        features['description_length'] = len(desc) if desc else 0
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        from collections import Counter
        import math
        counts = Counter(text)
        total = len(text)
        entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
        return entropy
    
    def train(self, training_datasets):
        """Train the ensemble on multiple datasets"""
        X = []
        y = []
        
        for dataset_file, bots_file in training_datasets:
            # Load data
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with open(bots_file, 'r', encoding='utf-8') as f:
                bot_ids = set(line.strip() for line in f if line.strip())
            
            # Organize posts by user
            user_posts = defaultdict(list)
            for post in data['posts']:
                user_posts[post['author_id']].append(post)
            
            # Create user lookup
            users_by_id = {u['id']: u for u in data['users']}
            
            # Extract features
            for user_id, posts in user_posts.items():
                if user_id not in users_by_id:
                    continue
                    
                user_info = users_by_id[user_id]
                features = self.extract_features(user_id, posts, user_info)
                
                # Convert to list maintaining order
                if not self.feature_names:
                    self.feature_names = sorted(features.keys())
                
                feature_vector = [features[f] for f in self.feature_names]
                X.append(feature_vector)
                y.append(1 if user_id in bot_ids else 0)
        
        if SKLEARN_AVAILABLE:
            X = np.array(X)
            y = np.array(y)
            
            # Normalize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble
            print(f"Training on {len(X)} samples ({sum(y)} bots, {len(y)-sum(y)} humans)")
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, 
                                       class_weight='balanced')
            rf.fit(X_scaled, y)
            self.models.append(('rf', rf))
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            gb.fit(X_scaled, y)
            self.models.append(('gb', gb))
            
            # Logistic Regression
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr.fit(X_scaled, y)
            self.models.append(('lr', lr))
            
            # Print feature importance from RF
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nTop 10 Most Important Features:")
            for i in range(min(10, len(self.feature_names))):
                idx = indices[i]
                print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        else:
            # Store raw data for rule-based detection
            self.X = X
            self.y = y
            
        return self
    
    def predict(self, dataset_file, threshold=0.5):
        """Predict bots in a dataset"""
        # Load data
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Organize posts by user
        user_posts = defaultdict(list)
        for post in data['posts']:
            user_posts[post['author_id']].append(post)
        
        # Create user lookup
        users_by_id = {u['id']: u for u in data['users']}
        
        detected_bots = []
        
        for user_id, posts in user_posts.items():
            if user_id not in users_by_id:
                continue
                
            user_info = users_by_id[user_id]
            features = self.extract_features(user_id, posts, user_info)
            
            # Convert to feature vector
            feature_vector = [features[f] for f in self.feature_names]
            
            if SKLEARN_AVAILABLE:
                # Ensemble prediction
                X_test = np.array([feature_vector])
                X_test_scaled = self.scaler.transform(X_test)
                
                # Average predictions from all models
                predictions = []
                for name, model in self.models:
                    pred_proba = model.predict_proba(X_test_scaled)[0][1]
                    predictions.append(pred_proba)
                
                avg_prediction = sum(predictions) / len(predictions)
                
                if avg_prediction > threshold:
                    detected_bots.append(user_id)
            else:
                # Rule-based detection (fallback)
                is_bot = self._rule_based_detection(features)
                if is_bot:
                    detected_bots.append(user_id)
        
        return detected_bots
    
    def _rule_based_detection(self, features):
        """Simple rule-based detection as fallback"""
        # Bot indicators based on analysis
        score = 0
        
        # High post count
        if features['post_count'] > 30:
            score += 2
        
        # High z-score
        if features['z_score'] > 0.3:
            score += 2
        
        # Many hashtags
        if features['avg_hashtags'] > 0.5:
            score += 2
        
        # Few URLs
        if features['avg_urls'] < 0.4:
            score += 1
        
        # Numbers in username
        if features['username_has_numbers']:
            score += 1
        
        # Long intervals
        if features['avg_interval_seconds'] > 30000:
            score += 1
        
        # High min interval
        if features['min_interval_seconds'] > 500:
            score += 1
        
        return score >= 5


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python bot_detector.py <dataset_file> [output_file]")
        print("   OR: python bot_detector.py --train")
        sys.exit(1)
    
    if sys.argv[1] == '--train':
        # Training mode - train on all practice datasets
        detector = BotDetector()
        
        training_datasets = [
            ('dataset.posts&users.30.json', 'dataset.bots.30.txt'),
            ('dataset.posts&users.31.json', 'dataset.bots.31.txt'),
            ('dataset.posts&users.32.json', 'dataset.bots.32.txt'),
            ('dataset.posts&users.33.json', 'dataset.bots.33.txt'),
        ]
        
        detector.train(training_datasets)
        
        # Save model
        import pickle
        with open('bot_detector_model.pkl', 'wb') as f:
            pickle.dump(detector, f)
        
        print("\nModel trained and saved to bot_detector_model.pkl")
        
    else:
        # Detection mode
        dataset_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'detected_bots.txt'
        
        # Load trained model
        import pickle
        try:
            with open('bot_detector_model.pkl', 'rb') as f:
                detector = pickle.load(f)
            print("Loaded trained model")
        except FileNotFoundError:
            print("No trained model found. Training on practice datasets...")
            detector = BotDetector()
            training_datasets = [
                ('dataset.posts&users.30.json', 'dataset.bots.30.txt'),
                ('dataset.posts&users.32.json', 'dataset.bots.32.txt'),
            ]
            detector.train(training_datasets)
        
        # Detect bots
        print(f"Analyzing {dataset_file}...")
        detected_bots = detector.predict(dataset_file, threshold=0.5)
        
        # Save results
        with open(output_file, 'w') as f:
            for bot_id in detected_bots:
                f.write(f"{bot_id}\n")
        
        print(f"Detected {len(detected_bots)} bots")
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()