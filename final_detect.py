"""
Usage:
    python final_detect.py <dataset_file> <output_file> <language>
    
    'en' for english or 'fr' for french
"""

import json
import sys
import pickle
import os

# Import BotDetector class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot_detector import BotDetector

def main():
    if len(sys.argv) != 4:
        print("Usage: python final_detect.py <dataset_file> <output_file> <language>")
        print("  language: 'en' for english or 'fr' for french")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    output_file = sys.argv[2]
    language = sys.argv[3].lower()
    
    if language not in ['en', 'fr']:
        print("Error: Language must be 'en' or 'fr'")
        sys.exit(1)
    
    print(f"Bot Detection System - {language.upper()}")
    print("=" * 50)
    
    # Load appropriate model and threshold
    if language == 'en':
        model_file = 'bot_detector_english.pkl'
        threshold_default = 0.4
    else:
        model_file = 'bot_detector_french.pkl'
        threshold_default = 0.45
    
    # Load threshold config
    try:
        with open('optimal_thresholds.json', 'r') as f:
            config = json.load(f)
            threshold = config.get(f'{language}ish_threshold', threshold_default)
    except:
        threshold = threshold_default
    
    print(f"Loading model: {model_file}")
    print(f"Detection threshold: {threshold}")
    
    # Load model
    try:
        with open(model_file, 'rb') as f:
            detector = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model file {model_file} not found!")
        sys.exit(1)
    
    # Detect bots
    print(f"\nAnalyzing dataset: {dataset_file}")
    detected_bots = detector.predict(dataset_file, threshold=threshold)
    
    # Save results
    with open(output_file, 'w') as f:
        for bot_id in detected_bots:
            f.write(f"{bot_id}\n")
    
    print(f"Detected {len(detected_bots)} bot accounts")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
