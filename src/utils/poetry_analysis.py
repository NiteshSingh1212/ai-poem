import spacy
import nltk
from nltk.corpus import cmudict
from typing import List, Tuple, Dict
import re

# Download required NLTK data
nltk.download('cmudict')
nltk.download('punkt')

class PoetryAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.pronouncing_dict = cmudict.dict()
        
    def analyze_rhyme_scheme(self, poem: str) -> List[str]:
        """Analyze the rhyme scheme of a poem"""
        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        last_words = [line.split()[-1].lower() for line in lines]
        
        rhyme_scheme = []
        rhyme_map = {}
        current_rhyme = 'A'
        
        for word in last_words:
            if word not in rhyme_map:
                rhyme_map[word] = current_rhyme
                current_rhyme = chr(ord(current_rhyme) + 1)
            rhyme_scheme.append(rhyme_map[word])
            
        return rhyme_scheme
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU pronouncing dictionary"""
        word = word.lower()
        if word in self.pronouncing_dict:
            return len([ph for ph in self.pronouncing_dict[word][0] if ph[-1].isdigit()])
        return 0
    
    def analyze_meter(self, poem: str) -> List[List[int]]:
        """Analyze the meter of each line in the poem"""
        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        meter_pattern = []
        
        for line in lines:
            words = line.split()
            line_pattern = [self.count_syllables(word) for word in words]
            meter_pattern.append(line_pattern)
            
        return meter_pattern
    
    def get_poem_stats(self, poem: str) -> Dict:
        """Get comprehensive statistics about the poem"""
        doc = self.nlp(poem)
        
        stats = {
            'num_lines': len([line for line in poem.split('\n') if line.strip()]),
            'num_words': len(doc),
            'num_unique_words': len(set([token.text.lower() for token in doc])),
            'rhyme_scheme': self.analyze_rhyme_scheme(poem),
            'meter_pattern': self.analyze_meter(poem),
            'sentiment': self.analyze_sentiment(doc),
            'literary_devices': self.find_literary_devices(doc)
        }
        
        return stats
    
    def analyze_sentiment(self, doc) -> str:
        """Basic sentiment analysis of the poem"""
        # This is a simplified version - could be enhanced with proper sentiment analysis
        positive_words = set(['love', 'joy', 'happy', 'light', 'bright', 'beautiful'])
        negative_words = set(['dark', 'sad', 'pain', 'grief', 'death', 'fear'])
        
        pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
        neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def find_literary_devices(self, doc) -> Dict:
        """Identify common literary devices in the poem"""
        devices = {
            'alliteration': [],
            'assonance': [],
            'consonance': []
        }
        
        # Simple alliteration detection
        for i in range(len(doc) - 1):
            if doc[i].text[0].lower() == doc[i+1].text[0].lower():
                devices['alliteration'].append(f"{doc[i].text} {doc[i+1].text}")
        
        return devices
