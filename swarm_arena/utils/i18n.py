"""
Internationalization (i18n) support for global-first multi-agent systems.

Provides sentiment analysis and emotional state management across
multiple languages and cultural contexts.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Language(Enum):
    """Supported languages for global sentiment analysis."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


class CulturalContext(Enum):
    """Cultural contexts affecting emotional expression and interpretation."""
    WESTERN_INDIVIDUALISTIC = "western_ind"
    WESTERN_COLLECTIVISTIC = "western_col"
    EAST_ASIAN = "east_asian"
    LATIN_AMERICAN = "latin_am"
    MIDDLE_EASTERN = "middle_east"
    SOUTH_ASIAN = "south_asian"
    AFRICAN = "african"
    NORDIC = "nordic"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    
    primary_language: Language = Language.ENGLISH
    secondary_languages: List[Language] = None
    cultural_context: CulturalContext = CulturalContext.WESTERN_INDIVIDUALISTIC
    enable_cultural_adaptation: bool = True
    sentiment_cultural_weight: float = 0.3
    emotion_display_rules: Dict[str, float] = None
    
    def __post_init__(self):
        if self.secondary_languages is None:
            self.secondary_languages = []
        
        if self.emotion_display_rules is None:
            self.emotion_display_rules = self._get_default_display_rules()
    
    def _get_default_display_rules(self) -> Dict[str, float]:
        """Get default emotion display rules based on cultural context."""
        rules = {
            "joy_expression": 1.0,
            "anger_suppression": 0.5,
            "sadness_expression": 0.7,
            "fear_suppression": 0.6,
            "surprise_expression": 0.8,
            "disgust_suppression": 0.4
        }
        
        # Adjust based on cultural context
        if self.cultural_context == CulturalContext.EAST_ASIAN:
            rules.update({
                "anger_suppression": 0.8,
                "joy_expression": 0.7,
                "sadness_suppression": 0.6
            })
        elif self.cultural_context == CulturalContext.LATIN_AMERICAN:
            rules.update({
                "joy_expression": 1.2,
                "anger_expression": 0.9,
                "sadness_expression": 1.0
            })
        elif self.cultural_context == CulturalContext.NORDIC:
            rules.update({
                "anger_suppression": 0.9,
                "joy_expression": 0.6,
                "sadness_suppression": 0.8
            })
        
        return rules


class MultilingualSentimentLexicon:
    """Multilingual sentiment lexicon for cross-cultural sentiment analysis."""
    
    def __init__(self):
        self.lexicons = self._build_multilingual_lexicons()
        self.cultural_modifiers = self._build_cultural_modifiers()
    
    def _build_multilingual_lexicons(self) -> Dict[Language, Dict[str, Dict[str, float]]]:
        """Build sentiment lexicons for multiple languages."""
        lexicons = {}
        
        # English lexicon
        lexicons[Language.ENGLISH] = {
            'positive': {
                'excellent': 0.9, 'amazing': 0.9, 'wonderful': 0.8, 'great': 0.7,
                'good': 0.6, 'nice': 0.5, 'cooperative': 0.7, 'helpful': 0.6,
                'successful': 0.8, 'effective': 0.7, 'beneficial': 0.6,
                'harmony': 0.8, 'teamwork': 0.7, 'collaboration': 0.8
            },
            'negative': {
                'terrible': -0.9, 'awful': -0.8, 'horrible': -0.8, 'bad': -0.6,
                'poor': -0.5, 'failed': -0.7, 'conflict': -0.6, 'harmful': -0.7,
                'ineffective': -0.6, 'destructive': -0.8, 'chaos': -0.7,
                'competitive': -0.4  # Context dependent
            },
            'neutral': {
                'system': 0.0, 'data': 0.0, 'position': 0.0, 'location': 0.0,
                'agent': 0.0, 'environment': 0.0, 'step': 0.0, 'time': 0.0
            }
        }
        
        # Spanish lexicon
        lexicons[Language.SPANISH] = {
            'positive': {
                'excelente': 0.9, 'increíble': 0.9, 'maravilloso': 0.8, 'bueno': 0.6,
                'cooperativo': 0.7, 'útil': 0.6, 'exitoso': 0.8, 'efectivo': 0.7,
                'beneficioso': 0.6, 'armonía': 0.8, 'trabajo_en_equipo': 0.7
            },
            'negative': {
                'terrible': -0.9, 'horrible': -0.8, 'malo': -0.6, 'fallido': -0.7,
                'conflicto': -0.6, 'dañino': -0.7, 'inefectivo': -0.6, 'caos': -0.7
            },
            'neutral': {
                'sistema': 0.0, 'datos': 0.0, 'posición': 0.0, 'agente': 0.0,
                'ambiente': 0.0, 'tiempo': 0.0
            }
        }
        
        # French lexicon
        lexicons[Language.FRENCH] = {
            'positive': {
                'excellent': 0.9, 'merveilleux': 0.8, 'bon': 0.6, 'coopératif': 0.7,
                'utile': 0.6, 'réussi': 0.8, 'efficace': 0.7, 'bénéfique': 0.6,
                'harmonie': 0.8, 'collaboration': 0.8
            },
            'negative': {
                'terrible': -0.9, 'horrible': -0.8, 'mauvais': -0.6, 'échoué': -0.7,
                'conflit': -0.6, 'nuisible': -0.7, 'inefficace': -0.6, 'chaos': -0.7
            },
            'neutral': {
                'système': 0.0, 'données': 0.0, 'position': 0.0, 'agent': 0.0,
                'environnement': 0.0, 'temps': 0.0
            }
        }
        
        # German lexicon
        lexicons[Language.GERMAN] = {
            'positive': {
                'ausgezeichnet': 0.9, 'wunderbar': 0.8, 'gut': 0.6, 'kooperativ': 0.7,
                'hilfreich': 0.6, 'erfolgreich': 0.8, 'effektiv': 0.7, 'vorteilhaft': 0.6,
                'harmonie': 0.8, 'zusammenarbeit': 0.8
            },
            'negative': {
                'schrecklich': -0.9, 'furchtbar': -0.8, 'schlecht': -0.6, 'gescheitert': -0.7,
                'konflikt': -0.6, 'schädlich': -0.7, 'unwirksam': -0.6, 'chaos': -0.7
            },
            'neutral': {
                'system': 0.0, 'daten': 0.0, 'position': 0.0, 'agent': 0.0,
                'umgebung': 0.0, 'zeit': 0.0
            }
        }
        
        # Japanese lexicon (romanized)
        lexicons[Language.JAPANESE] = {
            'positive': {
                'subarashii': 0.9, 'yoi': 0.6, 'kyoryoku': 0.7, 'tasukeru': 0.6,
                'seikou': 0.8, 'kouka': 0.7, 'rieki': 0.6, 'chouwa': 0.8,
                'teamwork': 0.7  # Borrowed word
            },
            'negative': {
                'warui': -0.6, 'shippai': -0.7, 'tairitsu': -0.6, 'gai': -0.7,
                'mukouryoku': -0.6, 'konton': -0.7
            },
            'neutral': {
                'system': 0.0, 'data': 0.0, 'ichi': 0.0, 'agent': 0.0,
                'kankyou': 0.0, 'jikan': 0.0
            }
        }
        
        # Chinese lexicon (pinyin)
        lexicons[Language.CHINESE] = {
            'positive': {
                'youxiu': 0.9, 'hao': 0.6, 'hezuo': 0.7, 'bangzhu': 0.6,
                'chenggong': 0.8, 'youxiao': 0.7, 'youli': 0.6, 'hexie': 0.8,
                'tuandui': 0.7
            },
            'negative': {
                'bad': -0.6, 'shibai': -0.7, 'chongtu': -0.6, 'youhai': -0.7,
                'wuxiao': -0.6, 'hunluan': -0.7
            },
            'neutral': {
                'xitong': 0.0, 'shuju': 0.0, 'weizhi': 0.0, 'daili': 0.0,
                'huanjing': 0.0, 'shijian': 0.0
            }
        }
        
        return lexicons
    
    def _build_cultural_modifiers(self) -> Dict[CulturalContext, Dict[str, float]]:
        """Build cultural modifiers for emotion interpretation."""
        modifiers = {}
        
        modifiers[CulturalContext.WESTERN_INDIVIDUALISTIC] = {
            'individual_achievement': 1.2,
            'competition': 1.0,
            'direct_expression': 1.1,
            'emotional_openness': 1.0
        }
        
        modifiers[CulturalContext.EAST_ASIAN] = {
            'group_harmony': 1.3,
            'face_saving': 1.2,
            'indirect_expression': 1.2,
            'emotional_restraint': 1.3
        }
        
        modifiers[CulturalContext.LATIN_AMERICAN] = {
            'family_community': 1.2,
            'emotional_expressiveness': 1.3,
            'relationship_focus': 1.2,
            'warmth': 1.2
        }
        
        modifiers[CulturalContext.NORDIC] = {
            'equality': 1.2,
            'consensus': 1.2,
            'emotional_restraint': 1.1,
            'pragmatism': 1.1
        }
        
        modifiers[CulturalContext.MIDDLE_EASTERN] = {
            'honor': 1.2,
            'hospitality': 1.2,
            'respect_hierarchy': 1.1,
            'family_loyalty': 1.2
        }
        
        return modifiers
    
    def get_sentiment_score(self, word: str, language: Language) -> float:
        """Get sentiment score for word in specified language."""
        if language not in self.lexicons:
            # Fallback to English if language not supported
            language = Language.ENGLISH
        
        lexicon = self.lexicons[language]
        word_lower = word.lower()
        
        # Check positive words
        if word_lower in lexicon.get('positive', {}):
            return lexicon['positive'][word_lower]
        
        # Check negative words
        if word_lower in lexicon.get('negative', {}):
            return lexicon['negative'][word_lower]
        
        # Check neutral words
        if word_lower in lexicon.get('neutral', {}):
            return lexicon['neutral'][word_lower]
        
        # Unknown word
        return 0.0
    
    def analyze_multilingual_sentiment(self, text: str, language: Language) -> Dict[str, float]:
        """Analyze sentiment in specified language."""
        words = text.lower().split()
        
        sentiment_scores = []
        for word in words:
            score = self.get_sentiment_score(word, language)
            if score != 0.0:
                sentiment_scores.append(score)
        
        if not sentiment_scores:
            return {'polarity': 0.0, 'intensity': 0.0, 'confidence': 0.0}
        
        # Calculate aggregate sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        intensity = abs(avg_sentiment)
        confidence = min(1.0, len(sentiment_scores) / len(words))
        
        return {
            'polarity': avg_sentiment,
            'intensity': intensity,
            'confidence': confidence
        }


class CulturalEmotionAdapter:
    """Adapts emotional expressions based on cultural context."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.cultural_modifiers = self._get_cultural_emotion_modifiers()
    
    def _get_cultural_emotion_modifiers(self) -> Dict[str, float]:
        """Get emotion expression modifiers based on cultural context."""
        base_modifiers = {
            'joy': 1.0, 'anger': 1.0, 'sadness': 1.0, 'fear': 1.0,
            'surprise': 1.0, 'disgust': 1.0, 'trust': 1.0, 'anticipation': 1.0
        }
        
        context = self.config.cultural_context
        
        if context == CulturalContext.EAST_ASIAN:
            base_modifiers.update({
                'anger': 0.7,  # More restrained anger expression
                'joy': 0.8,    # More modest joy expression
                'trust': 1.2,  # Higher value on trust/harmony
                'sadness': 0.7  # Less direct sadness expression
            })
        
        elif context == CulturalContext.LATIN_AMERICAN:
            base_modifiers.update({
                'joy': 1.3,      # More expressive joy
                'anger': 1.1,    # More expressive anger
                'sadness': 1.1,  # More expressive sadness
                'trust': 1.2     # Strong trust/family values
            })
        
        elif context == CulturalContext.NORDIC:
            base_modifiers.update({
                'joy': 0.8,      # More reserved joy
                'anger': 0.6,    # Very restrained anger
                'sadness': 0.7,  # Reserved sadness
                'trust': 1.1     # High trust in institutions
            })
        
        elif context == CulturalContext.MIDDLE_EASTERN:
            base_modifiers.update({
                'anger': 0.8,    # Controlled anger expression
                'trust': 1.3,    # High value on trust/honor
                'joy': 1.1,      # Expressive positive emotions
                'fear': 0.8      # Less open fear expression
            })
        
        return base_modifiers
    
    def adapt_emotional_expression(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Adapt emotion scores based on cultural display rules."""
        if not self.config.enable_cultural_adaptation:
            return emotion_scores
        
        adapted_scores = {}
        
        for emotion, score in emotion_scores.items():
            modifier = self.cultural_modifiers.get(emotion, 1.0)
            
            # Apply cultural modifier
            adapted_score = score * modifier
            
            # Apply display rules
            display_rule_key = f"{emotion}_expression"
            if display_rule_key in self.config.emotion_display_rules:
                display_modifier = self.config.emotion_display_rules[display_rule_key]
                adapted_score *= display_modifier
            
            # Apply suppression rules
            suppress_rule_key = f"{emotion}_suppression"
            if suppress_rule_key in self.config.emotion_display_rules:
                suppress_modifier = self.config.emotion_display_rules[suppress_rule_key]
                adapted_score *= (1.0 - suppress_modifier)
            
            adapted_scores[emotion] = max(0.0, min(2.0, adapted_score))
        
        return adapted_scores
    
    def get_cultural_behavior_modifiers(self) -> Dict[str, float]:
        """Get behavioral modifiers based on cultural context."""
        modifiers = {
            'cooperation_tendency': 0.5,
            'competition_tolerance': 0.5,
            'hierarchy_respect': 0.5,
            'individual_autonomy': 0.5,
            'group_conformity': 0.5,
            'direct_communication': 0.5,
            'risk_taking': 0.5,
            'long_term_orientation': 0.5
        }
        
        context = self.config.cultural_context
        
        if context == CulturalContext.EAST_ASIAN:
            modifiers.update({
                'cooperation_tendency': 0.8,
                'hierarchy_respect': 0.8,
                'group_conformity': 0.8,
                'direct_communication': 0.3,
                'long_term_orientation': 0.9
            })
        
        elif context == CulturalContext.WESTERN_INDIVIDUALISTIC:
            modifiers.update({
                'individual_autonomy': 0.9,
                'direct_communication': 0.8,
                'competition_tolerance': 0.7,
                'hierarchy_respect': 0.4,
                'risk_taking': 0.6
            })
        
        elif context == CulturalContext.LATIN_AMERICAN:
            modifiers.update({
                'cooperation_tendency': 0.7,
                'hierarchy_respect': 0.6,
                'direct_communication': 0.7,
                'group_conformity': 0.6,
                'risk_taking': 0.5
            })
        
        elif context == CulturalContext.NORDIC:
            modifiers.update({
                'cooperation_tendency': 0.9,
                'hierarchy_respect': 0.3,
                'direct_communication': 0.9,
                'group_conformity': 0.4,
                'long_term_orientation': 0.8
            })
        
        return modifiers


class GlobalSentimentManager:
    """Manages sentiment analysis across multiple languages and cultures."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.lexicon = MultilingualSentimentLexicon()
        self.cultural_adapter = CulturalEmotionAdapter(config)
        
        logger.info(f"GlobalSentimentManager initialized for {config.primary_language.value} "
                   f"with cultural context {config.cultural_context.value}")
    
    def analyze_sentiment(self, text: str, language: Optional[Language] = None) -> Dict[str, Any]:
        """Analyze sentiment with cultural and linguistic adaptation."""
        if language is None:
            language = self.config.primary_language
        
        # Perform multilingual sentiment analysis
        raw_sentiment = self.lexicon.analyze_multilingual_sentiment(text, language)
        
        # Get cultural behavior modifiers
        cultural_modifiers = self.cultural_adapter.get_cultural_behavior_modifiers()
        
        # Combine sentiment with cultural context
        adapted_sentiment = {
            'polarity': raw_sentiment['polarity'],
            'intensity': raw_sentiment['intensity'] * (1.0 + cultural_modifiers['direct_communication'] * 0.2),
            'confidence': raw_sentiment['confidence'],
            'cultural_context': self.config.cultural_context.value,
            'language': language.value,
            'cultural_modifiers': cultural_modifiers
        }
        
        return adapted_sentiment
    
    def adapt_emotional_behavior(self, base_emotions: Dict[str, float]) -> Dict[str, float]:
        """Adapt emotional behavior based on cultural context."""
        return self.cultural_adapter.adapt_emotional_expression(base_emotions)
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages."""
        return list(self.lexicon.lexicons.keys())
    
    def get_localization_info(self) -> Dict[str, Any]:
        """Get comprehensive localization information."""
        return {
            'primary_language': self.config.primary_language.value,
            'secondary_languages': [lang.value for lang in self.config.secondary_languages],
            'cultural_context': self.config.cultural_context.value,
            'supported_languages': [lang.value for lang in self.get_supported_languages()],
            'cultural_adaptation_enabled': self.config.enable_cultural_adaptation,
            'emotion_display_rules': self.config.emotion_display_rules
        }