"""
Legislative context analysis service for mapping comments to specific provisions
and providing context-aware analysis for draft legislation feedback.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict, Counter

# NLP libraries
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False

from backend.app.services.sentiment_service import SentimentAnalyzer
from backend.app.services.summarization_service import SummarizationService

class ProvisionType(str, Enum):
    SECTION = "section"
    CLAUSE = "clause" 
    SUB_CLAUSE = "sub_clause"
    ARTICLE = "article"
    CHAPTER = "chapter"
    SCHEDULE = "schedule"
    AMENDMENT = "amendment"

@dataclass
class LegislativeProvision:
    """Represents a specific provision in legislation."""
    provision_id: str
    provision_type: ProvisionType
    title: str
    content: str
    parent_provision: Optional[str] = None
    sub_provisions: List[str] = None

@dataclass
class ProvisionMapping:
    """Maps comment to specific legislative provision."""
    comment_text: str
    provision_id: str
    provision_type: ProvisionType
    provision_title: str
    mapping_confidence: float
    mapping_method: str  # "explicit_reference", "keyword_match", "semantic_similarity"
    context_snippet: str

@dataclass
class ProvisionAnalysis:
    """Analysis of comments for a specific provision."""
    provision: LegislativeProvision
    comment_count: int
    sentiment_distribution: Dict[str, int]
    dominant_sentiment: str
    key_concerns: List[str]
    suggested_amendments: List[str]
    stakeholder_perspectives: Dict[str, Dict[str, Any]]
    provision_summary: str

@dataclass
class LegislativeContextResult:
    """Complete legislative context analysis result."""
    total_comments: int
    mapped_comments: int
    unmapped_comments: int
    provision_analyses: List[ProvisionAnalysis]
    cross_provision_themes: List[str]
    legislative_recommendations: List[str]
    implementation_considerations: List[str]

class LegislativeContextService:
    """Service for legislative context analysis and provision mapping."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm") if _SPACY_AVAILABLE else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarization_service = SummarizationService()
        
        # Legislative patterns for identifying references
        self.provision_patterns = {
            'section': re.compile(r'\b(?:section|sec\.?)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            'clause': re.compile(r'\b(?:clause|cl\.?)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            'sub_clause': re.compile(r'\b(?:sub-?clause|sub-?section)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            'article': re.compile(r'\b(?:article|art\.?)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            'chapter': re.compile(r'\b(?:chapter|ch\.?)\s*(\d+(?:\w+)*)\b', re.IGNORECASE),
            'schedule': re.compile(r'\b(?:schedule|sch\.?)\s*(\d+(?:\w+)*)\b', re.IGNORECASE),
            'amendment': re.compile(r'\b(?:amendment|amend\.?)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE)
        }
        
        # Keywords for different legal concepts
        self.legal_keywords = {
            'rights': ['right', 'rights', 'liberty', 'freedom', 'privilege'],
            'obligations': ['duty', 'obligation', 'responsibility', 'liable', 'accountable'],
            'procedures': ['procedure', 'process', 'method', 'mechanism', 'protocol'],
            'penalties': ['penalty', 'fine', 'punishment', 'sanction', 'violation'],
            'definitions': ['definition', 'means', 'includes', 'refers to', 'interpreted'],
            'exemptions': ['exemption', 'exception', 'excluded', 'not applicable', 'waiver'],
            'compliance': ['compliance', 'conform', 'adhere', 'follow', 'observe'],
            'enforcement': ['enforcement', 'implement', 'execute', 'apply', 'ensure']
        }
    
    async def analyze_legislative_context(
        self, 
        comments: List[str],
        legislation_structure: Optional[List[LegislativeProvision]] = None
    ) -> LegislativeContextResult:
        """
        Perform comprehensive legislative context analysis.
        
        Args:
            comments: List of comment texts
            legislation_structure: Optional structure of the legislation
            
        Returns:
            LegislativeContextResult: Complete analysis
        """
        # Step 1: Map comments to provisions
        provision_mappings = await self.map_comments_to_provisions(comments, legislation_structure)
        
        # Step 2: Group comments by provision
        comments_by_provision = self._group_comments_by_provision(provision_mappings)
        
        # Step 3: Analyze each provision
        provision_analyses = []
        for provision_id, provision_comments in comments_by_provision.items():
            if provision_comments:
                analysis = await self._analyze_provision_comments(
                    provision_id, provision_comments, legislation_structure
                )
                provision_analyses.append(analysis)
        
        # Step 4: Find cross-provision themes
        cross_themes = self._find_cross_provision_themes(provision_analyses)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_legislative_recommendations(provision_analyses)
        implementation_considerations = self._generate_implementation_considerations(provision_analyses)
        
        mapped_count = sum(len(mappings) for mappings in comments_by_provision.values())
        
        return LegislativeContextResult(
            total_comments=len(comments),
            mapped_comments=mapped_count,
            unmapped_comments=len(comments) - mapped_count,
            provision_analyses=provision_analyses,
            cross_provision_themes=cross_themes,
            legislative_recommendations=recommendations,
            implementation_considerations=implementation_considerations
        )
    
    async def map_comments_to_provisions(
        self,
        comments: List[str],
        legislation_structure: Optional[List[LegislativeProvision]] = None
    ) -> List[ProvisionMapping]:
        """Map individual comments to specific legislative provisions."""
        mappings = []
        
        for comment in comments:
            # Try explicit reference mapping first
            explicit_mappings = self._find_explicit_provision_references(comment)
            
            if explicit_mappings:
                mappings.extend(explicit_mappings)
            else:
                # Try keyword-based mapping
                keyword_mapping = self._find_keyword_based_mapping(comment, legislation_structure)
                if keyword_mapping:
                    mappings.append(keyword_mapping)
                else:
                    # Try semantic similarity (if structure provided)
                    if legislation_structure:
                        semantic_mapping = await self._find_semantic_mapping(comment, legislation_structure)
                        if semantic_mapping:
                            mappings.append(semantic_mapping)
        
        return mappings
    
    def _find_explicit_provision_references(self, comment: str) -> List[ProvisionMapping]:
        """Find explicit references to provisions in comment text."""
        mappings = []
        
        for provision_type, pattern in self.provision_patterns.items():
            matches = pattern.finditer(comment)
            
            for match in matches:
                provision_number = match.group(1)
                provision_id = f"{provision_type}_{provision_number}"
                
                # Extract context around the match
                start = max(0, match.start() - 50)
                end = min(len(comment), match.end() + 50)
                context = comment[start:end]
                
                mapping = ProvisionMapping(
                    comment_text=comment,
                    provision_id=provision_id,
                    provision_type=ProvisionType(provision_type),
                    provision_title=f"{provision_type.title()} {provision_number}",
                    mapping_confidence=0.9,  # High confidence for explicit references
                    mapping_method="explicit_reference",
                    context_snippet=context
                )
                mappings.append(mapping)
        
        return mappings
    
    def _find_keyword_based_mapping(
        self, 
        comment: str, 
        legislation_structure: Optional[List[LegislativeProvision]]
    ) -> Optional[ProvisionMapping]:
        """Find provision mapping based on legal concept keywords."""
        comment_lower = comment.lower()
        
        # Score provisions based on keyword matches
        provision_scores = {}
        
        if legislation_structure:
            for provision in legislation_structure:
                score = 0
                
                # Check if provision content contains keywords found in comment
                provision_content = provision.content.lower()
                
                for concept, keywords in self.legal_keywords.items():
                    comment_keyword_count = sum(1 for keyword in keywords if keyword in comment_lower)
                    provision_keyword_count = sum(1 for keyword in keywords if keyword in provision_content)
                    
                    if comment_keyword_count > 0 and provision_keyword_count > 0:
                        score += comment_keyword_count * provision_keyword_count
                
                if score > 0:
                    provision_scores[provision.provision_id] = score
        
        if provision_scores:
            # Select provision with highest score
            best_provision_id = max(provision_scores, key=provision_scores.get)
            best_provision = next(p for p in legislation_structure if p.provision_id == best_provision_id)
            
            confidence = min(0.8, provision_scores[best_provision_id] / 10)  # Normalize confidence
            
            return ProvisionMapping(
                comment_text=comment,
                provision_id=best_provision_id,
                provision_type=best_provision.provision_type,
                provision_title=best_provision.title,
                mapping_confidence=confidence,
                mapping_method="keyword_match",
                context_snippet=comment[:100]  # First 100 characters as context
            )
        
        return None
    
    async def _find_semantic_mapping(
        self, 
        comment: str, 
        legislation_structure: List[LegislativeProvision]
    ) -> Optional[ProvisionMapping]:
        """Find provision mapping using semantic similarity."""
        if not self.nlp:
            return None
        
        try:
            comment_doc = self.nlp(comment)
            best_similarity = 0
            best_provision = None
            
            for provision in legislation_structure:
                provision_doc = self.nlp(provision.content)
                similarity = comment_doc.similarity(provision_doc)
                
                if similarity > best_similarity and similarity > 0.3:  # Minimum threshold
                    best_similarity = similarity
                    best_provision = provision
            
            if best_provision:
                return ProvisionMapping(
                    comment_text=comment,
                    provision_id=best_provision.provision_id,
                    provision_type=best_provision.provision_type,
                    provision_title=best_provision.title,
                    mapping_confidence=best_similarity,
                    mapping_method="semantic_similarity",
                    context_snippet=comment[:100]
                )
        
        except Exception as e:
            print(f"Error in semantic mapping: {e}")
        
        return None
    
    def _group_comments_by_provision(self, mappings: List[ProvisionMapping]) -> Dict[str, List[str]]:
        """Group comments by their mapped provisions."""
        grouped = defaultdict(list)
        
        for mapping in mappings:
            grouped[mapping.provision_id].append(mapping.comment_text)
        
        return dict(grouped)
    
    async def _analyze_provision_comments(
        self,
        provision_id: str,
        comments: List[str],
        legislation_structure: Optional[List[LegislativeProvision]]
    ) -> ProvisionAnalysis:
        """Analyze comments for a specific provision."""
        
        # Find provision details
        provision = None
        if legislation_structure:
            provision = next((p for p in legislation_structure if p.provision_id == provision_id), None)
        
        if not provision:
            # Create a default provision object
            provision = LegislativeProvision(
                provision_id=provision_id,
                provision_type=ProvisionType.SECTION,
                title=provision_id.replace('_', ' ').title(),
                content=""
            )
        
        # Analyze sentiment for all comments
        sentiment_results = []
        for comment in comments:
            result = await self.sentiment_analyzer.analyze_policy_sentiment(comment)
            sentiment_results.append(result)
        
        # Calculate sentiment distribution
        sentiment_counts = Counter()
        for result in sentiment_results:
            sentiment_counts[result.sentiment_label.value] += 1
        
        dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
        
        # Extract key concerns
        key_concerns = self._extract_provision_concerns(comments)
        
        # Extract suggested amendments
        suggested_amendments = self._extract_suggested_amendments(comments)
        
        # Analyze by stakeholder type
        stakeholder_perspectives = await self._analyze_stakeholder_perspectives(comments)
        
        # Generate provision-specific summary
        combined_text = " ".join(comments)
        summary_result = await self.summarization_service.policy_summarization(combined_text)
        
        return ProvisionAnalysis(
            provision=provision,
            comment_count=len(comments),
            sentiment_distribution=dict(sentiment_counts),
            dominant_sentiment=dominant_sentiment,
            key_concerns=key_concerns,
            suggested_amendments=suggested_amendments,
            stakeholder_perspectives=stakeholder_perspectives,
            provision_summary=summary_result.summary_text
        )
    
    def _extract_provision_concerns(self, comments: List[str]) -> List[str]:
        """Extract key concerns from provision-specific comments."""
        concern_indicators = [
            'concern', 'worry', 'issue', 'problem', 'difficulty', 'challenge',
            'unclear', 'ambiguous', 'vague', 'confusing', 'contradictory'
        ]
        
        concerns = []
        for comment in comments:
            comment_lower = comment.lower()
            
            # Find sentences with concern indicators
            sentences = comment.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in concern_indicators):
                    # Extract the concern phrase
                    concern = sentence.strip()
                    if len(concern) > 10 and len(concern) < 200:
                        concerns.append(concern)
        
        # Return most frequent concerns
        concern_counts = Counter(concerns)
        return [concern for concern, count in concern_counts.most_common(5)]
    
    def _extract_suggested_amendments(self, comments: List[str]) -> List[str]:
        """Extract suggested amendments from comments."""
        amendment_indicators = [
            'suggest', 'recommend', 'propose', 'should', 'could', 'modify',
            'change', 'add', 'remove', 'delete', 'replace', 'amend'
        ]
        
        amendments = []
        for comment in comments:
            comment_lower = comment.lower()
            
            # Find sentences with amendment indicators
            sentences = comment.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in amendment_indicators):
                    amendment = sentence.strip()
                    if len(amendment) > 15 and len(amendment) < 300:
                        amendments.append(amendment)
        
        # Return unique amendments
        return list(set(amendments))[:5]
    
    async def _analyze_stakeholder_perspectives(self, comments: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze perspectives by stakeholder type for provision-specific comments."""
        stakeholder_comments = defaultdict(list)
        
        # Group comments by stakeholder type
        for comment in comments:
            stakeholder_type = self.sentiment_analyzer._detect_stakeholder_type(comment)
            stakeholder_comments[stakeholder_type].append(comment)
        
        # Analyze each stakeholder group
        perspectives = {}
        for stakeholder_type, group_comments in stakeholder_comments.items():
            if group_comments:
                # Analyze sentiment for this stakeholder group
                sentiment_results = []
                for comment in group_comments:
                    result = await self.sentiment_analyzer.analyze_policy_sentiment(comment)
                    sentiment_results.append(result)
                
                # Calculate group statistics
                sentiments = [r.sentiment_label.value for r in sentiment_results]
                sentiment_counts = Counter(sentiments)
                dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
                
                perspectives[stakeholder_type] = {
                    "comment_count": len(group_comments),
                    "dominant_sentiment": dominant_sentiment,
                    "sentiment_distribution": dict(sentiment_counts),
                    "concerns": self._extract_provision_concerns(group_comments)[:3]
                }
        
        return perspectives
    
    def _find_cross_provision_themes(self, provision_analyses: List[ProvisionAnalysis]) -> List[str]:
        """Find themes that appear across multiple provisions."""
        all_concerns = []
        for analysis in provision_analyses:
            all_concerns.extend(analysis.key_concerns)
        
        # Find concerns that appear in multiple provisions
        concern_counts = Counter(all_concerns)
        cross_themes = [concern for concern, count in concern_counts.items() 
                       if count >= 2 and count <= len(provision_analyses)]
        
        return cross_themes[:5]
    
    def _generate_legislative_recommendations(self, provision_analyses: List[ProvisionAnalysis]) -> List[str]:
        """Generate recommendations for the legislation based on provision analyses."""
        recommendations = []
        
        # Analyze overall sentiment patterns
        negative_provisions = [a for a in provision_analyses if a.dominant_sentiment == "negative"]
        positive_provisions = [a for a in provision_analyses if a.dominant_sentiment == "positive"]
        
        if len(negative_provisions) > len(positive_provisions):
            recommendations.append("Significant concerns identified across multiple provisions - consider comprehensive review")
        
        # Identify provisions with most concerns
        high_concern_provisions = sorted(provision_analyses, key=lambda x: len(x.key_concerns), reverse=True)[:3]
        if high_concern_provisions:
            provision_names = [p.provision.title for p in high_concern_provisions]
            recommendations.append(f"Priority provisions for revision: {', '.join(provision_names)}")
        
        # Check for stakeholder consensus
        unanimous_concerns = []
        for analysis in provision_analyses:
            if len(analysis.stakeholder_perspectives) >= 2:
                # Check if all stakeholder types have the same dominant sentiment
                sentiments = [sp["dominant_sentiment"] for sp in analysis.stakeholder_perspectives.values()]
                if len(set(sentiments)) == 1 and sentiments[0] == "negative":
                    unanimous_concerns.append(analysis.provision.title)
        
        if unanimous_concerns:
            recommendations.append(f"Unanimous stakeholder concerns for: {', '.join(unanimous_concerns)}")
        
        # Amendment suggestions
        total_amendments = sum(len(a.suggested_amendments) for a in provision_analyses)
        if total_amendments > 0:
            recommendations.append(f"Review {total_amendments} specific amendment suggestions from stakeholders")
        
        recommendations.append("Conduct targeted consultations for provisions with mixed feedback")
        recommendations.append("Consider phased implementation for complex provisions")
        
        return recommendations
    
    def _generate_implementation_considerations(self, provision_analyses: List[ProvisionAnalysis]) -> List[str]:
        """Generate implementation considerations based on analysis."""
        considerations = []
        
        # Business implementation concerns
        business_concerns = []
        for analysis in provision_analyses:
            if "business" in analysis.stakeholder_perspectives:
                business_perspective = analysis.stakeholder_perspectives["business"]
                if business_perspective["dominant_sentiment"] == "negative":
                    business_concerns.append(analysis.provision.title)
        
        if business_concerns:
            considerations.append(f"Address business implementation challenges for: {', '.join(business_concerns)}")
        
        # Compliance complexity
        complex_provisions = [a for a in provision_analyses if "unclear" in " ".join(a.key_concerns).lower()]
        if complex_provisions:
            considerations.append("Provide implementation guidelines for provisions with clarity concerns")
        
        # Resource requirements
        considerations.append("Assess resource requirements for provisions with stakeholder opposition")
        considerations.append("Develop compliance monitoring mechanisms for high-concern provisions")
        considerations.append("Create stakeholder engagement plan for ongoing implementation feedback")
        
        return considerations