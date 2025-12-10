"""
Stakeholder analysis router for categorizing and analyzing comments by stakeholder type.
Provides insights into different stakeholder perspectives on policy proposals.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict, Counter

from backend.app.routers.auth import get_current_user
from backend.app.models.user import User
from backend.app.services.sentiment_service import SentimentAnalyzer
from backend.app.services.summarization_service import SummarizationService
from backend.app.services.visualization_service import VisualizationService

router = APIRouter()

class StakeholderComment(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class StakeholderAnalysisRequest(BaseModel):
    comments: List[StakeholderComment]
    auto_detect_stakeholders: bool = True
    predefined_stakeholders: Optional[Dict[str, List[int]]] = None  # stakeholder_type -> comment_indices

class StakeholderProfile(BaseModel):
    stakeholder_type: str
    comment_count: int
    sentiment_distribution: Dict[str, int]
    dominant_sentiment: str
    average_confidence: float
    key_concerns: List[str]
    key_phrases: List[str]
    representative_comments: List[str]
    policy_stance: str  # support, oppose, neutral

class StakeholderComparison(BaseModel):
    stakeholder_profiles: Dict[str, StakeholderProfile]
    consensus_areas: List[str]
    conflict_areas: List[str]
    stakeholder_alignment: Dict[str, Dict[str, float]]  # similarity matrix
    recommendations: List[str]

class StakeholderInsight(BaseModel):
    total_stakeholders: int
    most_active_stakeholder: str
    most_supportive_stakeholder: str
    most_critical_stakeholder: str
    stakeholder_diversity_score: float
    cross_stakeholder_themes: List[str]
    policy_implications: List[str]

# Initialize services
sentiment_analyzer = SentimentAnalyzer()
summarization_service = SummarizationService()
visualization_service = VisualizationService()

@router.post("/analyze-stakeholders", response_model=Dict[str, Any])
async def analyze_stakeholders(
    request: StakeholderAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze comments by stakeholder type to understand different perspectives.
    
    This endpoint provides:
    - Automatic stakeholder type detection
    - Sentiment analysis by stakeholder group
    - Comparative analysis across stakeholder types
    - Policy implications and recommendations
    """
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    # Step 1: Categorize comments by stakeholder type
    stakeholder_comments = await categorize_comments_by_stakeholder(
        request.comments, 
        request.auto_detect_stakeholders,
        request.predefined_stakeholders
    )
    
    # Step 2: Analyze each stakeholder group
    stakeholder_profiles = {}
    for stakeholder_type, comments in stakeholder_comments.items():
        if comments:  # Only analyze if there are comments
            profile = await analyze_stakeholder_group(stakeholder_type, comments)
            stakeholder_profiles[stakeholder_type] = profile
    
    # Step 3: Generate comparative analysis
    comparison = generate_stakeholder_comparison(stakeholder_profiles)
    
    # Step 4: Generate insights and recommendations
    insights = generate_stakeholder_insights(stakeholder_profiles)
    
    return {
        "stakeholder_analysis": {
            "profiles": {k: v.__dict__ for k, v in stakeholder_profiles.items()},
            "comparison": comparison.__dict__,
            "insights": insights.__dict__,
            "timestamp": datetime.utcnow().isoformat(),
            "total_comments_analyzed": len(request.comments)
        }
    }

@router.post("/stakeholder-comparison", response_model=StakeholderComparison)
async def compare_stakeholders(
    stakeholder_data: Dict[str, List[str]],  # stakeholder_type -> comments
    current_user: User = Depends(get_current_user)
):
    """
    Compare sentiment and perspectives across different stakeholder types.
    """
    if not stakeholder_data:
        raise HTTPException(status_code=400, detail="No stakeholder data provided")
    
    # Analyze each stakeholder group
    stakeholder_profiles = {}
    for stakeholder_type, comments in stakeholder_data.items():
        if comments:
            comment_objects = [StakeholderComment(text=comment) for comment in comments]
            profile = await analyze_stakeholder_group(stakeholder_type, comment_objects)
            stakeholder_profiles[stakeholder_type] = profile
    
    # Generate comparison
    return generate_stakeholder_comparison(stakeholder_profiles)

@router.get("/stakeholder-wordcloud/{stakeholder_type}")
async def generate_stakeholder_wordcloud(
    stakeholder_type: str,
    comments: List[str],
    current_user: User = Depends(get_current_user)
):
    """Generate word cloud specific to a stakeholder type."""
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    # Analyze comments for sentiment
    analysis_results = []
    for comment in comments:
        result = await sentiment_analyzer.analyze_policy_sentiment(comment)
        analysis_results.append({
            "sentiment_score": result.positive_score - result.negative_score,
            "stakeholder_type": stakeholder_type
        })
    
    # Generate stakeholder-specific word cloud
    wordcloud_bytes, word_data = await visualization_service.generate_stakeholder_wordcloud(
        {stakeholder_type: comments},
        {stakeholder_type: analysis_results}
    )
    
    from fastapi.responses import Response
    return Response(content=wordcloud_bytes, media_type="image/png")

@router.post("/stakeholder-summary")
async def generate_stakeholder_summary(
    stakeholder_comments: Dict[str, List[str]],
    current_user: User = Depends(get_current_user)
):
    """
    Generate summaries for each stakeholder group showing their main concerns and positions.
    """
    summaries = {}
    
    for stakeholder_type, comments in stakeholder_comments.items():
        if not comments:
            continue
        
        # Combine comments for this stakeholder type
        combined_text = " ".join(comments)
        
        # Generate policy-specific summary
        summary_result = await summarization_service.policy_summarization(
            combined_text, 
            stakeholder_type=stakeholder_type
        )
        
        summaries[stakeholder_type] = {
            "summary": summary_result.summary_text,
            "key_phrases": summary_result.key_sentences,
            "stakeholder_metadata": summary_result.metadata,
            "comment_count": len(comments)
        }
    
    return {
        "stakeholder_summaries": summaries,
        "timestamp": datetime.utcnow().isoformat()
    }

async def categorize_comments_by_stakeholder(
    comments: List[StakeholderComment],
    auto_detect: bool,
    predefined: Optional[Dict[str, List[int]]]
) -> Dict[str, List[StakeholderComment]]:
    """Categorize comments by stakeholder type."""
    stakeholder_comments = defaultdict(list)
    
    if predefined and not auto_detect:
        # Use predefined categorization
        for stakeholder_type, indices in predefined.items():
            for idx in indices:
                if 0 <= idx < len(comments):
                    stakeholder_comments[stakeholder_type].append(comments[idx])
    else:
        # Auto-detect stakeholder types
        for comment in comments:
            stakeholder_type = sentiment_analyzer._detect_stakeholder_type(comment.text)
            stakeholder_comments[stakeholder_type].append(comment)
    
    return dict(stakeholder_comments)

async def analyze_stakeholder_group(
    stakeholder_type: str, 
    comments: List[StakeholderComment]
) -> StakeholderProfile:
    """Analyze a group of comments from a specific stakeholder type."""
    
    # Analyze sentiment for all comments
    sentiment_results = []
    all_text = []
    
    for comment in comments:
        text = comment.text
        all_text.append(text)
        
        # Get policy-specific sentiment
        policy_result = await sentiment_analyzer.analyze_policy_sentiment(text)
        sentiment_results.append(policy_result)
    
    # Calculate sentiment distribution
    sentiment_counts = Counter()
    confidences = []
    sentiments = []
    
    for result in sentiment_results:
        sentiment_label = result.sentiment_label.value
        sentiment_counts[sentiment_label] += 1
        confidences.append(result.confidence_score)
        sentiments.append(sentiment_label)
    
    # Determine dominant sentiment and policy stance
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
    
    # Determine policy stance
    positive_ratio = sentiment_counts.get("positive", 0) / len(comments)
    negative_ratio = sentiment_counts.get("negative", 0) / len(comments)
    
    if positive_ratio > 0.6:
        policy_stance = "support"
    elif negative_ratio > 0.6:
        policy_stance = "oppose"
    else:
        policy_stance = "neutral"
    
    # Extract key concerns and phrases
    combined_text = " ".join(all_text)
    key_concerns = extract_stakeholder_concerns(combined_text, stakeholder_type)
    key_phrases = extract_key_phrases(all_text)
    
    # Select representative comments
    representative_comments = select_representative_comments(comments, sentiment_results)
    
    return StakeholderProfile(
        stakeholder_type=stakeholder_type,
        comment_count=len(comments),
        sentiment_distribution=dict(sentiment_counts),
        dominant_sentiment=dominant_sentiment,
        average_confidence=sum(confidences) / len(confidences) if confidences else 0,
        key_concerns=key_concerns,
        key_phrases=key_phrases,
        representative_comments=representative_comments,
        policy_stance=policy_stance
    )

def generate_stakeholder_comparison(profiles: Dict[str, StakeholderProfile]) -> StakeholderComparison:
    """Generate comparative analysis across stakeholder types."""
    
    # Find consensus and conflict areas
    consensus_areas = find_consensus_areas(profiles)
    conflict_areas = find_conflict_areas(profiles)
    
    # Calculate stakeholder alignment (similarity matrix)
    alignment = calculate_stakeholder_alignment(profiles)
    
    # Generate recommendations
    recommendations = generate_comparison_recommendations(profiles)
    
    return StakeholderComparison(
        stakeholder_profiles=profiles,
        consensus_areas=consensus_areas,
        conflict_areas=conflict_areas,
        stakeholder_alignment=alignment,
        recommendations=recommendations
    )

def generate_stakeholder_insights(profiles: Dict[str, StakeholderProfile]) -> StakeholderInsight:
    """Generate insights about stakeholder participation and perspectives."""
    
    if not profiles:
        return StakeholderInsight(
            total_stakeholders=0,
            most_active_stakeholder="none",
            most_supportive_stakeholder="none",
            most_critical_stakeholder="none",
            stakeholder_diversity_score=0.0,
            cross_stakeholder_themes=[],
            policy_implications=[]
        )
    
    # Find most active stakeholder
    most_active = max(profiles.items(), key=lambda x: x[1].comment_count)
    
    # Find most supportive/critical stakeholders
    support_scores = {}
    for stakeholder_type, profile in profiles.items():
        positive_ratio = profile.sentiment_distribution.get("positive", 0) / profile.comment_count
        support_scores[stakeholder_type] = positive_ratio
    
    most_supportive = max(support_scores.items(), key=lambda x: x[1])[0] if support_scores else "none"
    most_critical = min(support_scores.items(), key=lambda x: x[1])[0] if support_scores else "none"
    
    # Calculate diversity score
    diversity_score = calculate_diversity_score(profiles)
    
    # Find cross-stakeholder themes
    cross_themes = find_cross_stakeholder_themes(profiles)
    
    # Generate policy implications
    implications = generate_policy_implications(profiles)
    
    return StakeholderInsight(
        total_stakeholders=len(profiles),
        most_active_stakeholder=most_active[0],
        most_supportive_stakeholder=most_supportive,
        most_critical_stakeholder=most_critical,
        stakeholder_diversity_score=diversity_score,
        cross_stakeholder_themes=cross_themes,
        policy_implications=implications
    )

def extract_stakeholder_concerns(text: str, stakeholder_type: str) -> List[str]:
    """Extract key concerns specific to stakeholder type."""
    concern_keywords = {
        'business': ['cost', 'compliance', 'burden', 'impact', 'implementation', 'timeline'],
        'individual': ['rights', 'privacy', 'access', 'fairness', 'transparency'],
        'ngo': ['social', 'environment', 'community', 'protection', 'welfare'],
        'academic': ['research', 'evidence', 'methodology', 'data', 'analysis'],
        'legal': ['constitution', 'jurisdiction', 'precedent', 'enforcement', 'liability'],
        'government': ['coordination', 'resources', 'authority', 'implementation', 'oversight']
    }
    
    text_lower = text.lower()
    keywords = concern_keywords.get(stakeholder_type, [])
    found_concerns = []
    
    for keyword in keywords:
        if keyword in text_lower:
            found_concerns.append(keyword)
    
    return found_concerns[:5]  # Return top 5 concerns

def extract_key_phrases(texts: List[str]) -> List[str]:
    """Extract key phrases from a collection of texts."""
    # Simple keyword extraction based on frequency
    word_freq = Counter()
    
    for text in texts:
        words = text.lower().split()
        # Filter out common words
        filtered_words = [w for w in words if len(w) > 3 and w.isalpha()]
        word_freq.update(filtered_words)
    
    # Return top frequent words as key phrases
    return [word for word, freq in word_freq.most_common(10)]

def select_representative_comments(
    comments: List[StakeholderComment], 
    sentiment_results: List[Any]
) -> List[str]:
    """Select representative comments for each sentiment."""
    representatives = []
    
    # Group by sentiment
    sentiment_groups = defaultdict(list)
    for comment, result in zip(comments, sentiment_results):
        sentiment = result.sentiment_label.value
        sentiment_groups[sentiment].append(comment.text)
    
    # Select one representative from each sentiment group
    for sentiment, group_comments in sentiment_groups.items():
        if group_comments:
            # Select the comment with moderate length (not too short or long)
            selected = min(group_comments, key=lambda x: abs(len(x.split()) - 20))
            representatives.append(selected)
    
    return representatives[:3]  # Max 3 representatives

def find_consensus_areas(profiles: Dict[str, StakeholderProfile]) -> List[str]:
    """Find areas where stakeholders agree."""
    consensus = []
    
    # Check if majority of stakeholders have same dominant sentiment
    sentiments = [profile.dominant_sentiment for profile in profiles.values()]
    sentiment_counts = Counter(sentiments)
    
    if len(sentiment_counts) > 0:
        most_common_sentiment, count = sentiment_counts.most_common(1)[0]
        if count >= len(profiles) * 0.7:  # 70% agreement
            consensus.append(f"Majority stakeholder sentiment: {most_common_sentiment}")
    
    # Find common concerns across stakeholders
    all_concerns = []
    for profile in profiles.values():
        all_concerns.extend(profile.key_concerns)
    
    concern_counts = Counter(all_concerns)
    common_concerns = [concern for concern, count in concern_counts.items() 
                      if count >= len(profiles) * 0.5]
    
    if common_concerns:
        consensus.append(f"Common concerns: {', '.join(common_concerns[:3])}")
    
    return consensus

def find_conflict_areas(profiles: Dict[str, StakeholderProfile]) -> List[str]:
    """Find areas where stakeholders disagree."""
    conflicts = []
    
    # Check for opposing policy stances
    stances = [profile.policy_stance for profile in profiles.values()]
    stance_counts = Counter(stances)
    
    if "support" in stance_counts and "oppose" in stance_counts:
        conflicts.append("Divided stakeholder opinion on policy support")
    
    # Check for sentiment polarization
    sentiments = [profile.dominant_sentiment for profile in profiles.values()]
    if "positive" in sentiments and "negative" in sentiments:
        conflicts.append("Polarized sentiment across stakeholder groups")
    
    # Find stakeholder-specific concerns (not shared)
    stakeholder_specific_concerns = {}
    for stakeholder_type, profile in profiles.items():
        unique_concerns = []
        for concern in profile.key_concerns:
            # Check if this concern appears in other stakeholder groups
            appears_elsewhere = any(
                concern in other_profile.key_concerns 
                for other_type, other_profile in profiles.items() 
                if other_type != stakeholder_type
            )
            if not appears_elsewhere:
                unique_concerns.append(concern)
        
        if unique_concerns:
            stakeholder_specific_concerns[stakeholder_type] = unique_concerns
    
    if stakeholder_specific_concerns:
        conflicts.append("Stakeholder-specific concerns with limited cross-support")
    
    return conflicts

def calculate_stakeholder_alignment(profiles: Dict[str, StakeholderProfile]) -> Dict[str, Dict[str, float]]:
    """Calculate similarity between stakeholder groups."""
    alignment = {}
    stakeholder_types = list(profiles.keys())
    
    for i, type1 in enumerate(stakeholder_types):
        alignment[type1] = {}
        for j, type2 in enumerate(stakeholder_types):
            if i == j:
                alignment[type1][type2] = 1.0
            else:
                # Calculate similarity based on sentiment and concerns
                profile1 = profiles[type1]
                profile2 = profiles[type2]
                
                # Sentiment similarity
                sentiment_similarity = 1.0 if profile1.dominant_sentiment == profile2.dominant_sentiment else 0.0
                
                # Concern overlap
                concerns1 = set(profile1.key_concerns)
                concerns2 = set(profile2.key_concerns)
                concern_similarity = len(concerns1.intersection(concerns2)) / len(concerns1.union(concerns2)) if concerns1.union(concerns2) else 0
                
                # Policy stance similarity
                stance_similarity = 1.0 if profile1.policy_stance == profile2.policy_stance else 0.0
                
                # Weighted average
                overall_similarity = (sentiment_similarity * 0.4 + concern_similarity * 0.4 + stance_similarity * 0.2)
                alignment[type1][type2] = round(overall_similarity, 2)
    
    return alignment

def calculate_diversity_score(profiles: Dict[str, StakeholderProfile]) -> float:
    """Calculate how diverse stakeholder perspectives are."""
    if len(profiles) <= 1:
        return 0.0
    
    # Diversity based on different policy stances
    stances = [profile.policy_stance for profile in profiles.values()]
    unique_stances = len(set(stances))
    stance_diversity = unique_stances / 3  # Max 3 stances (support, oppose, neutral)
    
    # Diversity based on sentiment distribution
    sentiments = [profile.dominant_sentiment for profile in profiles.values()]
    unique_sentiments = len(set(sentiments))
    sentiment_diversity = unique_sentiments / 3  # Max 3 sentiments
    
    # Diversity based on participation (comment count variation)
    comment_counts = [profile.comment_count for profile in profiles.values()]
    if max(comment_counts) > 0:
        participation_diversity = 1 - (min(comment_counts) / max(comment_counts))
    else:
        participation_diversity = 0
    
    # Overall diversity score
    return round((stance_diversity + sentiment_diversity + participation_diversity) / 3, 2)

def find_cross_stakeholder_themes(profiles: Dict[str, StakeholderProfile]) -> List[str]:
    """Find themes that appear across multiple stakeholder groups."""
    # Collect all phrases from all stakeholders
    all_phrases = []
    for profile in profiles.values():
        all_phrases.extend(profile.key_phrases)
    
    # Find phrases that appear across multiple stakeholders
    phrase_counts = Counter(all_phrases)
    cross_themes = [phrase for phrase, count in phrase_counts.items() 
                   if count >= min(3, len(profiles))]
    
    return cross_themes[:5]  # Return top 5 cross-themes

def generate_policy_implications(profiles: Dict[str, StakeholderProfile]) -> List[str]:
    """Generate policy implications based on stakeholder analysis."""
    implications = []
    
    # Analyze support vs opposition
    support_count = sum(1 for profile in profiles.values() if profile.policy_stance == "support")
    oppose_count = sum(1 for profile in profiles.values() if profile.policy_stance == "oppose")
    
    if support_count > oppose_count:
        implications.append("Policy has stakeholder support - consider expedited implementation")
    elif oppose_count > support_count:
        implications.append("Significant stakeholder opposition - review and address concerns")
    else:
        implications.append("Mixed stakeholder response - engage in dialogue and compromise")
    
    # Business stakeholder implications
    if "business" in profiles:
        business_profile = profiles["business"]
        if "cost" in business_profile.key_concerns or "burden" in business_profile.key_concerns:
            implications.append("Address business concerns about implementation costs and regulatory burden")
    
    # Individual stakeholder implications
    if "individual" in profiles:
        individual_profile = profiles["individual"]
        if "rights" in individual_profile.key_concerns or "privacy" in individual_profile.key_concerns:
            implications.append("Strengthen provisions for individual rights and privacy protection")
    
    # Cross-stakeholder engagement
    if len(profiles) >= 3:
        implications.append("Diverse stakeholder participation - establish ongoing consultation mechanism")
    
    return implications

def generate_comparison_recommendations(profiles: Dict[str, StakeholderProfile]) -> List[str]:
    """Generate recommendations based on stakeholder comparison."""
    recommendations = []
    
    # Engagement recommendations
    if len(profiles) < 3:
        recommendations.append("Increase outreach to ensure broader stakeholder representation")
    
    # Address opposing views
    opposing_stakeholders = [st for st, profile in profiles.items() if profile.policy_stance == "oppose"]
    if opposing_stakeholders:
        recommendations.append(f"Engage directly with opposing stakeholders: {', '.join(opposing_stakeholders)}")
    
    # Build on consensus
    consensus_items = find_consensus_areas(profiles)
    if consensus_items:
        recommendations.append("Leverage consensus areas to build broader support")
    
    # Address conflicts
    conflicts = find_conflict_areas(profiles)
    if conflicts:
        recommendations.append("Facilitate dialogue to address conflicting perspectives")
    
    recommendations.append("Consider stakeholder-specific communication strategies")
    recommendations.append("Monitor ongoing stakeholder sentiment through implementation")
    
    return recommendations