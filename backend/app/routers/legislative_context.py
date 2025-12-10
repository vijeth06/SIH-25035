"""
Legislative context analysis router for mapping comments to specific provisions
and providing context-aware analysis for draft legislation feedback.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime

from backend.app.routers.auth import get_current_user
from backend.app.models.user import User
from backend.app.services.legislative_context_service import (
    LegislativeContextService, 
    LegislativeProvision, 
    ProvisionType,
    LegislativeContextResult
)

router = APIRouter()

class LegislationStructureRequest(BaseModel):
    provision_id: str
    provision_type: ProvisionType
    title: str
    content: str
    parent_provision: Optional[str] = None

class LegislativeAnalysisRequest(BaseModel):
    comments: List[str]
    legislation_structure: Optional[List[LegislationStructureRequest]] = None
    legislation_title: Optional[str] = None
    legislation_type: Optional[str] = None  # "bill", "amendment", "regulation", etc.

class ProvisionMappingRequest(BaseModel):
    comments: List[str]
    target_provisions: Optional[List[str]] = None  # Specific provisions to focus on

class ProvisionAnalysisResponse(BaseModel):
    provision_id: str
    provision_title: str
    comment_count: int
    sentiment_distribution: Dict[str, int]
    dominant_sentiment: str
    key_concerns: List[str]
    suggested_amendments: List[str]
    stakeholder_breakdown: Dict[str, Dict[str, Any]]
    summary: str

# Initialize service
legislative_service = LegislativeContextService()

@router.post("/analyze-legislative-context")
async def analyze_legislative_context(
    request: LegislativeAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform comprehensive legislative context analysis to map comments to specific 
    provisions and provide context-aware insights for draft legislation.
    
    This addresses the MCA eConsultation requirement for systematic analysis
    to ensure no observations are overlooked.
    """
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    # Convert request structure to service objects
    legislation_structure = None
    if request.legislation_structure:
        legislation_structure = [
            LegislativeProvision(
                provision_id=item.provision_id,
                provision_type=item.provision_type,
                title=item.title,
                content=item.content,
                parent_provision=item.parent_provision
            )
            for item in request.legislation_structure
        ]
    
    # Perform analysis
    result = await legislative_service.analyze_legislative_context(
        request.comments, 
        legislation_structure
    )
    
    # Format response
    provision_responses = []
    for analysis in result.provision_analyses:
        provision_responses.append(ProvisionAnalysisResponse(
            provision_id=analysis.provision.provision_id,
            provision_title=analysis.provision.title,
            comment_count=analysis.comment_count,
            sentiment_distribution=analysis.sentiment_distribution,
            dominant_sentiment=analysis.dominant_sentiment,
            key_concerns=analysis.key_concerns,
            suggested_amendments=analysis.suggested_amendments,
            stakeholder_breakdown=analysis.stakeholder_perspectives,
            summary=analysis.provision_summary
        ))
    
    return {
        "legislative_analysis": {
            "legislation_title": request.legislation_title or "Draft Legislation",
            "legislation_type": request.legislation_type or "bill",
            "analysis_summary": {
                "total_comments": result.total_comments,
                "mapped_comments": result.mapped_comments,
                "unmapped_comments": result.unmapped_comments,
                "coverage_percentage": (result.mapped_comments / result.total_comments * 100) if result.total_comments > 0 else 0
            },
            "provision_analyses": [p.__dict__ for p in provision_responses],
            "cross_provision_themes": result.cross_provision_themes,
            "legislative_recommendations": result.legislative_recommendations,
            "implementation_considerations": result.implementation_considerations,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@router.post("/map-provisions")
async def map_comments_to_provisions(
    request: ProvisionMappingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Map individual comments to specific legislative provisions.
    
    Helps identify which parts of the legislation are receiving 
    the most feedback and ensures comprehensive coverage.
    """
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    # Perform provision mapping
    mappings = await legislative_service.map_comments_to_provisions(request.comments)
    
    # Format mappings for response
    formatted_mappings = []
    for mapping in mappings:
        formatted_mappings.append({
            "comment_text": mapping.comment_text,
            "provision_id": mapping.provision_id,
            "provision_type": mapping.provision_type.value,
            "provision_title": mapping.provision_title,
            "mapping_confidence": mapping.mapping_confidence,
            "mapping_method": mapping.mapping_method,
            "context_snippet": mapping.context_snippet
        })
    
    # Generate mapping statistics
    mapping_stats = {
        "total_comments": len(request.comments),
        "mapped_comments": len(mappings),
        "mapping_coverage": (len(mappings) / len(request.comments) * 100) if request.comments else 0,
        "provisions_referenced": len(set(m.provision_id for m in mappings)),
        "mapping_methods_used": list(set(m.mapping_method for m in mappings))
    }
    
    return {
        "provision_mappings": formatted_mappings,
        "mapping_statistics": mapping_stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/provision-specific-analysis/{provision_id}")
async def analyze_specific_provision(
    provision_id: str,
    comments: List[str],
    provision_title: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze comments for a specific provision in detail.
    
    Provides focused analysis for individual provisions to help
    understand stakeholder feedback on specific parts of legislation.
    """
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    # Create a mock provision for analysis
    from backend.app.services.legislative_context_service import LegislativeProvision, ProvisionType
    
    provision = LegislativeProvision(
        provision_id=provision_id,
        provision_type=ProvisionType.SECTION,  # Default type
        title=provision_title or f"Provision {provision_id}",
        content=""
    )
    
    # Analyze the provision
    analysis = await legislative_service._analyze_provision_comments(
        provision_id, comments, [provision]
    )
    
    return {
        "provision_analysis": {
            "provision_id": provision_id,
            "provision_title": analysis.provision.title,
            "comment_count": analysis.comment_count,
            "sentiment_analysis": {
                "distribution": analysis.sentiment_distribution,
                "dominant_sentiment": analysis.dominant_sentiment
            },
            "stakeholder_perspectives": analysis.stakeholder_perspectives,
            "key_concerns": analysis.key_concerns,
            "suggested_amendments": analysis.suggested_amendments,
            "provision_summary": analysis.provision_summary,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    }

@router.post("/cross-provision-analysis")
async def analyze_cross_provision_themes(
    comments_by_provision: Dict[str, List[str]],
    current_user: User = Depends(get_current_user)
):
    """
    Analyze themes and concerns that span across multiple provisions.
    
    Helps identify systemic issues or broad stakeholder concerns
    that affect multiple parts of the legislation.
    """
    if not comments_by_provision:
        raise HTTPException(status_code=400, detail="No provision-comment mapping provided")
    
    # Analyze each provision
    provision_analyses = []
    for provision_id, comments in comments_by_provision.items():
        if comments:
            analysis = await legislative_service._analyze_provision_comments(
                provision_id, comments, None
            )
            provision_analyses.append(analysis)
    
    # Find cross-provision themes
    cross_themes = legislative_service._find_cross_provision_themes(provision_analyses)
    
    # Generate recommendations
    recommendations = legislative_service._generate_legislative_recommendations(provision_analyses)
    
    # Calculate cross-provision statistics
    total_comments = sum(len(comments) for comments in comments_by_provision.values())
    provisions_with_concerns = len([a for a in provision_analyses if a.key_concerns])
    avg_concerns_per_provision = sum(len(a.key_concerns) for a in provision_analyses) / len(provision_analyses) if provision_analyses else 0
    
    return {
        "cross_provision_analysis": {
            "total_provisions_analyzed": len(provision_analyses),
            "total_comments_analyzed": total_comments,
            "provisions_with_concerns": provisions_with_concerns,
            "average_concerns_per_provision": round(avg_concerns_per_provision, 2),
            "cross_provision_themes": cross_themes,
            "legislative_recommendations": recommendations,
            "systemic_issues": [
                theme for theme in cross_themes 
                if any(keyword in theme.lower() for keyword in ['unclear', 'confusing', 'contradictory', 'ambiguous'])
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@router.get("/provision-coverage-report")
async def generate_provision_coverage_report(
    total_provisions: int,
    analyzed_provisions: int,
    current_user: User = Depends(get_current_user)
):
    """
    Generate a coverage report showing which provisions received feedback
    and which may need additional consultation.
    
    Helps ensure comprehensive review as required by MCA eConsultation objectives.
    """
    coverage_percentage = (analyzed_provisions / total_provisions * 100) if total_provisions > 0 else 0
    uncovered_provisions = total_provisions - analyzed_provisions
    
    # Generate recommendations based on coverage
    recommendations = []
    if coverage_percentage < 50:
        recommendations.append("Low provision coverage - consider targeted outreach for uncovered provisions")
    elif coverage_percentage < 80:
        recommendations.append("Moderate provision coverage - review uncovered provisions for importance")
    else:
        recommendations.append("Good provision coverage - ensure quality analysis of all feedback")
    
    if uncovered_provisions > 0:
        recommendations.append(f"Consider specific consultation for {uncovered_provisions} uncovered provisions")
    
    # Coverage assessment
    coverage_assessment = "excellent" if coverage_percentage >= 90 else \
                         "good" if coverage_percentage >= 70 else \
                         "moderate" if coverage_percentage >= 50 else "needs_improvement"
    
    return {
        "provision_coverage_report": {
            "total_provisions": total_provisions,
            "provisions_with_feedback": analyzed_provisions,
            "provisions_without_feedback": uncovered_provisions,
            "coverage_percentage": round(coverage_percentage, 2),
            "coverage_assessment": coverage_assessment,
            "recommendations": recommendations,
            "next_steps": [
                "Review provisions without feedback for potential consultation gaps",
                "Assess whether uncovered provisions require stakeholder input",
                "Consider targeted outreach for important uncovered provisions",
                "Document rationale for provisions with no feedback received"
            ],
            "report_timestamp": datetime.utcnow().isoformat()
        }
    }

@router.post("/amendment-impact-analysis")
async def analyze_amendment_impact(
    original_provision: str,
    suggested_amendments: List[str],
    related_comments: List[str],
    current_user: User = Depends(get_current_user)
):
    """
    Analyze the potential impact of suggested amendments on stakeholder concerns.
    
    Helps policymakers understand how proposed changes might address
    stakeholder feedback and concerns.
    """
    if not suggested_amendments:
        raise HTTPException(status_code=400, detail="No amendments provided")
    
    # Analyze sentiment of comments related to original provision
    original_sentiment_results = []
    for comment in related_comments:
        result = await legislative_service.sentiment_analyzer.analyze_policy_sentiment(comment)
        original_sentiment_results.append(result)
    
    # Extract concerns from comments
    concerns = legislative_service._extract_provision_concerns(related_comments)
    
    # Analyze each suggested amendment
    amendment_analyses = []
    for i, amendment in enumerate(suggested_amendments):
        # Simple analysis of how amendment addresses concerns
        addressed_concerns = []
        for concern in concerns:
            # Check if amendment text addresses the concern (basic keyword matching)
            concern_keywords = concern.lower().split()
            amendment_lower = amendment.lower()
            
            matches = sum(1 for keyword in concern_keywords if keyword in amendment_lower)
            if matches >= 2:  # At least 2 matching keywords
                addressed_concerns.append(concern)
        
        amendment_analyses.append({
            "amendment_id": f"amendment_{i+1}",
            "amendment_text": amendment,
            "potentially_addressed_concerns": addressed_concerns,
            "concern_coverage_score": len(addressed_concerns) / len(concerns) if concerns else 0
        })
    
    # Overall impact assessment
    total_concerns = len(concerns)
    total_addressed = len(set(
        concern 
        for analysis in amendment_analyses 
        for concern in analysis["potentially_addressed_concerns"]
    ))
    
    overall_coverage = (total_addressed / total_concerns * 100) if total_concerns > 0 else 0
    
    return {
        "amendment_impact_analysis": {
            "original_provision": original_provision,
            "total_related_comments": len(related_comments),
            "identified_concerns": concerns,
            "amendment_analyses": amendment_analyses,
            "impact_summary": {
                "total_concerns_identified": total_concerns,
                "concerns_potentially_addressed": total_addressed,
                "overall_coverage_percentage": round(overall_coverage, 2),
                "coverage_assessment": "high" if overall_coverage >= 70 else 
                                     "moderate" if overall_coverage >= 40 else "low"
            },
            "recommendations": [
                "Review amendments with highest concern coverage scores",
                "Consider combining amendments to address more concerns",
                "Assess feasibility of implementing suggested amendments",
                "Validate amendment effectiveness through follow-up consultation"
            ] if amendment_analyses else ["No amendments to analyze"],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    }