"""
Feature Dashboard API

FastAPI endpoints for the Feature Laboratory dashboard.
Provides access to feature statistics, quality scores, and analytics.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.db_sync import get_db
from database.models import FeatureMetadata as DBFeatureMetadata, FeatureQuality
from sqlalchemy import and_
from feature_layer.feature_generator import FeatureGenerator
from feature_layer.feature_analyzer import FeatureAnalyzer
from feature_layer.feature_importance import FeatureImportanceTracker
from feature_layer.feature_correlation import FeatureCorrelationAnalyzer
from feature_layer.feature_quality import FeatureQualityScorer
from feature_layer.feature_versioning import FeatureVersionManager


router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("/summary")
def get_feature_summary(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get overall feature laboratory summary.
    
    Returns:
        Dictionary with total features, categories, and top performers
    """
    # Count total features
    total_features = db.query(DBFeatureMetadata).count()
    
    # Count by category
    category_counts = {}
    for meta in db.query(DBFeatureMetadata.feature_category).distinct():
        category = meta[0]
        count = db.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_category == category
        ).count()
        category_counts[category] = count
    
    # Get top features by quality
    quality_scorer = FeatureQualityScorer(db)
    top_features = quality_scorer.get_quality_report(limit=10)
    
    return {
        "total_features": total_features,
        "categories": category_counts,
        "top_features": top_features.to_dict('records') if not top_features.empty else [],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/list")
def list_features(
    category: Optional[str] = Query(None, description="Filter by category"),
    enabled_only: bool = Query(True, description="Show only enabled features"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    List all features with metadata.
    
    Args:
        category: Optional category filter
        enabled_only: Whether to show only enabled features
        
    Returns:
        List of feature metadata
    """
    query = db.query(DBFeatureMetadata)
    
    if category:
        query = query.filter(DBFeatureMetadata.feature_category == category)
    
    if enabled_only:
        query = query.filter(DBFeatureMetadata.is_enabled == True)
    
    features = query.all()
    
    return [
        {
            "feature_name": f.feature_name,
            "category": f.feature_category,
            "description": f.description,
            "timeframe": f.timeframe,
            "version": f.version,
            "is_enabled": f.is_enabled,
            "author": f.author,
            "created_at": f.created_at.isoformat(),
            "last_updated": f.last_updated.isoformat(),
        }
        for f in features
    ]


@router.get("/{feature_name}")
def get_feature_details(
    feature_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Feature metadata and statistics
    """
    import json
    
    metadata = db.query(DBFeatureMetadata).filter(
        DBFeatureMetadata.feature_name == feature_name
    ).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Feature {feature_name} not found")
    
    # Get latest quality score
    quality = db.query(FeatureQuality).filter(
        FeatureQuality.feature_name == feature_name
    ).order_by(FeatureQuality.evaluation_period_end.desc()).first()
    
    return {
        "feature_name": metadata.feature_name,
        "category": metadata.feature_category,
        "description": metadata.description,
        "timeframe": metadata.timeframe,
        "required_columns": json.loads(metadata.required_columns),
        "output_range": metadata.output_range,
        "version": metadata.version,
        "author": metadata.author,
        "computation_method": metadata.computation_method,
        "assumptions": metadata.assumptions,
        "limitations": metadata.limitations,
        "references": metadata.references,
        "is_enabled": metadata.is_enabled,
        "created_at": metadata.created_at.isoformat(),
        "last_updated": metadata.last_updated.isoformat(),
        "quality_score": quality.quality_score if quality else None,
        "quality_grade": quality.quality_score if quality else None,
    }


@router.get("/{feature_name}/quality")
def get_feature_quality(
    feature_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get quality metrics for a specific feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Quality metrics and scores
    """
    quality_scorer = FeatureQualityScorer(db)
    
    # Get quality records
    quality_records = db.query(FeatureQuality).filter(
        FeatureQuality.feature_name == feature_name
    ).order_by(FeatureQuality.evaluation_period_end.desc()).limit(10).all()
    
    if not quality_records:
        raise HTTPException(status_code=404, detail=f"No quality data for {feature_name}")
    
    latest = quality_records[0]
    
    return {
        "feature_name": feature_name,
        "current_quality_score": latest.quality_score,
        "current_grade": quality_scorer._get_quality_grade(latest.quality_score),
        "win_rate": latest.win_rate,
        "sharpe_ratio": latest.sharpe_ratio,
        "profit_factor": latest.profit_factor,
        "average_return": latest.average_return,
        "max_drawdown": latest.max_drawdown,
        "sample_size": latest.sample_size,
        "evaluation_period": {
            "start": latest.evaluation_period_start.isoformat(),
            "end": latest.evaluation_period_end.isoformat(),
        },
        "history": [
            {
                "quality_score": q.quality_score,
                "evaluation_period_end": q.evaluation_period_end.isoformat(),
            }
            for q in quality_records
        ],
    }


@router.get("/quality/report")
def get_quality_report(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_score: Optional[float] = Query(None, description="Minimum quality score"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive quality report for all features.
    
    Args:
        category: Optional category filter
        min_score: Optional minimum quality score
        
    Returns:
        Quality report with recommendations
    """
    quality_scorer = FeatureQualityScorer(db)
    
    report = quality_scorer.get_quality_report(category=category, min_score=min_score)
    recommendations = quality_scorer.recommend_feature_actions()
    
    return {
        "report": report.to_dict('records') if not report.empty else [],
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/importance/{model_name}")
def get_feature_importance(
    model_name: str,
    model_version: Optional[str] = Query(None, description="Model version"),
    n: int = Query(10, description="Number of top features to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get feature importance for a specific model.
    
    Args:
        model_name: Name of the model
        model_version: Optional model version
        n: Number of top features to return
        
    Returns:
        Feature importance rankings
    """
    tracker = FeatureImportanceTracker(db)
    
    df = tracker.get_top_features(model_name, model_version, n)
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No importance data for model {model_name}"
        )
    
    return {
        "model_name": model_name,
        "model_version": model_version or "latest",
        "top_features": df.to_dict('records'),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/correlation/matrix")
def get_correlation_matrix(
    min_correlation: float = Query(0.5, description="Minimum correlation threshold"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get feature correlation matrix.
    
    Args:
        min_correlation: Minimum absolute correlation
        
    Returns:
        Correlation matrix as list of pairs
    """
    analyzer = FeatureCorrelationAnalyzer(db)
    
    df = analyzer.get_all_correlations(min_correlation=min_correlation)
    
    return {
        "correlations": df.to_dict('records') if not df.empty else [],
        "min_correlation": min_correlation,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/{feature_name}/enable")
def enable_feature(
    feature_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Enable a feature.
    
    Args:
        feature_name: Name of the feature to enable
        
    Returns:
        Success message
    """
    generator = FeatureGenerator(db)
    
    try:
        generator.enable_feature(feature_name)
        return {"message": f"Feature {feature_name} enabled successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{feature_name}/disable")
def disable_feature(
    feature_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Disable a feature.
    
    Args:
        feature_name: Name of the feature to disable
        
    Returns:
        Success message
    """
    generator = FeatureGenerator(db)
    
    try:
        generator.disable_feature(feature_name)
        return {"message": f"Feature {feature_name} disabled successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/categories/summary")
def get_category_summary(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get summary of features by category.
    
    Returns:
        Category-wise feature counts and statistics
    """
    categories = db.query(DBFeatureMetadata.feature_category).distinct()
    
    summary = {}
    for cat in categories:
        category_name = cat[0]
        
        total = db.query(DBFeatureMetadata).filter(
            DBFeatureMetadata.feature_category == category_name
        ).count()
        
        enabled = db.query(DBFeatureMetadata).filter(
            and_(
                DBFeatureMetadata.feature_category == category_name,
                DBFeatureMetadata.is_enabled == True
            )
        ).count()
        
        # Get average quality score for category
        quality_scores = db.query(FeatureQuality).join(
            DBFeatureMetadata,
            FeatureQuality.feature_name == DBFeatureMetadata.feature_name
        ).filter(
            DBFeatureMetadata.feature_category == category_name
        ).all()
        
        avg_quality = (
            sum(q.quality_score for q in quality_scores) / len(quality_scores)
            if quality_scores else 0
        )
        
        summary[category_name] = {
            "total_features": total,
            "enabled_features": enabled,
            "disabled_features": total - enabled,
            "average_quality_score": round(avg_quality, 2),
        }
    
    return {
        "categories": summary,
        "timestamp": datetime.now().isoformat(),
    }
