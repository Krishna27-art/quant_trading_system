"""
Research Project Manager

Manages research projects and groups related experiments.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import (
    ResearchProject,
    ProjectType,
    ExperimentStatus,
    ExperimentPriority,
)
from utils.logger import get_logger

logger = get_logger("experiments.project_manager")


class ProjectManager:
    """
    Research Project Manager.
    
    Manages:
    - Creating research projects
    - Organizing experiments by project
    - Project lifecycle management
    - Project metadata
    """
    
    def __init__(self):
        """Initialize project manager."""
        self.projects: Dict[str, ResearchProject] = {}
        self._logger = get_logger("experiments.project_manager")
    
    def create_project(
        self,
        name: str,
        project_type: ProjectType,
        description: str,
        created_by: str,
        tags: Optional[List[str]] = None,
        priority: ExperimentPriority = ExperimentPriority.MEDIUM,
    ) -> ResearchProject:
        """
        Create a new research project.
        
        Args:
            name: Project name
            project_type: Type of project
            description: Project description
            created_by: Creator name
            tags: Optional tags
            priority: Project priority
            
        Returns:
            ResearchProject object
        """
        project_id = f"PROJ-{uuid.uuid4().hex[:8].upper()}"
        
        project = ResearchProject(
            project_id=project_id,
            name=name,
            project_type=project_type,
            description=description,
            created_by=created_by,
            tags=tags or [],
            priority=priority,
        )
        
        self.projects[project_id] = project
        
        self._logger.info(
            f"Created project {project_id}: {name} ({project_type.value})"
        )
        
        return project
    
    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get a project by ID."""
        return self.projects.get(project_id)
    
    def update_project_status(
        self,
        project_id: str,
        status: ExperimentStatus,
    ) -> bool:
        """
        Update project status.
        
        Args:
            project_id: Project ID
            status: New status
            
        Returns:
            True if updated successfully
        """
        project = self.projects.get(project_id)
        if not project:
            self._logger.error(f"Project not found: {project_id}")
            return False
        
        project.status = status
        self._logger.info(f"Updated project {project_id} status to {status.value}")
        return True
    
    def update_project_notes(
        self,
        project_id: str,
        notes: str,
    ) -> bool:
        """
        Update project notes.
        
        Args:
            project_id: Project ID
            notes: New notes
            
        Returns:
            True if updated successfully
        """
        project = self.projects.get(project_id)
        if not project:
            self._logger.error(f"Project not found: {project_id}")
            return False
        
        project.notes = notes
        self._logger.info(f"Updated notes for project {project_id}")
        return True
    
    def list_projects(
        self,
        project_type: Optional[ProjectType] = None,
        status: Optional[ExperimentStatus] = None,
        created_by: Optional[str] = None,
    ) -> List[ResearchProject]:
        """
        List projects with optional filters.
        
        Args:
            project_type: Optional project type filter
            status: Optional status filter
            created_by: Optional creator filter
            
        Returns:
            List of matching projects
        """
        projects = list(self.projects.values())
        
        if project_type:
            projects = [p for p in projects if p.project_type == project_type]
        
        if status:
            projects = [p for p in projects if p.status == status]
        
        if created_by:
            projects = [p for p in projects if p.created_by == created_by]
        
        return projects
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if deleted successfully
        """
        if project_id not in self.projects:
            self._logger.error(f"Project not found: {project_id}")
            return False
        
        del self.projects[project_id]
        self._logger.info(f"Deleted project {project_id}")
        return True
    
    def get_project_stats(self, project_id: str) -> Optional[Dict]:
        """
        Get statistics for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with project statistics
        """
        project = self.projects.get(project_id)
        if not project:
            return None
        
        return {
            'project_id': project.project_id,
            'name': project.name,
            'type': project.project_type.value,
            'status': project.status.value,
            'priority': project.priority.value,
            'created_by': project.created_by,
            'created_at': project.created_at.isoformat(),
            'tags': project.tags,
            'age_days': (datetime.now() - project.created_at).days,
        }
    
    def search_projects(
        self,
        query: str,
    ) -> List[ResearchProject]:
        """
        Search projects by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching projects
        """
        query_lower = query.lower()
        
        matching_projects = []
        for project in self.projects.values():
            if (
                query_lower in project.name.lower() or
                query_lower in project.description.lower() or
                any(query_lower in tag.lower() for tag in project.tags)
            ):
                matching_projects.append(project)
        
        return matching_projects
