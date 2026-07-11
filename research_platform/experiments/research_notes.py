"""
Research Notes Manager

Manages research notes for experiments.
Captures learnings and conclusions.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import ResearchNote
from utils.logger import get_logger

logger = get_logger("experiments.research_notes")


class ResearchNotesManager:
    """
    Research Notes Manager.
    
    Manages:
    - Research notes for experiments
    - Learnings and conclusions
    - Author tracking
    - Note history
    """
    
    def __init__(self):
        """Initialize research notes manager."""
        self.notes: Dict[str, ResearchNote] = {}
        self._logger = get_logger("experiments.research_notes")
    
    def create_note(
        self,
        experiment_id: str,
        author: str,
        content: str,
    ) -> ResearchNote:
        """
        Create a research note.
        
        Args:
            experiment_id: Experiment ID
            author: Note author
            content: Note content
            
        Returns:
            ResearchNote object
        """
        note_id = f"RN-{uuid.uuid4().hex[:8].upper()}"
        
        note = ResearchNote(
            note_id=note_id,
            experiment_id=experiment_id,
            author=author,
            content=content,
        )
        
        self.notes[note_id] = note
        
        self._logger.info(
            f"Created research note {note_id} for experiment {experiment_id}"
        )
        
        return note
    
    def get_note(self, note_id: str) -> Optional[ResearchNote]:
        """Get a note by ID."""
        return self.notes.get(note_id)
    
    def get_notes_by_experiment(self, experiment_id: str) -> List[ResearchNote]:
        """Get all notes for an experiment."""
        return [
            note for note in self.notes.values()
            if note.experiment_id == experiment_id
        ]
    
    def get_notes_by_author(self, author: str) -> List[ResearchNote]:
        """Get all notes by an author."""
        return [
            note for note in self.notes.values()
            if note.author == author
        ]
    
    def update_note(
        self,
        note_id: str,
        content: str,
    ) -> bool:
        """
        Update a note's content.
        
        Args:
            note_id: Note ID
            content: New content
            
        Returns:
            True if updated successfully
        """
        note = self.notes.get(note_id)
        if not note:
            self._logger.error(f"Note not found: {note_id}")
            return False
        
        note.content = content
        self._logger.info(f"Updated note {note_id}")
        return True
    
    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note.
        
        Args:
            note_id: Note ID
            
        Returns:
            True if deleted successfully
        """
        if note_id not in self.notes:
            self._logger.error(f"Note not found: {note_id}")
            return False
        
        del self.notes[note_id]
        self._logger.info(f"Deleted note {note_id}")
        return True
    
    def search_notes(
        self,
        query: str,
        experiment_id: Optional[str] = None,
    ) -> List[ResearchNote]:
        """
        Search notes by content.
        
        Args:
            query: Search query
            experiment_id: Optional experiment ID filter
            
        Returns:
            List of matching notes
        """
        query_lower = query.lower()
        
        matching_notes = []
        for note in self.notes.values():
            if experiment_id and note.experiment_id != experiment_id:
                continue
            
            if query_lower in note.content.lower():
                matching_notes.append(note)
        
        return matching_notes
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """
        Get summary of notes for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with note summary
        """
        notes = self.get_notes_by_experiment(experiment_id)
        
        summary = {
            'experiment_id': experiment_id,
            'total_notes': len(notes),
            'authors': list(set(note.author for note in notes)),
            'latest_note': None,
            'all_notes': [note.to_dict() for note in notes],
        }
        
        if notes:
            latest = max(notes, key=lambda x: x.created_at)
            summary['latest_note'] = latest.to_dict()
        
        return summary
