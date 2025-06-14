from app.celery_app import celery_app
from typing import Dict, Optional
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TaskTracker:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        logger.info("TaskTracker initialized")

    def add_task(self, task_id: str, user_id: str) -> None:
        """Add a new task to tracking"""
        logger.info(f"Adding new task: {task_id} for user: {user_id}")
        self.tasks[task_id] = {
            'user_id': user_id,
            'status': 'PENDING',
            'start_time': datetime.now(),
            'progress': 0,
            'total_items': 0,
            'processed_items': 0,
            'remaining_items': 0,
            'current_operation': 'Initializing',
            'error': None,
            'last_updated': datetime.now()
        }
        logger.info(f"Current tasks: {list(self.tasks.keys())}")

    def update_progress(self, task_id: str, processed: int, total: int, current_operation: str = None) -> None:
        """Update task progress"""
        if task_id in self.tasks:
            logger.info(f"Updating progress for task {task_id}: {processed}/{total} - {current_operation}")
            self.tasks[task_id].update({
                'processed_items': processed,
                'total_items': total,
                'remaining_items': total - processed,
                'progress': (processed / total) * 100 if total > 0 else 0,
                'current_operation': current_operation or self.tasks[task_id]['current_operation'],
                'last_updated': datetime.now()
            })
        else:
            logger.warning(f"Task {task_id} not found in tracker")

    def complete_task(self, task_id: str, result: Dict) -> None:
        """Mark task as complete"""
        if task_id in self.tasks:
            logger.info(f"Completing task {task_id}")
            self.tasks[task_id].update({
                'status': 'COMPLETED',
                'end_time': datetime.now(),
                'result': result,
                'progress': 100,
                'current_operation': 'Completed',
                'last_updated': datetime.now()
            })
        else:
            logger.warning(f"Task {task_id} not found in tracker")

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed"""
        if task_id in self.tasks:
            logger.info(f"Failing task {task_id}: {error}")
            self.tasks[task_id].update({
                'status': 'FAILED',
                'end_time': datetime.now(),
                'error': error,
                'current_operation': 'Failed',
                'last_updated': datetime.now()
            })
        else:
            logger.warning(f"Task {task_id} not found in tracker")

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get current task status"""
        status = self.tasks.get(task_id)
        logger.info(f"Getting status for task {task_id}: {status is not None}")
        return status

    def get_user_tasks(self, user_id: str) -> list[Dict]:
        """Get all tasks for a user"""
        tasks = [
            {**task, 'task_id': task_id}
            for task_id, task in self.tasks.items()
            if task['user_id'] == user_id
        ]
        logger.info(f"Getting tasks for user {user_id}: {len(tasks)} tasks found")
        return tasks

# Initialize global task tracker
task_tracker = TaskTracker() 