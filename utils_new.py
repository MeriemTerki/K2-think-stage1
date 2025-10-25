# utils.py
import google.generativeai as genai
import asyncio
import logging
from typing import List, Dict, Set, Any
from collections import defaultdict
from datetime import datetime
import os

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure Gemini API
_raw_key = os.getenv("GEMINI_API_KEY", "").strip()
_raw_url = os.getenv("GEMINI_API_URL", "").strip()

# sanitize possible quotes/spaces in URL
if _raw_url:
    _raw_url = _raw_url.strip().strip('"').strip("'")

GEMINI_API_KEY = _raw_key
GEMINI_API_URL = _raw_url

# Only configure the SDK if a key is present; otherwise run in offline/mock mode.
model = None
try:
    if GEMINI_API_KEY:
        logger.info("Attempting to configure Gemini SDK with key=%s... and URL=%s", 
                   GEMINI_API_KEY[:6] if GEMINI_API_KEY else "None", 
                   GEMINI_API_URL or "default")
        
        # Configure SDK with API key only - the URL is handled internally
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Successfully configured Gemini SDK")
        except Exception as e:
            logger.error("Failed to configure SDK: %s", e)
            raise

        # Create model instance
        try:
            model = genai.GenerativeModel('gemini-pro')
            logger.info("Successfully created Gemini model instance")
            
            # Test the model with a simple prompt
            test_resp = model.generate_content("Say 'ok' if you can read this.")
            logger.info("Test response: %s", getattr(test_resp, 'text', str(test_resp)))
        except Exception as e:
            logger.error("Failed to create/test model: %s", e)
            model = None
    else:
        logger.warning("No GEMINI_API_KEY found: running in mock/offline mode for AI analyses.")
except Exception as e:
    logger.exception("Failed to configure Gemini SDK: %s", e)
    model = None

# Data Store Class
class DataStore:
    def __init__(self):
        self.active_connections: Dict[str, List] = defaultdict(list)
        self.user_sessions: Dict[str, dict] = {}
        self.active_files: Dict[str, Set[str]] = defaultdict(set)
        self.file_activities: List[dict] = []
        self.git_operations: List[dict] = []
        self.user_branches: Dict[str, str] = {}

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self, store: DataStore):
        self.store = store
    
    async def connect(self, websocket, project_id: str, user_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.store.active_connections[project_id].append(websocket)
        print(f"User {user_id} connected to project {project_id}")

    def disconnect(self, websocket, project_id: str):
        """Disconnect a WebSocket client"""
        if websocket in self.store.active_connections[project_id]:
            self.store.active_connections[project_id].remove(websocket)

    async def broadcast_to_project(self, project_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all users in a project"""
        disconnected = []
        for connection in self.store.active_connections[project_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, project_id)

    async def send_to_user(self, project_id: str, user_id: str, message: dict):
        """Send message to specific user"""
        await self.broadcast_to_project(project_id, message)

# AI Agent Classes
class ConflictAnalyzerAgent:
    @staticmethod
    async def analyze_potential_conflict(file_path: str, users_editing: List[str], project_id: str) -> dict:
        """Agent that analyzes if multiple users editing same file will cause conflicts"""
        
        if len(users_editing) <= 1:
            return {"risk_level": "low", "message": "No conflict risk"}

        prompt = f"""
        Analyze the potential for merge conflicts in this scenario:
        
        File: {file_path}
        Number of developers: {len(users_editing)}
        Developers: {', '.join(users_editing)}
        
        Consider:
        1. File type and typical conflict patterns
        2. Number of concurrent editors
        3. Common conflict-prone areas (imports, configuration, shared functions)
        
        Provide:
        - Risk level (low/medium/high)
        - Specific areas likely to conflict
        - Recommendations to avoid conflicts
        
        Format as JSON with keys: risk_level, conflict_areas, recommendations
        """
        
        # If the model is not configured (None), return a deterministic mock result.
        if model is None:
            logger.info("Model not configured, returning mock conflict analysis for file=%s", file_path)
            # Simple heuristic mock: medium risk if more than 1 user, high if filename or user contains 'conflict'
            mock_risk = "medium"
            if any("conflict" in u.lower() for u in users_editing) or "conflict" in file_path.lower():
                mock_risk = "high"
            mock_analysis = (
                f"Mock analysis: {len(users_editing)} editors detected. "
                f"Recommend coordinating edits and running tests locally."
            )
            return {
                "risk_level": mock_risk,
                "analysis": mock_analysis,
                "users_affected": users_editing
            }

        try:
            logger.info("Running conflict analysis for file=%s project=%s users=%s", file_path, project_id, users_editing)
            logger.debug("Prompt:\n%s", prompt)
            # Call the SDK via the unified wrapper in a thread to avoid blocking the event loop.
            response = await asyncio.to_thread(model.generate_content, prompt)
            result = getattr(response, "text", str(response))
            logger.debug("Model response (truncated): %s", (result or "")[0:2000])

            # Parse AI response heuristically
            if "high" in (result or "").lower():
                risk_level = "high"
            elif "medium" in (result or "").lower():
                risk_level = "medium"
            else:
                risk_level = "low"

            return {
                "risk_level": risk_level,
                "analysis": result,
                "users_affected": users_editing
            }
        except Exception as e:
            logger.exception("AI Analysis error for file=%s: %s", file_path, e)
            return {
                "risk_level": "medium",
                "analysis": "Unable to perform AI analysis",
                "users_affected": users_editing
            }

    @staticmethod
    async def predict_merge_conflict(file_path: str, base_content: str, 
                                    user1_changes: str, user2_changes: str) -> dict:
        """Deep analysis of actual code changes for conflict prediction"""
        
        prompt = f"""
        Analyze these concurrent changes to the same file for merge conflicts:
        
        File: {file_path}
        
        Base Content:
        ```
        {base_content[:1000]}
        ```
        
        User 1 Changes:
        ```
        {user1_changes[:1000]}
        ```
        
        User 2 Changes:
        ```
        {user2_changes[:1000]}
        ```
        
        Determine:
        1. Will these changes cause a merge conflict?
        2. What specific lines/sections will conflict?
        3. Can Git auto-merge or will manual intervention be needed?
        4. Severity of the conflict (trivial/moderate/severe)
        
        Provide actionable recommendations.
        """
        
        # If model isn't configured, return a mock deep analysis response
        if model is None:
            logger.info("Model not configured, returning mock deep analysis for file=%s", file_path)
            return {
                "will_conflict": True,
                "severity": "moderate",
                "analysis": (
                    "Mock deep analysis: concurrent edits detected. "
                    "Manual review recommended; run tests after merging."
                ),
                "recommendations": "Coordinate with team member before pushing"
            }

        try:
            logger.info("Running deep merge conflict prediction for file=%s", file_path)
            logger.debug("Deep analysis prompt:\n%s", prompt[:4000])
            response = await asyncio.to_thread(model.generate_content, prompt)
            result_text = getattr(response, "text", str(response))
            logger.debug("Deep model response (truncated): %s", (result_text or "")[0:2000])
            return {
                "will_conflict": True,
                "severity": "moderate",
                "analysis": result_text,
                "recommendations": "Coordinate with team member before pushing"
            }
        except Exception as e:
            logger.exception("Deep analysis error for file=%s: %s", file_path, e)
            return {"error": str(e)}

class NotificationAgent:
    @staticmethod
    async def should_notify(event_type: str, context: dict) -> bool:
        """Agent decides if notification is warranted"""
        
        if event_type == "file_editing":
            return len(context.get("users_editing", [])) > 1
        
        elif event_type == "git_push":
            return True
        
        elif event_type == "git_pull":
            return bool(context.get("affected_users"))
        
        return False

    @staticmethod
    def create_notification(event_type: str, data: dict) -> dict:
        """Generate notification message"""
        
        notifications = {
            "conflict_warning": {
                "type": "warning",
                "title": "Potential Conflict Detected",
                "message": f"{data.get('other_user')} is also editing {data.get('file_path')}",
                "action": "Consider coordinating changes",
                "priority": "high"
            },
            "push_warning": {
                "type": "warning",
                "title": "Push in Progress",
                "message": f"{data.get('username')} is pushing to {data.get('branch')}",
                "action": "Wait before pushing to avoid conflicts",
                "priority": "high"
            },
            "pull_notification": {
                "type": "info",
                "title": "Pull Detected",
                "message": f"{data.get('username')} pulled latest changes",
                "action": "Your local files may be outdated",
                "priority": "medium"
            }
        }
        
        return notifications.get(event_type, {})

# Helper Functions
def get_active_users_for_project(store: DataStore, project_id: str) -> List[dict]:
    """Get list of active users in a project"""
    return [
        {
            "user_id": uid,
            "username": session.username,
            "branch": session.branch
        }
        for uid, session in store.user_sessions.items()
        if session.project_id == project_id
    ]

def get_active_files_for_project(store: DataStore, project_id: str) -> List[dict]:
    """Get list of files being actively edited in a project"""
    active_files_list = []
    for file_key, users in store.active_files.items():
        proj_id, file_path = file_key.split(":", 1)
        if proj_id == project_id and users:
            active_files_list.append({
                "file_path": file_path,
                "users": list(users),
                "count": len(users)
            })
    return active_files_list