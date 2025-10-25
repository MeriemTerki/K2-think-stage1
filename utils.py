# utils.py
import google.generativeai as genai
import asyncio
import logging
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime
import os
from urllib.parse import urlparse


# === Configuration and Setup ===
class Config:
    """Configuration management for Gemini API"""
    
    def __init__(self):
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        self.GEMINI_API_URL = self._clean_api_url(os.getenv("GEMINI_API_URL", "").strip())
        
        # Updated model names based on your available models
        self.AVAILABLE_MODELS = [
            "models/gemini-2.0-flash-001",  # Stable and reliable
            "models/gemini-2.0-flash",      # Alternative
            "models/gemini-2.5-flash",      # Latest stable
            "models/gemini-2.0-flash-lite-001",  # Lite version
            "models/gemini-pro-latest",     # Fallback
        ]
        
        self.model = None
        self.selected_model = None
        self._setup_logging()
        self._configure_gemini()
    
    def _clean_api_url(self, url: str) -> str:
        """Clean and format API URL"""
        if url:
            return url.strip().strip('"').strip("'")
        return ""
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _configure_gemini(self):
        """Configure Gemini AI model with model discovery"""
        if not self.GEMINI_API_KEY:
            self.logger.warning("GEMINI_API_KEY not found. Running in MOCK mode.")
            return
        
        try:
            self.logger.info("Configuring Gemini with API key: %s...", self.GEMINI_API_KEY[:8])
            genai.configure(api_key=self.GEMINI_API_KEY)
            self.logger.info("Gemini SDK configured successfully.")
            
            # Try available models until one works
            self.model, self.selected_model = self._discover_working_model()
            
            if self.model:
                self._test_model()
            else:
                self.logger.warning("No working model found. Running in MOCK mode.")
                
        except Exception as e:
            self.logger.error("Failed to initialize Gemini: %s", e)
            self.model = None
    
    def _discover_working_model(self):
        """Try different model names to find one that works"""
        for model_name in self.AVAILABLE_MODELS:
            try:
                self.logger.info("Trying model: %s", model_name)
                model = genai.GenerativeModel(model_name)
                
                # Quick test to see if model works
                test_response = model.generate_content("Say 'ok' in one word.")
                if test_response.text:
                    self.logger.info("✅ Successfully initialized model: %s", model_name)
                    return model, model_name
                    
            except Exception as e:
                self.logger.warning("Model %s failed: %s", model_name, str(e))
                continue
        
        return None, None
    
    def _test_model(self):
        """Test the Gemini model with a simple request"""
        try:
            test_response = self.model.generate_content("Say 'ok' in one word.")
            test_text = test_response.text.strip().lower()
            self.logger.info("API test successful: '%s'", test_text)
            
        except Exception as e:
            self.logger.error("Model test failed: %s", e)
            self.model = None


# === Data Store ===
class DataStore:
    """Central data store for managing application state"""
    
    def __init__(self):
        self.active_connections: Dict[str, List] = defaultdict(list)
        self.user_sessions: Dict[str, dict] = {}
        self.active_files: Dict[str, Set[str]] = defaultdict(set)
        self.file_activities: List[dict] = []
        self.git_operations: List[dict] = []
        self.user_branches: Dict[str, str] = {}


# === Connection Manager ===
class ConnectionManager:
    """Manage WebSocket connections and broadcasting"""
    
    def __init__(self, store: DataStore):
        self.store = store
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket, project_id: str, user_id: str):
        """Accept connection and add to active connections"""
        await websocket.accept()
        self.store.active_connections[project_id].append(websocket)
        self.logger.info("User %s connected to project %s", user_id, project_id)

    def disconnect(self, websocket, project_id: str):
        """Remove connection from active connections"""
        if websocket in self.store.active_connections[project_id]:
            self.store.active_connections[project_id].remove(websocket)

    async def broadcast_to_project(self, project_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all connections in a project"""
        disconnected = []
        for conn in self.store.active_connections[project_id]:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, project_id)


# === AI Agents ===
class ConflictAnalyzerAgent:
    """AI-powered conflict analysis and prediction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def analyze_potential_conflict(self, file_path: str, users_editing: List[str], project_id: str) -> dict:
        """Analyze potential merge conflicts for a file"""
        if len(users_editing) <= 1:
            return {"risk_level": "low", "message": "No conflict risk"}

        prompt = self._build_conflict_analysis_prompt(file_path, users_editing)
        
        if self.config.model is None:
            self.logger.info("Using MOCK analysis for %s", file_path)
            return self._get_mock_conflict_analysis(file_path, users_editing)

        return await self._call_gemini_for_analysis(prompt, users_editing)
    
    def _build_conflict_analysis_prompt(self, file_path: str, users_editing: List[str]) -> str:
        """Build prompt for conflict analysis"""
        return f"""
        Analyze potential merge conflicts for a code file:
        File: {file_path}
        Current Editors: {', '.join(users_editing)}
        
        Provide a JSON response with:
        - risk_level: "low", "medium", or "high"
        - conflict_areas: list of potential conflict areas
        - recommendations: list of suggestions to avoid conflicts
        
        Keep the analysis concise and practical.
        """
    
    def _get_mock_conflict_analysis(self, file_path: str, users_editing: List[str]) -> dict:
        """Generate mock analysis when AI is unavailable"""
        risk_level = "high" if len(users_editing) > 2 else "medium"
        return {
            "risk_level": risk_level,
            "conflict_areas": ["Simultaneous edits to same file"],
            "recommendations": [f"Coordinate with {', '.join(users_editing)} before pushing changes"],
            "users_affected": users_editing,
            "analysis": f"Mock: {len(users_editing)} users editing {file_path}. Coordinate changes."
        }
    
    async def _call_gemini_for_analysis(self, prompt: str, users_editing: List[str]) -> dict:
        """Call Gemini API for conflict analysis"""
        try:
            self.logger.info("Calling Gemini for conflict analysis...")
            response = await asyncio.to_thread(
                self.config.model.generate_content, 
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=500,
                )
            )
            text = response.text.strip()
            self.logger.info("Gemini response received")
            
            # Parse the response
            return self._parse_ai_response(text, users_editing)
            
        except Exception as e:
            self.logger.error("Gemini API error: %s", e)
            return self._get_mock_conflict_analysis("unknown", users_editing)
    
    def _parse_ai_response(self, text: str, users_editing: List[str]) -> dict:
        """Parse AI response and extract structured data"""
        # Simple parsing - you might want to make this more robust
        text_lower = text.lower()
        
        if "high" in text_lower:
            risk_level = "high"
        elif "medium" in text_lower:
            risk_level = "medium"
        else:
            risk_level = "low"
            
        return {
            "risk_level": risk_level,
            "analysis": text,
            "users_affected": users_editing,
            "conflict_areas": ["Code sections being concurrently edited"],
            "recommendations": ["Coordinate changes with team members"]
        }
    
    async def predict_merge_conflict(self, file_path: str, base_content: str, 
                                   user1_changes: str, user2_changes: str) -> dict:
        """Predict merge conflicts between user changes"""
        prompt = self._build_merge_prediction_prompt(file_path, base_content, user1_changes, user2_changes)
        
        if self.config.model is None:
            return self._get_mock_merge_prediction()

        return await self._call_gemini_for_merge_prediction(prompt)
    
    def _build_merge_prediction_prompt(self, file_path: str, base_content: str, 
                                     user1_changes: str, user2_changes: str) -> str:
        """Build prompt for merge conflict prediction"""
        return f"""
        Analyze these code changes for potential merge conflicts:
        
        File: {file_path}
        
        Original Code:
        {base_content[:500]}
        
        User 1 Changes:
        {user1_changes[:500]}
        
        User 2 Changes:
        {user2_changes[:500]}
        
        Will these changes conflict? Provide a JSON response with:
        - will_conflict: true or false
        - severity: "low", "moderate", or "high" 
        - conflicting_lines: description of conflicting areas
        - recommendations: how to resolve
        
        Be concise and focus on practical advice.
        """
    
    def _get_mock_merge_prediction(self) -> dict:
        """Generate mock merge prediction"""
        return {
            "will_conflict": True,
            "severity": "moderate",
            "conflicting_lines": ["Simulated conflict in code changes"],
            "analysis": "Mock: Changes overlap. Manual merge needed.",
            "recommendations": ["Review diffs before pushing", "Coordinate with teammate"]
        }
    
    async def _call_gemini_for_merge_prediction(self, prompt: str) -> dict:
        """Call Gemini API for merge prediction"""
        try:
            response = await asyncio.to_thread(
                self.config.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )
            
            return {
                "will_conflict": True,  # Default to safe assumption
                "severity": "moderate",
                "analysis": response.text,
                "recommendations": ["Coordinate with teammate before merging"]
            }
        except Exception as e:
            self.logger.error("Merge prediction failed: %s", e)
            return self._get_mock_merge_prediction()


# === Notification Agent ===
class NotificationAgent:
    """Handle notification creation and management"""
    
    NOTIFICATION_TEMPLATES = {
        "conflict_warning": {
            "type": "warning",
            "title": "Conflict Risk",
            "message": "{other_user} is editing {file_path}",
            "action": "Coordinate now",
            "priority": "high"
        },
        "push_warning": {
            "type": "warning",
            "title": "Push Detected",
            "message": "{username} is pushing",
            "action": "Wait before pushing",
            "priority": "high"
        }
    }
    
    @staticmethod
    def create_notification(event_type: str, data: dict) -> dict:
        """Create notification based on event type"""
        template = NotificationAgent.NOTIFICATION_TEMPLATES.get(event_type, {})
        if not template:
            return {}
        
        # Format message with data
        notification = template.copy()
        if "message" in notification:
            notification["message"] = notification["message"].format(**data)
        
        return notification
    
    @staticmethod
    async def should_notify(event_type: str, data: dict) -> bool:
        """Determine if a notification should be sent"""
        # For now, always notify for these events
        return event_type in ["git_push", "git_pull", "conflict_warning"]


# === Helper Functions ===
def get_active_users_for_project(store: DataStore, project_id: str) -> List[dict]:
    """Get list of active users for a project"""
    return [
        {"user_id": uid, "username": s.get('username', 'Unknown'), "branch": s.get('branch', 'main')}
        for uid, s in store.user_sessions.items()
        if s.get('project_id') == project_id
    ]


def get_active_files_for_project(store: DataStore, project_id: str) -> List[dict]:
    """Get list of active files being edited in a project"""
    active_files = []
    for file_key, users in store.active_files.items():
        if file_key.startswith(project_id + ":") and users:
            file_path = file_key.split(":", 1)[1] if ":" in file_key else file_key
            active_files.append({
                "file_path": file_path, 
                "users": list(users), 
                "count": len(users)
            })
    return active_files


# === Global Configuration Instance ===
config = Config()
logger = config.logger

# Initialize agents
conflict_analyzer = ConflictAnalyzerAgent(config)

# For backward compatibility with existing code
model = config.model