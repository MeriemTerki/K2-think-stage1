# utils.py
import google.generativeai as genai
import asyncio
import logging
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from datetime import datetime
import os
import json
from urllib.parse import urlparse
import subprocess
import re


# === Configuration and Setup ===
class Config:
    """Configuration management for Gemini API"""
    
    def __init__(self):
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        self.GEMINI_API_URL = self._clean_api_url(os.getenv("GEMINI_API_URL", "").strip())
        
        self.AVAILABLE_MODELS = [
            "models/gemini-2.0-flash-001",
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash-lite-001",
            "models/gemini-pro-latest",
        ]
        
        self.model = None
        self.selected_model = None
        self._setup_logging()
        self._configure_gemini()
    
    def _clean_api_url(self, url: str) -> str:
        if url:
            return url.strip().strip('"').strip("'")
        return ""
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _configure_gemini(self):
        if not self.GEMINI_API_KEY:
            self.logger.warning("GEMINI_API_KEY not found. Running in MOCK mode.")
            return
        
        try:
            self.logger.info("Configuring Gemini with API key: %s...", self.GEMINI_API_KEY[:8])
            genai.configure(api_key=self.GEMINI_API_KEY)
            self.logger.info("Gemini SDK configured successfully.")
            
            self.model, self.selected_model = self._discover_working_model()
            
            if self.model:
                self._test_model()
            else:
                self.logger.warning("No working model found. Running in MOCK mode.")
                
        except Exception as e:
            self.logger.error("Failed to initialize Gemini: %s", e)
            self.model = None
    
    def _discover_working_model(self):
        for model_name in self.AVAILABLE_MODELS:
            try:
                self.logger.info("Trying model: %s", model_name)
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Say 'ok' in one word.")
                if test_response.text:
                    self.logger.info("✅ Successfully initialized model: %s", model_name)
                    return model, model_name
            except Exception as e:
                self.logger.warning("Model %s failed: %s", model_name, str(e))
                continue
        return None, None
    
    def _test_model(self):
        try:
            test_response = self.model.generate_content("Say 'ok' in one word.")
            test_text = test_response.text.strip().lower()
            self.logger.info("API test successful: '%s'", test_text)
        except Exception as e:
            self.logger.error("Model test failed: %s", e)
            self.model = None


# === Agent Memory ===
class AgentMemory:
    """Memory system for agent context and learning"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.conflict_patterns: Dict[str, Any] = {}
        self.user_behavior: Dict[str, Any] = {}
        self.resolution_strategies: Dict[str, Any] = {}
        self.agent_state: Dict[str, Any] = {
            "current_goals": [],
            "active_plans": {},
            "learned_lessons": []
        }
    
    def add_interaction(self, role: str, content: str, metadata: Dict = None):
        """Add an interaction to conversation history"""
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(interaction)
        
        # Keep only last 100 interactions
        if len(self.conversation_history) > 100:
            self.conversation_history.pop(0)
    
    def get_recent_context(self, limit: int = 10) -> str:
        """Get recent conversation context"""
        recent = self.conversation_history[-limit:]
        return "\n".join([f"{item['role']}: {item['content']}" for item in recent])
    
    def learn_conflict_pattern(self, file_path: str, users: List[str], resolution: str):
        """Learn from conflict patterns"""
        pattern_key = f"{file_path}:{','.join(sorted(users))}"
        if pattern_key not in self.conflict_patterns:
            self.conflict_patterns[pattern_key] = {
                "occurrences": 0,
                "resolutions": [],
                "success_rate": 0.0
            }
        
        self.conflict_patterns[pattern_key]["occurrences"] += 1
        self.conflict_patterns[pattern_key]["resolutions"].append(resolution)
    
    def get_learned_patterns(self) -> Dict:
        """Get learned conflict patterns"""
        return self.conflict_patterns


# === Agent Tools ===
class AgentTools:
    """Tools that the agent can use to interact with the system"""
    
    def __init__(self, store, manager, config):
        self.store = store
        self.manager = manager
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def analyze_code_structure(self, file_path: str) -> Dict:
        """Analyze code structure for conflict prediction"""
        # Simulate code analysis - in real implementation, you'd parse the actual file
        return {
            "file_type": self._detect_file_type(file_path),
            "complexity": "medium",
            "key_functions": ["main", "calculate", "process_data"],
            "dependencies": ["utils", "models"]
        }
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        return {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'c++',
            '.c': 'c',
            '.html': 'html',
            '.css': 'css'
        }.get(ext, 'unknown')
    
    async def check_git_status(self, project_id: str) -> Dict:
        """Check git status for the project"""
        # In a real implementation, you'd run actual git commands
        return {
            "branch": "main",
            "has_uncommitted_changes": True,
            "conflicting_files": [],
            "ahead_of_remote": False
        }
    
    async def suggest_code_refactor(self, file_path: str, conflict_areas: List[str]) -> Dict:
        """Suggest code refactoring to avoid conflicts"""
        prompt = f"""
        Suggest code refactoring strategies for file {file_path} to avoid conflicts in these areas: {conflict_areas}.
        Provide specific, actionable suggestions.
        """
        
        if self.config.model is None:
            return {
                "suggestions": [
                    "Break large functions into smaller ones",
                    "Use dependency injection to reduce coupling",
                    "Extract common functionality into separate modules"
                ],
                "confidence": 0.7
            }
        
        try:
            response = await asyncio.to_thread(
                self.config.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                )
            )
            return {
                "suggestions": response.text.split('\n'),
                "confidence": 0.8
            }
        except Exception as e:
            self.logger.error("Refactor suggestion failed: %s", e)
            return {"suggestions": [], "confidence": 0.0}
    
    async def coordinate_users(self, users: List[str], message: str, project_id: str):
        """Coordinate between users"""
        notification = {
            "type": "agent_coordination",
            "message": message,
            "users_involved": users,
            "timestamp": datetime.now().isoformat(),
            "priority": "medium"
        }
        
        await self.manager.broadcast_to_project(
            project_id,
            notification
        )
        
        return {"status": "message_sent", "users_notified": users}


# === Planning System ===
class AgentPlanner:
    """Planning system for goal-oriented behavior"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def create_resolution_plan(self, conflict_data: Dict, memory: AgentMemory) -> Dict:
        """Create a step-by-step plan for conflict resolution"""
        
        prompt = f"""
        You are a conflict resolution agent. Create a step-by-step plan to handle this situation:
        
        Conflict Data:
        - File: {conflict_data.get('file_path')}
        - Users Involved: {', '.join(conflict_data.get('users', []))}
        - Risk Level: {conflict_data.get('risk_level', 'unknown')}
        
        Available Tools:
        - Code analysis
        - User coordination
        - Refactoring suggestions
        - Git operations
        
        Create a JSON plan with:
        1. Overall goal
        2. Steps (each with: action, tool_to_use, expected_outcome)
        3. Success criteria
        4. Potential risks
        
        Be specific and actionable.
        """
        
        if self.config.model is None:
            return self._get_mock_plan(conflict_data)
        
        try:
            response = await asyncio.to_thread(
                self.config.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=800,
                )
            )
            
            return self._parse_plan_response(response.text, conflict_data)
        except Exception as e:
            self.logger.error("Planning failed: %s", e)
            return self._get_mock_plan(conflict_data)
    
    def _get_mock_plan(self, conflict_data: Dict) -> Dict:
        """Generate mock plan when AI is unavailable"""
        return {
            "goal": f"Resolve editing conflicts for {conflict_data.get('file_path')}",
            "steps": [
                {
                    "step": 1,
                    "action": "Analyze code structure and identify conflict-prone areas",
                    "tool": "code_analysis",
                    "expected_outcome": "Identify specific functions/variables at risk"
                },
                {
                    "step": 2,
                    "action": "Coordinate with users to establish editing priorities",
                    "tool": "user_coordination", 
                    "expected_outcome": "Users agree on who edits what and when"
                },
                {
                    "step": 3,
                    "action": "Suggest code refactoring to reduce coupling",
                    "tool": "refactoring_suggestions",
                    "expected_outcome": "Code structure improved to minimize conflicts"
                }
            ],
            "success_criteria": ["No simultaneous edits", "Clear ownership established", "Code structure improved"],
            "risks": ["Users don't coordinate", "Tight deadlines prevent refactoring"]
        }
    
    def _parse_plan_response(self, response_text: str, conflict_data: Dict) -> Dict:
        """Parse AI response into structured plan"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to structured response
                return {
                    "goal": "Resolve editing conflicts through coordination and code improvement",
                    "steps": [
                        {
                            "step": 1,
                            "action": "Coordinate users based on AI analysis",
                            "tool": "user_coordination",
                            "expected_outcome": "Clear communication established"
                        }
                    ],
                    "success_criteria": ["Conflict resolved", "Users coordinated"],
                    "risks": ["Communication breakdown"],
                    "raw_response": response_text
                }
        except Exception as e:
            self.logger.error("Plan parsing failed: %s", e)
            return self._get_mock_plan(conflict_data)


# === Main Reasoning Agent ===
class ConflictResolutionAgent:
    """Main agent that reasons about conflicts and takes action"""
    
    def __init__(self, store, manager, config: Config):
        self.store = store
        self.manager = manager
        self.config = config
        self.memory = AgentMemory()
        self.tools = AgentTools(store, manager, config)
        self.planner = AgentPlanner(config)
        self.logger = logging.getLogger(__name__)
        
        # Agent identity and goals
        self.identity = {
            "name": "ConflictResolver",
            "role": "AI agent for preventing and resolving code conflicts",
            "capabilities": ["reasoning", "planning", "coordination", "learning"],
            "goals": ["minimize merge conflicts", "improve team coordination", "learn conflict patterns"]
        }
    
    async def handle_conflict_situation(self, conflict_data: Dict) -> Dict:
        """Main agent entry point - reason about conflict and take action"""
        
        # Add to memory
        self.memory.add_interaction(
            "system", 
            f"New conflict detected: {conflict_data}",
            {"type": "conflict_detected"}
        )
        
        # Step 1: Reason about the situation
        reasoning = await self._reason_about_conflict(conflict_data)
        
        # Step 2: Create a plan
        plan = await self.planner.create_resolution_plan(conflict_data, self.memory)
        
        # Step 3: Execute the plan with reasoning loop
        results = await self._execute_plan(plan, conflict_data, reasoning)
        
        # Step 4: Learn from the experience
        await self._learn_from_experience(conflict_data, plan, results)
        
        return {
            "agent_reasoning": reasoning,
            "execution_plan": plan,
            "results": results,
            "agent_identity": self.identity
        }
    
    async def _reason_about_conflict(self, conflict_data: Dict) -> Dict:
        """Use Gemini as a reasoning engine to analyze the situation"""
        
        context = self.memory.get_recent_context()
        learned_patterns = self.memory.get_learned_patterns()
        
        prompt = f"""
        You are {self.identity['name']}, an AI agent with these goals: {self.identity['goals']}
        
        Current Situation:
        - File: {conflict_data.get('file_path')}
        - Users: {', '.join(conflict_data.get('users', []))}
        - Project: {conflict_data.get('project_id')}
        
        Recent Context:
        {context}
        
        Learned Patterns:
        {json.dumps(learned_patterns, indent=2)}
        
        Analyze this situation and provide your reasoning:
        1. What is the root cause of this conflict?
        2. What are the immediate risks?
        3. What information do you need to make better decisions?
        4. What tools would be most effective here?
        5. What's your recommended approach?
        
        Provide structured reasoning.
        """
        
        if self.config.model is None:
            return {
                "root_cause": "Multiple users editing same file",
                "immediate_risks": ["Merge conflicts", "Code duplication"],
                "needed_information": ["User intentions", "Code complexity"],
                "recommended_tools": ["coordination", "code_analysis"],
                "approach": "Coordinate users and suggest code organization"
            }
        
        try:
            response = await asyncio.to_thread(
                self.config.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                )
            )
            
            self.memory.add_interaction("agent_reasoning", response.text, {"type": "conflict_analysis"})
            
            return {
                "reasoning_text": response.text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error("Reasoning failed: %s", e)
            return {"error": str(e)}
    
    async def _execute_plan(self, plan: Dict, conflict_data: Dict, reasoning: Dict) -> Dict:
        """Execute the plan with monitoring and adaptation"""
        
        results = {
            "steps_completed": [],
            "adaptations_made": [],
            "final_outcome": "in_progress"
        }
        
        for step in plan.get("steps", []):
            self.logger.info("Executing step: %s", step.get("action"))
            
            step_result = await self._execute_step(step, conflict_data, reasoning)
            results["steps_completed"].append(step_result)
            
            # Check if we need to adapt the plan
            if step_result.get("status") == "failed":
                adaptation = await self._adapt_plan(step, step_result, conflict_data)
                results["adaptations_made"].append(adaptation)
            
            # Add delay between steps to simulate thoughtful execution
            await asyncio.sleep(1)
        
        results["final_outcome"] = "completed"
        return results
    
    async def _execute_step(self, step: Dict, conflict_data: Dict, reasoning: Dict) -> Dict:
        """Execute a single plan step using appropriate tools"""
        
        tool_to_use = step.get("tool", "")
        action = step.get("action", "")
        
        try:
            if tool_to_use == "code_analysis":
                result = await self.tools.analyze_code_structure(conflict_data.get("file_path"))
            
            elif tool_to_use == "user_coordination":
                message = f"Agent suggestion: {action}"
                result = await self.tools.coordinate_users(
                    conflict_data.get("users", []),
                    message,
                    conflict_data.get("project_id")
                )
            
            elif tool_to_use == "refactoring_suggestions":
                result = await self.tools.suggest_code_refactor(
                    conflict_data.get("file_path"),
                    conflict_data.get("conflict_areas", [])
                )
            
            elif tool_to_use == "git_operations":
                result = await self.tools.check_git_status(conflict_data.get("project_id"))
            
            else:
                result = {"status": "no_tool", "action": action}
            
            return {
                "step": step.get("step"),
                "action": action,
                "tool": tool_to_use,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Step execution failed: %s", e)
            return {
                "step": step.get("step"),
                "action": action,
                "tool": tool_to_use,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _adapt_plan(self, failed_step: Dict, step_result: Dict, conflict_data: Dict) -> Dict:
        """Adapt the plan when steps fail"""
        
        prompt = f"""
        Plan adaptation needed:
        
        Failed Step: {failed_step}
        Failure Reason: {step_result.get('error')}
        Current Situation: {conflict_data}
        
        Suggest an adaptation or alternative approach.
        """
        
        if self.config.model is None:
            return {
                "failed_step": failed_step.get("step"),
                "adaptation": "Retry with different parameters or skip to next step",
                "reason": "Generic fallback adaptation"
            }
        
        try:
            response = await asyncio.to_thread(
                self.config.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                )
            )
            
            return {
                "failed_step": failed_step.get("step"),
                "adaptation": response.text,
                "reason": "AI-suggested adaptation"
            }
        except Exception as e:
            return {
                "failed_step": failed_step.get("step"),
                "adaptation": "Continue with next step",
                "reason": f"Adaptation failed: {str(e)}"
            }
    
    async def _learn_from_experience(self, conflict_data: Dict, plan: Dict, results: Dict):
        """Learn from the execution and update memory"""
        
        resolution_success = results.get("final_outcome") == "completed"
        resolution_strategy = plan.get("goal", "unknown")
        
        self.memory.learn_conflict_pattern(
            conflict_data.get("file_path"),
            conflict_data.get("users", []),
            resolution_strategy if resolution_success else "failed"
        )
        
        # Add lesson to memory
        lesson = {
            "situation": conflict_data,
            "plan_used": plan,
            "outcome": results.get("final_outcome"),
            "timestamp": datetime.now().isoformat(),
            "key_insights": self._extract_insights(results)
        }
        
        self.memory.agent_state["learned_lessons"].append(lesson)
        
        self.memory.add_interaction(
            "system",
            f"Learning completed for conflict. Success: {resolution_success}",
            {"type": "learning", "success": resolution_success}
        )
    
    def _extract_insights(self, results: Dict) -> List[str]:
        """Extract key insights from execution results"""
        insights = []
        
        completed_steps = [s for s in results.get("steps_completed", []) if s.get("status") == "completed"]
        failed_steps = [s for s in results.get("steps_completed", []) if s.get("status") == "failed"]
        
        if completed_steps:
            insights.append(f"Successfully completed {len(completed_steps)} steps")
        
        if failed_steps:
            insights.append(f"Encountered {len(failed_steps)} failures needing adaptation")
        
        if results.get("adaptations_made"):
            insights.append("Plan adaptation was necessary and effective")
        
        return insights


# === Data Store & Connection Manager (Updated) ===
class DataStore:
    def __init__(self):
        self.active_connections: Dict[str, List] = defaultdict(list)
        self.user_sessions: Dict[str, dict] = {}
        self.active_files: Dict[str, Set[str]] = defaultdict(set)
        self.file_activities: List[dict] = []
        self.git_operations: List[dict] = []
        self.user_branches: Dict[str, str] = {}
        self.agent_activities: List[dict] = []  # Track agent actions


class ConnectionManager:
    def __init__(self, store: DataStore):
        self.store = store
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket, project_id: str, user_id: str):
        await websocket.accept()
        self.store.active_connections[project_id].append(websocket)
        self.logger.info("User %s connected to project %s", user_id, project_id)

    def disconnect(self, websocket, project_id: str):
        if websocket in self.store.active_connections[project_id]:
            self.store.active_connections[project_id].remove(websocket)

    async def broadcast_to_project(self, project_id: str, message: dict, exclude_user: str = None):
        disconnected = []
        for conn in self.store.active_connections[project_id]:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        
        for conn in disconnected:
            self.disconnect(conn, project_id)


# === Notification Agent ===
class NotificationAgent:
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
        },
        "agent_coordination": {
            "type": "info",
            "title": "Agent Suggestion",
            "message": "{message}",
            "action": "Review suggestion",
            "priority": "medium"
        }
    }
    
    @staticmethod
    def create_notification(event_type: str, data: dict) -> dict:
        template = NotificationAgent.NOTIFICATION_TEMPLATES.get(event_type, {})
        if not template:
            return {}
        
        notification = template.copy()
        if "message" in notification:
            notification["message"] = notification["message"].format(**data)
        
        return notification
    
    @staticmethod
    async def should_notify(event_type: str, data: dict) -> bool:
        return event_type in ["git_push", "git_pull", "conflict_warning", "agent_coordination"]


# === Helper Functions ===
def get_active_users_for_project(store: DataStore, project_id: str) -> List[dict]:
    return [
        {"user_id": uid, "username": s.get('username', 'Unknown'), "branch": s.get('branch', 'main')}
        for uid, s in store.user_sessions.items()
        if s.get('project_id') == project_id
    ]


def get_active_files_for_project(store: DataStore, project_id: str) -> List[dict]:
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


# === Global Configuration and Agent Initialization ===
config = Config()
logger = config.logger

# Initialize data store and manager first
store = DataStore()
manager = ConnectionManager(store)

# Initialize the main agent
conflict_agent = ConflictResolutionAgent(store, manager, config)

# For backward compatibility
model = config.model