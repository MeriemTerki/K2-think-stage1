# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
load_dotenv()


from utils import (
    DataStore,
    ConnectionManager,
    NotificationAgent,
    get_active_users_for_project,
    get_active_files_for_project,
    conflict_agent,  # The main agent
    store,
    manager
)

# Initialize FastAPI
app = FastAPI(title="Git Conflict Prevention API - Agentic Architecture")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class UserSession(BaseModel):
    user_id: str
    username: str
    project_id: str
    branch: str

class FileActivity(BaseModel):
    user_id: str
    username: str
    project_id: str
    file_path: str
    action: str
    branch: str
    timestamp: str = None

class GitOperation(BaseModel):
    user_id: str
    username: str
    project_id: str
    operation: str
    branch: str
    files: List[str] = []
    timestamp: str = None

class ConflictAnalysisRequest(BaseModel):
    project_id: str
    file_path: str
    user1_changes: str
    user2_changes: str
    base_content: str

class AgentActionRequest(BaseModel):
    project_id: str
    situation_description: str
    desired_outcome: str

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Git Conflict Prevention API - Agentic Architecture",
        "version": "2.0.0",
        "architecture": "agentic",
        "agent_capabilities": conflict_agent.identity["capabilities"],
        "endpoints": {
            "websocket": "/ws/{project_id}/{user_id}",
            "session": "/api/session/*",
            "file_tracking": "/api/file/activity", 
            "git_operations": "/api/git/operation",
            "conflict_analysis": "/api/conflict/analyze",
            "agent_actions": "/api/agent/*",
            "project_status": "/api/project/{project_id}/status"
        }
    }

@app.post("/api/session/start")
async def start_session(session: UserSession):
    session_dict = {
        "user_id": session.user_id,
        "username": session.username,
        "project_id": session.project_id,
        "branch": session.branch
    }
    store.user_sessions[session.user_id] = session_dict
    store.user_branches[session.user_id] = session.branch
    
    await manager.broadcast_to_project(
        session.project_id,
        {
            "type": "user_joined",
            "user_id": session.user_id,
            "username": session.username,
            "branch": session.branch,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    return {"status": "success", "message": "Session started"}

@app.post("/api/session/end")
async def end_session(user_id: str, project_id: str):
    if user_id in store.user_sessions:
        session = store.user_sessions[user_id]
        del store.user_sessions[user_id]
        
        for file_path in list(store.active_files.keys()):
            store.active_files[file_path].discard(user_id)
        
        await manager.broadcast_to_project(
            project_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "username": session['username'],
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return {"status": "success"}

@app.post("/api/file/activity")
async def track_file_activity(activity: FileActivity):
    activity.timestamp = datetime.now().isoformat()
    store.file_activities.append(activity.dict())
    
    file_key = f"{activity.project_id}:{activity.file_path}"
    
    if activity.action == "editing":
        store.active_files[file_key].add(activity.user_id)
        
        users_editing = list(store.active_files[file_key])
        if len(users_editing) > 1:
            usernames = [
                store.user_sessions[uid]['username'] 
                for uid in users_editing 
                if uid in store.user_sessions
            ]
            
            # Use the agentic system instead of simple analysis
            conflict_data = {
                "file_path": activity.file_path,
                "users": usernames,
                "project_id": activity.project_id,
                "risk_level": "medium",  # Initial assessment
                "timestamp": activity.timestamp
            }
            
            # Trigger agentic reasoning and action
            agent_result = await conflict_agent.handle_conflict_situation(conflict_data)
            
            # Store agent activity
            store.agent_activities.append({
                "type": "conflict_resolution",
                "conflict_data": conflict_data,
                "agent_result": agent_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Notify users about agent actions
            notification = NotificationAgent.create_notification(
                "agent_coordination",
                {
                    "message": f"Agent is analyzing conflict in {activity.file_path} and will suggest solutions"
                }
            )
            
            await manager.broadcast_to_project(
                activity.project_id,
                {
                    "type": "agent_action",
                    "data": notification,
                    "agent_reasoning": agent_result.get("agent_reasoning", {}),
                    "timestamp": activity.timestamp
                }
            )
    
    elif activity.action == "closing":
        store.active_files[file_key].discard(activity.user_id)
    
    return {
        "status": "success",
        "active_users": len(store.active_files[file_key]),
        "users": list(store.active_files[file_key])
    }

@app.post("/api/git/operation")
async def track_git_operation(operation: GitOperation):
    operation.timestamp = datetime.now().isoformat()
    store.git_operations.append(operation.dict())
    
    should_notify = await NotificationAgent.should_notify(
        f"git_{operation.operation}",
        {"operation": operation.dict()}
    )
    
    if should_notify:
        if operation.operation == "push":
            notification = NotificationAgent.create_notification(
                "push_warning",
                {
                    "username": operation.username,
                    "branch": operation.branch,
                    "files": operation.files
                }
            )
            
            await manager.broadcast_to_project(
                operation.project_id,
                {
                    "type": "git_operation",
                    "operation": "push",
                    "data": notification,
                    "timestamp": operation.timestamp
                }
            )
        
        elif operation.operation == "pull":
            notification = NotificationAgent.create_notification(
                "pull_notification",
                {
                    "username": operation.username,
                    "branch": operation.branch
                }
            )
            
            await manager.broadcast_to_project(
                operation.project_id,
                {
                    "type": "git_operation",
                    "operation": "pull",
                    "data": notification,
                    "timestamp": operation.timestamp
                }
            )
    
    return {"status": "success", "operation_logged": True}

# New Agentic Endpoints - FIXED ROUTE DEFINITIONS
@app.post("/api/agent/handle-conflict")
async def agent_handle_conflict(conflict_data: dict):
    """Direct endpoint to trigger agentic conflict resolution"""
    try:
        result = await conflict_agent.handle_conflict_situation(conflict_data)
        
        # Store the activity
        store.agent_activities.append({
            "type": "direct_agent_call",
            "conflict_data": conflict_data,
            "agent_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "agent_result": result,
            "agent_identity": conflict_agent.identity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/api/agent/memory")
async def get_agent_memory():
    """Get the agent's current memory state"""
    return {
        "conversation_history": conflict_agent.memory.conversation_history[-10:],  # Last 10
        "learned_patterns": conflict_agent.memory.conflict_patterns,
        "agent_state": conflict_agent.memory.agent_state,
        "memory_stats": {
            "total_interactions": len(conflict_agent.memory.conversation_history),
            "learned_patterns_count": len(conflict_agent.memory.conflict_patterns),
            "lessons_learned": len(conflict_agent.memory.agent_state["learned_lessons"])
        }
    }

@app.post("/api/agent/execute-action")
async def agent_execute_action(request: AgentActionRequest):
    """Execute a specific agent action"""
    conflict_data = {
        "project_id": request.project_id,
        "situation": request.situation_description,
        "desired_outcome": request.desired_outcome,
        "users": ["unknown"],  # Default
        "file_path": "unknown",
        "timestamp": datetime.now().isoformat()
    }
    
    result = await conflict_agent.handle_conflict_situation(conflict_data)
    
    return {
        "status": "success",
        "action_request": request.dict(),
        "agent_execution": result
    }

@app.get("/api/agent/status")
async def get_agent_status():
    """Get current agent status and capabilities"""
    return {
        "agent_identity": conflict_agent.identity,
        "is_operational": conflict_agent.config.model is not None,
        "model_in_use": conflict_agent.config.selected_model,
        "memory_usage": {
            "interactions": len(conflict_agent.memory.conversation_history),
            "patterns": len(conflict_agent.memory.conflict_patterns),
            "lessons": len(conflict_agent.memory.agent_state["learned_lessons"])
        },
        "goals": conflict_agent.identity["goals"]
    }

# Existing endpoints for compatibility
@app.post("/api/conflict/analyze")
async def analyze_conflict(request: ConflictAnalysisRequest):
    """Legacy endpoint - now uses agentic approach"""
    conflict_data = {
        "project_id": request.project_id,
        "file_path": request.file_path,
        "users": ["user1", "user2"],
        "changes": {
            "user1": request.user1_changes,
            "user2": request.user2_changes
        },
        "base_content": request.base_content,
        "analysis_type": "deep_merge_analysis"
    }
    
    result = await conflict_agent.handle_conflict_situation(conflict_data)
    return result

@app.get("/api/project/{project_id}/status")
async def get_project_status(project_id: str):
    active_users = get_active_users_for_project(store, project_id)
    active_files_list = get_active_files_for_project(store, project_id)
    agent_activities = [a for a in store.agent_activities if a.get('conflict_data', {}).get('project_id') == project_id]
    
    return {
        "project_id": project_id,
        "active_users": active_users,
        "active_files": active_files_list,
        "agent_activities_count": len(agent_activities),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{project_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str, user_id: str):
    await manager.connect(websocket, project_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
        print(f"User {user_id} disconnected from project {project_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)