# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()  # This loads .env

from utils import (
    DataStore,
    ConnectionManager,
    NotificationAgent,
    get_active_users_for_project,
    get_active_files_for_project,
    conflict_analyzer  # Import the instantiated analyzer
)

# Initialize FastAPI
app = FastAPI(title="Git Conflict Prevention API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data store and connection manager
store = DataStore()
manager = ConnectionManager(store)

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
    action: str  # 'editing', 'saving', 'closing'
    branch: str
    timestamp: str = None

class GitOperation(BaseModel):
    user_id: str
    username: str
    project_id: str
    operation: str  # 'pull', 'push', 'commit', 'checkout'
    branch: str
    files: List[str] = []
    timestamp: str = None

class ConflictAnalysisRequest(BaseModel):
    project_id: str
    file_path: str
    user1_changes: str
    user2_changes: str
    base_content: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Git Conflict Prevention API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/{project_id}/{user_id}",
            "session": "/api/session/*",
            "file_tracking": "/api/file/activity",
            "git_operations": "/api/git/operation",
            "conflict_analysis": "/api/conflict/analyze",
            "project_status": "/api/project/{project_id}/status"
        }
    }

@app.post("/api/session/start")
async def start_session(session: UserSession):
    """User starts a coding session"""
    # Convert Pydantic model to dict for storage
    session_dict = {
        "user_id": session.user_id,
        "username": session.username,
        "project_id": session.project_id,
        "branch": session.branch
    }
    store.user_sessions[session.user_id] = session_dict
    store.user_branches[session.user_id] = session.branch
    
    # Notify other users
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
    """User ends session"""
    if user_id in store.user_sessions:
        session = store.user_sessions[user_id]
        del store.user_sessions[user_id]
        
        # Remove from active files
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
    """Track when users open/edit/close files"""
    activity.timestamp = datetime.now().isoformat()
    store.file_activities.append(activity.dict())
    
    file_key = f"{activity.project_id}:{activity.file_path}"
    
    if activity.action == "editing":
        store.active_files[file_key].add(activity.user_id)
        
        # Check for conflicts
        users_editing = list(store.active_files[file_key])
        if len(users_editing) > 1:
            # Get usernames
            usernames = [
                store.user_sessions[uid]['username'] 
                for uid in users_editing 
                if uid in store.user_sessions
            ]
            
            # Run AI analysis
            conflict_analysis = await conflict_analyzer.analyze_potential_conflict(
                activity.file_path,
                usernames,
                activity.project_id
            )
            
            # Notify all users editing this file
            notification = NotificationAgent.create_notification(
                "conflict_warning",
                {
                    "file_path": activity.file_path,
                    "other_user": ", ".join([u for u in usernames if u != activity.username]),
                    "risk_level": conflict_analysis["risk_level"]
                }
            )
            
            notification["conflict_analysis"] = conflict_analysis
            
            await manager.broadcast_to_project(
                activity.project_id,
                {
                    "type": "conflict_warning",
                    "data": notification,
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
    """Track git operations (push, pull, commit)"""
    operation.timestamp = datetime.now().isoformat()
    store.git_operations.append(operation.dict())
    
    should_notify = await NotificationAgent.should_notify(
        f"git_{operation.operation}",
        {"operation": operation.dict()}
    )
    
    if should_notify:
        if operation.operation == "push":
            # Warn other users not to push
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
            # Notify users with uncommitted changes
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

@app.post("/api/conflict/analyze")
async def analyze_conflict(request: ConflictAnalysisRequest):
    """Deep analysis of potential merge conflicts"""
    
    analysis = await conflict_analyzer.predict_merge_conflict(
        request.file_path,
        request.base_content,
        request.user1_changes,
        request.user2_changes
    )
    return analysis

@app.get("/api/project/{project_id}/status")
async def get_project_status(project_id: str):
    """Get current status of project - who's editing what"""
    
    active_users = get_active_users_for_project(store, project_id)
    active_files_list = get_active_files_for_project(store, project_id)
    
    return {
        "project_id": project_id,
        "active_users": active_users,
        "active_files": active_files_list,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{project_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str, user_id: str):
    """WebSocket connection for real-time notifications"""
    await manager.connect(websocket, project_id, user_id)
    
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle ping/pong
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
        print(f"User {user_id} disconnected from project {project_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)