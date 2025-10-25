# simulate_real_conflicts.py
import asyncio
import httpx
import websockets
import json

API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
PROJECT_ID = "project1"

TIMEOUT = httpx.Timeout(60.0)

class RealConflictDemo:
    def __init__(self):
        self.messages_received = []
        self.conflicts_detected = 0
    
    async def ws_listener(self, user_id):
        uri = f"{WS_BASE}/ws/{PROJECT_ID}/{user_id}"
        try:
            async with websockets.connect(uri) as ws:
                print(f"🔗 [{user_id}] connected to WebSocket")
                async for message in ws:
                    data = json.loads(message)
                    self.messages_received.append((user_id, data))
                    
                    # Track actual conflict detection
                    if data.get('type') == 'conflict_warning':
                        self.conflicts_detected += 1
                        analysis = data.get('data', {}).get('conflict_analysis', {})
                        print(f"🚨 CONFLICT DETECTED! Risk: {analysis.get('risk_level', 'unknown')}")
                    
                    print(f"📨 [{user_id}] {data.get('type', 'unknown')}: {self._format_message(data)}")
        except Exception as e:
            print(f"WebSocket closed for {user_id}: {e}")
    
    def _format_message(self, data):
        msg_type = data.get('type')
        if msg_type == 'agent_action':
            reasoning = data.get('agent_reasoning', {})
            return f"🤖 {reasoning.get('approach', 'Analyzing conflict...')}"
        elif msg_type == 'agent_coordination':
            return f"🔄 {data.get('message', '')}"
        elif msg_type == 'conflict_warning':
            analysis = data.get('data', {}).get('conflict_analysis', {})
            return f"⚠️ {analysis.get('risk_level', 'unknown')} risk - {analysis.get('analysis', '')[:80]}..."
        elif msg_type == 'user_joined':
            return f"👤 {data.get('username')} joined project"
        else:
            return str(data)[:100] + "..."
    
    async def simulate_real_conflict(self):
        print("🚀 SIMULATING REAL CONFLICT SCENARIOS")
        print("=" * 60)
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Check agent status
            print("\n1. 🤖 AGENT STATUS")
            r = await client.get(f"{API_BASE}/api/agent/status")
            status = r.json()
            print(f"   Agent: {status['agent_identity']['name']}")
            print(f"   Model: {status['model_in_use']}")
            print(f"   Operational: {status['is_operational']}")
            
            # Start WebSocket listeners
            print("\n2. 🔗 STARTING REAL-TIME MONITORS")
            alice_task = asyncio.create_task(self.ws_listener("alice"))
            bob_task = asyncio.create_task(self.ws_listener("bob"))
            await asyncio.sleep(2)
            
            # Start user sessions
            print("\n3. 👥 USER SESSIONS STARTED")
            await client.post(f"{API_BASE}/api/session/start", json={
                "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID, "branch": "main"
            })
            await client.post(f"{API_BASE}/api/session/start", json={
                "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID, "branch": "feature/auth"
            })
            await asyncio.sleep(2)
            
            # SCENARIO 1: Basic file conflict
            print("\n4. 📄 SCENARIO 1: BASIC FILE CONFLICT")
            print("   Alice starts editing utils.py...")
            await client.post(f"{API_BASE}/api/file/activity", json={
                "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID,
                "file_path": "src/utils.py", "action": "editing", "branch": "main"
            })
            await asyncio.sleep(1)
            
            print("   Bob starts editing SAME FILE utils.py...")
            await client.post(f"{API_BASE}/api/file/activity", json={
                "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID,
                "file_path": "src/utils.py", "action": "editing", "branch": "feature/auth"
            })
            await asyncio.sleep(3)  # Wait for conflict detection
            
            # SCENARIO 2: Multiple file conflicts
            print("\n5. 📄 SCENARIO 2: MULTIPLE FILE CONFLICTS")
            print("   Alice starts editing config.py...")
            await client.post(f"{API_BASE}/api/file/activity", json={
                "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID,
                "file_path": "config.py", "action": "editing", "branch": "main"
            })
            await asyncio.sleep(1)
            
            print("   Bob ALSO starts editing config.py...")
            await client.post(f"{API_BASE}/api/file/activity", json={
                "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID,
                "file_path": "config.py", "action": "editing", "branch": "feature/auth"
            })
            await asyncio.sleep(3)
            
            # SCENARIO 3: Git operations during conflicts
            print("\n6. 🔄 SCENARIO 3: GIT OPERATIONS DURING CONFLICTS")
            print("   Alice tries to push while conflict exists...")
            await client.post(f"{API_BASE}/api/git/operation", json={
                "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID,
                "operation": "push", "branch": "main", "files": ["src/utils.py", "config.py"]
            })
            await asyncio.sleep(2)
            
            # Test agent's memory and learning
            print("\n7. 🧠 AGENT LEARNING CHECK")
            r = await client.get(f"{API_BASE}/api/agent/memory")
            if r.status_code == 200:
                memory = r.json()
                stats = memory.get('memory_stats', {})
                print(f"   📊 Total Interactions: {stats.get('total_interactions')}")
                print(f"   🎓 Patterns Learned: {stats.get('learned_patterns_count')}")
                print(f"   💡 Lessons: {stats.get('lessons_learned')}")
            
            # Results summary
            print("\n8. 📊 CONFLICT DETECTION RESULTS")
            conflict_warnings = [m for m in self.messages_received if m[1].get('type') == 'conflict_warning']
            agent_actions = [m for m in self.messages_received if m[1].get('type') == 'agent_action']
            coordinations = [m for m in self.messages_received if m[1].get('type') == 'agent_coordination']
            
            print(f"   🚨 Conflicts Detected: {self.conflicts_detected}")
            print(f"   ⚠️  Conflict Warnings: {len(conflict_warnings)}")
            print(f"   🤖 Agent Actions: {len(agent_actions)}")
            print(f"   🔄 Coordinations: {len(coordinations)}")
            
            # Show specific conflict details
            if conflict_warnings:
                print(f"\n   📋 CONFLICT DETAILS:")
                for i, (user, warning) in enumerate(conflict_warnings[:3]):
                    analysis = warning.get('data', {}).get('conflict_analysis', {})
                    print(f"     {i+1}. Risk: {analysis.get('risk_level')} - {analysis.get('analysis', '')[:60]}...")
            
            print("\n" + "=" * 60)
            if self.conflicts_detected > 0:
                print(f"🎉 SUCCESS! System detected {self.conflicts_detected} real conflicts!")
                print("   Agentic architecture is working correctly! 🚀")
            else:
                print("❌ No conflicts detected - system may need adjustment")
            print("=" * 60)
            
            # Cleanup
            alice_task.cancel()
            bob_task.cancel()
            try:
                await alice_task
                await bob_task
            except asyncio.CancelledError:
                pass

async def main():
    demo = RealConflictDemo()
    await demo.simulate_real_conflict()

if __name__ == "__main__":
    asyncio.run(main())