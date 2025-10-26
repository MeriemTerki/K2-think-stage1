# test_conflicts_direct.py
import asyncio
import httpx
import json

API_BASE = "http://127.0.0.1:8000"
PROJECT_ID = "project1"

async def test_conflict_detection():
    """Test conflict detection without WebSockets"""
    print("🔍 TESTING CONFLICT DETECTION DIRECTLY")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Clear any existing state
        print("\n1. 🧹 CLEARING EXISTING STATE...")
        await client.post(f"{API_BASE}/api/session/end", params={"user_id": "alice", "project_id": PROJECT_ID})
        await client.post(f"{API_BASE}/api/session/end", params={"user_id": "bob", "project_id": PROJECT_ID})
        await asyncio.sleep(1)
        
        # Start fresh sessions
        print("\n2. 👥 STARTING FRESH SESSIONS...")
        await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID, "branch": "main"
        })
        await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID, "branch": "feature/auth"
        })
        
        # Test 1: Single user editing (should be fine)
        print("\n3. 📄 TEST 1: SINGLE USER EDITING...")
        response = await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID,
            "file_path": "src/app.py", "action": "editing", "branch": "main"
        })
        result1 = response.json()
        print(f"   Alice editing app.py: {result1.get('active_users')} active users")
        
        await asyncio.sleep(1)
        
        # Test 2: Second user editing SAME file (should trigger conflict)
        print("\n4. ⚡ TEST 2: CONFLICT TRIGGER...")
        response = await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID,
            "file_path": "src/app.py", "action": "editing", "branch": "feature/auth"
        })
        result2 = response.json()
        print(f"   Bob editing app.py: {result2.get('active_users')} active users")
        
        # Check system state
        print("\n5. 📊 CHECKING SYSTEM STATE...")
        response = await client.get(f"{API_BASE}/api/project/{PROJECT_ID}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Active users: {len(status['active_users'])}")
            print(f"   Active files: {len(status['active_files'])}")
            for file in status['active_files']:
                print(f"     📁 {file['file_path']}: {file['count']} users - {file['users']}")
        
        # Check agent memory for conflict handling
        print("\n6. 🤖 CHECKING AGENT ACTIVITY...")
        response = await client.get(f"{API_BASE}/api/agent/memory")
        if response.status_code == 200:
            memory = response.json()
            print(f"   Total interactions: {memory['memory_stats']['total_interactions']}")
            print(f"   Patterns learned: {memory['memory_stats']['learned_patterns_count']}")
            
            # Show recent agent activities
            activities = memory.get('agent_state', {}).get('learned_lessons', [])
            if activities:
                print(f"   Recent activities: {len(activities)}")
                for activity in activities[-2:]:
                    situation = activity.get('situation', {})
                    print(f"     🎯 {situation.get('file_path', 'unknown')}: {activity.get('outcome', 'unknown')}")
        
        # Test direct conflict analysis
        print("\n7. 🧪 TESTING DIRECT CONFLICT ANALYSIS...")
        response = await client.post(f"{API_BASE}/api/conflict/analyze", json={
            "project_id": PROJECT_ID,
            "file_path": "src/app.py",
            "base_content": "def calculate_total():\n    return price * quantity",
            "user1_changes": "def calculate_total():\n    return price * quantity * tax_rate",
            "user2_changes": "def calculate_total():\n    return (price * quantity) - discount"
        })
        if response.status_code == 200:
            analysis = response.json()
            print(f"   Direct analysis result: {analysis.get('will_conflict', False)}")
            if 'analysis' in analysis:
                print(f"   AI Analysis: {analysis['analysis'][:100]}...")
        
        # Final assessment
        print("\n8. 📈 RESULTS SUMMARY")
        if result2.get('active_users', 0) > 1:
            print("   ✅ CONFLICT DETECTED: Multiple users editing same file")
            print("   🎉 Agentic system is working correctly!")
        else:
            print("   ❌ No conflict detected in system state")
        
        print("\n" + "=" * 50)

async def main():
    await test_conflict_detection()

if __name__ == "__main__":
    asyncio.run(main())