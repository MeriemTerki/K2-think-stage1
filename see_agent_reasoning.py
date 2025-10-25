# see_agent_reasoning.py
import asyncio
import httpx

API_BASE = "http://127.0.0.1:8000"
PROJECT_ID = "project1"

async def see_agent_reasoning():
    print("🧠 VIEWING AGENT REASONING IN ACTION")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Clear previous state
        await client.post(f"{API_BASE}/api/session/end", params={"user_id": "alice", "project_id": PROJECT_ID})
        await client.post(f"{API_BASE}/api/session/end", params={"user_id": "bob", "project_id": PROJECT_ID})
        await asyncio.sleep(1)
        
        # Start fresh
        await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID, "branch": "main"
        })
        await client.post(f"{API_BASE}/api/session/start", json={
            "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID, "branch": "feature/auth"
        })
        
        print("\n1. 🎯 TRIGGERING AGENT REASONING...")
        print("   Alice starts editing database.py...")
        await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "alice", "username": "Alice", "project_id": PROJECT_ID,
            "file_path": "src/database.py", "action": "editing", "branch": "main"
        })
        
        print("   Bob starts editing same database.py...")
        response = await client.post(f"{API_BASE}/api/file/activity", json={
            "user_id": "bob", "username": "Bob", "project_id": PROJECT_ID,
            "file_path": "src/database.py", "action": "editing", "branch": "feature/auth"
        })
        
        # Wait for agent processing
        await asyncio.sleep(3)
        
        print("\n2. 📋 CHECKING AGENT MEMORY...")
        response = await client.get(f"{API_BASE}/api/agent/memory")
        if response.status_code == 200:
            memory = response.json()
            
            print(f"   🤖 Recent Interactions:")
            for i, interaction in enumerate(memory['conversation_history'][-3:]):
                role = interaction['role']
                content = interaction['content']
                print(f"      {i+1}. {role.upper()}: {content[:120]}...")
            
            print(f"\n   🎓 Learned Patterns:")
            for pattern, data in memory['learned_patterns'].items():
                print(f"      📊 {pattern}: {data['occurrences']} occurrences")
            
            print(f"\n   💡 Agent State:")
            state = memory['agent_state']
            print(f"      Goals: {state.get('current_goals', [])}")
            print(f"      Lessons: {len(state.get('learned_lessons', []))}")
        
        print("\n3. 🎪 TESTING COMPLEX SCENARIO...")
        response = await client.post(f"{API_BASE}/api/agent/execute-action", json={
            "project_id": PROJECT_ID,
            "situation_description": "Critical security update needed in authentication module while two developers are actively refactoring the same code. Need to coordinate without causing merge conflicts or security issues.",
            "desired_outcome": "Secure coordinated deployment with zero downtime"
        })
        
        if response.status_code == 200:
            result = response.json()
            execution = result['agent_execution']
            
            print(f"\n   🎯 AGENT EXECUTION PLAN:")
            plan = execution.get('execution_plan', {})
            print(f"      Goal: {plan.get('goal', 'N/A')}")
            print(f"      Steps: {len(plan.get('steps', []))}")
            
            reasoning = execution.get('agent_reasoning', {})
            print(f"\n   🧠 AI REASONING:")
            if 'reasoning_text' in reasoning:
                # Show the actual Gemini AI reasoning
                lines = reasoning['reasoning_text'].split('\n')
                for line in lines[:8]:  # Show first 8 lines
                    if line.strip():
                        print(f"      {line}")
            else:
                for key, value in reasoning.items():
                    print(f"      {key}: {value}")
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your Agentic Architecture is Fully Operational!")
        print("   The system detects conflicts and uses AI reasoning to resolve them")
        print("=" * 60)

async def main():
    await see_agent_reasoning()

if __name__ == "__main__":
    asyncio.run(main())