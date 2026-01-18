"""Test script to verify LangSmith tracing is working."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set tracing environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'chefli-agents')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')

print("=" * 60)
print("LangSmith Tracing Test")
print("=" * 60)
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
print(f"LANGCHAIN_API_KEY: {'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET'}")
print(f"LANGCHAIN_ENDPOINT: {os.getenv('LANGCHAIN_ENDPOINT')}")
print()

# Test LangSmith connection
try:
    from langsmith import Client
    client = Client(api_key=os.getenv('LANGCHAIN_API_KEY'))
    print("✅ LangSmith Client initialized")
    
    # Test with a simple LangChain call
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    print("\nMaking test LLM call with tracing...")
    llm = ChatOpenAI(
        model='gpt-4-turbo-preview',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Say hello in one word")
    ])
    
    chain = prompt | llm
    result = chain.invoke({})
    
    print(f"✅ Test call completed!")
    print(f"Response: {result.content}")
    print()
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Go to: https://smith.langchain.com")
    print("2. Click on 'Projects' in the sidebar")
    print(f"3. Look for project: {os.getenv('LANGCHAIN_PROJECT')}")
    print("4. Click on the project to see traces")
    print("5. You should see a trace for this test call")
    print()
    print("If you don't see traces:")
    print("- Wait 10-30 seconds (traces may take time to appear)")
    print("- Check you're logged into the correct LangSmith account")
    print("- Verify the API key matches your LangSmith account")
    print("- Try refreshing the page")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
