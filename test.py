import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyBwLxZbcAzJKgo4GFVxtrDbM8ouSHvsNMw")

print("Available models that support generateContent:")
print("=" * 50)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print(f"Supported Methods: {model.supported_generation_methods}")
        print("-" * 30)