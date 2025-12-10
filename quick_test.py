import requests

print("🧠 Testing Contextual API...")

# Test with a complex sentiment that requires contextual understanding
test_text = "I support this but have concerns about implementation"

try:
    response = requests.post(
        'http://127.0.0.1:8002/api/v1/analyze',
        json={'text': test_text},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()['result']
        print(f"✅ API Status: {response.status_code}")
        print(f"Text: {test_text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Analysis Method: {result['method']}")
        print(f"Justification: {', '.join(result['justification_words'])}")
        print("🎉 Contextual Analysis API is working perfectly!")
    else:
        print(f"❌ API Error: {response.status_code}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")