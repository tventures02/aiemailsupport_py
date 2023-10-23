from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app) # allow all
#CORS(app, origins=["https://your-allowed-domain.com", "https://another-allowed-domain.com"])

@app.route('/api/callSupportScribeV1', methods=['GET','POST'])
def call_support_scribe():
    from callAI import main
    data = request.json
    # Use the data received in your Python script as needed.
    # Example: Assuming your data has a key called 'message'
    prompt = data.get('prompt', '')  # Defaulting to an empty string if 'message' key doesn't exist
    print(prompt)
    output = main(prompt, "0")
    #return jsonify(message='hello world')
    
    return jsonify(output), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on port 80
