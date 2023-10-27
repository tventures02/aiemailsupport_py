from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app) # allow all
#CORS(app, origins=["https://your-allowed-domain.com", "https://another-allowed-domain.com"])

@app.route('/callSupportScribeV1', methods=['GET','POST'])
def call_support_scribe():
    from callAI import main
    data = request.json
    prompt = data.get('prompt', '')  # Defaulting to an empty string if 'prompt' key doesn't exist
    collection = data.get('collection', '') 
    # print(prompt)
    output = main(prompt, "0", collection)
    
    return jsonify(output), 200

@app.route('/createAndSaveDocumentIndex', methods=['POST'])
def create_and_save_document_index():
    from aiemailsupport_vectorstore.createAndSaveIndex import main
    data = request.json
    docId = data.get('docId', '')
    collection = data.get('collection', '')
    refreshToken = data.get('refreshToken', '')
    output = main('', collection, docId, refreshToken)

    if output != None:
        return jsonify(success=True), 200
    else:
        return jsonify(success=False), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on port 5000
