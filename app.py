from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

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
    
    try:
        data = request.json
        email = data.get('email', '')
        if email == '':
            raise ValueError("No email found.")
        
        MONGO_URI='mongodb+srv://bpadmin:IMBF9iORG3TVS8XP@bp-dev-and-local.guo9v5t.mongodb.net/supportscribe?retryWrites=true&w=majority'
        client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
        mongodb = client['supportscribe']
        mongodbCollection = mongodb['users']
        
        
        user = mongodbCollection.find_one({"email": email})
        
        
        documents = user['documents']
        googleTokens = user['googleTokens']
        
        
        doc = documents[0]
        docId = doc['id']
        collection = user['email']
        refreshToken = googleTokens['refreshToken']
        
        print(docId)
        print(collection)
        print(refreshToken)
        return jsonify(success=True), 200
        
        # output = main( collection, docId, refreshToken)
        # success = output["success"]
    
        # if success:
        #     return jsonify(success=True, message=output["message"]), 200
        # else:
        #     return jsonify(success=False,error=output["error"]), 500
    except Exception as e:
        print(e)
        return jsonify(success=False,error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on port 5000
