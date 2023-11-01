from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi

load_dotenv() #load .env file

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
        
        MONGODB_URI = os.environ.get("MONGODB_URI")
        if MONGODB_URI == None:
            raise ValueError('No mongodb atlas url found. Please check env variables.')
        
        # Connect with mongodb atlas
        client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        mongodb = client['supportscribe']
        mongodbCollection = mongodb['users']
        
        # Get user document
        user = mongodbCollection.find_one({"email": email})
        
        # Parse user document for necessary values
        documents = user['documents']
        googleTokens = user['googleTokens']
        if not documents:
            raise ValueError('No documents to read from.')
        doc = documents[0] # TODO: 
        docId = doc['id']
        collection = str(user['_id']) # use user id as collection name
        refreshToken = googleTokens['refreshToken']
        if refreshToken == '':
            raise ValueError('No refresh token.')
        
        # Run the create and save chromadb function
        output = main(collection, docId, refreshToken)
        success = output["success"]
    
        if success:
            return jsonify(success=True, message=output["message"]), 200
        else:
            return jsonify(success=False,error=output["error"]), 500
    except Exception as e:
        # print(e)
        return jsonify(success=False,error=str(e)), 500


@app.route('/updateCollectionIndex', methods=['GET','POST'])
def update_collection_index():
    try:
        from aiemailsupport_vectorstore.updateIndex import main
        data = request.json
        collection = data.get('collection', '') 
        output = main(collection)
        success = output["success"]
        if success:
            return jsonify(success=True, message=output["message"]), 200
        else:
            return jsonify(success=False,error=output["error"]), 500
    except Exception as e:
        # print(e)
        return jsonify(success=False,error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on port 5000
