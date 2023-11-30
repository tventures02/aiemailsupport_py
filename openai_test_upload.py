from openai import OpenAI
import io
import os
import requests

client = OpenAI()

# Your API endpoint (replace with the actual endpoint)
api_endpoint = 'https://api.openai.com/v1/files'

# Your API key (replace with your actual API key)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
headers = {
    'Authorization': f'Bearer {OPENAI_API_KEY}'
}

file_content = "Jin guan's birthday is on November 4, 1980. His favorite color is green. He is 34.2 years old. He weighs 200 pounds. "
file_like_object = io.BytesIO(file_content.encode('utf-8'))
file_name = "example_filename.txt"

files = {'file': (file_name, file_like_object, 'application/octet-stream')}
data = {'purpose': 'assistants'}

response = requests.post(api_endpoint, headers=headers, files=files, data=data)

# Close the BytesIO object
file_like_object.close()

# Print the response from the server
print(response.text)
# Parse the JSON response
response_json = response.json()

# Access the "id" parameter
file_id = response_json.get('id', None) 
print(file_id)

my_updated_assistant = client.beta.assistants.update(
  "asst_fh5cjjxYy13QyhRux88vVxrQ",
  file_ids=[file_id],
)