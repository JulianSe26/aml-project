import requests

url = "http://localhost:5000/inference"

payload = {}
files = [
  ('file', open('./example.png','rb'))
]
headers = {
  'Authorization': 'Basic YW1scHJvamVjdDphbWxwcm9qZWN0'
}

response = requests.request("POST", url, headers=headers, data = payload, files = files)

print(response.text.encode('utf8'))