import requests
import json

url = "https://api.deepseek.com/chat/completions"

payload = json.dumps({
  "model": "deepseek-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant"
    },
    {
      "role": "user",
      "content": "请问1+1等于几？"
    }
  ]
})
headers = {
  'Authorization': 'Bearer sk-4132fd1b571b4f6e8486a45b061ab3d0',
  'Content-Type': 'application/json',
  'Cookie': 'HWWAFSESID=53c439d8038d07f7d0; HWWAFSESTIME=1752147167360'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
