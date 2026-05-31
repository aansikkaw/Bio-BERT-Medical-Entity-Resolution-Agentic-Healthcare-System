import requests

url = "http://localhost:8000/predict"
data = {
    "anchor": "Type 2 Diabetes Mellitus",
    "comparison": "T2DM"
}

response = requests.post(url, json=data)
print(response.json())
