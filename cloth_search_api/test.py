import requests

with open("img.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/search",
        files={"image": f},
    )

print(response)
