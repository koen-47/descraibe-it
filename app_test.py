import requests

# dictToSend = {"text": "This majestic predator hunts for prey on the plains of Africa."}
# res = requests.post('http://127.0.0.1:5000/predict', json=dictToSend)
# print('response from server:', res.text)
# dictFromServer = res.json()

dictToSend = {"text": "Small, sweet black fruit."}
res = requests.post('https://nlp-assignment-3.onrender.com/predict', json=dictToSend)
print('response from server:', res.text)
dictFromServer = res.json()
