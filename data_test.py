import requests

url = 'http://apis.data.go.kr/9710000/ProceedingInfoService/getLatestConInfoList'
params ={'serviceKey' : '서비스키', 'numOfRows' : '10', 'pageNo' : '1', 'class_code' : '1', 'commCode' : '' }

response = requests.get(url, params=params)
print(response.content)