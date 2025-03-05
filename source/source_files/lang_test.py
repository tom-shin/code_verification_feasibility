import requests
import re
from langdetect import detect_langs
import langid
import pycld2 as cld2


# ANSI 이스케이프 코드 및 숫자 제거를 위한 정규식 정의
ANSI_ESCAPE_AND_NUMBERS = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]|\d+')
OPENAI_API_KEY = "sk-"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

session = requests.Session()
session.headers.update(HEADERS)
###############################    기본 설정 끝 - 만지지 말 것    #############################################################


TEST_CONTENTS = """
Yoon Seok-yul 대통령 변호인단은 헌법재판소가 최상목 대통령 권한대행이 마은혁 
헌법재판관 후보자 임명을 보류한 것에 대해, '헌법재판소 구성권 침해'라며 판단한 것에 대해
 '결국 대통령 탄핵심판 정족수를 확보하기 위한 하명 결정'이라며 강하게 반발했다. 
 The defense team argued that the Constitutional Court’s ruling is essentially 
 a decision made to ensure the quorum of 3 required for the impeachment trial of the president. 
 他们认为，这一决定本质上是为了确保总统弹劾审判所需的法定人数，特别是3个法官。 
 Ils estiment que cette décision est en réalité un ordre visant à garantir les 3 
 juges nécessaires pour l'impeachment du président.
"""
cleaned_sentence = ANSI_ESCAPE_AND_NUMBERS.sub('', TEST_CONTENTS)   # ANSI 이스케이프 코드와 숫자 제거

lang_select = "english"
system_prompt = f"you can speak only {lang_select}. respond in {lang_select}. no matter situation"
user_prompt = f"summarize the followings. {cleaned_sentence}.\n respond in {lang_select}. no matter situation "
temperature = 0.1
model = "gpt-4"

###############################    사용자 설정 끝   #############################################################

# 2. 언어 판단
print("[입력 언어]")
langs = detect_langs(cleaned_sentence)
lang, _ = langid.classify(cleaned_sentence)
is_reliable, text_bytes_found, details = cld2.detect(cleaned_sentence)
# user_defined_lang_Detect(cleaned_sentence)  ==> 구현 필요 예를들면 char에 대해서 ascii 범위에 있는지 ...

print(f"detect_langs: {langs}")
print(f"langid: {lang}")
print(f"pycld2: {details[0][0]}")  # Detected language code
print("-----------------------------------------------------------------------------------------------")

# 3 pyload 만들기
payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }

response = session.post(OPENAI_API_URL, json=payload)  # 여기 수정됨!
answer = response.json()["choices"][0]["message"]["content"]

print("\n[출력 언어]")
langs = detect_langs(answer)
lang, _ = langid.classify(answer)
is_reliable, text_bytes_found, details = cld2.detect(answer)

print(f"detect_langs: {langs}")
print(f"langid: {lang}")
print(f"pycld2: {details[0][0]}")  # Detected language code
print("-----------------------------------------------------------------------------------------------")
print("\n[출력 문장]")
print(answer)
