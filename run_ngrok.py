from pyngrok import ngrok, conf
import time

# ngrok 인증 토큰 설정
conf.get_default().auth_token = "3H3SMLPFigCBBBYoOGfjAvCJCoG_2HQvkiAA2BE9b9e5H6fkU"

# ngrok 터널 오픈 (Streamlit 포트 8501)
public_url = ngrok.connect(8501)
print(f"NGROK_URL: {public_url.public_url}")

# 터널 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.kill()
