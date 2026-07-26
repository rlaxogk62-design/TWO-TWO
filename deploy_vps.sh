#!/bin/bash
# =============================================================================
# Ubuntu Linux VPS 자동 세팅 및 봇 가동 스크립트
# =============================================================================

echo "=== [1/4] 시스템 패키지 업데이트 및 필수 도구 설치 ==="
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo "=== [2/4] uv 패키지 매니저 설치 (빠른 배포용) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/4] 깃허브 저장소 클론 ==="
# 만약 개인 리포지토리라면 나중에 토큰 입력이 필요할 수 있습니다.
cd ~
git clone https://github.com/rlaxogk62-design/TWO-TWO-LIVE.git || git clone https://github.com/rlaxogk62-design/TWO-TWO.git
cd TWO-TWO-LIVE || cd TWO-TWO

echo "=== [4/4] 가상환경 구축 및 의존성 설치 ==="
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt

echo "============================================================================="
echo "설치가 완료되었습니다! 이제 아래 두 단계를 완료해 주세요."
echo "============================================================================="
echo "1. 현재 폴더에 .env 파일을 만들고 바이낸스 API 키를 입력하세요:"
echo "   nano .env"
echo "   (내용 입력 후 Ctrl+O -> Enter -> Ctrl+X로 저장 및 종료)"
echo ""
echo "2. 백그라운드에서 봇을 24시간 안정적으로 실행하세요 (SSH 종료 후에도 유지):"
echo "   nohup python3 binance_bot.py > bot.log 2>&1 &"
echo ""
echo "3. 실시간 작동 로그 확인법:"
echo "   tail -f bot.log"
echo "============================================================================="
