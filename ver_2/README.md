# TWO-TWO Version 2 (ver_2) Research & Development

이 디렉토리는 TWO-TWO 비트코인 선물 자동매매 보트의 **ver_2 연구 및 개발 작업공간**입니다.

## 📌 주요 연구 목표 (Research Goals)
1. **데이터 및 피처 고도화**: 추가 보조지표, 마이크로/마크로 지표, 온체인 데이터 결합
2. **모델링 개선**: XGBoost 파라미터 최적화, 신규 ML/DL 알고리즘(LightGBM, CatBoost 등) 실험
3. **리스크 관리 고도화**: 가변 레버리지, Dynamic Stop Loss / Take Profit, 변동성 기반 포지션 사이징
4. **백테스팅 고도화**: 슬리피지, 수수료, 미체결 및 펀딩비 정밀 반영 백테스트

## 📁 디렉토리 구조
- `ver_2/data/` : 연구용 OHLCV 및 피처 데이터 저장
- `ver_2/models/` : ver_2 모델 파일 (.pkl 등)
- `ver_2/backtest/` : 백테스팅 스크립트 및 결과
- `ver_2/strategies/` : 신규 매매 전략 및 시그널 로직
