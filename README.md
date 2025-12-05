# Claude Code Web Test Repository

Test environment for Claude Code web functionality with data analysis capabilities.

## 🚀 Quick Start with Google Colab

Pandas AI를 사용한 Boston 주택 가격 분석을 Google Colab에서 바로 실행해보세요!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yunho0130/claude-code-web-test2/blob/main/pandasai_analysis.ipynb)

## Components

### 1. Boston Housing Price Analysis Web Application

PandasAI를 활용한 Boston 집값 데이터 시각화 웹 애플리케이션

### 2. Flexible Data Stream Handler

A comprehensive data handling system supporting multiple formats and streaming operations

## 실행 방법

### 방법 1: Conda 환경 사용 (권장)

```bash
# 1. Conda 환경 생성
conda create -n boston-housing python=3.10 -y

# 2. 환경 활성화
conda activate boston-housing

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 앱 실행
streamlit run app.py
```

### 방법 2: venv 사용

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 앱 실행
streamlit run app.py
```

### 방법 3: 직접 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
```

## 접속

앱 실행 후 브라우저에서 `http://localhost:8501` 접속

## 기능

- **Data Overview**: 데이터 개요 및 통계
- **Visualizations**: 다양한 시각화 차트
- **PandasAI Chat**: 자연어로 데이터 질의 (OpenAI API 키 필요)
- **Correlation Analysis**: 상관관계 분석

## 요구사항

- Python 3.9+
- OpenAI API Key (PandasAI Chat 기능 사용 시)

## Data Stream Handler

이 저장소는 유연한 데이터 스트림 핸들러를 포함합니다:

### 지원 형식
- CSV, JSON, JSON Lines (JSONL)
- Parquet (컬럼 저장 형식)
- Excel (XLSX, XLS)

### 주요 기능
- **다중 형식 지원**: 자동 형식 감지 및 변환
- **스트리밍 처리**: 대용량 파일 청크 단위 처리
- **데이터 검증**: 스키마 기반 검증
- **설정 기반 로딩**: JSON 설정 파일 지원

### 빠른 시작

```python
from data_stream_handler import DataStreamHandler

# 핸들러 초기화
handler = DataStreamHandler()

# 데이터 로드 (자동 형식 감지)
df = handler.load_data("data.csv")

# 형식 변환
handler.convert_format("input.json", "output.csv")

# 대용량 파일 스트리밍
for chunk in handler.load_streaming("large_file.csv", chunk_size=10000):
    process(chunk)
```

### 사용 예제

```bash
python examples/usage_examples.py
```

자세한 내용은 [DATA_STREAM_GUIDE.md](DATA_STREAM_GUIDE.md) 참조

## 저장소 구조

```
claude-code-web-test2/
├── app.py                       # Streamlit 웹 애플리케이션
├── data_stream_handler.py       # 데이터 스트림 핸들러
├── boston_house_prices.csv      # Boston 주택 가격 데이터
├── test_data/                   # 테스트 데이터
│   ├── sample_data.json
│   ├── sample_metrics.csv
│   └── streaming_data.jsonl
├── config/                      # 설정 파일
│   └── data_config.json
├── fixtures/                    # 테스트 픽스처
│   └── test_fixtures.py
└── examples/                    # 사용 예제
    └── usage_examples.py
```