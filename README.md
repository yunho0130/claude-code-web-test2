# Boston Housing Price Analysis Web Application

PandasAI를 활용한 Boston 집값 데이터 시각화 웹 애플리케이션

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