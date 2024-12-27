# 🫀 **ECG Artifact Detection AI Model**

## 🚀 **1. 프로젝트 개요**

**목적:**  
12-Lead ECG 데이터를 분석하여 아티팩트(Artifact)를 검출하고, 클린 신호(0)와 아티팩트 포함 신호(1)를 분류하는 인공지능 모델을 개발합니다.  

**핵심 목표:**  
- ECG 신호 내 아티팩트 검출  
- 신호 분류 정확도 최적화  
- 진단 오류 최소화  

---

## 📊 **2. 데이터 설명**

- **출처:** 고려대학교 안암병원 (K-MEDiCon)  
- **포맷:** `.pkl` 형식  
- **샘플링 주파수:** 500Hz  
- **신호 형태:** `[5000,12]` (시간 포인트 5000 × 12 리드)  

**📌 라벨링:**  
- `0`: 클린 ECG 신호  
- `1`: 아티팩트 포함 ECG 신호  
- **조건:** 12개의 리드 중 하나라도 아티팩트가 감지되면 `1`로 라벨링  

**📌 데이터셋 구성:**  
| **Data Set** | **정상 심전도 (0)** | **Artifact 포함 (1)** |  
|--------------|-------------------|-----------------------|  
| Training set | 1,799             | 601                   |  
| Public test  | 225               | 75                    |  
| Private test | 225               | 75                    |

## ⚙️ **3. 프로젝트 설정**

### **3.1 환경 설정**

#### **필수 라이브러리 설치**

```
pip install -r requirements.txt
```

## ⚙️ **3.2 데이터 전처리**

- **노이즈 제거:** 저역통과 필터링  
- **신호 정규화:** Min-Max Scaling 적용  
- **데이터 분할:** Train / Validation / Test (8:1:1)  


## ⚙️ **3.3 모델 학습**

모델은 **CNN1D_Modified** 구조를 사용하며, 최적화 알고리즘으로 **Adam**을 사용합니다.



## 🛠️ **4. 프로젝트 구조**

```bash
├── data/
│   ├── raw/       # 원본 ECG 데이터
│   ├── processed/ # 전처리된 ECG 데이터
├── src/
│   ├── preprocessing.py # 데이터 전처리
│   ├── data_module.py   # 데이터 로드 및 분할
│   ├── model.py         # 모델 정의 (CNN1D_Modified)
│   ├── cpi_metric.py    # CPI 성능 메트릭
├── results/             # 분석 결과
├── README.md            # 프로젝트 설명
├── requirements.txt     # 의존성 목록
└── main.py              # 메인 실행 스크립트
```
