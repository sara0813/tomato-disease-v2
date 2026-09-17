"""클래스 정의와 외부 데이터셋 클래스 매핑.

- CLASS_NAMES: 모델 출력 순서 (PlantVillage 10개 클래스, 알파벳 순)
- CLASS_INFO: 웹 시스템(7단계)에서 보여줄 한국어 설명
- TAIWAN_CLASS_MAPPING / BANGLADESH_CLASS_ID_MAPPING / PLANTDOC_CLASS_MAPPING: 외부 평가(5단계) 라벨 정렬용
"""

# ============================================================
# 모델 출력 클래스 (순서 고정)
# ============================================================
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# 외부 데이터셋 → PlantVillage 클래스 매핑
# ============================================================
# Taiwan: 6개 클래스 중 PlantVillage와 겹치는 3개만 평가에 사용
# (Black mold / Gray spot / powdery mildew는 대응 클래스가 없어 제외)
TAIWAN_CLASS_MAPPING = {
    "Bacterial spot": "Tomato___Bacterial_spot",
    "Late blight": "Tomato___Late_blight",
    "health": "Tomato___healthy",
    "healthy": "Tomato___healthy",
    "Healthy": "Tomato___healthy",
}

# Bangladesh (YOLO bbox): class id → PlantVillage 클래스
# 0 Early Blight / 1 Black Spot / 2 Late Blight / 3 Leaf Mold
# 4 Bacterial Spot / 5 Target Spot / 6 Healthy
BANGLADESH_CLASS_ID_MAPPING = {
    0: "Tomato___Early_blight",
    1: None,  # Black Spot: 대응 클래스 없음 → 제외
    2: "Tomato___Late_blight",
    3: "Tomato___Leaf_Mold",
    4: "Tomato___Bacterial_spot",
    5: "Tomato___Target_Spot",
    6: "Tomato___healthy",
}

# PlantDoc: 9개 토마토 폴더 중 8개가 대응됨 (Taiwan 3/10, Bangladesh 6/10보다 훨씬 많이 겹침).
# "Tomato two spotted spider mites leaf"는 원본에 단 2장뿐이라(train 2, test 0) 통계적으로
# 의미가 없어 제외한다. Target_Spot에 대응하는 PlantDoc 폴더는 없다.
# 실사(Google/Bing 이미지 검색) 기반이라 검색 결과 특성상 진단표·비교 콜라주·삽화 같은
# "잎 사진이 아닌" 이미지가 섞여 있어, 수작업 콘택트시트 검수로 걸러낸 뒤 사용한다
# (PLANTDOC_EXCLUDE_FILES, prepare_plantdoc.py 참고).
PLANTDOC_CLASS_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": None,  # 원본 2장뿐 → 제외
}


# ============================================================
# 한국어 클래스 설명 (웹 시스템용)
# ============================================================
CLASS_INFO = {
    "Tomato___Bacterial_spot": {
        "name_ko": "세균성 점무늬병",
        "status": "비정상",
        "description": "잎에 작고 어두운 반점이 생기며 시간이 지나면 병반이 넓어질 수 있습니다.",
        "recommendation": "감염된 잎을 제거하고 과습을 피하며 통풍을 관리하세요.",
    },
    "Tomato___Early_blight": {
        "name_ko": "겹무늬병",
        "status": "비정상",
        "description": "잎에 둥근 갈색 병반과 동심원 모양의 무늬가 나타날 수 있습니다.",
        "recommendation": "감염 부위를 제거하고 잎이 젖어 있는 시간을 줄이세요.",
    },
    "Tomato___Late_blight": {
        "name_ko": "역병",
        "status": "비정상",
        "description": "잎에 어두운 갈색 또는 검은색 병반이 빠르게 퍼질 수 있습니다.",
        "recommendation": "감염된 식물체를 빠르게 제거하고 주변 식물로 확산되지 않게 관리하세요.",
    },
    "Tomato___Leaf_Mold": {
        "name_ko": "잎곰팡이병",
        "status": "비정상",
        "description": "잎 표면에 노란 반점이 생기고 뒷면에는 곰팡이성 병징이 나타날 수 있습니다.",
        "recommendation": "습도를 낮추고 환기를 강화하세요.",
    },
    "Tomato___Septoria_leaf_spot": {
        "name_ko": "셉토리아 잎반점병",
        "status": "비정상",
        "description": "작은 회색 또는 갈색 반점이 잎에 많이 생길 수 있습니다.",
        "recommendation": "감염 잎을 제거하고 토양에서 튀는 물이 잎에 닿지 않게 관리하세요.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name_ko": "점박이응애 피해",
        "status": "비정상",
        "description": "잎에 작은 점 형태의 피해가 나타나고 심하면 잎이 마르거나 변색될 수 있습니다.",
        "recommendation": "잎 뒷면을 확인하고 해충 확산을 막기 위해 초기 방제를 진행하세요.",
    },
    "Tomato___Target_Spot": {
        "name_ko": "타깃스팟",
        "status": "비정상",
        "description": "잎에 원형 갈색 반점이 생기며 병반 중심이 뚜렷하게 보일 수 있습니다.",
        "recommendation": "감염 잎을 제거하고 식물 간 간격과 통풍을 관리하세요.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_ko": "황화잎말림바이러스",
        "status": "비정상",
        "description": "잎이 노랗게 변하고 말리며 생장이 약해질 수 있습니다.",
        "recommendation": "매개충 관리가 중요하며 감염 의심 개체는 분리하세요.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name_ko": "토마토 모자이크 바이러스",
        "status": "비정상",
        "description": "잎에 얼룩덜룩한 모자이크 무늬와 변형이 나타날 수 있습니다.",
        "recommendation": "감염 의심 식물은 분리하고 도구 소독을 철저히 하세요.",
    },
    "Tomato___healthy": {
        "name_ko": "정상",
        "status": "정상",
        "description": "뚜렷한 병징이 관찰되지 않는 건강한 토마토 잎입니다.",
        "recommendation": "현재 상태를 유지하며 주기적으로 잎 상태를 확인하세요.",
    },
}
