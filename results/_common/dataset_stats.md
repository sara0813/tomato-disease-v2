# 데이터셋 클래스별 분포 및 불균형

이 리포트는 기록용이다. 클래스 가중치/오버샘플링으로 불균형을 보정하지 않고, PlantVillage의 자연스러운 분포를 그대로 학습에 사용한다.

| class | raw_plantvillage | plantvillage_train | plantvillage_val | plantvillage_test | taiwan_external | bangladesh_bbox_external | plantdoc_external |
|---|---|---|---|---|---|---|---|
| Tomato___Bacterial_spot | 2127 | 1488 | 319 | 320 | 110 | 93 | 100 |
| Tomato___Early_blight | 1000 | 700 | 150 | 150 | 0 | 84 | 86 |
| Tomato___Late_blight | 1909 | 1336 | 286 | 287 | 98 | 179 | 110 |
| Tomato___Leaf_Mold | 952 | 666 | 142 | 144 | 0 | 303 | 87 |
| Tomato___Septoria_leaf_spot | 1771 | 1239 | 265 | 267 | 0 | 0 | 146 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1676 | 1173 | 251 | 252 | 0 | 0 | 0 |
| Tomato___Target_Spot | 1404 | 982 | 210 | 212 | 0 | 47 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 | 3749 | 803 | 805 | 0 | 0 | 69 |
| Tomato___Tomato_mosaic_virus | 373 | 261 | 55 | 57 | 0 | 0 | 51 |
| Tomato___healthy | 1591 | 1113 | 238 | 240 | 106 | 395 | 60 |

### 불균형 비율 (max/min, 존재하는 클래스 기준)

| dataset | max/min ratio | max class | min class |
|---|---|---|---|
| raw_plantvillage | 14.36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus (5357) | Tomato___Tomato_mosaic_virus (373) |
| plantvillage_train | 14.36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus (3749) | Tomato___Tomato_mosaic_virus (261) |
| plantvillage_val | 14.60 | Tomato___Tomato_Yellow_Leaf_Curl_Virus (803) | Tomato___Tomato_mosaic_virus (55) |
| plantvillage_test | 14.12 | Tomato___Tomato_Yellow_Leaf_Curl_Virus (805) | Tomato___Tomato_mosaic_virus (57) |
| taiwan_external | 1.12 | Tomato___Bacterial_spot (110) | Tomato___Late_blight (98) |
| bangladesh_bbox_external | 8.40 | Tomato___healthy (395) | Tomato___Target_Spot (47) |
| plantdoc_external | 2.86 | Tomato___Septoria_leaf_spot (146) | Tomato___Tomato_mosaic_virus (51) |
