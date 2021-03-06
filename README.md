# Post_Editing Seq2Seq
### 목적

음성인식 결과 자동 교정을 위한 sequence to sequence 모델링

- [Bahdanau_Attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention )을 통한 교정
- [KorBert](http://aiopen.etri.re.kr/service_dataset.php)임베딩모델 사용한 성능 향상(형태소)
- OOV 문제 해결을 위한 [FastText](https://github.com/facebookresearch/fastText/blob/master/python/README.mdhttps://github.com/facebookresearch/fastText/blob/master/python/README.md)를 적용한 attention model

---

### 데이터

구글 음성인식기를 통한 뉴스의 음성정보 데이터 사용(9.6만 문장)

---

### 진행사항

```
2020.01.30		코드 수정 및 데이터 전처리
2020.01.31		전처리 코드 수정 평가 코드 추가 및 주석
2020.02.05		검증 집합의 비용계산 함수 추가, 데이터 증폭을 위한 함수 추가(음소)
2020.02.06		데이터 증폭을 위한 함수 추가(형태소)
2020.02.11 ~		어절 단위 Attention Seq2Seq를 위한 코드 작성
2020.02.17
2020.02.18		형태소 단위 Attention validation loss 변환
			형태소 단위 save_data_morph.py 파일 수정
2020.02.25 ~ 		Transformer 공부 및 [구현](https://www.tensorflow.org/tutorials/text/transformer#top_of_page)
2020.02.28
2020.03 ~ 		KorBERT 적용 
2020.03.12		FastText 적용
2020.03.17		성능 측정을 위한 WER알고리즘 추가
```

---

### 성능 측정

[WER(Word Error Rate)](https://github.com/zszyellow/WER-in-python) 알고리즘을 사용한 측정

