# Comparative ML/DL Approaches for Information Extraction in Real Estate

Dự án này thực hiện nghiên cứu, triển khai và so sánh hiệu quả giữa các phương pháp Học máy truyền thống (Machine Learning - ML) và Học sâu (Deep Learning - DL) cho bài toán Trích xuất thông tin (Information Extraction - IE) trên dữ liệu văn bản tin rao bất động sản tiếng Việt.
Hệ thống bao gồm hai tác vụ chính:
1. Nhận diện thực thể (NER): Xác định các thông tin như Vị trí, Giá cả, Diện tích, Loại bất động sản...
2. Trích xuất quan hệ (RE): Xác định mối liên kết giữa các thực thể (ví dụ: Nhà A có giá 5 tỷ, Nhà A nằm tại Quận 1).

## Project Structure

```
├── configs/               # Chứa các file cấu hình (YAML) cho training
│   ├── ml_config.yaml     # Tham số cho các mô hình ML (SVM, CRF, RF...)
│   └── dl_config.yaml     # Tham số cho các mô hình DL (PhoBERT, LR, Epochs...)
├── data/                  # Quản lý dữ liệu theo từng giai đoạn
│   ├── 01_raw/            # Dữ liệu thô ban đầu (CSV/Excel)
│   ├── 02_intermediate/   # Dữ liệu JSON đã chuyển đổi sơ bộ
│   ├── 03_primary/        # Dữ liệu đã gán nhãn (Labeled Data) theo vùng miền
│   └── 04_model_input/    # Dữ liệu chuẩn (Train/Dev/Test) để đưa vào mô hình
├── docs/                  # Tài liệu hướng dẫn gán nhãn, báo cáo
├── models/                # Nơi lưu trữ các model đã huấn luyện (.pkl, .bin) và metadata
├── notebooks/             # Jupyter Notebooks cho thực nghiệm và phân tích
│   ├── 01_EDA_Data_Analysis.ipynb         # Phân tích khám phá dữ liệu
│   ├── 02_Exp_NER_ML.ipynb                # Huấn luyện NER bằng ML (CRF, SVM...)
│   ├── 02_Exp_NER_DL.ipynb                # Huấn luyện NER bằng DL (PhoBERT)
│   ├── 03_Exp_RE_ML.ipynb                 # Huấn luyện RE bằng ML (SVM, RF + PhoBERT Vectors)
│   ├── 03_Exp_RE_DL.ipynb                 # Huấn luyện RE bằng DL (PhoBERT Fine-tuning)
│   ├── 04_Benchmark_Comparison_ML.ipynb   # So sánh hiệu năng các model ML
│   └── 04_Benchmark_Comparison_DL.ipynb   # So sánh hiệu năng các model DL & Tổng kết
├── scripts/               
├── src/                   
│   ├── data_loader/       
│   ├── features/          
│   ├── models/            
│   └── ui/                
├── ui/                    # Giao diện
├── requirements.txt       # Danh sách thư viện phụ thuộc
└── README.md              # Tài liệu dự án
```
## Phương pháp Tiếp cận
### 1. Nhận diện thực thể (Named Entity Recognition - NER)
Mục tiêu: Gán nhãn cho từng token trong câu (B-LOC, I-LOC, B-PRICE, O...).
Machine Learning:
- Mô hình: CRF (Conditional Random Fields), SVM, MaxEnt.
- Đặc trưng: Hand-crafted features (word shape, prefix, suffix) kết hợp vector phẳng.
Deep Learning:
- Mô hình: PhoBERT-base (Fine-tuning cho bài toán Token Classification).
- Đặc trưng: Contextual Embeddings tự động từ PhoBERT.

3. Trích xuất quan hệ (Relation Extraction - RE)
Mục tiêu: Phân loại quan hệ giữa 2 thực thể (HAS_PRICE, LOCATED_AT, NO_RELATION...).

Machine Learning:
- Mô hình: SVM, Logistic Regression (MaxEnt), Random Forest.
- Kỹ thuật: Sử dụng vector trích xuất từ PhoBERT (Feature Extraction) kết hợp kỹ thuật Entity Markers ([E1], [E2]) + PCA giảm chiều + Hybrid Sampling (để xử lý mất cân bằng dữ liệu).

Deep Learning:
- Mô hình: PhoBERT-base (Fine-tuning cho bài toán Sequence Classification).
- Input: [CLS] Câu văn chứa cặp thực thể [SEP].

## Cài đặt & Sử dụng
### 1. Yêu cầu hệ thống
```
Python 3.8+
GPU (Khuyến nghị cho Deep Learning)
```
### 2. Cài đặt thư viện
```
git clone https://github.com/tommyhuy1705/comparative_ml_deep_learning_approaches_for_information_extraction_in_real_estate.git
cd comparative_ml_deep_learning_approaches_for_information_extraction_in_real_estate
pip install -r requirements.txt
```
### 3. Chuẩn bị dữ liệu
Chạy script để chuyển đổi dữ liệu thô sang định dạng model input:
```
python scripts/prepare_data.py
```
### 4. Huấn luyện & Đánh giá (Experiments)
Chạy các Jupyter Notebook trong thư mục notebooks/ để tái lập kết quả thực nghiệm:
- Bước 1: Chạy `01_EDA_Data_Analysis.ipynb` để hiểu dữ liệu.
- Bước 2: Chạy các file `02_Exp_NER_.ipynb` để huấn luyện mô hình NER.
- Bước 3: Chạy các file `03_Exp_RE_.ipynb` để huấn luyện mô hình RE.
- Bước 4: Chạy `04_Benchmark_Comparison_ML.ipynb` và `04_Benchmark_Comparison_DL.ipynb` để xem biểu đồ so sánh kết quả.
### 5. Chạy Demo Ứng dụng
Dự án cung cấp giao diện trực quan (UI) để test hệ thống IE hoàn chỉnh:
```
streamlit run ui/app.py
```
### Tính năng nổi bật
- End-to-End Pipeline: Tích hợp trọn vẹn từ Raw Text -> NER -> RE -> Structured Info.
- Hybrid Approach: Kết hợp điểm mạnh của ML (tốc độ, ít dữ liệu) và DL (độ chính xác cao).
- Data Handling: Xử lý tốt các vấn đề đặc thù của dữ liệu bất động sản Việt Nam (viết tắt, lỗi chính tả, format lộn xộn).
- Optimization: Áp dụng các kỹ thuật tối ưu như PCA, SMOTE, Class Weights cho ML để cải thiện F1-Score trên dữ liệu mất cân bằng.

## License
MIT License

