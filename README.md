# HOMEWORK 2: NUMPY FOR DATA SCIENCE

**Sinh viên:** Trần Hồng Phương.

**MSSV:** 23127250

## Mục lục

1.  [Giới thiệu](#giới-thiệu)
2.  [Dataset](#dataset)
3.  [Cấu trúc dự án](#cấu-trúc-dự-án)
4.  [Phương pháp thực hiện](#phương-pháp-thực-hiện)
5.  [Cài đặt và Hướng dẫn sử dụng](#cài-đặt-và-hướng-dẫn-sử-dụng)
6.  [Kết quả](#kết-quả)
7.  [Khó khăn và Giải pháp](#khó-khăn-và-giải-pháp)
8.  [Hướng phát triển](#hướng-phát-triển)
9.  [Thông tin tác giả Dataset](#thông-tin-tác-giả-dataset)
10. [Thông tin liên hệ](#thông-tin-liên-hệ)

-----

## Giới thiệu

Dự án này là bài tập thực hành kỹ năng xử lý dữ liệu và xây dựng mô hình học máy cơ bản với yêu cầu kỹ thuật nghiêm ngặt: **Chỉ sử dụng thư viện NumPy** cho toàn bộ quá trình xử lý, tính toán và xây dựng mô hình (không sử dụng Pandas hay Scikit-learn cho các tác vụ này).

**Bài toán:** Dự đoán giá thuê phòng và phân tích các yếu tố ảnh hưởng đến giá dựa trên dữ liệu Airbnb tại New York City (2019).
**Mục tiêu:**

  - Xây dựng quy trình xử lý dữ liệu thô (ETL) hoàn toàn bằng NumPy.
  - Trực quan hóa dữ liệu để tìm ra các insight về vị trí, loại phòng và hành vi đánh giá.
  - Cài đặt thủ công (from scratch) hai thuật toán **Ridge Regression** và **Lasso Regression** để dự đoán giá.

## Dataset

  - **Nguồn dữ liệu:** New York City Airbnb Open Data (Kaggle).
  - **Kích thước:** 48,895 dòng, 16 cột.
  - **Mô tả đặc trưng (Features):**

| Index | Tên cột | Mô tả ngắn gọn |
|-------|------------------------------------|--------------------------------------------------------------------|
| 0 | **id** | Mã định danh của từng listing. |
| 1 | **name** | Tên tiêu đề của chỗ ở trên Airbnb. |
| 2 | **host\_id** | Mã định danh của chủ nhà. |
| 3 | **host\_name** | Tên hiển thị của chủ nhà trên Airbnb. |
| 4 | **neighbourhood\_group** | Khu vực lớn/ quận (ví dụ: Manhattan, Brooklyn). |
| 5 | **neighbourhood** | Khu phố cụ thể nơi listing nằm. |
| 6 | **latitude** | Vĩ độ vị trí của listing. |
| 7 | **longitude** | Kinh độ vị trí của listing. |
| 8 | **room\_type** | Loại phòng cho thuê (Entire home/apt, Private room...). |
| 9 | **price** | Giá thuê theo đêm (Target Variable). |
| 10 | **minimum\_nights** | Số đêm tối thiểu phải đặt. |
| 11 | **number\_of\_reviews** | Tổng số đánh giá mà listing nhận được. |
| 12 | **last\_review** | Ngày đánh giá gần nhất. |
| 13 | **reviews\_per\_month** | Số lượng đánh giá trung bình mỗi tháng. |
| 14 | **calculated\_host\_listings\_count** | Số lượng listing khác của cùng chủ nhà. |
| 15 | **availability\_365** | Số ngày trống để đặt trong 1 năm. |

## Cấu trúc dự án

```text
23127250/
├── README.md               # rít mi
├── requirements.txt        # Danh sách thư viện cần thiết
├── data/
│   ├── raw/                # Dữ liệu gốc (AB_NYC_2019.csv)
│   └── processed/          # Dữ liệu sau khi làm sạch và chuẩn hóa
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Khám phá dữ liệu (EDA) và Trực quan hóa
│   ├── 02_preprocessing.ipynb     # Làm sạch, tạo đặc trưng và chuẩn hóa
│   └── 03_modeling.ipynb          # Cài đặt thuật toán và đánh giá kết quả
└── src/
    ├── __init__.py
    ├── data_processing.py  # Hàm load, xử lý missing, one-hot encoding
    ├── visualization.py    # Hàm vẽ biểu đồ
    └── models.py           # Class RidgeRegression và LassoRegression tự cài đặt
```

## Phương pháp thực hiện

### 1\. Xử lý dữ liệu (Data Processing)

  - **Đọc dữ liệu:** Sử dụng module `csv` kết hợp với `np.array` để xử lý định dạng hỗn hợp (chuỗi và số) mà `np.loadtxt` thông thường không đọc được.
  - **Làm sạch dữ liệu:**
      - Sử dụng kỹ thuật **Boolean Masking** (Vectorization) để lọc và xử lý các giá trị `nan`, `null` hoặc chuỗi rỗng.
      - Xử lý các giá trị ngoại lai (Outliers) dựa trên phân phối thực tế của dữ liệu.
  - **Feature Engineering:**
      - Tạo đặc trưng mới: Khoảng cách từ căn hộ đến trung tâm thành phố.
      - Mã hóa One-Hot (One-Hot Encoding) cho các biến phân loại (`neighbourhood_group`, `room_type`) sử dụng NumPy thuần.
      - Chuẩn hóa dữ liệu số (Numerical features) bằng phương pháp Log transformation (`np.log1p`) để giảm độ lệch (skewness).

### 2\. Thuật toán sử dụng (Modeling)

Tôi tự cài đặt hai lớp mô hình hồi quy tuyến tính có điều chuẩn (Regularized Linear Regression) bằng NumPy:

**a. Ridge Regression (L2 Regularization):**

  - Sử dụng công thức nghiệm đóng (Closed-form solution) để tìm trọng số tối ưu $w$:
    $$w = (X^T X + \alpha I)^{-1} X^T y$$
  - Cài đặt bằng `np.linalg.solve` để tối ưu hiệu suất tính toán ma trận.

**b. Lasso Regression (L1 Regularization):**

  - Sử dụng thuật toán **Coordinate Descent** để tối ưu hàm mất mát.
  - Cập nhật trọng số dựa trên toán tử Soft Thresholding:
    $$S(w_j, \lambda) = \text{sign}(w_j)(|w_j| - \lambda)_+$$
  - Mô hình hỗ trợ chọn lọc đặc trưng (Feature Selection) bằng cách đưa các trọng số không quan trọng về 0.

### 3\. Đánh giá mô hình

  - Tự cài đặt hàm chia tập dữ liệu `train_test_split` và K-Fold Cross Validation.
  - Sử dụng metrics: $R^2$ Score và RMSE (Root Mean Squared Error).

## Cài đặt và Hướng dẫn sử dụng

1.  **Cài đặt thư viện:**
    Bài tập được làm trên Python 3.13.2 và các thư viện cơ bản liệt kê trong `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Quy trình chạy:**

      - **Bước 1:** Chạy `notebooks/01_data_exploration.ipynb` để có cái nhìn tổng quan về dữ liệu.
      - **Bước 2:** Chạy `notebooks/02_preprocessing.ipynb`. Notebook này sẽ lưu các file dữ liệu đã xử lý vào thư mục `data/processed/`.
      - **Bước 3:** Chạy `notebooks/03_modeling.ipynb` để huấn luyện mô hình Ridge/Lasso và xem kết quả dự đoán.

## Kết quả

  - **Phân tích:**
      - Yếu tố **Vị trí** (khoảng cách đến trung tâm) và **Loại phòng** (Entire home/apt) có ảnh hưởng lớn nhất đến giá thuê.
      - Mối quan hệ giữa các biến độc lập và giá tiền không hoàn toàn tuyến tính.
  - **Hiệu năng mô hình:**
      - Cả Ridge và Lasso đều cho kết quả $R^2$ ở mức trung bình (khoảng 0.5 - 0.6) trên tập kiểm thử.
      - Lasso Regression thực hiện tốt việc loại bỏ các biến nhiễu (đưa trọng số về 0), giúp mô hình đơn giản hơn nhưng vẫn giữ nguyên độ chính xác so với Ridge.

## Khó khăn và Giải pháp

Trong quá trình thực hiện bài tập với ràng buộc chỉ dùng NumPy, tôi đã gặp các khó khăn sau:

1.  **Đọc dữ liệu hỗn hợp:**

      - *Vấn đề:* Dữ liệu gốc chứa cả ký tự xuống dòng, dấu phẩy trong chuỗi văn bản khiến `np.genfromtxt` bị lỗi cấu trúc.
      - *Giải pháp:* Xin thầy sử dụng thư viện `csv`. Viết hàm `load_data` riêng sử dụng `csv.reader` để parse từng dòng chính xác trước khi chuyển sang NumPy array.

2.  **Dữ liệu phân tán, phi tuyến tính:**

      - *Vấn đề:* Dữ liệu có nhiều giá trị bằng `0` và mối quan hệ giữa các đặc trưng với giá tiền không hoàn toàn là tuyến tính. Điều này gây khó khăn cho các mô hình hồi quy tuyến tính.
      - *Giải pháp:* Dùng `np.log1p` cho cả biến mục tiêu và các đặc trưng numeric để giảm độ lệch của dữ liệu.

3.  **Vectorization:**

      - *Vấn đề:* Các thao tác xử lý chuỗi ban đầu dùng vòng lặp `for` rất chậm. Và cách làm này cũng vi phạm vào một trong những tiêu chí tính điểm của thầy.
      - *Giải pháp:* Chuyển đổi sang sử dụng Boolean Masking và các hàm Universal Functions (ufuncs) của NumPy để xử lý trên toàn bộ mảng cùng lúc.

## Hướng phát triển

  - Tối ưu hóa code để xử lý lượng dữ liệu lớn hơn.
  - Nếu không bị giới hạn bởi yêu cầu chỉ dùng NumPy, tôi sẽ áp dụng các mô hình phi tuyến tính như **Random Forest** hoặc **XGBoost**. Các mô hình này có khả năng tự học các tương tác phức tạp và xử lý ngoại lai tốt hơn, dự kiến sẽ cải thiện đáng kể điểm số $R^2$.

## Thông tin tác giả Dataset

  - **Nguồn:** Kaggle - New York City Airbnb Open Data.
  - **Tác giả:** Dgomonov.
  - **License:** CC0: Public Domain.

## Thông tin liên hệ
  - Github: [phuongth05](https://github.com/phuongth05)
  - email: thphuong23@clc.fitus.edu.vn

Trần Hồng Phương - lữ khách giang hồ, nay đây mai đó.
