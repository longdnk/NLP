# Kiểm thử mô hình

## 1. Phương pháp Kiểm thử

- Kiểm thử bằng cách đưa văn bản, kiểm tra nghĩa giữa văn bản mẫu và văn bản đầu ra có phù hợp hay không.
- Các dữ liệu đưa vào là các dữ liệu không hề tồn tại trong qúa trình huấn luyện.

## 2. Kết quả kiểm thử
- Vì là mô hình ngôn ngữ nên kết quả kiểm thử có phần chủ quan, khi thực hiện lại trên api thì có thể sẽ ra kết quả không khớp như trong file đã mô tả.
- Kết quả có thể chỉ gần giống với kết quả đầu ra.

## 3. Thông tin các trường của API
``` python
class ModelEntity(BaseModel):
    text: str
    type: str
    model: str | None = None
    compression: float
```
**text**: là văn bản đầu vào. <br>

**type**: là loại tóm tắt, mặc định là tóm tắt ngắn gọn. <br>

**model**: mô hình dùng để tóm tắt, mặc định là dùng mô hình 2 tỉ tham số. <br>

**compression**: tỉ lệ nén của văn bản so với văn bản gốc