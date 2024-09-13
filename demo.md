# Demo đề tài xây dựng api tóm tắt văn bản tiếng Việt dùng cho tóm tắt và tóm lượt đơn văn bản

## 1. Giới thiệu

- Trong demo này sử dụng GPU Nvidia Rtx 3080ti để chạy cả 2 mô hình cùng một lúc.
- Không khuyến khích chạy cả 2 mô hình trên một máy nếu đem đi triển khai thực tế.
- Đối với GPU có lượng VRam từ 4 - 8GB nên dùng mô hình 2 tỉ tham số.
- Đối với GPU có lượng VRam trên 10GB thì mới chạy được mô hình 7 tỉ tham số trong demo.

## 2. Các tính năng demo
- Tóm tắt văn bản với loại tóm tắt **chi tiết** và tóm tắt **ngắn gọn**.
- Kết hợp thêm tóm tắt với độ nén, trong đó tóm tắt **chi tiết** có độ nén tối thiểu là **40%**, tóm tắt **ngắn gọn** thì có độ nén tối thiểu lên đến **10%**.
- Độ nén mặc định của tính năng tóm tắt **chi tiết** là 70%, đối với tóm tắt **ngắn gọn** là **50%**.
- Có thể đọc thông tin từ tệp docx, dotx, pdf, txt.
- Hệ thống không hỗ trợ đọc các tệp .xlsx, .md, .tex hay các định dang tương tự.
- Kiểm soát hoạt động của hệ thống bằng **reverse proxy** và **hệ thống file log**.

## 3. Bảng thống kê thời gian phản hồi
Lưu ý: bảng thống kê này được thực hiện dựa trên **GPU Nvidia 3080 ti** và được thời gian phản hồi được tính là thời gian phản hồi trung bình trên các văn bản có độ dài tối đa 3000 từ.

Bảng thống kê cho loại **tóm tắt chi tiết**
| Số lượng từ nén (%) | Mô hình 2 tỷ tham số (giây) | Mô hình 7 tỷ tham số (giây) |
|---------------------|-----------------------------|-----------------------------|
| 40                  | 9.25                         | 17.4                       |
| 50                  | 12                           | 21.7                       |
| 60                  | 13.5                         | 26                         |
| 70                  | 15.8                         | 30                         |
| 80                  | 17.8                         | 34.7                       |
| 90                  | 20                           | 40                         |

Bảng thống kê cho loại tóm tắt **ngắn gọn**
| Số lượng từ nén (%) | Mô hình 2 tỷ tham số (giây) | Mô hình 7 tỷ tham số (giây) |
|---------------------|-----------------------------|-----------------------------|
| 10                  | 2.5                         | 4.5                         |
| 20                  | 5                           | 8.7                         |
| 30                  | 7                           | 13                          |
| 40                  | 9.3                         | 17.4                        |
| 50                  | 11.2                        | 21.7                        |
| 60                  | 13.7                        | 26                          |

Thông tin chi tiết mã nguồn: [Github](https://github.com/longdnk/NLP)