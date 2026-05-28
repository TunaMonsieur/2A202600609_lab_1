# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, câu trả lời thường ổn định, ít biến thể và thiên về thông tin an toàn. Ở mức 0.5-1.0, phản hồi cân bằng hơn giữa tính chính xác và sự tự nhiên, nội dung đa dạng hơn nhưng vẫn bám ý chính. Khi lên 1.5, mô hình sáng tạo hơn rõ rệt nhưng cũng dễ thêm chi tiết ít chắc chắn hoặc diễn đạt lan man.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt khoảng 0.2-0.4 (thường chọn 0.3) để ưu tiên tính nhất quán, giảm rủi ro "bịa" thông tin và giữ giọng trả lời ổn định. Với chatbot hỗ trợ khách hàng, độ chính xác và dự đoán được quan trọng hơn sự sáng tạo.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Tổng output token/ngày = 10,000 x 3 x 350 = 10,500,000 token (~10,500 K token).  
> Chi phí GPT-4o: 10,500 x $0.010 = **$105/ngày**.  
> Chi phí GPT-4o-mini: 10,500 x $0.0006 = **$6.3/ngày**.  
> Tỷ lệ: 105 / 6.3 ≈ **16.7 lần** (GPT-4o đắt hơn khoảng 16-17 lần).

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> GPT-4o xứng đáng khi tác vụ cần suy luận chất lượng cao và sai số thấp (ví dụ: trợ lý phân tích hợp đồng, tư vấn nghiệp vụ quan trọng, hoặc tạo nội dung cần độ chính xác cao).  
> GPT-4o-mini phù hợp hơn cho các tác vụ volume lớn, yêu cầu phản hồi nhanh và chấp nhận chất lượng "đủ tốt" (ví dụ: FAQ, phân loại yêu cầu, tóm tắt ngắn, chatbot hỗ trợ cấp 1).

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang tương tác thời gian thực và nhạy với độ trễ cảm nhận, như chatbot hỗ trợ, copilot viết nội dung, hoặc giao diện hội thoại dài; việc thấy token xuất hiện dần giúp người dùng cảm giác hệ thống phản hồi ngay và giảm thời gian chờ chủ quan. Ngược lại, non-streaming phù hợp hơn khi cần kết quả hoàn chỉnh để xử lý một lần (ví dụ: trả về JSON chuẩn cho backend, chạy pipeline tự động, hoặc các tác vụ ngắn mà tổng thời gian đã rất thấp), vì cách này đơn giản hơn cho kiểm soát lỗi, kiểm thử và tích hợp hệ thống.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
