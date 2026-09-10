Dưới đây là khung sườn và hướng dẫn chi tiết cách viết báo cáo Đồ án ngành **hoàn toàn tổng quát**, được xây dựng **nghiêm ngặt theo đúng file hướng dẫn chính thức của Khoa Công nghệ Thông tin — Trường Đại học Mở TP.HCM (`DO_AN_Huong_dan_Do_an_nganh_CNTT_OU.pdf`)**, không dựa vào bất kỳ dự án cụ thể nào.

---

### I. THÔNG TIN CHUNG & QUY CHUẨN ĐỊNH DẠNG BÁO CÁO

#### 1. Quy định về cấu trúc tổng thể và độ dài
* **Độ dài quyển báo cáo**: Từ **50 đến 80 trang** phần nội dung chính (chưa tính phần phụ lục). Quyển dày không đồng nghĩa với điểm cao vì hội đồng chấm mật độ thông tin.
* **Nguyên tắc viết báo cáo**: Báo cáo được chấm dựa trên 3 yếu tố: sản phẩm chạy được, báo cáo viết đúng chuẩn và khả năng bảo vệ. Tuyệt đối **không dồn việc viết báo cáo vào 2 tuần cuối** mà phải viết song song ngay từ tuần đầu tiên của học kỳ.
* **Bố cục tổng thể 3 phần**:
  1. **Phần đầu**: Bìa chính, bìa phụ, Lời cảm ơn, Nhận xét của GVHD, Mục lục, Danh mục hình vẽ/bảng biểu, Danh mục từ viết tắt.
  2. **Phần nội dung (5 chương)**: Chương 1 đến Chương 5.
  3. **Phần cuối**: Tài liệu tham khảo, Phụ lục (mã nguồn tiêu biểu, bảng dữ liệu, hướng dẫn cài đặt, hình ảnh giao diện đầy đủ).

#### 2. Quy định về định dạng trang và trình bày
* **Khổ giấy & Font chữ**: Khổ giấy A4 (210 × 297 mm); phông chữ **Times New Roman, cỡ 13** cho phần nội dung (không dùng nhiều phông chữ trong một quyển).
* **Giãn dòng & Căn lề**: Giãn dòng **1,5 lines**; giãn đoạn dưới **6 pt**; căn đều hai bên (justify).
* **Lề trang**: Lề Trái **3 cm** (để đóng gáy); Phải **2 cm**; Trên **2 cm**; Dưới **2 cm**.
* **Đánh số trang**: Phần đầu sử dụng **chữ số La Mã thường** (`i`, `ii`, `iii`...); phần nội dung chính sử dụng **chữ số Ả Rập** (`1`, `2`, `3`...) bắt đầu từ trang đầu tiên của Chương 1.
* **Đánh số mục (Headings)**: Tối đa **3 cấp** (ví dụ: `1.` \\(\to\\) `1.1.` \\(\to\\) `1.1.1.`), tuyệt đối không dùng từ cấp 4 trở đi. Mỗi tiêu đề chương bắt đầu ở một trang mới, in đậm và cỡ chữ lớn hơn.
* **Quy tắc Bảng, Hình, Công thức & Mã nguồn**:
  * **Bảng**: Đánh số theo chương (Bảng 3.1, Bảng 3.2...). Tên bảng đặt **PHÍA TRÊN** bảng, căn giữa. Ghi nguồn ngay dưới bảng nếu lấy từ tài liệu khác.
  * **Hình vẽ**: Đánh số theo chương (Hình 4.1, Hình 4.2...). Chú thích đặt **PHÍA DƯỚI** hình, căn giữa. Sơ đồ phải tự vẽ bằng công cụ vector (draw.io, PlantUML), không vẽ tay rồi chụp lại; ảnh chụp giao diện phải rõ nét và cắt gọn.
  * **Công thức**: Đánh số thụt lề bên phải dạng `(3.1)`, `(3.2)`... và giải thích rõ ràng từng ký hiệu.
  * **Mã nguồn**: Phông chữ đơn cách (Consolas hoặc Courier New), cỡ **10–11**; chỉ trích đoạn cốt lõi từ 10–20 dòng (đoạn dài đưa xuống phụ lục); mỗi đoạn mã có tên (ví dụ: *Đoạn mã 4.1 — Hàm tiền xử lý*).
  * **Quy tắc vàng**: Mọi bảng, hình, công thức và đoạn mã **bắt buộc phải được nhắc đến trong phần văn bản trước khi xuất hiện** (ví dụ: "...như trong Hình 3.2...").

#### 3. Văn phong khoa học và trích dẫn chuẩn IEEE
* **Văn phong**: Khách quan, **không dùng ngôi thứ nhất** (dùng "hệ thống được xây dựng..." thay cho "em đã làm..."). Dùng câu ngắn (một ý một câu), không dùng các từ cảm tính như "rất tốt", "khá nhanh" mà thay bằng con số cụ thể.
* **Trích dẫn chuẩn IEEE**: Đánh số trích dẫn tăng dần trong ngoặc vuông ``, ``, `` ngay trong văn bản. Danh mục tài liệu tham khảo ở cuối quyển liệt kê chi tiết theo đúng thứ tự đó. Chỉ liệt kê tài liệu thực sự được trích dẫn trong bài.

---

### II. HƯỚNG DẪN CHI TIẾT CÁCH VIẾT BÁO CÁO 5 CHƯƠNG

```
BỐ CỤC NỘI DUNG BÁO CÁO PHÂN BỔ THEO TỶ LỆ DUNG LƯỢNG
├── Chương 1: Giới thiệu (Khoảng 10% dung lượng)
├── Chương 2: Cơ sở lý thuyết & Công trình liên quan (Khoảng 20% dung lượng)
├── Chương 3: Phân tích và thiết kế hệ thống (Khoảng 25% dung lượng)
├── Chương 4: Hiện thực và kiểm thử (Khoảng 25% dung lượng)
└── Chương 5: Kết quả, đánh giá và kết luận (Khoảng 15% dung lượng)
```

---

#### 📌 CHƯƠNG 1: GIỚI THIỆU (Khoảng 10% dung lượng quyển)

##### 1. Nội dung bắt buộc cần có:
* **Bối cảnh và vấn đề**: Hiện trạng thực tế ra sao, bất cập hoặc khó khăn đang nằm ở đâu, những ai bị ảnh hưởng bởi vấn đề này.
* **Mục tiêu đề tài**: Đưa ra **3–5 mục tiêu phát biểu rõ ràng, đo được** (không phát biểu chung chung).
* **Đối tượng và phạm vi**: Đề tài làm cho ai, làm trên bộ dữ liệu/môi trường nào, và nêu rõ **những gì nằm ngoài phạm vi** (danh sách những gì KHÔNG làm).
* **Phương pháp thực hiện**: Tóm tắt vắn tắt cách tiếp cận giải quyết bài toán trong vài câu.
* **Bố cục báo cáo**: Viết một đoạn văn ngắn giới thiệu tóm tắt nội dung chính sẽ trình bày từ Chương 1 đến Chương 5.

##### 2. Lưu ý từ hướng dẫn:
* Tránh tên đề tài và mục tiêu quá rộng hoặc mơ hồ.
* Mọi hình vẽ và bảng biểu xuất hiện trong chương này bắt buộc phải được nhắc tới trong văn bản.

---

#### 📌 CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN (Khoảng 20% dung lượng quyển)

##### 1. Nội dung bắt buộc cần có:
* **Cơ sở lý thuyết & Công nghệ**: Trình bày các khái niệm, mô hình, thuật toán, nền tảng công nghệ mà đồ án sử dụng — **chỉ đưa vào những kiến thức thật sự được sử dụng**.
* **Khảo sát công trình / sản phẩm tương tự**: Khảo sát các đề tài/sản phẩm đã có; với mỗi công trình nêu rõ ý tưởng chính, ưu điểm và hạn chế.
* **Bảng so sánh giải pháp**: Xây dựng bảng so sánh các công trình/giải pháp theo một số tiêu chí cụ thể, từ đó kết lại bằng **khoảng trống mà đồ án này sẽ lấp vào**.
* **Lý do chọn công nghệ**: Phân tích và so sánh có căn cứ khoa học/kỹ thuật rõ ràng, không viết lý do cảm tính (như "vì em quen dùng").

##### 2. Lưu ý từ hướng dẫn:
* Đây **không phải là nơi chép lại giáo trình**. Mọi kiến thức lý thuyết đưa vào Chương 2 bắt buộc phải được dùng lại ở Chương 3 hoặc Chương 4; nếu không dùng đến thì phải bỏ đi.
* Đưa quá nhiều lý thuyết chép từ tài liệu khác mà không liên quan tới phần hiện thực là lỗi khiến sinh viên bị trừ điểm nhiều nhất.

---

#### 📌 CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG (Khoảng 25% dung lượng quyển)

##### 1. Nội dung bắt buộc cần có:
* **Yêu cầu hệ thống**: Liệt kê chi tiết yêu cầu chức năng, yêu cầu phi chức năng, xác định các Actor, Use Case (hoặc User Story) kèm bảng đặc tả chi tiết.
* **Kiến trúc tổng thể**: Sơ đồ khối kiến trúc, luồng dữ liệu (Data Flow), các thành phần mô-đun và cơ chế giao tiếp giữa chúng.
* **Thiết kế dữ liệu**: Sơ đồ ERD / Lược đồ CSDL, mô tả chi tiết các bảng và ràng buộc dữ liệu (đối với đề tài phần mềm/web/app); hoặc mô tả cấu trúc dữ liệu, các bước tiền xử lý và đặc trưng (đối với đề tài AI/Machine Learning).
* **Thiết kế xử lý và giao diện**: Vẽ sơ đồ tuần tự (Sequence Diagram) / sơ đồ hoạt động (Activity Diagram) cho các chức năng chính; vẽ phác thảo giao diện (Wireframe) các màn hình chính.

##### 2. Lưu ý từ hướng dẫn:
* **Mỗi sơ đồ bắt buộc phải có ít nhất một đoạn văn giải thích**. Sơ đồ đứng một mình không có đoạn văn giải thích đi kèm sẽ không được chấm điểm.

---

#### 📌 CHƯƠNG 4: HIỆN THỰC VÀ KIỂM THỬ (Khoảng 25% dung lượng quyển)

##### 1. Nội dung bắt buộc cần có:
* **Môi trường triển khai**: Đặc tả chi tiết thông số phần cứng, hệ điều hành, ngôn ngữ lập trình, các thư viện phụ thuộc và phiên bản cụ thể.
* **Cấu trúc mã nguồn & Mô-đun cốt lõi**: Trình bày sơ đồ cây cấu trúc thư mục mã nguồn; trích dẫn và giải thích các đoạn mã xử lý cốt lõi (chỉ trích đoạn ngắn 10–20 dòng, có đánh số và tiêu đề).
* **Giao diện & Chức năng hoàn thành**: Chụp ảnh màn hình các giao diện/chức năng đã lập trình hoàn chỉnh (ảnh rõ nét, cắt gọn) kèm mô tả kịch bản sử dụng chi tiết.
* **Bảng ca kiểm thử (Test Cases)**: Xây dựng bảng kiểm thử chi tiết bao gồm các cột: **[Mã test | Đầu vào \\(\to\\) Kết quả mong đợi \\(\to\\) Kết quả thực tế \\(\to\\) Đạt / Không đạt]**. Bao phủ các mức kiểm thử đơn vị (Unit test) và kiểm thử tích hợp (Integration test).

##### 2. Lưu ý từ hướng dẫn:
* Không dán hàng trang mã nguồn dài lê thê vào chương này; mã nguồn dài bắt buộc phải đưa xuống phần Phụ lục.

---

#### 📌 CHƯƠNG 5: KẾT QUẢ, ĐÁNH GIÁ VÀ KẾT LUẬN (Khoảng 15% dung lượng quyển)

##### 1. Nội dung bắt buộc cần có:
* **Đối chiếu kết quả với mục tiêu**: Lập bảng đối chiếu chi tiết kết quả thực tế đạt được so với từng mục tiêu đã đề ra ở Chương 1 (đạt, đạt một phần, hay chưa đạt).
* **Số liệu đánh giá định lượng**: Đưa ra các con số đo đạc thực nghiệm cụ thể như độ chính xác, thời gian phản hồi, mức độ hoàn thành chức năng, hoặc phản hồi thử nghiệm từ người dùng.
* **Bàn luận & Phân tích nguyên nhân**: Phân tích rõ tại sao đạt được kết quả đó; nêu các trường hợp hệ thống xử lý sai/lỗi và giải thích nguyên nhân gốc rễ.
* **Hạn chế & Hướng phát triển**: Thừa nhận thẳng thắn những mặt chưa làm được, kèm theo các đề xuất hướng phát triển cụ thể cho giai đoạn tiếp theo.

##### 2. Lưu ý từ hướng dẫn:
* Việc thừa nhận hạn chế một cách có phân tích khoa học luôn được hội đồng đánh giá cao hơn là việc làm đẹp kết quả hoặc che giấu lỗi.

---

### III. TÀI LIỆU THAM KHẢO, PHỤ LỤC VÀ PHÂN BỔ ĐIỂM

#### 1. Phần cuối quyển (Back Matter):
* **Tài liệu tham khảo**: Xếp theo chuẩn IEEE. Ưu tiên các bài báo khoa học, sách, tiêu chuẩn kỹ thuật; hạn chế dùng các trang blog không rõ tác giả. Chỉ đưa vào danh mục những tài liệu thực sự được trích dẫn trong bài.
* **Phụ lục**: Mã nguồn tiêu biểu dài, bảng dữ liệu lớn, hướng dẫn cài đặt & vận hành hệ thống.

#### 2. Phân bổ trọng số điểm của Hội đồng:
* **35%**: Nội dung & mức độ hoàn thành nhiệm vụ đề tài.
* **25%**: Sản phẩm / Demo chạy thực tế.
* **20%**: Chất lượng quyển báo cáo (đúng quy chuẩn, trình bày đẹp, không lỗi văn phong).
* **15%**: Bảo vệ & khả năng trả lời câu hỏi phản biện trước hội đồng.
* **5%**: Thái độ & tiến độ làm việc trong suốt 15 tuần.

