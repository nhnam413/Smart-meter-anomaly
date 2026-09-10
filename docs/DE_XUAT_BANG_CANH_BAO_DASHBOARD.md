# Đề xuất bảng giám sát và xử lý cảnh báo điện năng

## 1. Kết luận đề xuất

Dashboard nên xoay quanh ba thành phần: **bảng cảnh báo cần xử lý**, **bảng bằng chứng của cảnh báo được chọn**, và **bảng lịch sử xử lý có thể xuất báo cáo**. Bảng tổng hợp theo loại phù hợp ở trang thống kê, nhưng không cần chiếm chỗ trên màn hình trực tiếp. Danh sách cảnh báo hiện có là nền tảng tốt; điểm thiếu quan trọng nhất là trạng thái xử lý, khả năng chọn một cảnh báo để kiểm tra, và lưu lịch sử bền vững.

Phạm vi đánh giá là mã nguồn `src/04_dashboard.py`, đối chiếu với `src/features.py`, `src/config.py`, `src/02_train.py` và tài liệu chính thức của các thư viện. Đây là đánh giá tĩnh và đề xuất thiết kế; chưa phải kết quả chạy thử giao diện hoặc benchmark. Luồng hiện tại phát lại CSV theo mẫu giờ, nên các đề xuất xử lý trực tiếp cần phân biệt thời gian dữ liệu mô phỏng với thời gian thao tác của người vận hành.[^1][^2]

## 2. Thành phần hiện có và mức phù hợp

| Thành phần hiện có | Vị trí trong mã | Quyết định đề xuất | Lý do |
|---|---|---|---|
| Danh sách điểm bất thường lịch sử | `render_history_view`, dòng 514–528 | Giữ và nâng cấp thành lịch sử có lọc/xuất | Cần tra cứu từng điểm và đối chiếu kết quả |
| Nhật ký bất thường trong phiên | `render_realtime_view`, dòng 713–727 | Nâng cấp thành hàng đợi cảnh báo | Hiện chỉ liệt kê, chưa giúp xác nhận hoặc theo dõi xử lý |
| Sáu cột dùng chung | `TABLE_COLS`, dòng 37–40 | Giữ dữ liệu đo, sửa nhãn, thêm trạng thái | Thiếu thông tin “cảnh báo nào cần làm tiếp” |
| Biểu đồ công suất | `build_power_line_chart` | Giữ, cho xem quanh điểm được chọn | Thể hiện bối cảnh trước và sau biến động |
| Biểu đồ điện áp | `build_voltage_line_chart` | Giữ nhưng sửa cách đánh dấu | Hiện mọi điểm bất thường đều được gọi là sự cố điện áp |
| Donut phân bố loại | `build_anomaly_donut_chart` | Chuyển sang tab thống kê | Không chỉ ra cảnh báo nào đang chờ xử lý |
| Heatmap thứ × giờ | `build_anomaly_heatmap` | Chỉ giữ ở phân tích lịch sử | Hữu ích tìm xu hướng, ít hỗ trợ xử lý một cảnh báo cụ thể |
| Cột phân bố 24 giờ ở trực tiếp | `build_hourly_distribution_chart` | Bỏ khỏi màn hình trực tiếp mặc định | Đang thống kê cửa sổ hiển thị ngắn, không phải toàn bộ phiên |
| KPI công suất hiện tại | `render_realtime_view` | Đưa gần biểu đồ hoặc vùng thông số | Dành KPI chính cho số cảnh báo đang chờ và mức ưu tiên |
| Play/Pause | Bảng điều khiển phát luồng | Giữ dưới nhãn “Mô phỏng” | Là điều khiển phát dữ liệu, không phải thao tác xử lý cảnh báo |

Hai bảng hiện có dùng cùng `render_anomaly_table()`, nhưng khác phạm vi: bảng lịch sử lấy khoảng ngày đã chọn, bảng trực tiếp lấy sự kiện tích lũy trong phiên. Không nên gọi cả hai là lịch sử sự cố đã xác nhận.[^1]

## 3. Bảng 1 — Cảnh báo cần xử lý

Đây là bảng chính của chế độ trực tiếp. Mỗi dòng đại diện một cảnh báo hoặc một nhóm điểm liên tiếp được định nghĩa rõ. Mặc định hiển thị cảnh báo mới và đang xem xét; người vận hành có thể lọc loại và mức cảnh báo.

| Cột mặc định | Nội dung | Nguồn/khả năng hiện tại |
|---|---|---|
| Mã cảnh báo | Định danh ổn định để chọn, cập nhật và tra cứu | Cần bổ sung; không dùng vị trí dòng |
| Thời điểm dữ liệu | Timestamp của điểm hoặc lần đầu trong nhóm | Có trong index |
| Trạng thái xử lý | Mới / Đã tiếp nhận / Đã đóng | Cần bổ sung |
| Mức cảnh báo | Cảnh báo / Cao; tooltip giải thích điểm quy đổi | Có `severity`, cần đồng bộ ngưỡng |
| Dạng gợi ý | Đột biến công suất / Sụt điện áp / Đột biến đêm | Có `classify_type()`; là quy tắc |
| Công suất (kW) | Giá trị của điểm đại diện | Có |
| Điện áp (V) | Giá trị của điểm đại diện | Có |
| Dấu hiệu nổi bật | Một mô tả ngắn; xem đủ khi chọn dòng | Có chuỗi giải thích, cần làm gọn |

Không thêm cả chín đặc trưng, raw score, median, IQR vào bảng chính. Những giá trị này cần để điều tra nhưng sẽ làm hàng đợi khó đọc. Các cột “số điểm trong nhóm”, “lần cuối”, “người tiếp nhận” có thể bật khi triển khai nhóm cảnh báo hoặc nhiều người sử dụng.

Đề xuất thứ tự mặc định: cảnh báo mới trước, mức cao trước, rồi cảnh báo chờ lâu hơn. Cung cấp lựa chọn xem mới nhất trước cho nhu cầu theo dõi luồng. Đây là lựa chọn thiết kế, chưa phải quy tắc nghiệp vụ đã được thử nghiệm.

Chọn một dòng sẽ mở phần chi tiết và các nút “Tiếp nhận”, “Ghi chú”, “Đóng cảnh báo”. Tiếp nhận nghĩa là đã thấy và bắt đầu xem xét, không có nghĩa sự cố điện đã được khắc phục. Khi đóng, lưu kết luận như “đã kiểm tra”, “biến động dự kiến”, “chưa đủ bằng chứng” cùng ghi chú; tránh tự ghi “cảnh báo giả” chỉ vì mẫu kế tiếp bình thường.

### Tách trạng thái tín hiệu và trạng thái xử lý

Tín hiệu có thể đã hết bất thường trong khi người vận hành chưa xem xét. Vì vậy cần hai trường riêng:

- `signal_state`: đang xuất hiện / không còn ở mẫu mới nhất / chưa đủ dữ liệu;
- `workflow_status`: mới / đã tiếp nhận / đã đóng.

Một mẫu bình thường không tự đóng cảnh báo chưa được xử lý. Nút Pause cũng không được đánh dấu cảnh báo là đã hết. Đối với luồng dữ liệu thực trong tương lai, mất dữ liệu phải có trạng thái riêng; không suy ra bình thường từ việc không nhận thêm mẫu.

## 4. Bảng 2 — Bằng chứng của cảnh báo được chọn

Bảng này nằm trong vùng chi tiết, không hiển thị hàng loạt. Nó trả lời “điểm này khác dữ liệu tham chiếu ở đâu?” và hỗ trợ nhận xét có căn cứ.

| Cột | Ý nghĩa |
|---|---|
| Đại lượng hoặc đặc trưng | Công suất, điện áp, độ lệch 24 mẫu, Z-score… |
| Giá trị tại thời điểm chọn | Giá trị số chưa chuyển thành chuỗi |
| Giá trị tham chiếu | Giá trị trước đó, trung vị Train hoặc thống kê rolling tương ứng |
| Chênh lệch | Chênh lệch có dấu, kèm đơn vị |
| Loại tham chiếu | Mẫu trước / 24 mẫu trước / median Train |
| Diễn giải | Dấu hiệu quan sát được, không suy luận hỏng thiết bị |

Nên mở đầu bằng công suất hiện tại so với công suất trước đó và 24 mẫu trước; điện áp so với mẫu trước. Sau đó có thể mở rộng xem chín đặc trưng, median, IQR và độ lệch chuẩn hóa. Không gọi độ lệch Median/IQR là phần trăm đóng góp của mô hình: `explain_anomaly()` chỉ xếp hạng khoảng cách thống kê.[^2]

Trong bảng chi tiết, giữ `raw_score` và ngưỡng 0 để đối chiếu quyết định. Theo Scikit-learn, `decision_function = score_samples - offset_`; đây là điểm quyết định, không phải xác suất.[^5] `severity` của project là sigmoid của điểm này, nên hiển thị dạng “Điểm quy đổi: …/100” thay vì “xác suất nguy hiểm”.

Trường hợp IQR bằng hoặc gần 0 cần chú thích “tham chiếu ít biến thiên”. Cờ nhị phân như `is_night` có thể có IQR bằng 0, khiến phép chia cho epsilon tạo độ lệch rất lớn. Không nên để thứ hạng này tự động được hiểu là bằng chứng vật lý mạnh nhất.[^2]

Biểu đồ bên cạnh nên hiển thị cửa sổ bối cảnh quanh điểm chọn, đánh dấu thời điểm đó. Trong mô phỏng trực tiếp, không hiển thị dữ liệu tương lai chưa được phát. Các tham chiếu `shift(24)` phải ghi đúng là 24 mẫu nếu chưa xác minh đủ 24 giờ liên tục.

## 5. Bảng 3 — Lịch sử xử lý và xuất báo cáo

Bảng này là trang tra cứu riêng, dùng cùng kho cảnh báo với bảng cần xử lý, thay vì duy trì hai danh sách độc lập.

Các cột đề xuất: mã cảnh báo, thời điểm dữ liệu, dạng gợi ý, mức cao nhất, trạng thái xử lý, thời điểm tiếp nhận, thời điểm đóng, kết luận và ghi chú. Khi có nhóm điểm, thêm lần đầu, lần cuối, số điểm và liên kết tới từng mẫu thành viên.

Bộ lọc: khoảng thời gian dữ liệu, trạng thái xử lý, dạng gợi ý, mức cảnh báo. Khi cần đánh giá công việc người vận hành, bổ sung bộ lọc thời gian thao tác riêng. Xuất CSV theo đúng bộ lọc đang áp dụng, kèm một bản tổng hợp phạm vi.

Định dạng số và thời gian phải được giữ dưới dạng dữ liệu có kiểu, chỉ định dạng khi hiển thị. `format_anomaly_table_data()` hiện đổi công suất, điện áp, severity và timestamp thành chuỗi; `_step_stream_engine()` còn lưu sự kiện trực tiếp dưới dạng chuỗi đã trình bày. Cách này bất tiện cho lọc theo ngưỡng, sắp xếp số và xuất báo cáo có thể tính toán.[^1]

Nên lưu dữ liệu thô của cảnh báo cùng thông tin phiên mô phỏng, model/version, đặc trưng tại thời điểm đánh giá và các thao tác xử lý. Với nguyên mẫu cục bộ, một cơ sở dữ liệu nhỏ như SQLite là phương án triển khai đề xuất; khi có nhiều phiên/người dùng, cần đánh giá yêu cầu đồng thời và phân quyền trước khi chọn hạ tầng.

`st.session_state` gắn với phiên người dùng và kết nối WebSocket; không phù hợp làm kho lịch sử duy nhất. Trong code hiện tại, `_reset_stream()` còn xóa toàn bộ danh sách sự kiện của phiên.[^1][^8] Nút khởi động lại mô phỏng nên tạo phiên mới nếu đã có lưu trữ, không âm thầm xóa lịch sử xử lý.

## 6. Bảng phụ chỉ cần ở trang thống kê

Một bảng nhỏ tổng hợp theo dạng gợi ý là đủ: loại, tổng số cảnh báo, số mới, số đã tiếp nhận, số đã đóng, mức cao nhất. Tổng phải có nhãn phạm vi: toàn phiên hay khoảng thời gian đã chọn. Tỷ trọng theo loại dùng mẫu số là tổng cảnh báo, không phải tổng mẫu.

Nếu cần thống kê theo giờ/thứ, dùng tỷ lệ số điểm bất thường trên số mẫu đã được đánh giá trong mỗi ô. Ô không có mẫu phải là “không có dữ liệu”, không phải 0%. Cách này giúp tránh nhầm vùng có nhiều bản ghi hơn với vùng có mức bất thường cao hơn. Đây là đề xuất từ cấu trúc dữ liệu, không phải kết quả chứng minh rủi ro theo giờ.

Confusion matrix, Precision/Recall và bảng TP/FP/FN nên nằm trong tab “Đánh giá mô hình” dành cho Demo có nhãn. Hiện `render_history_view()` ghi đè `is_anomaly` và `anomaly_type` bằng dự đoán; muốn đối chiếu cần giữ `true_is_anomaly`, `true_type` tách khỏi `pred_is_anomaly`, `pred_type`.[^1][^3] Ma trận nhầm lẫn cần nhãn thật và dự đoán riêng; không lấy danh sách chỉ có cảnh báo để tính Recall vì sẽ thiếu FN.[^9]

## 7. Những sai lệch cần sửa trước khi thêm bảng

| Ưu tiên | Quan sát từ code | Hệ quả | Đề xuất |
|---|---|---|---|
| P0 | Bảng tô đỏ từ 60%, hàm mức độ dùng 70% | Một điểm có hai mức biểu diễn | Dùng chung `get_severity_level()` và cùng nhãn |
| P0 | 24 mẫu đầu được ghi `normal`, severity 0 | Chưa đánh giá bị hiểu là bình thường | Thêm `evaluation_status=warming_up` và giá trị dự đoán rỗng |
| P0 | Mẫu số tỷ lệ trực tiếp là tất cả mẫu đã phát | Tỷ lệ bị pha bởi mẫu chưa đánh giá | Chia cho số mẫu suy luận hợp lệ |
| P0 | Banner xanh khẳng định vận hành an toàn khi không báo lỗi | Kết luận mạnh hơn khả năng mô hình | Ghi “Mẫu mới nhất không bị mô hình đánh dấu bất thường” |
| P0 | Biểu đồ điện áp gắn mọi bất thường là sự cố điện áp | Bất thường công suất bị gọi sai | Đổi nhãn marker chung hoặc lọc đúng loại gợi ý |
| P0 | Dải 220–250 V được gọi “vùng an toàn” | Không có cấu hình/nguồn tiêu chuẩn trong code | Gọi dải tham chiếu cấu hình hoặc ẩn đến khi xác định cơ sở |
| P1 | KPI và bảng tích lũy toàn phiên, biểu đồ chỉ 100 mẫu | Tổng giữa các vùng không khớp | Ghi phạm vi rõ và cung cấp tổng hợp toàn phiên riêng |
| P1 | Lọc ngày trước khi trích xuất đặc trưng | Mỗi lần chọn khoảng lại bỏ 24 dòng đầu; khoảng ngắn có thể không còn mẫu | Tính đặc trưng trên dữ liệu có lịch sử rồi lọc hiển thị; chặn ma trận rỗng |
| P1 | `rt_event_keys` chỉ chống trùng timestamp | Nhiều điểm liên tiếp vẫn thành nhiều cảnh báo | Định nghĩa nhóm cảnh báo, giữ bảng mẫu thành viên |
| P1 | Nhãn “Phân loại lỗi (AI)”, “Nguyên nhân chính (XAI)” | Dễ hiểu là phân loại học máy và chẩn đoán nguyên nhân | Đổi thành “Dạng gợi ý”, “Dấu hiệu nổi bật” |
| P1 | Chưa có xử lý tiếp nhận/đóng/xuất | Có nhật ký nhưng chưa có quy trình làm việc | Bổ sung thao tác và kho lưu trữ |

Các điểm trên được xác định từ luồng mã, chưa phải lỗi đã được tái hiện trong trình duyệt. Trường hợp bộ lọc ngắn cần được kiểm thử runtime khi triển khai.[^1][^2]

## 8. Nhóm cảnh báo và tránh lặp

Alertmanager là một ví dụ chính thức về việc tách nhận diện cảnh báo khỏi khử trùng, nhóm và điều phối thông báo.[^7] Với project này có thể áp dụng nguyên tắc nhóm mà chưa cần cài Alertmanager.

Đề xuất ban đầu: gom những điểm bất thường cùng công tơ, cùng dạng gợi ý, liên tiếp theo timestamp mong đợi. Bắt đầu nhóm mới khi đổi dạng, có điểm bình thường ở giữa hoặc có khoảng trống thời gian. Giữ lần đầu, lần cuối, số mẫu và mẫu có điểm quy đổi cao nhất; không chỉ lưu một giá trị công suất tối đa rồi ghép điện áp thấp nhất từ thời điểm khác thành “mẫu đại diện”.

Quy tắc nhóm là giả thuyết thiết kế cần kiểm thử. Nó không chứng minh các điểm trong nhóm có cùng nguyên nhân vật lý. Với dữ liệu giờ, tránh gọi một chuỗi điểm là sự cố liên tục kéo dài chính xác N giờ nếu chưa xác định cách diễn giải khoảng đo.

## 9. Bố cục màn hình đề xuất

```text
GIÁM SÁT VÀ XỬ LÝ CẢNH BÁO
Nguồn: mô phỏng CSV | thời điểm dữ liệu mới nhất | trạng thái dữ liệu

[Mới] [Đã tiếp nhận] [Mức cao chưa đóng] [Mẫu đã đánh giá]

[Lọc trạng thái] [Lọc mức] [Lọc dạng] [Tìm mã]
BẢNG CẢNH BÁO CẦN XỬ LÝ
Chọn dòng → thông tin, bằng chứng và thao tác

[Tiếp nhận] [Ghi chú] [Đóng cảnh báo]
Bảng bằng chứng | Biểu đồ công suất/điện áp quanh điểm chọn

Tab riêng: Lịch sử & xuất báo cáo | Thống kê | Đánh giá Demo
Điều khiển mô phỏng: Play / Pause / Bước tiếp / Phiên mới
```

Trong bản đầu tiên, có thể chỉ triển khai hàng đợi, chi tiết và lịch sử; thống kê và đánh giá mô hình là các vùng phụ. Không cần đồng thời donut, biểu đồ cột 24 giờ và bảng phân loại trên màn hình xử lý.

## 10. Hướng triển khai và tiêu chí nghiệm thu

HTML hiện tại chỉ tạo bảng cuộn, chưa có thao tác chọn dòng, lọc hoặc tải dữ liệu do code cung cấp. Đề xuất dùng `st.dataframe` cho bảng tương tác, định dạng cột bằng `column_config`, và chọn dòng mở chi tiết. Các khả năng này có trong tài liệu Streamlit hiện hành; requirements của project chỉ ghi `streamlit>=1.30.0`, nên cần xác minh phiên bản thực tế trước khi dùng API lựa chọn dòng.[^4]

Xuất báo cáo có thể dùng `st.download_button` với dữ liệu CSV đã lọc.[^6] Tệp xuất cần lưu timestamp rõ ràng, giá trị số nguyên bản, trạng thái xử lý và phạm vi; thời gian dữ liệu và thời gian thao tác không được dùng thay nhau. Các trường kết quả mô hình nên chỉ đọc; trạng thái và ghi chú được sửa qua thao tác riêng để tránh làm thay đổi bằng chứng.

Thứ tự thực hiện đề xuất:

1. Chuẩn hóa trường dữ liệu, ngưỡng màu, trạng thái chưa đánh giá và phạm vi thống kê.
2. Nâng cấp bảng nhật ký thành hàng đợi chọn được dòng, thêm bảng bằng chứng.
3. Bổ sung lưu trữ cảnh báo, tiếp nhận, đóng và lịch sử thao tác.
4. Thêm lọc, xuất báo cáo và quy tắc nhóm sau khi xác nhận yêu cầu.

Tiêu chí nghiệm thu quan trọng:

- Điểm có severity 0,65 luôn cùng mức trên bảng, banner và chi tiết.
- Warm-up không được tính là bình thường hoặc đưa vào mẫu số đã đánh giá.
- Tải lại ứng dụng không làm mất cảnh báo đã tiếp nhận khi đã bật lưu trữ.
- Đổi thứ tự bảng không khiến thao tác áp dụng nhầm cảnh báo; thao tác bám mã ổn định.
- Nhận mẫu bình thường không tự đóng cảnh báo chưa được xem xét.
- Chọn một ngày vẫn sử dụng ngữ cảnh trước ngày đó nếu nguồn có sẵn; không có đặc trưng thì báo đúng trạng thái.
- Tổng xuất CSV khớp bộ lọc; biểu đồ cửa sổ gần nhất không bị trình bày là toàn phiên.
- Khi không có cảnh báo, hiển thị danh sách rỗng; khi không có dữ liệu hoặc chưa đánh giá, hiển thị trạng thái khác.
- Nếu có nhóm cảnh báo, có thể xem lại toàn bộ mẫu thành viên và tái lập quy tắc nhóm.

## Nguồn

[^1]: Project Smart-meter-anomaly, [src/04_dashboard.py](../src/04_dashboard.py). Bản mã cục bộ đã đọc: các hàm `render_anomaly_table`, `render_history_view`, `_step_stream_engine`, `_reset_stream`, `render_realtime_view`. Tham chiếu dòng trong tài liệu áp dụng cho phiên bản được đánh giá.
[^2]: Project Smart-meter-anomaly, [src/features.py](../src/features.py) và [src/config.py](../src/config.py). Nguồn định nghĩa đặc trưng, severity, gán loại, giải thích và giới hạn hiển thị.
[^3]: Project Smart-meter-anomaly, [src/02_train.py](../src/02_train.py). Hàm `evaluate_model`, đánh giá nhị phân và Recall theo nhãn tiêm.
[^4]: Streamlit, [st.dataframe](https://docs.streamlit.io/develop/api-reference/data/st.dataframe) và [Dataframes](https://docs.streamlit.io/develop/concepts/design/dataframes). Tài liệu trực tuyến hiện hành, truy cập 09/09/2026; hỗ trợ bảng tương tác, cấu hình cột và chọn dòng. Phiên bản cài trong project chưa xác minh.
[^5]: Scikit-learn, [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html). Tài liệu API hiện hành, truy cập 09/09/2026; quan hệ score, offset và quyết định bất thường.
[^6]: Streamlit, [st.download_button](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button). Truy cập 09/09/2026; cơ chế cung cấp tệp tải xuống.
[^7]: Prometheus Authors, [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/). Truy cập 09/09/2026; nguyên tắc khử trùng và nhóm cảnh báo. Dùng làm tham chiếu thiết kế, không phải yêu cầu triển khai thêm dịch vụ.
[^8]: Streamlit, [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state). Truy cập 09/09/2026; trạng thái theo phiên và giới hạn gắn với WebSocket.
[^9]: Scikit-learn, [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html). Truy cập 09/09/2026; đầu vào nhãn thật và dự đoán.
