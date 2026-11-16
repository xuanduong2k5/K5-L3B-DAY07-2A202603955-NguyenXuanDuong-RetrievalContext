# Báo cáo Day 7 — Agentic RAG

## Dữ liệu và benchmark

- Nguồn: Ariane Burel, *Equidistribution of curves in hypersurfaces of low degree*, arXiv:2611.17120v1. Day 6 manifest ghi 94 trang nội dung. PDF chỉ được giữ cục bộ; không đưa PDF hoặc trích đoạn nguồn vào Git.
- Day 6 có 198 chunk cho chiến lược C; PostgreSQL 16.16 + pgvector 0.8.7 truy cập được qua cấu hình Day 6. Embedding: `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều; cấu hình phải khớp index.
- Benchmark gồm 20 câu có đáp án của Day 6 (15 dev / 5 test) và 4 câu không đủ evidence trong [abstention_questions.json](../../data/abstention_questions.json) (2 dev / 2 test). Các câu vắng mặt được soát bằng toàn văn PDF cục bộ; cần tiếp tục đọc trace để xác nhận agent từ chối đúng.
- Agent model: GPT-4 qua OpenAI Chat Completions. `OPENAI_API_KEY` đã được cấu hình trong `.env` cục bộ; giá trị không được đưa vào báo cáo. Chưa gọi model từ phiên này.

## Retrieval và thử nghiệm

Đã triển khai và chạy Dense qua pgvector, TF-IDF trên cùng 198 chunk và Hybrid RRF (dense + TF-IDF, RRF constant 60, candidate pool 12, top-3). Cả ba cùng dùng chiến lược C và metadata filter `section` khi câu hỏi có filter. Context mặc định 800 ký tự; agent cho phép tối đa 3 lượt search. Kết quả retrieval chỉ tính 15 câu có đáp án của dev; median là thời gian mỗi truy vấn, chưa gồm model/API. Phí dịch vụ retrieval là $0.

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median ms / chi phí |
|---|---:|---:|---:|---:|
| Dense, top-3, budget 800 | 7/15 | 8/15 | 0/15 | 26.13 ms / $0 |
| TF-IDF, top-3, budget 800 | 10/15 | 12/15 | 0/15 | 0.55 ms / $0 |
| Hybrid RRF, candidate 12, top-3, budget 800 | 9/15 | 10/15 | 0/15 | 42.25 ms / $0 |
| Hybrid RRF, candidate 20, top-3, budget 800 | 9/15 | 10/15 | 0/15 | 27.37 ms / $0 |
| Dense, top-3, budget 1200 | 7/15 | 8/15 | 3/15 | 31.04 ms / $0 |
| TF-IDF, top-3, budget 1200 | 10/15 | 12/15 | 7/15 | 0.36 ms / $0 |
| Hybrid RRF, candidate 12, top-3, budget 1200 | 9/15 | 10/15 | 5/15 | 32.88 ms / $0 |

Tăng candidate pool 12→20 không đổi Hit@1, Evidence Hit@3 hoặc Context Evidence Rate cho Hybrid. Tăng budget 800→1200 giúp evidence vào context ở 3/15 Dense, 7/15 TF-IDF và 5/15 Hybrid. Cấu hình TF-IDF + budget 1200 là lựa chọn retrieval tạm thời trên dev (12/15 Evidence Hit@3, 7/15 Context Evidence Rate, median 0.36 ms), chưa phải cấu hình agent cuối vì GPT-4 chưa chạy.

Hai lỗi dev được đọc đối chiếu:

- **D04** (câu hỏi về nội dung Section 3): TF-IDF tìm được evidence ở top-1, nhưng Hybrid không có evidence trong top-3; các chunk Dense khác được RRF đẩy lên trước. Candidate pool 20 không sửa được lỗi này.
- **D11** (hỏi hai đóng góp theo major/minor arcs): TF-IDF có evidence trong top-3, Dense không; Hybrid cũng đẩy evidence ra khỏi top-3. Đây là lỗi xếp hạng khi hợp nhất, không phải lỗi citation.

Hai cấu hình cải thiện đã đo: tăng candidate pool không cải thiện hit rate; tăng budget cải thiện context evidence. Chưa chạy rerank/query rewrite. Chưa chạy test Day 7 vì chưa chốt cấu hình agent trên dev.

## Agent và đánh giá

- Tool search chọn `dense`, `tfidf` hoặc `hybrid`; controller dùng function calling qua endpoint tương thích OpenAI. Nội dung tài liệu được xem là dữ liệu không đáng tin cậy, không phải chỉ dẫn.
- Agent chưa chạy trong phiên này nên chưa có tool-use rate, grounded answer, citation validity, abstention rate hoặc đánh giá thủ công. Không gửi credential hay chunk tài liệu tới API bên ngoài. Test split Day 7 chưa bị dùng để chọn cấu hình.
- Retrieval traces và summaries đã lưu trong `starter/results/` (Git ignored). Sau khi điền API key, chạy agent trên dev với TF-IDF + budget 1200 và so sánh cấu hình; đọc lỗi agent rồi chốt trước khi chạy test. Tự chấm `reviews.json` trước khi tổng hợp metric.
- Các test tự động trong repository kiểm tra cấu trúc/ranking và parsing tool calls giả lập; chúng không phải bằng chứng về chất lượng model hay benchmark thật.

## Cách tái lập

Lệnh PowerShell và thiết lập database/model được ghi trong [starter/README.md](../../starter/README.md). `.env` cục bộ đã có API key; không đưa key vào báo cáo, Git hoặc chat. Chạy benchmark agent cục bộ sau khi xác nhận quyền gửi câu hỏi và chunk context tới OpenAI. Kết quả Day 6 test (1/5 Hit@1, 4/5 Evidence Hit@3) không dùng làm kết quả Day 7.
