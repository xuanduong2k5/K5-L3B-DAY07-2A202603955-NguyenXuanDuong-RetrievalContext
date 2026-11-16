# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## 5. Chạy ba phương pháp với kho Day 6

`day6_retrieval.py` đọc database URL và embedding model từ `starter/.env` của Day 6, dùng nguyên bảng `day6_chunks`, không tạo index giả hoặc sửa schema. Mặc định dùng chiến lược `C`, đã được chọn trên dev ở Day 6. Bộ truy xuất có:

- `dense`: embedding `sentence-transformers/all-MiniLM-L6-v2` (384 chiều) và tìm cosine bằng pgvector Day 6.
- `tfidf`: TF-IDF trên các chunk của cùng chiến lược.
- `hybrid`: hợp nhất dense và TF-IDF bằng Reciprocal Rank Fusion (RRF, hằng số 60); mặc định lấy tối đa 12 ứng viên mỗi phương pháp rồi xuất top-3.

Cài phụ thuộc nếu cần. Thay `$day6` bằng đường dẫn starter Day 6 thật:

```powershell
python -m pip install -r starter\requirements.txt
$day6 = "..\K5-L3B-DAY06-NguyenXuanDuong-2A202603955-DataIngestion\starter"
Set-Location $day6
docker compose -p day6_student_starter up -d --wait
python -m src.run
Set-Location -
```

Chạy cùng benchmark Day 6 trên dev cho từng phương pháp:

```powershell
$questions = Join-Path $day6 "benchmark\questions.json"
python starter\benchmark_retrieval.py --day6-starter $day6 --questions $questions --split dev --method dense --output starter\results\dense-dev
python starter\benchmark_retrieval.py --day6-starter $day6 --questions $questions --split dev --method tfidf --output starter\results\tfidf-dev
python starter\benchmark_retrieval.py --day6-starter $day6 --questions $questions --split dev --method hybrid --output starter\results\hybrid-dev
```

Chạy agent dùng GPT-4 qua OpenAI Chat Completions. Điền key trong file `.env` ở thư mục gốc; file này đã được Git ignore:

```powershell
# .env
OPENAI_API_KEY=điền_key_cục_bộ_của_bạn
OPENAI_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1

# Chạy tại workspace root sau khi lưu .env
python starter\run_agent_benchmark.py --day6-starter $day6 --day6-questions $questions --abstentions data\abstention_questions.json --split dev --method hybrid --output starter\results\agent-hybrid-dev
```

Có thể đặt `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` để dùng endpoint OpenAI-compatible khác. Câu hỏi và các chunk context sẽ được gửi đến provider khi chạy agent; xác nhận quyền sử dụng trước. Không commit `.env` hoặc gửi API key trong chat. Đầu ra trong `starter/results/` gồm `traces.json`, `cases.json` và `run.json`; thư mục này bị Git ignore.

So sánh ít nhất ba phương pháp và thử ít nhất hai cấu hình trên dev trước; ghi giả thuyết, đọc lỗi và chốt cấu hình. Chỉ sau đó chạy baseline và cấu hình chọn trên test. Benchmark agent có 20 câu Day 6 có đáp án cùng bốn câu không có evidence ở `data/abstention_questions.json` (15+2 dev, 5+2 test). Script chỉ chấm retrieval trên câu có đáp án, không đưa `gold` hay đáp án vào agent. Nhãn `reviews.json` cho agent phải do người đọc trace, context và nguồn tự chấm rồi mới chạy `agent_metrics.py`.

Chạy toàn bộ test unit:

```powershell
python -m unittest discover -s tests -v
```

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
