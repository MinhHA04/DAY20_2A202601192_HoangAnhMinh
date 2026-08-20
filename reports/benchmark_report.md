# Báo cáo benchmark

> Được tạo bởi `malab benchmark`. Quality score là một structural heuristic minh bạch, không phải human/LLM judge score.

## Kết quả

| Run | Latency (s) | Input tokens | Output tokens | Cost (USD) | Quality /10 | Citation coverage | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline-q1 | 11.41 | 1073 | 653 |  | 7.3 | 100% | 0% | heuristic; modes=openai; errors=0 |
| multi-agent-q1 | 33.78 | 3581 | 1821 |  | 8.3 | 100% | 0% | heuristic; modes=openai; errors=0 |
| baseline-q2 | 11.26 | 1069 | 646 |  | 7.3 | 40% | 0% | heuristic; modes=openai; errors=0 |
| multi-agent-q2 | 17.82 | 2999 | 1272 |  | 10.0 | 100% | 0% | heuristic; modes=offline-fallback,openai; errors=1; first_error=writer online fallback: writer citation validation failed: none valid |
| baseline-q3 | 5.69 | 1008 | 411 |  | 8.7 | 80% | 0% | heuristic; modes=openai; errors=0 |
| multi-agent-q3 | 20.82 | 2989 | 1262 |  | 8.5 | 100% | 0% | heuristic; modes=openai; errors=0 |

## Trung bình theo kiến trúc

| Architecture | Mean latency (s) | Mean tokens | Mean quality /10 | Mean citation coverage |
|---|---:|---:|---:|---:|
| Baseline | 9.45 | 1620 | 7.8 | 73% |
| Multi-agent | 24.14 | 4641 | 8.9 | 100% |

## Phương pháp đo

- Latency là end-to-end wall-clock time. Token counts lấy từ provider khi dùng online model; offline fallback báo 0 tokens.
- Citation coverage là tỷ lệ retrieved source IDs được cite ít nhất một lần. Chỉ số này không chứng minh mỗi citation thực sự entail claim tương ứng.
- Quality heuristic (0–10) xét độ dài, query-term coverage, valid source-ID coverage, counterargument và explicit limitations. Khi chấm cuối, peer review nên thay thế heuristic này.
- Failure rate bằng 1 khi run không có final answer hoặc gặp uncaught exception; các trường hợp còn lại bằng 0.

## Diễn giải

Multi-agent workflow được dự kiến sẽ dùng nhiều calls/tokens hơn. Kiến trúc này chỉ tạo thêm giá trị nếu các research/analysis handoffs cải thiện evidence coverage hoặc review quality đủ để bù coordination overhead. Cần so sánh paired runs của cùng query và không suy ra multi-agent luôn vượt trội từ mẫu lab nhỏ này.

Trong lần chạy này, multi-agent tăng mean citation coverage từ 73% lên 100% và quality heuristic
từ 7.8 lên 8.9, nhưng latency tăng từ 9.45 giây lên 24.14 giây và token/run tăng từ 1,620 lên
4,641. Query 2 kích hoạt citation guard: Writer online không tạo được citation ID hợp lệ nên output
bị loại, deterministic fallback được dùng và Critic xác nhận coverage 100%. Điểm 10.0 của fallback
cũng cho thấy structural heuristic có thể bị template tối ưu hóa; cần peer/human review trước khi
kết luận chất lượng nội dung thực sự cao hơn.

## Failure modes và cách giảm thiểu

- **Cascading hallucination:** research note yếu có thể làm nhiễm các downstream stages. Hệ thống giữ source IDs và audit citations với source ledger gốc.
- **Coordination overhead:** extra calls có thể tăng latency/cost mà không tạo thông tin mới. Cần giữ single-agent baseline và ablate agent không cải thiện metrics.
- **Provider/search outage:** requests có bounded retries và timeout; workflow chuyển sang versioned local corpus và ghi lỗi trong state.
- **Infinite routing:** Supervisor có hard iteration cap và graph có recursion limit.

## Trace artifacts

Per-run JSON state, gồm route events và timed spans, được lưu trong `reports/traces/`.

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

Trả lời: multi-agent phù hợp khi task phân rã thành các phần cần chuyên môn hoặc kiểm chứng độc
lập và phần tăng chất lượng đo được lớn hơn overhead. Không nên dùng cho task hẹp, ít tool/nguồn,
độ trễ thấp hoặc khi một deterministic workflow/single agent đã đạt acceptance criteria.