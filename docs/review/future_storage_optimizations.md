# 중장기 스토리지 최적화 검토

> data_loader parquet I/O hard floor(10-15s)를 돌파하기 위한 아키텍처 레벨 대안.

---

## 1. DuckDB 도입 — Parquet 쿼리 엔진 교체

**현재 문제:**
- `pd.read_parquet(path)` → 개별 parquet 파일마다 전체 로드 + pandas filtering
- 50심볼 × 3TFs = 150회 file open + decompress
- `columns=`, `filters=` kwarg 미활용으로 불필요 I/O 발생

**DuckDB 대안:**
```python
import duckdb
# 50개 심볼의 4h 데이터를 단일 쿼리로 로드
df = duckdb.sql("""
    SELECT timestamp, open, high, low, close, volume, datetime
    FROM 'data/futures/*_4h.parquet'
    WHERE datetime BETWEEN '2024-02-01' AND '2024-06-01'
""").df()
```
- Column pruning + predicate pushdown 자동 (파일 레벨에서 필요한 row group만 스캔)
- 150회 file open → 3회 (TF별 1회)
- PyArrow보다 2-5× 빠른 OLAP 쿼리 엔진

**도입 비용:**
- `uv add duckdb` 1줄 (pyarrow 이미 dependency)
- `DataCollector` 내부 `_load_cache`를 DuckDB 기반으로 교체 (기존 API 보존)
- **리스크:** parquet schema 불일치 시 쿼리 오류. schema migration 전략 필요.

**예상 절감:** 10-15s → 3-5s (hard floor를 ~5s로 낮춤)

---

## 2. universe_ledger.db 인덱스 최적화

**현재 상태:**
- 177MB SQLite, `ledger` table (35 columns)
- 쿼리 패턴: `WHERE tf=? AND date<=? AND knowledge_date<=? AND symbol IN (...)`
- `storage.py:1152-1163`에서 테이블 존재만 체크, **인덱스 생성 코드 없음**

**필요 인덱스:**
```sql
CREATE INDEX IF NOT EXISTS idx_ledger_tf_date_kdate
    ON ledger(tf, date, knowledge_date);
CREATE INDEX IF NOT EXISTS idx_ledger_symbol_tf_date
    ON ledger(symbol, tf, date);
```
- 177MB 풀스캔 → index seek (50-100× faster for date range queries)
- WAL 모드 활성화 검토 (현재 `storage.py:1148` raw connect, WAL 미설정)

**예상 절감:** `_ensure_universe_ledger_sync` 체크 시간 수백 ms → 수 ms

---

## 3. In-Memory Parquet Cache (LRU)

**현재 상태:**
- 동일 심볼의 parquet 파일이 여러 TF에서 반복 로드됨
- `_load_cache` 결과가 메모리에 캐시되지 않음
- `@lru_cache` 없음 (codebase 전체 미사용)

**대안:**
```python
from functools import lru_cache

class DataCollector:
    @lru_cache(maxsize=128)
    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ...
```
- 3 TFs(1h,4h,1d) 로드 시 첫 TF만 disk I/O, 나머지는 cache hit
- maxsize=128 → 약 128 × 1.2MB(1h avg) = 150MB 메모리
- 프로세스 수명 동안 유지. `_load_cache`가 pure function이므로 안전

**리스크:**
- `_load_cache`가 `self._cache_path()` 호출 → instance method라 `lru_cache` 가능하나 `self`가 hash key에 포함됨
- `DataCollector` 인스턴스가 매번 새로 생성되면 캐시 무효화 → 인스턴스 재사용 전략 필요

**대안 2:** 모듈 레벨 dict (인스턴스 독립):
```python
_PARQUET_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
```
- `(symbol, timeframe, path_mtime)` 키 → mtime 기반 invalidation

**예상 절감:** 50심볼 3TFs → 150회 read가 50회 read로 감소 (3× I/O 절감)

---

## 4. Parquet Row Group 재구성

**현재 상태:**
- 모든 OHLCV parquet 파일이 **single row group** (BTC 1h: 35K rows, 4.4MB)
- predicate pushdown 사용 불가 (row group이 1개면 전체 읽기와 동일)

**최적화:**
```python
df.to_parquet(path, index=False, row_group_size=5000)
```
- BTC 1h: 7개 row group (각 ~0.6MB)
- Parquet predicate pushdown: date range 쿼리 시 필요한 row group만 읽기
- 단, DuckDB 도입 시 자동 해결되므로 DuckDB 도입 전 과도기 대안

---

## 5. 사전 로딩 (Warmup Pool)

**현재:** 매 run마다 parquet 전체 로드

**대안:** startup 시 모든 심볼의 parquet을 메모리에 pre-load
```python
# L1 startup 시 1회 실행
_PRELOADED_POOL: dict[str, dict[str, pd.DataFrame]] = {}
for sym in FUTURES_ANCHOR_SYMBOLS:
    for tf in ("4h", "1d", "1h"):
        _PRELOADED_POOL.setdefault(sym, {})[tf] = _load_cache(sym, tf)
```
- 메모리: 50심볼 × (1.2MB + 0.3MB + 0.06MB) ≈ 78MB
- `load_single_symbol_data`에서 `collect_and_save` 대신 pool 참조
- OS page cache warmup 효과도 겸함

---

## 6. PyArrow Table 공유 메모리 (mmap)

**대안:**
```python
import pyarrow as pa
# /dev/shm에 memory-mapped Arrow table
buf = pa.memory_map("/dev/shm/btc_4h.arrow", "r")
table = pa.ipc.open_stream(buf).read_all()
```
- 여러 프로세스가 동일 물리 메모리를 참조 (COW 회피)
- ProcessPool fork 시 DataFrame copy 방지
- 단, 현재는 ThreadPool만 사용 중이므로 급하지 않음

---

## 우선순위

| 순위 | 항목 | 난이도 | 예상 절감 |
|------|------|--------|----------|
| 1 | DuckDB 도입 | 중 | 10-15s → 3-5s |
| 2 | ledger.db 인덱스 | 하 (SQL 2줄) | sync 체크 100× |
| 3 | In-memory parquet cache | 하 | 3× I/O 절감 |
| 4 | DuckDB + warmup pool 통합 | 중 | 추가 1-2s |
| 5 | Row group 재구성 | 중 | DuckDB 없을 시 대안 |
| 6 | mmap 공유 메모리 | 상 | multiprocess 경로에만 유효 |
