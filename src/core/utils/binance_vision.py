import io
import logging
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta

import pandas as pd


class BinanceVisionDownloader:
    """Binance Vision(data.binance.vision)에서 과거 통계 데이터를 수집하는 유틸리티."""

    BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

    def __init__(self) -> None:
        """Binance Vision 다운로더 초기화."""
        self.logger = logging.getLogger("BinanceVision")

    def fetch_daily_metrics(self, symbol: str, date: datetime) -> pd.DataFrame:
        """특정 날짜의 metrics ZIP 파일을 다운로드하여 DataFrame으로 반환합니다."""
        date_str = date.strftime("%Y-%m-%d")
        
        # [Fix] URL 인코딩 처리 (비 ASCII 문자 포함 시 오류 방지)
        safe_symbol = urllib.parse.quote(symbol)
        url = f"{self.BASE_URL}/{safe_symbol}/{safe_symbol}-metrics-{date_str}.zip"

        try:
            self.logger.info(f"Downloading Vision metrics: {symbol} @ {date_str}")
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                zip_data = response.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # ZIP 내부의 첫 번째 CSV 파일 로드
                csv_names = z.namelist()
                if not csv_names:
                    return pd.DataFrame()
                csv_name = csv_names[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)

            # 컬럼명 정규화 (시스템 표준인 timestamp 기반으로 변환)
            if "create_time" in df.columns:
                df["timestamp"] = df["create_time"]
            return df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # 404 is expected for dates before listing or missing data
                self.logger.debug(f"Vision data not found for {symbol} on {date_str} (404)")
            else:
                msg = f"HTTP Error fetching Vision data for {symbol} on {date_str}: {e}"
                self.logger.warning(msg)
            return pd.DataFrame()
        except Exception as e:
            # [Fix] 에러 로그 출력 시 인코딩 안전성 확보
            try:
                sym_log = symbol.encode("ascii", "ignore").decode("ascii") or "Unknown"
            except Exception:
                sym_log = "EncodingError"
            msg = f"Unexpected error fetching Vision data for {sym_log} on {date_str}: {e}"
            self.logger.warning(msg)
            return pd.DataFrame()

    def fetch_range_metrics(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """지정된 기간 전체의 metrics 수집 및 병합합니다."""
        all_dfs = []
        current = start_date
        while current <= end_date:
            df = self.fetch_daily_metrics(symbol, current)
            if not df.empty:
                all_dfs.append(df)
            current += timedelta(days=1)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["timestamp"])
