import pandas as pd
import numpy as np

def generate_ideas():
    print("Generating structural alpha feature ideas based on Binance Futures data...")
    ideas = {
        "OI (Open Interest)": [
            "OI_Momentum_1h/24h: 미결제약정 변화율 (추세 강도 파악)",
            "OI_Price_Divergence: 가격은 상승하나 OI는 하락 (추세 소진 시그널)",
            "OI_Vol_Ratio: OI 대비 거래량 비율 (단기 투기 거래 vs 장기 포지션)",
            "OI_Z_Score_72h: 과거 72시간 대비 현재 OI의 극단값 (과열/침체 판단)",
            "OI_Funding_Cross: OI 증가 + 극단적 펀딩비 (스퀴즈 임박 시그널)"
        ],
        "LSR (Long/Short Ratio)": [
            "Top_Trader_LSR_Z_Score_24h: 탑 트레이더 포지션 쏠림 현상 (역발상 지표)",
            "Global_vs_Top_LSR_Spread: 일반 개미와 스마트 머니 간의 포지션 괴리",
            "LSR_Price_Correlation_12h: 가격 방향과 LSR 변화의 상관계수",
            "LSR_Momentum: 최근 4시간 동안의 LSR 급변 (패닉 바잉/셀링 감지)",
            "Retail_Sentiment_Index: (Global Longs - Top Longs) / Total"
        ],
        "Taker Flow (Microstructure)": [
            "Taker_Buy_Sell_Imbalance_Z_24h: 시장가 매수/매도 불균형의 극단값",
            "CVD_Price_Divergence: 누적 델타(CVD)와 가격 간의 괴리 (Exhaustion)",
            "Taker_Volume_Acceleration: 테이커 거래량의 가속도 (Momentum of Momentum)",
            "VPIN_Proxy_1h/12h: 독성 주문 흐름 (Toxic Order Flow) 감지",
            "Absorption_Ratio: 캔들 꼬리(Shadow) / Taker Volume (지정가 흡수 물량 추정)"
        ],
        "Funding & Premium": [
            "Funding_Rate_Momentum_24h: 펀딩비의 변화율 (추세 가속/둔화)",
            "Funding_Trap_24h: 가격 역행 + 펀딩비 역행 (강력한 반전 시그널)",
            "Basis_Premium_Z_Score: 현선물 가격차(Basis)의 롤링 Z-Score",
            "Funding_Yield_Curve: 단기/장기 펀딩비 평균의 스프레드 (시장 심리 곡선)",
            "Funding_Intensity_EWMA: 절대 펀딩비 * 거래량의 지수이동평균 (스트레스 강도)"
        ],
        "Liquidation & Volatility (If available)": [
            "Liquidation_Cluster_Proximity: 대규모 청산 발생 후 경과 시간 및 가격 거리",
            "Vol_Surface_Skewness: 고가-시가 vs 저가-시가 변동성의 비대칭성",
            "Tail_Risk_Jump_24h: 최근 24시간 내 발생한 극단적 꼬리 위험(하방 점프)",
            "Range_Position_Fractal: 다중 타임프레임에서의 레인지 내 현재 위치",
            "Hurst_Exponent_OI: OI 시계열에 대한 허스트 지수 (OI의 지속성/평균회귀성)"
        ]
    }
    
    for category, feature_list in ideas.items():
        print(f"\n--- {category} ---")
        for f in feature_list:
            print(f"  * {f}")

if __name__ == "__main__":
    generate_ideas()
