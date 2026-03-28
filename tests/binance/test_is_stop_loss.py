class Mock:
    @staticmethod
    def _is_stop_loss_order(o: dict) -> bool:
        ccxt_type = str(o.get("type", "")).upper()
        raw_type = str(o.get("info", {}).get("type", "")).upper()
        is_stop = "STOP" in ccxt_type or "STOP" in raw_type
        is_take_profit = "TAKE_PROFIT" in ccxt_type or "TAKE_PROFIT" in raw_type
        return is_stop and not is_take_profit

o1 = {"type": "market", "info": {"type": "STOP_MARKET"}}
o2 = {"type": "market", "info": {"origType": "STOP_MARKET"}}
o3 = {"type": "limit", "info": {"type": "STOP_MARKET"}}
o4 = {"type": "stop", "info": {"type": "STOP"}}

print("o1:", Mock._is_stop_loss_order(o1))
print("o2:", Mock._is_stop_loss_order(o2))
print("o3:", Mock._is_stop_loss_order(o3))
print("o4:", Mock._is_stop_loss_order(o4))
