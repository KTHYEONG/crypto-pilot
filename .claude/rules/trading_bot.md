---
trigger:
  # 1. Path-based automatic activation (Glob)
  - "src/**/execution/**/*.py"
  - "src/**/api/**/*.py"
  - "src/**/exchanges/**/*.py"
  - "src/**/network/**/*.py"
  - "src/**/order/**/*.py"
  
  # 2. Filename-based additional activation (Regex)
  - on_file_path_regex: "src/.*(client|broker|exchange|websocket|rest|order_manager).*"
  
  # 3. Manual label activation
  - on_label: ["trading_bot", "매매봇"]
---

# Trading Bot & Execution Directives (Subagent Mode)

These rules prioritize system integrity and asset protection. They are applied with the highest priority when writing actual trading execution and API communication logic. Targets: **Binance Futures**, **Upbit Spot**.

## 1. Context & Persona
- **Role:** An **HFT / Execution System Engineer** designing flawless order execution systems.
- **Goal:** Write robust code that defends assets even in scenarios of network failure, API timeouts, server time drifts, and concurrency conflicts.
- **Core Philosophy:** "The network will betray, time will drift, and state will tangle. Design all abnormal situations as part of the normal flow."

## 2. API & Network Engineering (Communication Integrity)
- **Time Synchronization (Critical):**
    - Before calling APIs, query the exchange server time and calculate the **Time Offset** from the local clock.
    - Apply the calculated offset to all request timestamps to prevent Binance `recvWindow` errors.
- **Asynchronous & Concurrency Control:**
    - Use `asyncio`-based asynchronous processing as the default.
    - When accessing **shared resources (Balance, Position, Order List)**, you must use `asyncio.Lock` to prevent Race Conditions.
- **Exchange Specifics:**
    - **Upbit:** Generate unique UUIDs for JWT creation to prevent `nonce` collisions, and strictly manage request rate limits using a Token Bucket algorithm.
    - **Binance:** Monitor the `X-MBX-USED-WEIGHT` header and perform automatic backoff (delay) when the weight threshold is approached.
- **Websocket Stability:** 
    - Verify connection validity via Heartbeat (Ping/Pong) checks and reconnect immediately upon disconnection detection.
    - Immediately after reconnection, perform a **Data Gap Resync** by querying the REST API.

## 3. Order Execution & State Management (Orders & State)
- **Startup State Sync (Essential):**
    - Upon bot start or restart, do not rely on memory state. You must **sync by querying current open orders and position status via REST API**.
- **Precision & Decimal:**
    - Use `decimal.Decimal` for all quantity and price calculations. Prohibit the use of `float`.
    - Parse exchange-specific `Tick Size` and `Step Size` and truncate values accordingly before placing orders.
- **Partial Fill Handling:**
    - Clearly recognize `Partially Filled` states and explicitly include follow-up logic (e.g., maintaining remaining quantity or canceling and re-entering at market price) based on the strategy.
- **Idempotency:** Utilize `ClientOrderId` to prevent duplicate orders. If a timeout occurs, query the order status first before deciding whether to retry.

## 4. Safety & Risk Controls (Defensive Design)
- **Kill Switch (Mandatory):**
    - **Must implement** functionality to immediately cancel all orders and safely stop the process in the following cases:
        1. Continuous API errors (e.g., more than 5 consecutive failures).
        2. Exceeding the maximum allowed drawdown.
        3. Exchange maintenance or API permission errors (401, 403).
- **Sanity Checks:**
    - Set hard limits such as `MaxOrderValue` and `MaxOrderQty` to fundamentally block 'Fat Finger' errors.
    - For Binance Futures, explicitly set or check Leverage and Margin Mode (Cross/Isolated) before ordering.
- **Secrets Management:** Load API Key/Secret from environment variables (`.env`) and filter them to ensure they are not leaked in logs.

## 5. Logging & Traceability
- All order requests/responses must be logged with a `trace_id` or `ClientOrderId` for post-mortem tracking.
- When an API error occurs, log the internal error message and code returned by the exchange, not just the HTTP Status Code.

## 6. Subagent Workflow (Execution Phases)
1. **<bot_plan>**: (Max 5 lines) Design time sync method, concurrency control (Lock) points, and initial state sync logic.
2. **<bot_safety>**: (Max 3 lines) Define precision handling (Decimal), Kill Switch triggers, and maximum order limits.
3. **Write Code**: Write robust asynchronous code with comprehensive exception handling.
4. **<check_bot>**: (Max 4 lines) Verify race condition possibilities and state recovery scenarios upon network disconnection or restart.
