# X25 — Agentic Architecture Diagrams

---

## 1. Per-Call Agentic Routing Flow

LangGraph state machine — what happens every time you call `agent.complete()`.

```mermaid
flowchart TD
    A([SDK: agent.complete\nprompt + optimize_for]) --> B

    subgraph GATEWAY["🔐 Gateway — FastAPI"]
        B[Auth\nvalidate Bearer key\nresolve org] --> C
        C[Perceive\nparse prompt\nextract context features]
        C --> D[Classify\ndetect task type\nsummary / code / extraction\nclassification / qa / other]
        D --> E

        subgraph BANDIT["🎲 Thompson Sampling Bandit"]
            E[Select Arm\nsample Beta α,β per tier\npick highest sample\nskip already-tried tiers]
        end

        E --> F[Dispatch\ncall OpenRouter\ntier → live model ID\nvia model registry]
        F --> G[Judge\nquality score\nlatency check\ncost vs frontier]

        G --> H{Pass threshold?\nq ≥ 0.6}
        H -- Yes --> I[Learn\nupdate Beta arm\nBaRP partial feedback\nonly dispatched arm]
        H -- No / timeout --> J{More tiers\nleft?}
        J -- Yes: escalate --> E
        J -- No: best effort --> I

        I --> K[Audit\nwrite tamper-proof\nhash chain record\nrecord_call → stages]
    end

    K --> L([Return to SDK\ntext + model_used\nquality + cost\naudit_hash])

    style BANDIT fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style GATEWAY fill:#1a1a2e,color:#fff,stroke:#555
```

---

## 2. System Architecture — All Components

How the five phases fit together as a single self-learning system.

```mermaid
flowchart TB
    subgraph CLIENT["📦 Client (Your Code)"]
        SDK["X25 SDK\nx25/client.py\nagent.complete(prompt)"]
    end

    subgraph GATEWAY["🖥️ X25 Gateway — FastAPI :8000"]
        direction TB
        AUTH["🔑 Auth\ngateway/auth.py\nSQLite key store\nper-org rate limiting"]
        AGENT["🤖 Routing Agent\ngateway/agent.py\nLangGraph 8-node\nstate machine"]
        REGISTRY["📋 Model Registry\ngateway/model_registry.py\n349 models, 4 tiers\nhourly refresh"]
        THOMPSON["🎲 Thompson Sampling\ngateway/thompson.py\nBeta(α,β) per tier\nper-org state"]
        STAGES["📈 Stage Tracker\ngateway/stages.py\n4 stages, auto-advance\ndrift detection"]
        AUDIT["🔒 Audit Store\ngateway/audit.py\nhash chain\ntamper-proof log"]
        FINETUNE["⚙️ Fine-tune Manager\ngateway/finetune.py\nJSONL extraction\nUnsloth LoRA script"]
    end

    subgraph OPENROUTER["🌐 OpenRouter API"]
        OR_MODELS["300+ Models\nfree / slm / mid\nfrontier / vlm\nOpenAI, Anthropic\nDeepSeek, Qwen, Llama…"]
    end

    subgraph OUTPUTS["📊 Outputs"]
        DASH["Dashboard\nlocalhost:8000/dashboard\nlive metrics + stage"]
        TRAINING["Training Artifacts\nJSONL data\nUnsloth .py script\nLoRA weights"]
        CUSTOM["Custom SLM\nregistered back\ninto routing pool\n~$0.01/1M tokens"]
    end

    SDK -- "POST /route\nBearer sk-x25-…" --> AUTH
    AUTH --> AGENT
    AGENT <--> REGISTRY
    AGENT <--> THOMPSON
    AGENT -- "dispatch call" --> OR_MODELS
    OR_MODELS -- "response + latency" --> AGENT
    AGENT --> AUDIT
    AUDIT --> STAGES
    STAGES -- "Stage 4: trigger" --> FINETUNE
    FINETUNE -- "extract from audit" --> AUDIT
    FINETUNE --> TRAINING
    TRAINING --> CUSTOM
    CUSTOM -- "register tier=slm" --> REGISTRY
    REGISTRY -- "live catalog" --> AGENT
    AUDIT --> DASH
    STAGES --> DASH
    THOMPSON --> DASH

    style CLIENT fill:#0d2137,color:#fff,stroke:#4a90d9
    style GATEWAY fill:#1a1a2e,color:#fff,stroke:#555
    style OPENROUTER fill:#1e3a1e,color:#fff,stroke:#4a9d4a
    style OUTPUTS fill:#2d1e3a,color:#fff,stroke:#9d4ad9
```

---

## 3. Stage Progression Timeline

How an org autonomously moves through improvement stages over time.

```mermaid
timeline
    title X25 Org Improvement Lifecycle

    section Stage 1 — Explore
        Call 1    : X25 warms up Thompson priors
                  : Tries all 3 tiers (slm / mid / frontier)
                  : Learns your task mix
        Call 25   : Routing converges toward best tier
                  : Quality baseline established

    section Stage 2 — Exploit
        Call 50   : Stage auto-advances
                  : Drift monitor starts (weekly)
                  : Router personalised to your patterns
        Call 150  : Consistent quality + cost savings
                  : Model selection stabilises

    section Stage 3 — Feedback
        Call 200  : POST /feedback unlocked
                  : Submit labelled examples
                  : "For THIS prompt, THAT model was best"
        Call 400  : 50+ examples → fine-tune ready

    section Stage 4 — Fine-tune
        Call 500  : POST /improve/{org} unlocked
                  : Audit log → Alpaca JSONL
                  : Unsloth LoRA script generated
        Training  : ~45 min on Colab T4
                  : Custom Llama 3.2 3B trained on your tasks
                  : Model registered → enters routing pool
                  : $0.01/1M tokens, 85–92% task coverage
```

---

## 4. Thompson Sampling — Bandit Learning Loop

How the router learns across calls within a single org.

```mermaid
sequenceDiagram
    participant SDK as SDK
    participant Agent as Routing Agent
    participant TS as Thompson Sampler<br/>(per-org Beta distributions)
    participant OR as OpenRouter<br/>(live model)
    participant Audit as Audit + Stage Tracker

    SDK->>Agent: complete("summarise this…")
    Agent->>TS: select_arm(tried=[])
    Note over TS: Sample θ_slm ~ Beta(2,2) = 0.41<br/>Sample θ_mid ~ Beta(3,2) = 0.71<br/>Sample θ_frontier ~ Beta(5,1) = 0.89<br/>→ select frontier
    TS-->>Agent: arm=2 (frontier)
    Agent->>OR: call(tier="frontier")
    OR-->>Agent: response, quality=0.95, cost=$0.004
    Agent->>TS: update(arm=2, reward=0.81)
    Note over TS: Beta(5,1) → Beta(5.81, 1.19)<br/>frontier arm gets stronger
    Agent->>Audit: write hash-chained record
    Audit->>Audit: record_call(org, q=0.95)<br/>check stage threshold
    Agent-->>SDK: text + model_used + audit_hash

    Note over TS: After 50 calls: if slm handles<br/>your tasks well, its α grows.<br/>Router naturally shifts cheaper.
```

---

## 5. Fine-tuning Data Pipeline

What Phase 5 does with your call history.

```mermaid
flowchart LR
    A["Audit Log\nSQLite\nevery call recorded"] --> B

    B["Extract Examples\nfor org at Stage 4\ntask_type + tier + reward\nquality + optimize_for"]

    B --> C["Format as\nAlpaca JSONL\ninstruction tuning format\n{instruction, input, output}"]

    C --> D["Generate\nUnsloth Script\nLlama 3.2 3B\nr=16 LoRA, 3 epochs\n4-bit quantised"]

    D --> E{GPU available?}

    E -- "Colab T4\n~45 min" --> F["LoRA Weights\noutput/lora_weights/"]
    E -- "Local RTX 3060+\nor Apple M1+" --> F
    E -- "Together AI API\n~$1–3, no GPU" --> F

    F --> G["POST /improve/{org}/register\nmodel_path=lora_weights/"]

    G --> H["Custom SLM\nregistered in registry\ntier=slm\n$0.01/1M tokens"]

    H --> I["Thompson Sampler\nexplores custom model\nif reward ≥ 0.6 consistently\n→ selected for 85–92% of calls"]

    style A fill:#1a1a2e,color:#fff
    style H fill:#1e3a1e,color:#fff
    style I fill:#1e3a5f,color:#fff
```
