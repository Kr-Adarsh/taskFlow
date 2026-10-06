# TaskFlow architecture

TaskFlow is a local prototype that carries out a natural-language objective across Finance, CRM, Support and document applications. Browser actions write to SQLite; generated Python analyzes registered CSV files locally.

```mermaid
flowchart TD
    subgraph UI_Layer ["Interface & Control"]
        UI["Execution<br/>Dashboard"]
        API["FastAPI<br/>Run API"]
        UI <-->|Stream / Control| API
    end

    subgraph Core ["Agent Runtime"]
        Graph["LangGraph<br/>Runtime"]
        Model["Selected LLM<br/>Provider"]
        API --> Graph
        Graph <--> Model
    end

    subgraph Tools_Layer ["Capability Registry"]
        Tools["Capability<br/>Registry"]
        Browser["Browser<br/>Actions"]
        Docs["Local Document<br/>Retrieval"]
        Python["Isolated<br/>Python Worker"]

        Graph --> Tools
        Tools --> Browser
        Tools --> Docs
        Tools --> Python
    end

    subgraph Workspace ["Target Applications"]
        Apps[("Company Apps<br/>& SQLite")]
        Browser --> Apps
    end

    subgraph Assurance ["Independent Verification & Audit"]
        Verify["Independent<br/>Verification"]
        Audit["Final Mutation<br/>Audit"]

        Graph --> Verify
        Verify -.->|Inspect State| Apps
        Verify -.->|Read Sources| Docs
        Verify -.->|Validate Output| Python
        Verify --> Audit
    end

    Audit -.->|Results & Evidence| UI
```

## Who owns what

- The planner creates typed subtasks. The executor selects one action from current observations; it has no invoice or complaint workflow script.
- The runtime enforces dependencies, budgets and tool schemas, retains evidence, and records each decision and outcome.
- Only the current browser page exposes actionable control IDs. Historical pages retain evidence and observed values.
- The provider adapter handles transport, structured responses, validation and bounded repair. Provider selection is explicit.
- Browser forms own business writes. A confirmed mutation plus an observed persisted delta triggers independent verification before another executor decision.
- Successful full Python execution owns its result. The model's final echo is diagnostic and cannot replace computed metrics.
- Verifiers independently read sources and persisted state or recompute calculations. Completion also requires a deterministic audit of all business mutations.

## Boundaries

Documents and webpages are untrusted task data. Raw CSVs remain local; the model receives a bounded profile, generated code and compact execution results. Python runs under namespace and syscall restrictions and fails closed when isolation is unavailable.

The workspace supports one active run. There is no production authentication, automatic interrupted-run resume or unrestricted verification of arbitrary domains. Read the README for setup and limits.
