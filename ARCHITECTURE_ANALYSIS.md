# GASCO Codebase Architecture Analysis

**Analysis Date:** 2026-05-25  
**Current Phase:** Phase 1 - Rule-based MVP  
**Analysis Scope:** Current state + target modular architecture

---

## 1. CURRENT ARCHITECTURE SUMMARY

### 1.1 Project Purpose
GASCO is a **Group Audit Scope Calculator** that determines which entities within a multinational group should be audited and at what depth. The system evaluates ~10 group subsidiaries using financial metrics and risk factors to recommend audit scope and generate audit instructions.

### 1.2 Current Execution Flow

```
main.py (Orchestrator)
    ├─ data_ingestion.py::load_group_data()
    │   └─ Loads: group_structure.json, findings_repo.csv
    │
    ├─ significance_agent.py::recommend_scope(group_df)
    │   └─ Rule-based scope classification (asset % threshold + risk level)
    │
    ├─ coverage_agent.py::calculate_coverage(group_df)
    │   └─ Aggregates coverage metrics (70% threshold check)
    │
    ├─ instruction_agent.py::generate_instructions(group_df)
    │   └─ Template-based instruction drafting
    │
    └─ Output Generation
        ├─ outputs/significance_scope_recommendation.csv
        ├─ outputs/group_audit_instructions.csv
        └─ outputs/coverage_summary.csv
```

### 1.3 Data Inputs

| File | Type | Purpose | Records |
|------|------|---------|---------|
| `group_structure.json` | JSON | Entity metadata (revenue, assets, risk_level, country) | 10 entities |
| `findings_repo.csv` | CSV | Prior audit findings (entity, finding text, severity) | 7 findings |

**Data Schemas:**

**group_structure.json:**
```json
[
  {
    "Entity": "string",
    "Revenue": "int (USD)",
    "Assets": "int (USD)",
    "Country": "string",
    "Risk_Level": "string (Low|Medium|High)"
  }
]
```

**findings_repo.csv:**
```
Entity, Finding, Severity (High|Medium|Low)
```

### 1.4 Data Outputs

| File | Purpose | Content |
|------|---------|---------|
| `significance_scope_recommendation.csv` | Scope decisions per entity | Entity, Country, Assets, Risk_Level, Asset_Percentage, Recommended_Scope |
| `group_audit_instructions.csv` | Audit instructions | Entity, Country, Risk_Level, Recommended_Scope, Audit_Instruction |
| `coverage_summary.csv` | Coverage metrics | Total_Assets, Covered_Assets, Coverage_Percentage, Status |

---

## 2. DETAILED COMPONENT ANALYSIS

### 2.1 significance_agent.py (Scope Recommendation)

**Logic Type:** Hardcoded rule-based  

**Scope Classification Rules:**
```python
if asset_percentage > 15%:
    scope = "Full Scope"
elif risk_level == "High":
    scope = "Specific Procedures"
else:
    scope = "Analytical Procedures"
```

**Issues:**
- Thresholds (15%) hardcoded
- Risk level not combined with asset %; independent branches
- No weighting between factors
- No consideration of findings history
- Three-tier bucketing not configurable

### 2.2 coverage_agent.py (Coverage Aggregation)

**Logic Type:** Hardcoded threshold  

**Coverage Rules:**
```python
covered = entities where scope in ["Full Scope", "Specific Procedures"]
coverage_pct = sum(covered_assets) / sum(total_assets)
status = "Sufficient" if coverage_pct >= 70 else "Warning"
```

**Issues:**
- 70% threshold hardcoded
- Only 2-tier scope inclusion (Full/Specific in, Analytical out)
- No graduated coverage weighting
- Circular dependency: calls `recommend_scope()` internally

### 2.3 instruction_agent.py (Instruction Generation)

**Logic Type:** Template-based deterministic  

**Instruction Templates:**
- "Full Scope" → comprehensive audit template
- "Specific Procedures" → high-risk focused template  
- "Analytical Procedures" → analytical review template

**Issues:**
- Templates hardcoded as strings
- No dynamic content based on prior findings
- Findings repository loaded but never used
- No customization by country/context

### 2.4 data_ingestion.py (Data Loading)

**Issues:**
- Hard-wired file paths
- Assumes files always exist (no validation)
- Findings loaded but not used
- No error handling for missing/malformed data

---

## 3. IDENTIFIED PROBLEMS IN CURRENT STRUCTURE

### 3.1 Architectural Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| **No separation of concerns** | High | Business logic tightly coupled to data; hard to test |
| **Circular dependencies** | High | coverage_agent imports significance_agent; instruction_agent imports significance_agent; creates hard-to-refactor knots |
| **Hardcoded parameters** | High | Rules not configurable; changing thresholds requires code edits |
| **No config layer** | High | Cannot change behavior without modifying .py files |
| **Monolithic agents** | Medium | Cannot incrementally upgrade agents to ML/CrewAI |
| **Data flow unclear** | Medium | Main orchestrator doesn't fully specify inputs/outputs; findings loaded but unused |
| **No data validation** | Medium | Bad inputs silently fail or produce unexpected results |
| **No error handling** | Medium | No graceful failure paths |
| **Instruction templates hardcoded** | Low | Cannot support dynamic or ML-generated instructions |

### 3.2 Extensibility Blockers for ML/CrewAI Integration

1. **Cannot inject ML model** - No abstraction layer for scope recommendation
2. **Cannot parallelize agents** - Sequential orchestration; no task queue
3. **Cannot train supervised models** - No feature engineering layer; no labeling pipeline
4. **Cannot use local models** - No model loading/inference infrastructure
5. **Cannot add Azure ML** - No cloud service integration points
6. **Cannot reuse rules** - Rules embedded in agent logic, not as configurable policies

---

## 4. HARDCODED & RULE-BASED LOGIC INVENTORY

### 4.1 Hardcoded Thresholds

```python
# significance_agent.py:15
if asset_percentage > 0.15:  # ← 15% threshold hardcoded

# coverage_agent.py:48 and main.py:40
if coverage_results["Coverage_Percentage"] >= 70:  # ← 70% threshold hardcoded
```

### 4.2 Hardcoded Scope Buckets

```python
# significance_agent.py:12-20
return_values = ["Full Scope", "Specific Procedures", "Analytical Procedures"]
# ← No way to add/remove scope categories without code change
```

### 4.3 Hardcoded Instruction Templates

```python
# instruction_agent.py:21-40
if scope == "Full Scope": instruction = "For {entity}...comprehensive..."
elif scope == "Specific Procedures": instruction = "For {entity}...high-risk..."
else: instruction = "For {entity}...analytical..."
# ← Template strings hardcoded; cannot load from external source
```

### 4.4 Hardcoded File Paths

```python
# data_ingestion.py:11
"data/group_structure.json"
# data_ingestion.py:14
"data/findings_repo.csv"
# main.py:29
"outputs/"
```

### 4.5 Hardcoded Coverage Inclusion Logic

```python
# coverage_agent.py:18-21
covered_components = scoped_df[
    scoped_df["Recommended_Scope"].isin(["Full Scope", "Specific Procedures"])
]
# ← Hard to change coverage weight formula
```

### 4.6 Unused/Underutilized Data

```
findings_repo.csv loaded in data_ingestion but NEVER used downstream
- Could inform scope decisions
- Could enhance instruction generation
- Could track compliance with prior findings
```

---

## 5. PROPOSED MODULAR TARGET ARCHITECTURE

### 5.1 Design Goals

1. **Separation of Concerns** - Data, config, rules, models, orchestration
2. **Dependency Injection** - Pass configs, models, rules as parameters
3. **Pluggable Scope Engines** - Swap rules ↔ ML models ↔ CrewAI
4. **Feature Engineering Layer** - Prepare data for ML training/inference
5. **Configuration-driven** - All thresholds externalized
6. **Local & Cloud Ready** - Abstraction for model inference backends
7. **Testable** - Each layer can be unit tested independently

### 5.2 Proposed Folder Structure

```
gasco-phase-1/
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_schema.py          # Pydantic schemas for validation
│   │   └── defaults.yaml             # Default thresholds, rules, paths
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Load/validate input files
│   │   ├── models.py                 # Pydantic models for input/output schemas
│   │   └── repository.py             # Abstract data persistence
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entity.py                 # Group entity domain objects
│   │   ├── finding.py                # Prior finding domain objects
│   │   ├── scope.py                  # Scope recommendation domain object
│   │   ├── coverage.py               # Coverage metrics domain object
│   │   └── instruction.py            # Audit instruction domain object
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py               # Transform raw data → ML features
│   │   ├── scalers.py                # Feature normalization
│   │   └── encoders.py               # Categorical encoding
│   │
│   ├── scope_engine/
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract scope recommendation engine
│   │   ├── rules_engine.py            # Rule-based implementation (current logic)
│   │   ├── ml_engine.py               # ML model-based implementation
│   │   └── crewai_engine.py           # CrewAI agent orchestration
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract model inference interface
│   │   ├── local_inference.py         # Local model loader & predictor
│   │   ├── azure_ml_inference.py      # Azure ML endpoint client
│   │   └── training.py               # Model training pipeline (future)
│   │
│   ├── coverage/
│   │   ├── __init__.py
│   │   ├── calculator.py             # Coverage calculation logic
│   │   └── validators.py             # Coverage validation rules
│   │
│   ├── instructions/
│   │   ├── __init__.py
│   │   ├── generator.py              # Instruction generation
│   │   ├── templates.py              # Template management
│   │   └── enhancer.py               # Find-informed instruction enhancement
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── main.py                   # Main orchestration logic
│   │   ├── pipeline.py               # Composition of pipeline steps
│   │   └── context.py                # Pipeline execution context
│   │
│   └── export/
│       ├── __init__.py
│       ├── exporter.py               # Abstract output exporter
│       ├── csv_exporter.py           # CSV export
│       └── json_exporter.py          # JSON export (future)
│
├── data/
│   ├── group_structure.json
│   ├── findings_repo.csv
│   └── config.yaml                   # (Create) Externalizes thresholds & rules
│
├── models/
│   ├── scope_model.pkl               # (Future) Trained ML model
│   └── feature_config.json           # (Future) Feature engineering config
│
├── tests/
│   ├── unit/
│   │   ├── test_scope_engine.py
│   │   ├── test_coverage_calculator.py
│   │   ├── test_instruction_generator.py
│   │   └── test_feature_engineering.py
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py
│   │   └── test_orchestrator.py
│   └── fixtures/
│       ├── sample_group.json
│       └── sample_findings.csv
│
├── config/
│   ├── defaults.yaml                 # Default configuration
│   ├── dev.yaml                      # Development overrides
│   └── prod.yaml                     # Production overrides
│
├── main.py                           # Entry point (thin wrapper)
├── requirements.txt
├── README.md
└── ARCHITECTURE_ANALYSIS.md
```

### 5.3 Key Design Patterns

#### A. Abstract Base Classes (Engine Pattern)

```python
# src/scope_engine/base.py
class ScopeRecommendationEngine(ABC):
    @abstractmethod
    def recommend_scope(self, entity: Entity, context: PipelineContext) -> Scope:
        """Recommend audit scope for entity. Swappable implementations."""
        pass
```

Three implementations:
- `RulesEngine` - Current hardcoded logic (phase 1)
- `MLEngine` - Local/cloud model inference (phase 2)
- `CrewAIEngine` - Multi-agent orchestration (phase 2)

#### B. Configuration Objects (not dicts)

```python
# src/config/config_schema.py
class ScopeRecommendationConfig(BaseModel):
    asset_percentage_threshold: float = 0.15
    risk_level_weights: Dict[str, float] = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
    scope_buckets: List[str] = ["Full Scope", "Specific Procedures", "Analytical"]
    
class CoverageConfig(BaseModel):
    coverage_threshold: float = 0.70
    included_scopes: List[str] = ["Full Scope", "Specific Procedures"]
```

#### C. Dependency Injection (Constructor)

```python
# Current (bad):
def recommend_scope(group_df):  # Hardcoded thresholds inside

# Proposed (good):
class ScopeService:
    def __init__(self, engine: ScopeRecommendationEngine, config: ScopeRecommendationConfig):
        self.engine = engine
        self.config = config
    
    def recommend_scope(self, entities: List[Entity]) -> List[Scope]:
        return [self.engine.recommend_scope(e, self.config) for e in entities]
```

#### D. Pipeline Composition

```python
# src/orchestrator/pipeline.py
class AuditScopePipeline:
    def __init__(self, 
                 scope_engine: ScopeRecommendationEngine,
                 coverage_calc: CoverageCalculator,
                 instruction_gen: InstructionGenerator,
                 exporter: Exporter):
        self.scope_engine = scope_engine
        self.coverage_calc = coverage_calc
        self.instruction_gen = instruction_gen
        self.exporter = exporter
    
    def execute(self, group_df: pd.DataFrame) -> ExecutionResult:
        # Composable, testable, swappable
        context = PipelineContext(group_df)
        scopes = self.scope_engine.recommend_all(context)
        coverage = self.coverage_calc.calculate(context, scopes)
        instructions = self.instruction_gen.generate(context, scopes)
        self.exporter.export(context, scopes, coverage, instructions)
```

#### E. Feature Engineering for ML

```python
# src/features/engineer.py
class FeatureEngineer:
    def extract_features(self, entity: Entity, findings: List[Finding]) -> Dict[str, float]:
        return {
            "asset_percentage": entity.assets / total_assets,
            "revenue_percentage": entity.revenue / total_revenue,
            "risk_level_numeric": self.encode_risk(entity.risk_level),
            "finding_count": len(findings),
            "prior_severity_max": max([f.severity for f in findings] or [0]),
            "prior_severity_avg": mean([f.severity for f in findings] or [0]),
            "country_risk_score": self.get_country_risk(entity.country),
        }
```

#### F. Model Abstraction for Swappable Inference

```python
# src/model/base.py
class ModelInference(ABC):
    @abstractmethod
    def predict_scope(self, features: Dict[str, float]) -> str:
        pass

# src/model/local_inference.py
class LocalModelInference(ModelInference):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict_scope(self, features: Dict[str, float]) -> str:
        return self.model.predict([features])[0]

# src/model/azure_ml_inference.py
class AzureMLInference(ModelInference):
    def __init__(self, endpoint_url: str, api_key: str):
        self.client = AuthenticatedHttpClient(endpoint_url, api_key)
    
    def predict_scope(self, features: Dict[str, float]) -> str:
        response = self.client.invoke(features)
        return response["prediction"]
```

### 5.4 Configuration Externalization

**Create `data/config.yaml`:**
```yaml
scope_recommendation:
  engine_type: "rules"  # or "ml" or "crewai"
  asset_percentage_threshold: 0.15
  risk_level_weights:
    Low: 0.3
    Medium: 0.6
    High: 1.0
  scope_categories:
    - "Full Scope"
    - "Specific Procedures"
    - "Analytical Procedures"

coverage:
  threshold: 0.70
  included_scopes:
    - "Full Scope"
    - "Specific Procedures"

instruction:
  use_prior_findings: true
  template_source: "file"  # or "ml_model"

model_inference:
  backend: "local"  # or "azure_ml"
  local:
    model_path: "models/scope_model.pkl"
  azure_ml:
    endpoint_url: "https://gasco-scope.eastus.inference.ml.azure.com"
    api_key_env: "AZURE_ML_API_KEY"

data:
  input_group_structure: "data/group_structure.json"
  input_findings: "data/findings_repo.csv"
  output_dir: "outputs/"
```

---

## 6. REFACTOR PLAN (Safe, Incremental)

### Phase 0: Prep (No Breaking Changes)
**Goal:** Set up infrastructure without breaking current code

#### Step 0.1: Create New Module Structure
```bash
mkdir -p src/{config,data,domain,features,scope_engine,model,coverage,instructions,orchestrator,export}
touch src/__init__.py
# + all submodule __init__.py files
```

#### Step 0.2: Create Config Infrastructure
- Write `src/config/config_schema.py` (Pydantic models)
- Write `src/config/loader.py` (load YAML)
- Create `data/config.yaml` with current hardcoded values
- No changes to existing agents yet

#### Step 0.3: Create Domain Objects
- Write `src/domain/models.py` (Entity, Finding, Scope, Coverage, Instruction classes)
- These wrap raw DataFrames; non-breaking

#### Step 0.4: Create Abstract Interfaces
- Write `src/scope_engine/base.py` (abstract ScopeRecommendationEngine)
- Write `src/model/base.py` (abstract ModelInference)
- Write `src/coverage/base.py` (abstract CoverageCalculator)

**At end of Phase 0:**
- Old code still works unchanged
- New infrastructure exists alongside
- No migration yet

---

### Phase 1: Migrate to Dependency Injection
**Goal:** Extract hardcoded logic into injectable components

#### Step 1.1: Create Rules Engine
Write `src/scope_engine/rules_engine.py`:
```python
class RulesEngine(ScopeRecommendationEngine):
    def __init__(self, config: ScopeRecommendationConfig):
        self.config = config
    
    def recommend_scope(self, entity: Entity) -> Scope:
        # Move logic from significance_agent.py here
        # Use config thresholds instead of hardcoded values
```

#### Step 1.2: Create Coverage Calculator
Write `src/coverage/calculator.py`:
```python
class CoverageCalculator:
    def __init__(self, config: CoverageConfig):
        self.config = config
    
    def calculate(self, scopes: List[Scope]) -> Coverage:
        # Move logic from coverage_agent.py
        # Use config instead of hardcoded 70%
```

#### Step 1.3: Create Instruction Generator
Write `src/instructions/generator.py`:
```python
class InstructionGenerator:
    def __init__(self, config: InstructionConfig):
        self.config = config
        self.templates = self._load_templates()
    
    def generate(self, entity: Entity, scope: Scope) -> Instruction:
        # Move logic from instruction_agent.py
        # Use template loader instead of hardcoded strings
```

#### Step 1.4: Create Data Loader
Write `src/data/loader.py`:
```python
class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
    
    def load_group_structure(self) -> List[Entity]:
        # Replace hardcoded path with config
        # Add validation
        # Return domain objects, not raw DataFrames
    
    def load_findings(self) -> List[Finding]:
        # Replace hardcoded path with config
        # Add validation
```

#### Step 1.5: Create New Orchestrator
Write `src/orchestrator/pipeline.py`:
```python
class AuditScopePipeline:
    def __init__(self, engines: Dict, config: Dict):
        self.scope_engine = engines["scope"]
        self.coverage_calc = engines["coverage"]
        self.instruction_gen = engines["instruction"]
        self.exporter = engines["exporter"]
    
    def execute(self, group_df):
        # Call new components
        # Export results
```

#### Step 1.6: Create Entry Point
Write new `src/main_v2.py` that uses pipeline:
- Keep old `src/main.py` untouched
- New entry point calls pipeline with injected components
- Produces same output format (verify parity)

**Migration Check:**
- Run old `main.py` → captures output A
- Run new `main_v2.py` → should produce output A identical
- If not identical: debug until parity achieved
- Then delete old code

---

### Phase 2: Add Feature Engineering & ML Readiness
**Goal:** Prepare for supervised model training

#### Step 2.1: Create Feature Engineer
Write `src/features/engineer.py`:
- Extract features from Entity + Finding combinations
- Document feature meanings
- Add feature scaling/normalization

#### Step 2.2: Create Model Abstraction
- Implement `src/model/local_inference.py`
- Implement `src/model/azure_ml_inference.py`
- ML scope engine wraps model inference

#### Step 2.3: Create Training Pipeline
Write `src/model/training.py`:
- Load historical entity × scope labels
- Extract features
- Train classifier
- Save model + config

---

### Phase 3: CrewAI Integration
**Goal:** Allow multi-agent orchestration

#### Step 3.1: Create CrewAI Engine
- Implement `src/scope_engine/crewai_engine.py`
- Define agents: significance_agent, risk_agent, coverage_agent
- Compose crew with tasks
- CrewAI engine calls crew.kickoff()

---

### Phase 4: Azure ML Integration  
**Goal:** Support cloud model inference

#### Step 4.1: Wire Azure ML Endpoint
- Configure `config.yaml` with endpoint
- AzureMLInference handles auth + API calls
- ML engine uses Azure backend

---

## 7. EXECUTION CHECKLIST (Phase 0-1 Rollout)

### Phase 0: Build Infrastructure (3 days)
- [ ] Create folder structure
- [ ] Write `config_schema.py` (Pydantic models)
- [ ] Write `config.yaml` with current values
- [ ] Write `domain/models.py` (Entity, Finding, Scope, etc.)
- [ ] Write abstract base classes (ScopeRecommendationEngine, etc.)
- [ ] Write `DataLoader` with validation
- [ ] Add unit tests for all new modules

### Phase 1: Migrate Components (5 days)
- [ ] Extract `RulesEngine` from significance_agent.py
- [ ] Extract `CoverageCalculator` from coverage_agent.py
- [ ] Extract `InstructionGenerator` from instruction_agent.py
- [ ] Write `AuditScopePipeline` orchestrator
- [ ] Create `main_v2.py` entry point
- [ ] Verify output parity with original `main.py`
- [ ] Write integration tests for pipeline
- [ ] Delete old agents (significance_agent.py, coverage_agent.py, instruction_agent.py)
- [ ] Update `main.py` to use pipeline

### Post-Phase 1: Code Cleanup (1 day)
- [ ] Remove unused imports
- [ ] Update README with new architecture
- [ ] Document config schema
- [ ] Update requirements.txt with any new deps (pydantic, pyyaml)

---

## 8. BENEFITS SUMMARY

| Benefit | Before | After |
|---------|--------|-------|
| **Change thresholds** | Edit .py + restart | Update config.yaml + restart |
| **Add new scope categories** | Code change + test | Config update + test |
| **Swap scope algorithm** | Rewrite function | Inject different engine |
| **Train ML model** | No infrastructure | Feature engineer + training pipeline |
| **Deploy to Azure ML** | Not possible | Configure endpoint + use Azure backend |
| **Use CrewAI** | No support | Inject CrewAI engine |
| **Unit test agents** | Hard (coupled) | Easy (injected) |
| **Parallel execution** | Sequential only | Pipeline supports async |
| **Debug data flow** | Embedded in functions | Explicit via context |

---

## 9. RISKS & MITIGATIONS

| Risk | Mitigation |
|------|-----------|
| **Refactoring breaks output** | Verify parity before deleting old code; keep old code as reference |
| **Over-engineering** | Start with Phase 0-1 only; only add layers when needed for next phase |
| **Dependencies grow** | Minimize external deps (Pydantic, PyYAML only; avoid heavy frameworks yet) |
| **Config too complex** | Start with simple config; add complexity only as needed |
| **Circular imports** | Use dependency injection; no module-level imports of sibling modules |

---

## 10. NEXT STEPS

1. **Review this analysis** - Confirm target architecture aligns with vision
2. **Approve Phase 0 plan** - Set up infrastructure
3. **Begin Phase 0** - Build without modifying existing code
4. **Run Phase 1 migrations** - Incrementally migrate with parity checks
5. **Phase 2+ deferred** - Only after Phase 1 stabilizes and Phase 2 requirements finalized
