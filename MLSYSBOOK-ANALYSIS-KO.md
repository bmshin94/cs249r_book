# MLSysBook (cs249r_book) 저장소 전수조사 분석 정리

> 이 문서는 `bmshin94/cs249r_book` 저장소를 전수조사하고,
> "이게 무엇이고 / 어떻게 쓰고 / 나에게 무슨 도움이 되고 / 어떻게 수익화할 수 있는가"를
> 정리한 한국어 분석 문서입니다.
>
> - **원본 저장소**: https://github.com/harvard-edge/cs249r_book
> - **이 포크**: https://github.com/bmshin94/cs249r_book
> - **공식 사이트**: https://mlsysbook.ai
> - 작성일: 2026-10-08
> - 조사 기준 커밋: `a15fc8eb` (branch `dev` 기준 포크)

---

## 목차

1. [저장소 정체 — 전수조사 결과](#1-저장소-정체--전수조사-결과)
2. [쉬운 설명 (비유 중심)](#2-쉬운-설명-비유-중심)
3. [실무 질문 7개 답변](#3-실무-질문-7개-답변)
4. [수익화 아이디어](#4-수익화-아이디어)
5. [라이선스 요약 — 가장 중요한 표](#5-라이선스-요약--가장-중요한-표)
6. [참고 링크 모음](#6-참고-링크-모음)

---

## 1. 저장소 정체 — 전수조사 결과

### 한 줄 결론

**도구나 플러그인이 아니라, 하버드 대학교 CS249r 강의에서 출발한
"ML 시스템 공학(ML Systems Engineering)" 교과서 + 교육과정 전체가 담긴 거대한 모노레포.**

| 항목 | 값 |
|---|---|
| 원저자 | Prof. Vijay Janapa Reddi (Harvard University) |
| 규모 | **2.3 GB, 18,044 파일, 기여자 140명** |
| 출판 | **2026년 MIT Press** 종이책 출간 예정 |
| 성격 | 문서/콘텐츠 90% + 파이썬 패키지 + 웹앱 + 빌드 툴체인 |
| 목표 | README 명시 — "올해 10만 명, 2030년까지 100만 명 학습자" |

> ⚠️ 저장소 루트의 `CLAUDE.md`는 이전 세션의 자동 생성물(PR #1)로,
> "최신 엔지니어링 비법 집대성" 같은 마케팅 문구를 담고 있으나 실제 성격과 다릅니다.
> 이건 비법서가 아니라 **대학 정규 교재 + 부속 실습 환경 전체**입니다.

### 1-1. 교재 본문 (`books/`, 590 MB)

4권 시리즈 전체 원고 (Quarto `.qmd`). **Vol I만 36개 파일 / 약 72만 5천 단어**
(일반 기술서 1권이 10~15만 단어이므로, Vol I 하나가 단행본 5~7권 분량).

| 권 | 상태 | 챕터 | 작업 단위 | 핵심 질문 | 실패의 대가 |
|---|---|---|---|---|---|
| **Vol I — Foundations** | 출간완료 | 16 | The Model | 한 대의 노드에서 지능을 효율적으로 실행하는 법 | 오답 / 성능 저하 |
| **Vol II — Scaling** | 프리뷰 | 17 | The Fleet | 수천 가속기 분산 클러스터로 확장하는 법 | 수십억 원 클러스터 정지 |
| **Vol III — Agentic** | 개발중 | 18 | The Trajectory | 자율 에이전트를 장기 호라이즌에서 통제하는 법 | 궤적 드리프트 / 무허가 부작용 |
| **Vol IV — Physical AI** | 개발중 | 17 | The Physical Plant | 지능이 물질에 안전하게 작용하는 법 | **되돌릴 수 없는 물리적 파괴** |

**Vol I 챕터**: introduction → ml_systems → ml_workflow → data_engineering →
nn_computation → nn_architectures → frameworks → training → data_selection →
model_compression → hw_acceleration → benchmarking → model_serving → ml_ops →
responsible_engr → conclusion

**Vol II 챕터**: compute_infrastructure, network_fabrics, data_storage,
distributed_training, collective_communication, fault_tolerance,
fleet_orchestration, performance_engineering, inference, edge_intelligence,
ops_scale, security_privacy, robust_ai, sustainable_ai, responsible_ai

**Vol III (에이전트) 챕터 — 이 저장소의 가장 독창적인 부분**:
`02_processor`, `03_deliberation`, `04_working_sets`, `05_virtual_memory`,
`06_episodic_memory`, `07_checkpointing`, `08_actuation`, `09_virtualization`,
`10_interrupts`, `11_scheduling`, `12_data_flywheel`, `13_sft`, `14_rlvr`,
`15_multi_agent`, `16_observability`, `17_tokenomics`

→ 챕터 제목이 전부 **운영체제(OS) 용어**입니다. 우연이 아니고,
**"에이전트를 프롬프트 기법이 아니라 운영체제로 보라"**는 것이 Vol III의 핵심 주장입니다.

**Vol IV 챕터**: boundary, body, brain, nervous, data, training, evaluation,
perception, memory, intent, planning, enforcement, placement, intervention,
verification, release, frontier

> ⚠️ 저자가 README에 **"Vol III, IV는 아직 인용하거나 강의에 쓰지 말라"**고 명시.
> 빠르게 변경 중입니다.

### 1-2. 실습 코드

| 폴더 | 정체 | 규모 | 라이선스 |
|---|---|---|---|
| `tinytorch/` | **NumPy만으로 PyTorch를 처음부터 재구현하는 20단계 과정** | 53 MB / 784 파일 | **MIT** 🟢 |
| `mlsysim/` | **ML 시스템 해석적(analytical) 시뮬레이터** (PyPI 배포) | 4.5 MB / 489 파일 | **Apache 2.0** 🟢 |
| `labs/` | **34개 인터랙티브 노트북** (Marimo + WebAssembly) | 5.5 MB | CC-BY-NC-SA 🔴 |
| `kits/` | **실제 하드웨어 실습** (Arduino / Seeed / Grove / Raspberry Pi) | 267 MB / 1,097 파일 | CC-BY-NC-SA 🔴 |
| `mlperf-edu/` | **노트북에서 돌아가는 MLPerf식 벤치마크** (14 워크로드) | 8 MB | — |

#### TinyTorch 20개 모듈 (`tinytorch/src/`)

```
01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader
→ 06_autograd → 07_optimizers → 08_training → 09_convolutions
→ 10_tokenization → 11_embeddings → 12_attention → 13_transformers
→ 14_profiling → 15_quantization → 16_compression → 17_acceleration
→ 18_memoization → 19_benchmarking → 20_capstone
```

텐서 → 자동미분(autograd) 직접 구현 → 어텐션/트랜스포머 → 양자화·압축·가속.
`tito` 전용 CLI로 모듈별 테스트/채점. **nbgrader 연동**이 있어
(`NBGRADER_RELEASE_TIERS.md`) 대학 과제 자동채점용으로 설계됨.
**PyTorch / TensorFlow 의존성 0.**

#### MLSys·im 5계층 구조 (`mlsysim/`)

| Layer | 도메인 | 내용 |
|---|---|---|
| **A** | 워크로드 (`mlsysim.models`) | FLOPs, 파라미터, 연산 강도 — `Models.Language.Llama3_70B`, `Models.Vision.ResNet50` |
| **B** | 하드웨어 (`mlsysim.hardware`) | 실제 실리콘 스펙 — `Hardware.Cloud.H100`, Jetson, ESP32 |
| **C** | 인프라 (`mlsysim.infrastructure`) | PUE, 탄소집약도(Carbon Intensity), 물 사용량(WUE) |
| **D** | 시스템/토폴로지 (`mlsysim.systems`) | `Systems.Racks.DGX_H100_4Node`, `Systems.Clusters.Frontier_8K` |
| **E** | 실행/리졸버 (`mlsysim.engine.solver`) | 3-tier 수학 엔진: Models / Solvers / Optimizers (설계공간 탐색) |

→ **H100 8천 장 클러스터를 빌릴 돈이 없어도 "메모리 병목이 어디서 터지는가"를
수식으로 계산**해보게 만든 것. 이 프로젝트의 숨은 보석.

#### Co-Labs (`labs/`)

Marimo 노트북 34개 (Vol I 17개 + Vol II 17개). **Pyodide/WebAssembly로 브라우저 실행
→ 설치 불필요.** 구조: Briefing → Parts A~E (예측 잠금 → 계측 탐색 → 공개) → Synthesis.
**모든 예측은 구조화된 선택(라디오/숫자), 자유 서술 없음.**
"예측과 현실의 차이가 학습의 순간"이라는 설계 철학.

### 1-3. 가르치는 사람용

| 폴더 | 내용 |
|---|---|
| `instructors/` | **16주 실러버스 2종**(foundations / scale / tinyml), 교수법 가이드, 평가 루브릭, TA 핸드북 — "The AI Engineering Blueprint" |
| `slides/` | **챕터별 Beamer 강의 슬라이드** (4개 테마 변형), 22 MB / 393 파일 |
| `staffml/` | **ML 시스템 면접 문제 9,000개+**, 70 MB / 11,404 파일 |
| `socratiq/` | **AI 학습 위젯** — `<script>` 한 줄로 어떤 정적 사이트에도 삽입 |

- **StaffML**: Next.js 16 + React 19 + Tailwind v4 웹앱. Bloom 분류법 L1~L6+ 난이도,
  Cloud / Edge / Mobile / TinyML 트랙. 기능: Vault(문제 창고) · Practice(간격 반복) ·
  Gauntlet(시간제 모의면접) · Progress(커버리지 추적) · Chains(L1→L6+ 심화 시퀀스).
  Cloudflare Worker로 API 서빙. 테스트는 Vitest + Playwright.
- **Socratiq**: Vite + Shadow DOM 격리. AI 채팅, 퀴즈 자동생성(객관식/단답/플래시카드),
  하이라이트 후 질문, 지식그래프, IndexedDB 진도 추적, 간격 반복 스케줄러,
  KaTeX/Mermaid 렌더링. LLM 프로바이더: Cloudflare AI Gateway / Gemini / Groq.
  `socratiq=true` 쿠키로 게이팅.

### 1-4. 빌드·운영 인프라

| 폴더 | 내용 |
|---|---|
| `binder/` | **자체 제작 책 빌드 CLI** (`./binder/binder`), 22 MB / 532 파일 |
| `.github/workflows/` | **GitHub Actions 워크플로우 66개** |
| `shared/` / `site/` / `scripts/` / `docs/` | 브랜드 스타일·네비게이션, mlsysbook.ai 랜딩, 버전관리 유틸, 레포 문서 |

`binder` 서브커맨드: `build` `render` `preview` `validate` `audit` `bib` `doctor`
`clean` `headings` `layout` `reference_check` `release` `newsletter` `maintenance` `debug`

→ **출판사 수준의 자동화 파이프라인을 직접 만들어 운영.**
교차참조 검증, 참고문헌 검사, 오타(codespell), 문체(Vale), 링크 썩음(link-rot) 야간 점검,
Cloudflare 캐시 퍼지, 시각 회귀 스모크 테스트까지 포함.

> 📌 레포 구조를 가장 잘 설명한 문서는 **`docs/REPO_LAYOUT.md`** 입니다. 먼저 읽으세요.

### 1-5. 어떨 때 쓰는가

| 대상 | 경로 |
|---|---|
| **독학자** | Vol I 읽기 → Lab 00 → TinyTorch 구현 → MLSys·im 계산 → 하드웨어 키트 → StaffML 면접 |
| **현업 엔지니어** | "왜 4,000장 학습이 멈추나", "KV 캐시가 서빙 메모리를 왜 잡아먹나" 같은 **원리 레벨 디버깅 감각** |
| **교수 / 강사** | 16주 커리큘럼 + 슬라이드 + 루브릭을 통째로 가져다 강의 개설 |
| **에이전트 개발자** | Vol III = **"에이전트를 OS처럼 설계하는 법"** 교과서 |

학습 루프: **Read → Explore → Build → Model → Deploy → Practice → Teach**

저자가 FAQ에서 직접 그은 선:
- *"MLOps 책은 오늘의 툴 레시피고, 이 책은 그 레시피가 왜 존재하는지(대역폭·지연·전력·고장률)를 가르친다"*
- *"딥러닝 책이 끝나는 지점에서 이 책이 시작한다"*
- *"요리 레시피를 따르는 것과 요리의 원리를 이해하는 것의 차이"*

---

## 2. 쉬운 설명 (비유 중심)

### 비유: 요리 학교를 통째로 받은 것

| 받은 것 | 요리 학교로 치면 |
|---|---|
| `books/` 교재 4권 | **교과서** (열·소금·산·시간의 원리) |
| `labs/` 노트북 34개 | **시뮬레이션 주방** (소금 2배 넣으면 어떻게 되는지 클릭으로 확인) |
| `tinytorch/` | **칼과 냄비를 직접 만들어보기** |
| `kits/` | **실제 주방에서 실제 불로 요리** (아두이노, 라즈베리파이) |
| `mlsysim/` | **계산기** (10만 인분 만들면 가스비 얼마인지 계산) |
| `staffml/` | **자격증 시험 문제집** (9천 문제) |
| `instructors/` | **교사용 지도서 + 16주 커리큘럼** |
| `binder/` | **교과서 인쇄·검수 공장** |

저자의 말: **"The repository is the curriculum"** (이 저장소 자체가 커리큘럼이다)

### 바로잡을 오해 3개

**❌ 오해 1: "설치해서 쓰는 프로그램이겠지"**
→ 아닙니다. 2.3GB 중 대부분이 글·그림·참고문헌입니다. 실행되는 건 일부:
`mlsysim`(pip 패키지) / `tinytorch`(빈칸 채우기 과제) / `staffml/app`(Next.js) / `socratiq`(웹 위젯)

**❌ 오해 2: "AI를 더 똑똑하게 만드는 기법 모음이겠지"**
→ 모델을 **똑똑하게** 만드는 책이 아니라 **돌아가게** 만드는 책입니다.

```
딥러닝 책:  "트랜스포머가 어떻게 학습하는가"
이 책:      "그 트랜스포머를 GPU 4,000장에 올렸을 때
            왜 멈추는가, 전기료가 얼마인가, 메모리가 왜 터지는가"
```

**❌ 오해 3: "최신 AI 트렌드 정리 자료겠지"**
→ 반대입니다. 저자가 **"지나가는 업계 유행과 지속되는 공학적 기초를 분리하려고 썼다"**고 명시.
트렌드가 아니라 **변하지 않는 물리 제약**(대역폭·지연·전력·고장률)을 다룹니다.

### 4권을 한 문장씩

> **Vol I**: 컴퓨터 **한 대**에서 AI를 돌린다. 실패하면 → 느려지거나 틀린 답.
> **Vol II**: 컴퓨터 **수천 대**로 늘린다. 실패하면 → 수십억 원짜리 클러스터가 멈춤.
> **Vol III**: AI가 **스스로 여러 단계를 실행**한다(= 에이전트). 실패하면 → 엉뚱한 방향으로 계속 가거나 허가 없는 일을 저지름.
> **Vol IV**: AI가 **물리 세계를 조작**한다(로봇). 실패하면 → **되돌릴 수 없는 물리적 파괴**.

**실패의 대가가 커지는 순서로 배열**한 구조입니다.

### Vol III가 특히 흥미로운 이유 — OS 용어 매핑

| Vol III 챕터 | OS의 그 개념 | 에이전트에선 무슨 뜻 |
|---|---|---|
| `02_processor` | CPU | LLM이 연산 장치 역할 |
| `03_deliberation` | 명령 실행 사이클 | 추론·숙고 루프 |
| `04_working_sets` | 워킹셋 | 지금 당장 필요한 컨텍스트 |
| `05_virtual_memory` | 가상 메모리 | 컨텍스트 윈도우 넘는 정보를 어떻게 스왑 |
| `06_episodic_memory` | — | 장기 기억 저장/검색 |
| `07_checkpointing` | 체크포인트 | 중간 상태 저장 → 실패 시 복구 |
| `08_actuation` | I/O | 툴 호출로 외부에 영향 |
| `09_virtualization` | 가상화/샌드박스 | 툴 권한 격리 |
| `10_interrupts` | 인터럽트 | 사람이 중간에 끼어들기 / 긴급 중단 |
| `11_scheduling` | 스케줄러 | 여러 작업 순서 결정 |
| `15_multi_agent` | 멀티프로세스 | 에이전트 여러 개 협업 |
| `16_observability` | 추적/프로파일링 | 뭐가 잘못됐는지 파악 |
| `17_tokenomics` | — | **토큰 = 돈.** 비용 관리 |

### 지금 당장 뭘 하면 되나 (5분 버전)

```bash
# 1. 뭐가 있는지 지도 보기 (제일 잘 쓴 문서)
cat docs/REPO_LAYOUT.md

# 2. 돌아가는 걸 하나 체험
pip install mlsysim

# 3. 에이전트 책 목차 훑기
ls books/vol3/

# 4. 2.3GB 안 받고 웹에서 읽기
#    → https://mlsysbook.ai/vol1/
```

> 💡 **2.3GB 중 본문 읽기용으로 필요한 건 사실 0에 가깝습니다.**
> mlsysbook.ai에 이미 렌더링돼 공개되어 있으니, 저장소는
> "고쳐서 PR 보낼 때"나 "내 강의로 개조할 때"만 필요합니다.

---

## 3. 실무 질문 7개 답변

### Q1. 설치 및 사용법?

#### 경로 ⓐ 그냥 읽기 — **설치 불필요** (99%에게 정답)

```
https://mlsysbook.ai/vol1/      ← Vol I 전문
https://mlsysbook.ai/vol2/      ← Vol II 프리뷰
https://mlsysbook.ai/labs/      ← 34개 랩, 브라우저에서 바로 실행
https://mlsysbook.ai/tinytorch/ ← TinyTorch 과정
https://mlsysbook.ai/kits/      ← 하드웨어 실습
https://mlsysbook.ai/mlsysim/   ← 시뮬레이터 문서
https://mlsysbook.ai/staffml/   ← 면접 문제 9천개
https://mlsysbook.ai/instructors/ ← 강사용 Blueprint
https://mlsysbook.ai/slides/    ← 강의 슬라이드
```

Co-Labs는 Pyodide/WebAssembly로 브라우저 실행 → **파이썬 설치조차 불필요.**

#### 경로 ⓑ MLSys·im만 쓰기 — **가장 가성비 좋음**

```bash
pip install mlsysim
```

2.3GB 클론 불필요. PyPI 정식 배포 패키지.

```python
from mlsysim import Models, Hardware, Systems, Scenarios
# Models.Language.Llama3_70B, Hardware.Cloud.H100,
# Systems.Clusters.Frontier_8K 등으로 계산
```

#### 경로 ⓒ TinyTorch 과정 수강

```bash
git clone https://github.com/harvard-edge/cs249r_book
cd cs249r_book
pip install -e tinytorch/

tito --version          # 설치 확인
tito system health      # 환경 점검
tito module status      # 20개 모듈 진행률
tito module test 01     # 1번 모듈 채점
```

⚠️ **함정 2개** (`CONTRIBUTING.md` 명시):
1. `pip install -e` 후 `tito`가 PATH에 없으면 **venv 재활성화**.
2. `tinytorch/src/` 수정 후 **`tito dev export` 반드시 실행.**
   `tinytorch/tinytorch/*`는 gitignore 대상이고 **진짜 원본은 `src/`**.

#### 경로 ⓓ 책 전체 직접 빌드 (난이도 높음)

```bash
pip install -r requirements.txt

./binder/binder doctor              # 환경 진단 먼저
./binder/binder build pdf --vol1    # Vol I PDF 생성
./binder/binder check refs          # 교차참조 검증
./binder/binder preview             # 로컬 미리보기
```

**Quarto + Pandoc + LaTeX(TeX Live)** 필요. 리눅스/윈도우용 **도커 컨테이너**
(`binder/docker/`)가 제공되니 그걸 쓰는 게 훨씬 편함.

#### 클론 용량 줄이기

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/harvard-edge/cs249r_book
cd cs249r_book
git sparse-checkout set books/vol3 mlsysim
```

---

### Q2. 이거 플러그인이야? 스킬이야? MCP야?

#### **셋 다 아닙니다.**

| 질문 | 답 | 근거 (전수조사) |
|---|---|---|
| Claude Code **플러그인**? | ❌ | `.claude-plugin/`, `plugin.json` 없음 |
| Claude **스킬**? | ❌ | `.claude/skills/`, `SKILL.md` 없음 |
| **MCP 서버**? | ❌ | MCP 서버 구현·설정 파일 없음 |

저장소 전체에서 Claude 관련 파일은 **루트의 `CLAUDE.md` 단 하나**뿐이고,
그조차 **원본 저장소에 없던 것** — 이전 세션의 자동 생성물입니다
(`git log`: `Merge PR #1: docs: add CLAUDE.md project guide`).

#### 그럼 정체는

```
📚 문서/콘텐츠 프로젝트 (90%)  ← Quarto .qmd 원고
📦 파이썬 패키지 (mlsysim, tinytorch)
🌐 웹 애플리케이션 (staffml = Next.js, socratiq = Vite 위젯)
🏭 빌드 툴체인 (binder CLI + GH Actions 66개)
```

#### 혼동하기 쉬운 지점

- **MCP가 책 내용에 나옵니다.** Vol III가 "tool execution protocols (MCP)"를
  **주제로 가르칩니다.** 즉 **MCP를 설명하는 교재**이지 **MCP 서버 자체는 아닙니다.**
- `mlsysim/mlsysim/agents/` 폴더 존재. 하지만 `registry.py`, `types.py`,
  `__init__.py` 3개뿐 — **시뮬레이터 내부에서 에이전트 워크로드를 모델링하기 위한
  타입 정의**이고, LLM 에이전트 프레임워크가 아닙니다.
- `socratiq`는 LLM을 호출하지만 **웹페이지용 학습 위젯**이고 Claude 생태계와 무관.

> 💡 **반대로 보면 기회**: 이 저장소를 **Claude Code 스킬 / MCP 서버로 감싼 사람이
> 아무도 없습니다.** → 4장 수익화 아이디어 ②

---

### Q3. API 토큰을 사용해야 돼?

**대부분 필요 없습니다.** 코드베이스 grep 실측 결과:

#### 🟢 토큰 전혀 불필요

| 대상 | 이유 |
|---|---|
| 책 읽기 (웹/PDF) | 정적 사이트 |
| `mlsysim` | **순수 해석적 계산.** 수식으로 푸는 거라 외부 호출 0 |
| `tinytorch` | NumPy만 사용. 네트워크 미사용 |
| Co-Labs | 브라우저 WASM 내부 실행 |
| `kits` | 보드에 직접 플래시 |
| StaffML 조회 | 정적 corpus JSON |

#### 🟡 토큰 필요 (선택적 기능)

| 환경변수 | 위치 | 용도 |
|---|---|---|
| `GROQ_API_KEY` | `socratiq/` | AI 채팅 / 퀴즈 생성 |
| `GEMINI_API_KEY` | `socratiq/` | 동일 (프로바이더 대안) |
| `OPENAI_API_KEY` | 일부 스크립트 | 보조 생성 작업 |
| `TOGETHER_API_KEY` | 일부 스크립트 | 동일 |
| `BUTTONDOWN_API_KEY` | 뉴스레터 워크플로우 | **저자 전용** |
| `GITHUB_TOKEN` | GH Actions | CI가 자동 제공 |
| `SEMANTIC_SCHOLAR_API_KEY` / `PUBMED_API_KEY` / `ELSEVIER_API_KEY` | 참고문헌 검증 도구 | 인용 메타데이터 조회 |

**정리**: **SocratiQ(AI 위젯)를 직접 띄울 때만** LLM 키가 필요.
Cloudflare AI Gateway / Gemini / Groq 중 택1이고 **Groq 무료 티어로 충분.**
나머지는 원저자의 저장소 운영용이며 학습자에겐 불필요.

> ⚠️ `MAX_TOKEN`, `NUM_TOKEN`, `UNIT_TOKEN`, `CONFLICT_TOKEN` 등은 API 키가 아니라
> **책 본문의 "토큰"(LLM 토큰, 파서 토큰) 설명 변수**입니다. grep 결과에 섞여 나오니 주의.

---

### Q4. AI 에이전트를 구축하는 데 도움이 될까?

#### **예. 단, "코드 복붙"이 아니라 "설계 사고"로 도움이 됩니다.**

#### ✅ 크게 도움되는 부분

**1) Vol III = 사실상 "에이전트 시스템 설계 교과서"**

| 당신이 겪을 문제 | 해당 챕터 |
|---|---|
| "컨텍스트가 터진다" | `04_working_sets`, `05_virtual_memory` |
| "대화 기억을 어디에 저장하지" | `06_episodic_memory` |
| "중간에 죽으면 처음부터 다시?" | `07_checkpointing` |
| "툴 호출이 위험한 일을 하면?" | `08_actuation`, `09_virtualization` |
| "사람이 중간에 멈추게 하려면" | `10_interrupts` |
| "여러 작업 순서를 누가 정하나" | `11_scheduling` |
| "API 비용이 폭발한다" | `17_tokenomics` |
| "뭐가 잘못됐는지 알 수가 없다" | `16_observability` |
| "에이전트 여러 개 붙이면?" | `15_multi_agent` |
| "우리 데이터로 개선하려면" | `12_data_flywheel`, `13_sft`, `14_rlvr` |

**이 목록이 곧 에이전트 프로덕션 체크리스트입니다.**
프레임워크 튜토리얼 100개를 봐도 이 구조는 안 나옵니다.

**2) "에이전트 = OS" 프레임이 실전에서 먹힙니다.**
컨텍스트 관리를 "가상 메모리 페이징"으로 보면 → LRU 축출, 요약 압축,
디스크(=벡터DB) 스왑이 자연스럽게 따라옵니다. 즉흥적 땜질을
**40년치 OS 연구 자산으로 설계**할 수 있게 됩니다.

**3) MLSys·im으로 비용·지연 사전 계산.**
에이전트는 한 작업에 LLM을 수십 번 호출 → 배포 전 추정은 실무 가치가 큼.
**Apache 2.0이라 상업 제품에 넣어도 됩니다.**

#### ❌ 기대하면 안 되는 부분

| 기대 | 현실 |
|---|---|
| "에이전트 프레임워크 코드가 있겠지" | **없습니다.** Vol III는 글(.qmd) |
| "바로 쓸 스타터 템플릿" | 없음 |
| "LangChain/AutoGen 대체" | 성격이 다름 — 이론서 |
| "최신 모델 벤치마크" | 원리 중심 |
| "안정적인 레퍼런스" | ⚠️ 저자가 **"Vol III/IV 인용·강의 금지"** 명시 |

#### 결론

> **코드는 에이전트 프레임워크/SDK에서 가져오고, 설계 판단은 Vol III에서 가져오세요.**
> 경쟁 관계가 아니라 **상하 관계**입니다.
> 특히 **PoC는 됐는데 프로덕션에서 무너지는 단계**라면 가치가 매우 큽니다.

---

### Q5. 수익화할만한 아이디어가 있어?

→ **4장에서 전체 전개.** 핵심 원칙만:

```
🔴 하지 마세요: books/, slides/, labs/, kits/, staffml/ 내용을 복사·재가공해 판매
   → CC-BY-NC 위반. staffml은 ND라 변형 자체가 금지

🟢 하세요:
   1. tinytorch(MIT) / mlsysim(Apache 2.0) 코드 기반 제품
   2. 책에서 배운 "지식"으로 만든 내 오리지널 콘텐츠
      (지식·사실·아이디어 자체는 저작권 대상 아님)
   3. 서비스·구축·교육 노동 판매 (콘텐츠 판매가 아님)
   4. 이 저장소를 "둘러싸는" 도구 (내용 재배포 아님)
```

---

### Q6. 우리가 React나 PHP로 만들 수 있어?

#### 🟢 ① 이미 React로 되어 있습니다 — 만들 필요 없음

`staffml/app/package.json` 실측:

```json
"next": "^16.3.5",  "react": "^19",  "react-dom": "^19",
"framer-motion": "^12.38.0",  "katex": "^0.17.0",
"@react-sigma/core": "^5.0.6",  "sigma": "^3.0.3",
"graphology": "^0.26.0",  "lucide-react": "^1.16.0"
```

**StaffML = Next.js 16 + React 19 앱.** Tailwind v4, Vitest, Playwright 완비,
Cloudflare Worker로 API 서빙. `socratiq`도 Vite 기반 JS 위젯.

→ **React 하신다면 바로 기여하거나 포크해서 개조 가능.**
단 StaffML은 **ND(변형금지)** 라이선스라 공개 배포 ❌, 사내/개인용은 ✅.

#### 🟡 ② React로 재구현 가능한 것

| 만들 것 | 난이도 | 비고 |
|---|---|---|
| MLSys·im 계산기 웹 프론트 | **쉬움** | 백엔드 `mlsysim`(FastAPI) + React. **가장 추천** |
| 랩 UI를 React로 포팅 | 중 | 현재 Marimo(Python) |
| 면접 연습 앱 (내 문제로) | 중 | 문제는 직접 제작 (StaffML 재사용 ❌) |
| 학습 진도 대시보드 | 쉬움 | |

#### 🔴 ③ React/PHP로 "할 수 없는" 것

| 대상 | 이유 |
|---|---|
| `tinytorch` 재구현 | NumPy 수치연산. JS는 성능·생태계 모두 불리 → **의미 없음** |
| `mlsysim` 재구현 | 과학계산 Python 생태계 의존 → **포팅 대신 API로 감싸는 게 정답** |
| `books/` 빌드 파이프라인 | Quarto + Pandoc + LaTeX. 대체 불가 |
| 교재 본문 재배포 사이트 | 기술은 가능, **라이선스가 막음**(NC) |

#### PHP에 대해 — 직설적으로

**저장소 전체에 PHP 파일이 0개입니다.** 억지로 쓸 수 있는 영역은
유료 강의 사이트의 **회원/결제/LMS 백엔드**(Laravel 등) 또는 워드프레스 블로그 발행 정도.
ML 시스템 영역 자체는 Python 생태계이므로 **PHP로 핵심을 만드는 시도는 권하지 않습니다.**

#### 👉 권장 아키텍처 (React 기준)

```
┌─────────────────────────────┐
│  React / Next.js 프론트엔드    │  ← 여기를 당신이 만듦
│  (계산기 UI, 차트, 시나리오 입력)  │
└──────────────┬──────────────┘
               │ REST / JSON
┌──────────────▼──────────────┐
│  FastAPI 얇은 래퍼 (Python)   │  ← 100줄 정도
└──────────────┬──────────────┘
┌──────────────▼──────────────┐
│  mlsysim (Apache 2.0) 🟢     │  ← 그대로 사용, 상업 OK
└─────────────────────────────┘
```

Python을 몰라도 FastAPI 래퍼 100줄이면 됩니다.
**React 개발자가 이 저장소에서 가치를 뽑는 가장 짧은 경로.**

---

### Q7. 유튜브 강의 영상으로 제작 가능할까?

#### **가능합니다. 단, 조건이 붙습니다.**

| 하는 것 | 가능? | 설명 |
|---|---|---|
| 책 읽고 **내 말·내 슬라이드로** 설명 | ✅ **완전 자유** | **지식·사실·아이디어는 저작권 대상이 아님** |
| `slides/`의 Beamer 슬라이드를 **그대로** 띄움 | ⚠️ **위험** | CC-BY-NC-SA. 광고 수익 붙으면 NC 위반 소지 |
| 책 그림/도표를 **그대로** 사용 | ⚠️ 위험 | 동일 |
| 비수익 채널(수익창출 OFF) + 출처 표기 | ✅ 가능 | NC 충족. SA 때문에 동일 라이선스 표기 권장 |
| `tinytorch` 코드를 띄우고 설명 | ✅ **자유** | **MIT** |
| `mlsysim` 사용법 영상 | ✅ **자유** | **Apache 2.0** |
| StaffML 문제를 읽어주는 영상 | ❌ **안 됨** | **CC-BY-NC-ND** — 변형·편집 금지, 가장 엄격 |

#### 💡 가장 안전하고 효과적인 포맷

```
✅ 추천 ① "TinyTorch 완주 시리즈" (MIT — 완전 안전)
   20개 모듈을 직접 코딩하며 20편.
   "PyTorch를 처음부터 만들기" = 조회수 잘 나오는 주제

✅ 추천 ② "MLSys·im으로 계산해보기" (Apache 2.0)
   "H100 8천장 클러스터 전기료 계산해봤습니다" 류
   숫자가 나오는 영상은 신뢰도가 높음

✅ 추천 ③ "에이전트를 OS처럼 설계하기" (내 해석, 내 자료)
   Vol III의 프레임을 읽고 → 내 다이어그램·내 코드 예제로 재구성
   = 당신의 오리지널 저작물

✅ 추천 ④ 하드웨어 실습 영상
   아두이노에 모델 올리는 과정 = 내가 찍은 영상 = 내 저작물
   레시피(사실)는 저작권 대상 아님
```

#### ⚠️ 반드시 지킬 것

1. **슬라이드·그림을 그대로 쓰지 말고 다시 만드세요.** (가장 중요)
2. **출처 명확히**: "하버드 MLSysBook (mlsysbook.ai) 기반,
   Prof. Vijay Janapa Reddi" — CC의 **BY** 조건이고 신뢰도도 상승.
3. **공식 교재라고 오해시키지 마세요.** "제가 공부하며 정리한 내용"으로 포지셔닝.
4. **Vol III/IV는 "개발 중"이라고 고지.** 저자가 인용을 만류함.
5. 애매하면 **저자에게 문의.** 교육 목적 개방에 매우 적극적
   (Open Collective 후원, 기여자 140명, GitHub Discussions 개방).

#### 한국 시장 관점의 기회

- 한국어 ML **시스템**(≠ MLOps) 콘텐츠는 **거의 비어 있음.**
  모델 학습 강의는 포화, 인프라·서빙·비용 최적화는 공백.
- README에 이미 **한국어 번역본**(`README/README_ko.md`) 존재
  → 한국 수요를 저자도 인식.
- "GPU 비용 줄이는 법", "에이전트 토큰 비용 90% 줄이기" 같은 주제는
  **B2B 리드까지 연결**됨.

---

## 4. 수익화 아이디어

> **전제**: 이 저장소를 "상품"으로 파는 건 불가능합니다(NC 라이선스).
> 돈이 되는 건 ① 🟢 라이선스 코드로 만든 **제품**,
> ② 이 지식으로 만든 **내 오리지널 콘텐츠**, ③ **노동/서비스**,
> ④ 저장소를 **둘러싸는 도구**입니다.

### 🥇 티어 1 — 지금 바로, 리스크 낮고 현실적

#### ① ML 시스템 비용 계산기 SaaS

**근거**: `mlsysim` = **Apache 2.0** → 상업적 사용·수정·재배포 전부 자유.

```
제품: "우리 모델을 서빙하면 월 얼마인가" 계산기
입력: 모델(파라미터/토큰) + 하드웨어 + QPS + 지연 목표
출력: GPU 수, 메모리 병목, 월 비용, 전력, 탄소배출
```

- **왜 팔리나**: CTO/인프라 리드가 매주 하는 계산인데 전부 엑셀로 함.
  벤더 계산기는 자사 유리하게 편향됨.
- **가격**: 무료(계산 3회) / $29월(무제한+내보내기) / $299월(팀+API+사설 하드웨어 등록)
- **구현**: `mlsysim` + FastAPI 래퍼 + React 프론트 → **React 하신다면 바로 이것**
- **차별점**: "벤더 중립 + 탄소/물 사용량 포함" → ESG 보고 수요와 직결
  (Layer C에 PUE·탄소집약도가 이미 내장)
- 난이도 ★★☆☆☆ / 수익 ★★★★☆ / 리스크 ★☆☆☆☆

#### ② MCP 서버 / Claude 스킬로 패키징 ⭐ **가장 추천**

**근거**: Q2에서 확인 — **아무도 안 했습니다.**
그리고 이건 내용 **재배포가 아니라 질의 도구 제공**입니다.

```
"ML Systems Advisor" MCP 서버
  tool: query_mlsys_knowledge(question)   → Vol I~IV 인덱스 검색 + 출처 링크
  tool: estimate_cost(model, hw, qps)     → mlsysim 계산
  tool: check_agent_design(spec)          → Vol III 체크리스트 대조
  tool: find_bottleneck(config)           → 병목 분석
```

- **왜 지금인가**: 코딩 에이전트 사용자가 폭증하는데
  "ML 인프라 설계를 물어볼 믿을 만한 도구"가 없음.
  에이전트가 웹검색으로 블로그를 긁어오는 것보다
  **교과서 인덱스를 치는 게 질이 압도적.**
- **라이선스 안전 설계**: 전문을 뿌리지 말고 **요약 + mlsysbook.ai 딥링크** 반환
  → BY(출처표기) 충족, NC 회피.
- **수익 모델**: 오픈소스 공개로 인지도 → 유료 호스팅($19월) / 기업 사설 배포($2k)
- **부수 효과**: 포트폴리오 가치 큼 — "하버드 교재를 에이전트 도구로 만든 사람"
- 난이도 ★★☆☆☆ / 수익 ★★★☆☆ / 리스크 ★☆☆☆☆

#### ③ 한국어 ML 시스템 콘텐츠 (유튜브 + 뉴스레터)

```
시리즈 A: "PyTorch를 처음부터 만들기" (TinyTorch 20편) ← MIT, 안전
시리즈 B: "AI 에이전트를 운영체제처럼 설계하기" (Vol III 재해석)
시리즈 C: "GPU 비용 계산 실전" (mlsysim)
```

- **수익 경로**: 애드센스(작음) → **유료 멤버십/패트리온** →
  **기업 강의 리드(여기가 본체)** → 컨설팅
- **현실적 숫자**: 틈새 기술 채널은 구독 1만이 한계지만 **전환율이 높음.**
  기업 강의 1건 300~800만원. 영상은 영업 자산.
- ⚠️ 슬라이드·그림은 **반드시 직접 재제작**
- 난이도 ★★★☆☆ / 수익 ★★★☆☆ / 리스크 ★★☆☆☆

### 🥈 티어 2 — 중기, 수익성 높음

#### ④ 기업 교육 / 사내 강의

**가장 확실하게 돈이 되는 경로.** 콘텐츠를 파는 게 아니라
**가르치는 노동**을 파는 것이므로 NC와 무관.

```
타깃: AI 도입했는데 비용·지연으로 고생하는 중견 테크 기업
      MLOps 팀, 플랫폼 팀, 임베디드 AI 팀

패키지 A: "ML 시스템 비용 최적화" 2일 워크숍     → 500~800만원
패키지 B: "에이전트 프로덕션화" 3일 과정          → 800~1,500만원
패키지 C: 분기 리테이너 (상시 자문)              → 월 300~500만원
```

- **차별점**: "하버드 CS249r 커리큘럼 기반"은 강력한 신뢰 신호
  (단, 공식 인증이 아님을 명확히 — **허위 제휴 표시는 절대 금지**)
- `instructors/`의 16주 실러버스와 루브릭을 **참고해 내 커리큘럼 설계**
  (참고는 자유, 복사 배포는 ❌)
- 난이도 ★★★☆☆ / 수익 ★★★★★ / 리스크 ★★☆☆☆

#### ⑤ 에이전트 아키텍처 컨설팅

Vol III 18챕터를 **그대로 진단 체크리스트**로 사용.

```
"에이전트 프로덕션 준비도 감사(Audit)"
  □ 컨텍스트 관리 전략 (working sets / virtual memory)
  □ 영속 기억 설계 (episodic memory)
  □ 실패 복구 (checkpointing)
  □ 툴 권한 격리 (actuation / virtualization)
  □ 휴먼 인더루프 중단 (interrupts)
  □ 작업 스케줄링 (scheduling)
  □ 토큰 비용 구조 (tokenomics)
  □ 관측성·추적 (observability)
  □ 개선 루프 (data flywheel / SFT / RLVR)

산출물: 리스크 등급 + 우선순위 개선안 + 비용 추정(mlsysim)
가격: 500~1,500만원 / 건
```

- **왜 팔리나**: PoC는 다 성공하는데 프로덕션에서 무너지는 회사가 지금 매우 많고,
  **체계적 점검 프레임이 없음.** 이 저장소가 그 프레임을 공짜로 제공.
- 난이도 ★★★★☆ / 수익 ★★★★★ / 리스크 ★★☆☆☆

#### ⑥ 하드웨어 교육 키트 판매

`kits/`에 Arduino / Seeed / Grove / Raspberry Pi 실습이 1,097 파일로 정리됨.

```
상품: "엣지 AI 입문 키트" — 보드 + 센서 + 케이스 + 한국어 가이드
가격: 15~25만원
타깃: 대학 실습, 고등학교 AI 과목, 메이커, 기업 신입교육
```

- **하드웨어 판매는 NC와 무관** (물건 판매). **단, 가이드 문서는 직접 작성.**
- 유통: 네이버 스마트스토어, 디바이스마트/엘레파츠 제휴, 대학 구매팀 직판
- 난이도 ★★★★☆ / 수익 ★★★☆☆ / 리스크 ★★★☆☆ (재고 부담)

### 🥉 티어 3 — 장기 / 큰 판

#### ⑦ 한국형 ML 시스템 부트캠프

```
8~12주, 온라인 + 오프라인 혼합
커리큘럼: Vol I 요약 → TinyTorch → MLSys·im → 키트 → 캡스톤
수강료: 250~400만원 × 20명 = 기수당 5,000~8,000만원
```

- **차별점**: 국내 부트캠프는 전부 "모델 학습·웹개발". **시스템/인프라 레이어는 공백.**
- 국비지원(K-디지털 트레이닝) 연계 가능성
- ⚠️ 커리큘럼 **구조 참고는 자유**, 교재 **배포는 금지**.
  "mlsysbook.ai에서 무료로 읽으세요"로 안내하면 완전히 합법이고 오히려 신뢰 상승.
- 난이도 ★★★★★ / 수익 ★★★★★ / 리스크 ★★★★☆

#### ⑧ 오픈소스 → 취업/이직 레버리지 (숨은 최고 ROI)

```
기여자 140명인 활발한 하버드 프로젝트에 의미 있는 기여를 남김
  → TinyTorch 모듈 개선, mlsysim에 한국 하드웨어/전력 데이터 추가,
     한국어 번역 품질 개선, 랩 신규 작성

결과: 이력서에 "하버드 MLSysBook 컨트리뷰터"
      + .all-contributorsrc에 영구 기재
```

- **직접 수익 0원, 기대수익 최고**: 시니어 ML 인프라 포지션 연봉 1~2억.
  연봉 상승분 2천만원이면 ROI는 어떤 SaaS보다 높음.
- `.all-contributorsrc`에 140명 등재, 봇이 자동 크레딧 → **진입 장벽 낮음.**
- 난이도 ★★☆☆☆ / 수익 ★★★★★(간접) / 리스크 ☆☆☆☆☆

### 비교표

| # | 아이디어 | 착수 | 수익 | 리스크 | 추천 |
|---|---|---|---|---|---|
| ② | **MCP 서버 / Claude 스킬** | 1~2주 | 중 | 낮음 | 🥇🥇 |
| ① | **mlsysim 계산기 SaaS** | 3~6주 | 중상 | 낮음 | 🥇 |
| ⑧ | **OSS 기여 → 커리어** | 즉시 | 간접 최상 | 없음 | 🥇 |
| ④ | 기업 교육 | 2~3개월 | 최상 | 중 | 🥈 |
| ⑤ | 에이전트 컨설팅 | 2~3개월 | 최상 | 중 | 🥈 |
| ③ | 유튜브 + 뉴스레터 | 1개월~ | 중 | 중 | 🥈 |
| ⑥ | 하드웨어 키트 | 3~6개월 | 중 | 중상 | 🥉 |
| ⑦ | 부트캠프 | 6개월+ | 최상 | 높음 | 🥉 |

### 🎯 권장 실행 순서

```
[1~2주]   ② MCP 서버를 오픈소스로 공개
          → 비용 0, 기술 검증, 인지도 확보, 포트폴리오
          ⑧ 동시에 저장소에 작은 PR 1~2건 (한국어 번역 개선 등)

[1~2개월] ③ 유튜브 "TinyTorch 완주" 5편 (MIT라 안전)
          → 신뢰 자산 축적. 이게 ④⑤의 영업 채널이 됨

[3~4개월] ① mlsysim 계산기 SaaS 런칭 (React + FastAPI)
          → 유튜브 시청자가 첫 고객

[5개월~]  ④⑤ 기업 교육·컨설팅으로 수익 본체 전환
          → ①②③이 전부 영업 자산으로 작동
```

**핵심 논리**: ①②③은 **돈을 버는 수단이 아니라 ④⑤를 팔기 위한 신뢰 자산**입니다.
공짜로 뿌린 도구와 영상이 "이 사람은 ML 시스템을 안다"를 증명하고,
실제 돈은 기업 교육·컨설팅에서 나옵니다.
콘텐츠 직판 경로가 라이선스로 막혀 있으니,
**오히려 이 구조가 유일하면서 동시에 가장 수익성 높은 길**입니다.

### ⚖️ 법적 체크리스트

```
❌ 절대 하지 말 것
   · books/, slides/, labs/, kits/ 내용을 유료 상품에 포함
   · StaffML 문제 재가공·번역 배포 (ND = 변형금지)
   · "하버드 공인 / 제휴" 표현 — 허위표시
   · 교재 PDF를 유료 자료로 배포

✅ 안전
   · tinytorch(MIT), mlsysim(Apache 2.0) 코드 상업적 사용
   · 읽고 배운 지식으로 만든 내 오리지널 자료
   · 커리큘럼 구조·순서 참고 (사실·아이디어는 저작권 대상 아님)
   · 가르치는 노동, 구축 서비스 판매
   · "mlsysbook.ai에서 무료로 읽으세요" 안내 (오히려 장려됨)

💡 애매하면
   · GitHub Discussions 또는 저자에게 문의 → 교육 목적엔 매우 개방적
   · Open Collective 후원도 받고 있으니 상업적 활용 협의 여지 있음
```

---

## 5. 라이선스 요약 — 가장 중요한 표

| 대상 | 라이선스 | 상업적 사용 | 2차 변형 | 신호 |
|---|---|---|---|---|
| `books/` (교재 본문) | CC-BY-NC-SA 4.0 | ❌ | ✅ (동일조건) | 🔴 |
| `labs/` (Co-Labs) | CC-BY-NC-SA 4.0 | ❌ | ✅ | 🔴 |
| `kits/` (하드웨어) | CC-BY-NC-SA 4.0 | ❌ | ✅ | 🔴 |
| `slides/` | CC-BY-NC-SA 4.0 | ❌ | ✅ | 🔴 |
| `instructors/` | CC-BY-NC-SA 4.0 | ❌ | ✅ | 🔴 |
| `binder/` | CC-BY-NC-SA 4.0 | ❌ | ✅ | 🔴 |
| **`staffml/`** | **CC-BY-NC-ND 4.0** | ❌ | ❌ **변형도 금지** | 🔴🔴 |
| **`tinytorch/`** | **MIT** | ✅ | ✅ | 🟢 |
| **`mlsysim/`** | **Apache 2.0** | ✅ | ✅ | 🟢 |

저작권자: President and Fellows of Harvard College (2024-2026) /
mlsysim은 Vijay Janapa Reddi and MLSys·im contributors (2026)

> **NC = NonCommercial.** 책 내용·슬라이드·실습·면접문제는 **그대로 가져다 돈 벌 수 없음.**
> 반면 **TinyTorch(MIT)와 MLSys·im(Apache 2.0) 코드는 상업적 사용이 자유.**
> **"내가 이걸로 뭘 할 수 있나"의 답은 거의 전부 이 표에서 나옵니다.**

---

## 6. 참고 링크 모음

### GitHub

| 용도 | URL |
|---|---|
| **원본 저장소** | https://github.com/harvard-edge/cs249r_book |
| **이 포크** | https://github.com/bmshin94/cs249r_book |
| 이슈 | https://github.com/harvard-edge/cs249r_book/issues |
| 새 이슈(피드백 양식) | https://github.com/harvard-edge/cs249r_book/issues/new/choose |
| 디스커션 | https://github.com/harvard-edge/cs249r_book/discussions |
| TinyTorch 피드백 스레드 | https://github.com/harvard-edge/cs249r_book/discussions/1076 |
| 릴리스 | https://github.com/harvard-edge/cs249r_book/releases |
| GitHub Actions | https://github.com/harvard-edge/cs249r_book/actions |

### 사이트 / 패키지

| 용도 | URL |
|---|---|
| 메인 사이트 | https://mlsysbook.ai |
| Vol I (출간완료) | https://mlsysbook.ai/vol1/ |
| Vol II (프리뷰) | https://mlsysbook.ai/vol2/ |
| TinyTorch | https://mlsysbook.ai/tinytorch/ |
| Co-Labs | https://mlsysbook.ai/labs/ |
| 하드웨어 키트 | https://mlsysbook.ai/kits/ |
| MLSys·im | https://mlsysbook.ai/mlsysim/ |
| MLSys·im 시작하기 | https://mlsysbook.ai/mlsysim/getting-started.html |
| 강사 Blueprint | https://mlsysbook.ai/instructors/ |
| 강의 슬라이드 | https://mlsysbook.ai/slides/ |
| StaffML | https://mlsysbook.ai/staffml/ |
| MLSys·im PyPI | https://pypi.org/project/mlsysim/ |
| 뉴스레터 | https://buttondown.email/mlsysbook |
| 후원 (Open Collective) | https://opencollective.com/mlsysbook |

### 저장소 내부 필독 문서

```
docs/REPO_LAYOUT.md          ← 전체 구조 지도 (가장 먼저 읽을 문서)
docs/CI-VARIABLES.md         ← CI 변수 규칙
docs/VERSIONING.md           ← 버전 정책
CONTRIBUTING.md              ← 기여 라우팅 + 초보자 함정 모음
tinytorch/CONTRIBUTING.md    ← tito CLI 전체 명령
tinytorch/MODULE_ANATOMY.md  ← 모듈 구조
labs/PROTOCOL.md             ← 랩 릴리스 불변식
staffml/ARCHITECTURE.md      ← StaffML 아키텍처
mlperf-edu/SPEC.md           ← 벤치마크 스펙
```

---

*이 문서는 Claude Code 세션에서 저장소 전수조사를 통해 작성되었습니다.*
*라이선스 해석은 참고용이며, 실제 상업적 활용 전에는 원문 라이선스 확인과*
*필요 시 법률 자문을 권장합니다.*
