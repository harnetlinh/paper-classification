# Codebook v2.1 — Educational Research Bibliometric Classification

**Status: FROZEN** — Không thay đổi sau khi pipeline đã chạy.
**Thay đổi từ v2:** Drop `research`, BROAD definition cho `psychology in education`.

## Schema overview

| Task | Type | Số lớp |
|---|---|---|
| Fields | Multi-label | 12 |
| Educational Level | Multi-label | 6 |
| Method | Single-label | 5 |

## Quy tắc tổng

1. **Abstract is PRIMARY** (~190 words, detailed). Title is SECONDARY (~14 words).
   Phân loại PRIMARILY dựa trên Abstract. Title chỉ dùng để disambiguate.
2. **20% threshold**: Gắn label nếu topic chiếm ≥20% nội dung paper.
3. **STEM/Non-STEM**: Default mutually exclusive. Cho phép cả 2 cho comparative/integration.
4. **Reasoning required**: Mỗi label TRUE phải cite specific text từ Abstract.

## 12 FIELDS

### 1. teaching & learning (UMBRELLA)
- Definition: Mọi paper có classroom-level dạy/học
- Inclusion: Teaching methods, learning processes, learner engagement, learning outcomes
- Disambiguation: T&L = HOW; Curriculum = WHAT; T&A = primarily measurement

### 2. management, leadership & policy
- Definition: Institutional/system-level management, leadership, governance, policy
- Inclusion: School management, dean/principal roles, education policy, QA, accreditation

### 3. test and assessment
- Definition: PRIMARILY về measurement methodology — design, validation, psychometrics
- Inclusion: Psychometric validation, IRT/Rasch, rubrics, assessment methodology

### 4. Technology in education
- Definition: Digital tools as MEANS for teaching/learning/admin
- Inclusion: LMS, MOOCs, AI tutoring, VR/AR, learning analytics
- vs STEM: Tech-in-edu = tech là MEANS; STEM = tech/CS là SUBJECT

### 5. English Education
- Definition: Teaching/learning English as language (ESL/EFL/EMI/EAP/ESP/CLIL)
- Inclusion: TESOL, English-medium instruction, English language testing

### 6. curriculum
- Definition: Curriculum design — content/learning outcomes/educational activities planned
- Inclusion: Program design, syllabus, curriculum reform, CBE/OBE

### 7. psychology in education (BROAD definition)
- Definition: Psychological/emotional aspects in educational contexts
- Inclusion:
  - Student wellbeing, mental health, academic anxiety
  - Motivation, engagement, self-efficacy in learning
  - Social-emotional learning (SEL), emotional intelligence
  - Psychological assessment for students/teachers
  - Stress, burnout, emotional support
- Note: BROAD per Q2 decision — covers psych ASPECTS, not only "teaching of psych subject"

### 8. Special education
- Definition: Education for exceptional learners — disabilities AND gifted
- Inclusion: Disabilities, gifted/talented, inclusive education, IEP

### 9. International education
- Definition: Cross-border, comparative, transnational education
- Inclusion: International students, comparative education, internationalization of HE

### 10. Education economically
- Definition: Education from economic perspective — ROE, labor outcomes, costs, equity
- Inclusion: Wage premium, education funding, employability, human capital

### 11. STEM education
- Definition: Teaching/learning trong Science/Technology/Engineering/Mathematics
- Inclusion: Math/physics/chem/bio/engineering/CS education; integrated STEM

### 12. Non-STEM Education
- Definition: Education in arts, humanities, social sciences (non-STEM)
- Inclusion: History, philosophy, social sciences, arts, business, humanities

## 6 EDUCATIONAL LEVELS

| Code | Name | Definition |
|---|---|---|
| ECE | Early Childhood Education | 0-6 tuổi, mầm non |
| GE | General Education | K-12 (tiểu học + THCS + THPT) |
| HE | Higher Education | Đại học, sau đại học |
| TVET | Technical/Vocational | Đào tạo nghề |
| LLL | Lifelong Learning | Người lớn, continuing edu, professional development |
| ALL | All levels | Bao trùm 3+ cấp hoặc không phân biệt cấp cụ thể |

Rules:
- Multi-label allowed
- Teacher PD → LLL (training adults)
- 3+ cấp → ALL thay vì list từng cấp
- Default fallback: ALL

## 5 METHODS (single-label)

| Method | Indicators |
|---|---|
| Quantitative | Stats tests, regression, surveys N=, ANOVA, SEM, p-value |
| Qualitative | Interviews, focus groups, ethnography, thematic analysis |
| Mixed | Both quan + qual systematically combined |
| Review | Systematic/scoping review, meta-analysis, bibliometric (no primary data) |
| Other | Conceptual papers, theoretical models, position papers |

## Decisions trong codebook v2.1 vs v2

1. **Drop `research`** (per Q1 decision) — broad definition không informative
2. **Rename `psychology eduation` → `psychology in education`** — sửa typo + clarify
3. **`psychology in education` BROAD** (per Q2 decision) — match gold cũ
4. **Special education includes gifted** (per codebook v2 maintained)
5. **STEM/Non-STEM exclusive default**, cho phép cả 2 cho comparative

## Audit trail

- File này được hash SHA-256, lưu tại `configs/codebook_hash.txt`
- Bất kỳ revision nào sẽ bump version v2.1 → v2.2 và yêu cầu re-run pipeline
