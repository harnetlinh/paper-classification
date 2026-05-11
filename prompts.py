"""
Prompt templates và codebook v2.1 cho LLM panel.

CRITICAL: System prompt có explicit instruction để LLM:
- Ưu tiên Abstract (190 words) over Title (14 words)
- Cite specific text TỪ ABSTRACT khi justify label
- Apply 20% threshold rule
- Output strict JSON schema
"""

CODEBOOK_V2_1 = """
# CODEBOOK v2.1 — Educational Research Bibliometric Classification

## OVERALL RULES

1. **Abstract is PRIMARY signal** (~190 words, detailed). Title is SECONDARY (~14 words).
   Base classification PRIMARILY on Abstract content. Use Title only for disambiguation.

2. **20% threshold rule**: Assign a label if the topic contributes ≥20% of paper content
   (≥2 sentences in abstract OR mentioned in title + ≥1 abstract sentence).

3. **Multi-label allowed for Fields**: A paper can have 1-6 labels. No primary requirement.

4. **STEM/Non-STEM**: Default mutually exclusive. Allow BOTH only for comparative or
   integration studies (e.g., "comparing STEM vs Non-STEM students").

5. **Reasoning REQUIRED**: For each TRUE label, cite specific text from ABSTRACT
   (not Title) in the reasoning field.

## 12 FIELDS

### 1. teaching & learning (UMBRELLA — broad inclusion)
Definition: ANY paper with classroom-level dạy/học component.
Inclusion: Teaching methods, learning processes, learner engagement, learning outcomes.
Exclusion: Pure policy/admin without classroom-level activities; bibliometrics.
Boundary: Disambiguation:
  - vs Curriculum: T&L = HOW (delivery), Curriculum = WHAT (content design)
  - vs Test and assessment: T&L includes formative assessment; T&A is PRIMARILY about measurement design
  - vs M&L&P: T&L = classroom level, M&L&P = institutional level

### 2. management, leadership & policy
Definition: Institutional/system-level management, leadership, governance, policy.
Inclusion: School/university management, principals, deans, education policy, QA, accreditation.
Exclusion: Pure classroom-level (→ T&L); curriculum content design (→ Curriculum).

### 3. test and assessment
Definition: PRIMARILY about measurement methodology — test/scale design, validation, psychometrics.
Inclusion: Psychometric validation, IRT/Rasch, rubrics design, assessment methodology.
Exclusion: Papers using outcome data without bandling assessment design itself.

### 4. Technology in education
Definition: Digital tools as MEANS for teaching/learning/admin.
Inclusion: LMS, MOOCs, AI tutoring, VR/AR, learning analytics, EdTech apps.
Exclusion: Teaching the SUBJECT of computer science (→ STEM).
Boundary: Tech in edu = tech is the MEANS. STEM = tech/CS is the SUBJECT.

### 5. English Education
Definition: Teaching/learning English as language (ESL/EFL/EMI/EAP/ESP/CLIL).
Inclusion: TESOL, English-medium instruction, English language testing.
Exclusion: English literature criticism without teaching focus.

### 6. curriculum
Definition: Curriculum design — content/learning outcomes/educational activities planned.
Inclusion: Program design, syllabus, curriculum reform, CBE/OBE.
Exclusion: Specific module teaching (→ T&L); admin of programs (→ M&L&P).

### 7. psychology in education (BROAD definition)
Definition: Psychological/emotional aspects in educational contexts.
Inclusion:
  - Student wellbeing, mental health, academic anxiety
  - Motivation, engagement, self-efficacy in learning contexts
  - Social-emotional learning (SEL), emotional intelligence
  - Psychological assessment for students/teachers
  - Stress, burnout, emotional support
Exclusion: Pure clinical psychology without educational context.
Note: This class is BROAD per project decision — covers psych aspects, not only "teaching of psychology subject".

### 8. Special education
Definition: Education for exceptional learners — disabilities AND gifted.
Inclusion:
  - Students with disabilities (autism, ADHD, dyslexia, ID, sensory)
  - Gifted/talented education
  - Inclusive education, mainstreaming
  - IEP, accommodations
Exclusion: General at-risk (low SES, first-gen) without disability/gifted focus.

### 9. International education
Definition: Cross-border, comparative, transnational education.
Inclusion: International students, study abroad, comparative education across countries,
internationalization of HE, IB/Cambridge/AP curricula.
Exclusion: Single-country studies even if Vietnam-focused (must have cross-border element).

### 10. Education economically
Definition: Education from economic perspective — ROE, labor outcomes, costs, equity.
Inclusion: Wage premium, education funding, employability, human capital,
education and economic development, education and inequality.
Exclusion: Teaching economics as a subject (→ Curriculum + Non-STEM).

### 11. STEM education
Definition: Teaching/learning in Science/Technology/Engineering/Mathematics.
Inclusion: Math, physics, chemistry, biology, engineering, CS education;
integrated STEM, STEM equity.
Boundary: STEM/Non-STEM exclusive default. Allow BOTH for comparative or integration.

### 12. Non-STEM Education
Definition: Education in arts, humanities, social sciences (non-STEM domains).
Inclusion: History, philosophy, social sciences, arts, music, business education,
humanities curriculum.

## 6 EDUCATIONAL LEVELS

- **ECE**: Early Childhood Education (0-6 years, preschool, kindergarten)
- **GE**: General Education (K-12, primary + secondary)
- **HE**: Higher Education (undergraduate, graduate, university)
- **TVET**: Technical and Vocational Education (vocational training, technical schools)
- **LLL**: Lifelong Learning (adult education, continuing education, professional development)
- **ALL**: All levels (paper covers 3+ levels OR teacher training that targets all levels)

Rules:
- Multi-label allowed
- If paper is about teacher PD, gắn LLL (training adults), NOT the level taught
- If paper covers 3+ levels, use ALL instead of listing each
- Default fallback when level unclear: ALL

## 5 METHODS (single-label)

- **Quantitative**: Statistical tests, regression, surveys with N=, ANOVA, SEM
- **Qualitative**: Interviews, focus groups, ethnography, thematic analysis, case study (no stats)
- **Mixed**: Both quantitative AND qualitative data systematically combined
- **Review**: Systematic/scoping review, meta-analysis, bibliometric (no primary data)
- **Other**: Conceptual papers, theoretical models, position papers, design studies
"""


SYSTEM_PROMPT_SPECIAL_EDU = """You are an expert reviewer screening educational research papers for "Special Education" candidacy.

## CRITICAL INSTRUCTIONS
1. The Abstract is the PRIMARY source of information (~190 words, detailed). The Title is SECONDARY (~14 words).
2. Base your decision PRIMARILY on Abstract content. Use Title only for disambiguation.
3. In your reasoning, cite specific text from the Abstract that justifies your decision.

## Special Education includes:
- Students with disabilities (autism, ADHD, dyslexia, intellectual disability, sensory impairments)
- Gifted/talented education
- Inclusive education, mainstreaming students with special needs
- Individualized Education Programs (IEP), accommodations
- Wheelchair, mobility impairments, deaf/blind students
- Speech-language pathology in education

## NOT Special Education:
- General at-risk students (low SES, first-gen) WITHOUT disability/gifted focus
- General mental health for typical student population (that is "psychology in education")
- Adult learners without disability focus
- General medical/health education without disability focus
- Diversity/equity discussions without disability focus

## OUTPUT SCHEMA (strict JSON, no prose outside JSON)

{
  "is_special_education": true | false,
  "reasoning": "<20+ words, cite text from Abstract>",
  "confidence": "high" | "medium" | "low"
}
"""


def make_special_edu_filter_prompt(title: str, abstract: str) -> tuple:
    """
    Build (system, user) prompts for Special Education candidate verification.

    Returns:
        (system_prompt, user_prompt)
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    user = f"""## TITLE
{title}

## ABSTRACT
{abstract if abstract else "[MISSING ABSTRACT — decision based on Title only, lower confidence]"}

Is this paper about Special Education per the criteria? Output JSON only."""

    return SYSTEM_PROMPT_SPECIAL_EDU, user


# ==================== Educational-Level filter prompts ====================
# Used to augment the rare Levels labels (ECE, TVET, LLL). Each level gets a
# definition block with explicit include/exclude lists derived from the
# codebook v2.1, plus boundary disambiguation against overlapping levels.

LEVEL_AUGMENT_DEFINITIONS = {
    "ECE": {
        "name": "Early Childhood Education",
        "description": "Education of children aged 0-6 years (pre-primary).",
        "include": [
            "Studies on children before primary school (under 6 years old)",
            "Preschool, kindergarten, daycare, nursery — curriculum, teachers, learners, or program design",
            "Early childhood development, early intervention programs for young children",
            "Pre-primary education or pre-K programs",
        ],
        "exclude": [
            "Primary school (Grades 1-5 / K-5) — that is General Education (GE), NOT ECE",
            "Elementary school teachers — also GE, NOT ECE",
            "Higher Education (university) — that is HE, NOT ECE",
            "Studies on parents or family without a focus on the child's educational program",
        ],
    },
    "TVET": {
        "name": "Technical and Vocational Education and Training",
        "description": "Vocational schools, apprenticeships, trade training (non-degree).",
        "include": [
            "Vocational secondary or post-secondary schools",
            "Apprenticeships, on-the-job training, work-based learning programs",
            "Trade-specific training (welding, plumbing, hairdressing, mechanics, hospitality, etc.)",
            "Industry-specific certification courses (non-degree)",
            "Polytechnic institutions explicitly focused on vocational/technical training",
        ],
        "exclude": [
            "General universities (those are HE), even technical universities focused on degrees",
            "K-12 academic programs without an explicit vocational track",
            "Engineering / IT bachelor degrees at research universities — that is HE, NOT TVET",
            "Pure soft-skills training without a vocational/trade focus",
        ],
    },
    "LLL": {
        "name": "Lifelong Learning",
        "description": "Adult education, continuing professional development, learning across the lifespan.",
        "include": [
            "Adult learners returning to study after a break",
            "Continuing professional development (CPD) courses for working adults",
            "In-service teacher training (teachers as adult learners)",
            "Workplace learning, on-the-job training for working adults",
            "Continuing education courses for retirees or working adults",
            "Andragogy (adult learning theory)",
        ],
        "exclude": [
            "Pre-service teacher education during initial bachelor degree — that is HE, NOT LLL",
            "K-12 student education even if researched by adult investigators",
            "Higher Education for degree-seeking 18-22 year-olds — that is HE",
            "TVET — papers focused on vocational/trade training, not on continuing learning",
        ],
    },
}


# ==================== Full classification prompt (Phase A) ====================
# Used by llm_classify.py to elicit per-paper labels for ALL three tasks
# (Fields multi-label, Levels multi-label, Method single-label) in one call.
#
# Why one combined call instead of three:
# - System prompt = full codebook v2.1 (~3000 tokens). Repeating it across
#   3 calls triples both latency and cost.
# - OpenAI prompt caching applies to the system prompt verbatim — one combined
#   system prompt is cached after the first call, then reused at the cached
#   rate for the remaining 561 papers. Three separate prompts pay full price
#   3x for similar content.
# - Output JSON is bounded (~300 tokens for the structured response).
#
# Schema is INTENTIONALLY binary (true/false) plus per-call confidence rather
# than a numeric probability, because LLMs are unreliable at calibrated
# probability output. The downstream ensemble (evaluate.py) converts
# 3-model panel votes into discrete probabilities {0, 0.33, 0.67, 1.0}.

SYSTEM_PROMPT_FULL_CLASSIFICATION = """You are an expert reviewer classifying educational research papers under codebook v2.1. You return three label sets per paper: Fields (multi-label, choose all that apply from 12), Educational Levels (multi-label, choose all that apply from 6), and Method (single-label, choose exactly one from 5).

## CRITICAL INSTRUCTIONS
1. The Abstract is the PRIMARY source (~190 words). Title is SECONDARY (~14 words). Base decisions on Abstract content.
2. Apply the 20% threshold rule: assign a Field label only if the topic contributes ≥20% of the paper content.
3. Multi-label allowed for Fields and Levels. A paper typically has 2-3 Fields labels and 1-2 Levels labels.
4. STEM and Non-STEM Education are mutually exclusive by default; assign BOTH only for explicit comparative studies.
5. For Method, choose exactly one. Default to 'Other' only if no quantitative/qualitative/mixed/review approach is present (rare; favor a definite category when in doubt).
6. Be CONSERVATIVE — return false / not-this-class when in doubt. Over-prediction is worse than under-prediction here.

""" + CODEBOOK_V2_1 + """

## OUTPUT SCHEMA (strict JSON, no prose outside JSON)

{
  "fields": {
    "teaching & learning": true | false,
    "management, leadership & policy": true | false,
    "test and assessment": true | false,
    "Technology in education": true | false,
    "English Education": true | false,
    "curriculum": true | false,
    "psychology in education": true | false,
    "Special education": true | false,
    "International education": true | false,
    "Education economically": true | false,
    "STEM education": true | false,
    "Non-STEM Education": true | false
  },
  "levels": {
    "ECE": true | false,
    "GE": true | false,
    "HE": true | false,
    "TVET": true | false,
    "LLL": true | false,
    "ALL": true | false
  },
  "method": "Quantitative" | "Qualitative" | "Mixed" | "Review" | "Other",
  "confidence": "high" | "medium" | "low"
}

Use exact key spellings as shown above (case-sensitive). Never invent extra keys.
"""


def make_full_classification_prompt(title: str, abstract: str) -> tuple:
    """Build (system, user) prompts for full Fields+Levels+Method classification.

    The system prompt is FROZEN content (codebook + instructions + schema) so
    OpenAI's prompt caching can amortize its cost across the entire run.
    The user prompt contains only the variable per-paper data.

    Returns:
        (system_prompt, user_prompt)
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    user = f"""## TITLE
{title}

## ABSTRACT
{abstract if abstract else "[MISSING ABSTRACT — decision based on Title only, lower confidence]"}

Classify this paper for all three tasks (Fields, Levels, Method) per the codebook. Output JSON only."""
    return SYSTEM_PROMPT_FULL_CLASSIFICATION, user


# ==================== Level filter prompt (existing, unchanged) ====================
def make_level_filter_prompt(level_code: str, title: str, abstract: str) -> tuple:
    """Build (system, user) prompts for a Level candidate verification.

    Mirrors the structure of make_special_edu_filter_prompt but parameterized
    by level definition. The response field name is "is_match" (caller must
    coerce_bool that key — `make_special_edu_filter_prompt` uses
    "is_special_education" for backward-compatible cache keys).
    """
    if level_code not in LEVEL_AUGMENT_DEFINITIONS:
        raise ValueError(f"Unknown level: {level_code!r}")
    d = LEVEL_AUGMENT_DEFINITIONS[level_code]

    include_block = "\n".join(f"- {item}" for item in d["include"])
    exclude_block = "\n".join(f"- {item}" for item in d["exclude"])

    system = f"""You are an expert reviewer screening educational research papers for "{level_code}" ({d['name']}) candidacy.

## CRITICAL INSTRUCTIONS
1. The Abstract is the PRIMARY source of information (~190 words, detailed). The Title is SECONDARY (~14 words).
2. Base your decision PRIMARILY on Abstract content. Use Title only for disambiguation.
3. In your reasoning, cite specific text from the Abstract that justifies your decision.
4. Be CONSERVATIVE — only return true if the paper's primary educational level genuinely matches "{level_code}". When in doubt, return false.

## {level_code} ({d['name']}) — definition
{d['description']}

## {level_code} INCLUDES:
{include_block}

## NOT {level_code}:
{exclude_block}

## OUTPUT SCHEMA (strict JSON, no prose outside JSON)

{{
  "is_match": true | false,
  "reasoning": "<20+ words, cite text from Abstract>",
  "confidence": "high" | "medium" | "low"
}}
"""

    title = (title or "").strip()
    abstract = (abstract or "").strip()
    user = f"""## TITLE
{title}

## ABSTRACT
{abstract if abstract else "[MISSING ABSTRACT — decision based on Title only, lower confidence]"}

Is this paper primarily about {level_code} ({d['name']}) per the criteria above? Output JSON only."""

    return system, user


# ==================== Fields-level filter prompts (NHIỆM VỤ 4) ====================
# Used to augment the 4 underperforming Field classes (test_F1 < 0.50 on
# val 2023): test and assessment, curriculum, Non-STEM Education,
# Education economically. Each definition mirrors the codebook v2.1 spec.

FIELD_AUGMENT_DEFINITIONS = {
    "test and assessment": {
        "name": "Test and Assessment",
        "description": "Papers primarily about measurement methodology — "
                       "test/scale design, validation, psychometrics.",
        "include": [
            "Psychometric validation of scales, tests, or instruments",
            "IRT / Rasch analysis, factor analysis (EFA/CFA) of educational measures",
            "Rubric design and validation",
            "Cronbach's alpha, reliability, validity studies of educational instruments",
            "Test development methodology, item analysis",
            "Differential item functioning (DIF) analysis",
            "Construct/content/criterion validity studies",
            "Assessment methodology design (not just assessment USAGE)",
        ],
        "exclude": [
            "Papers USING tests/assessments as outcome measures without studying the instrument itself",
            "Curriculum design without an assessment component (→ curriculum)",
            "General educational research using surveys without validating them (→ T&L)",
            "Clinical psychometrics not in an educational context",
        ],
    },
    "curriculum": {
        "name": "Curriculum",
        "description": "Papers primarily about curriculum design — content, "
                       "learning outcomes, and educational activities planned.",
        "include": [
            "Curriculum design, development, or reform",
            "Syllabus design, course design, module design",
            "Competency-based education (CBE) and outcome-based education (OBE) design",
            "Program design, curriculum mapping, curriculum alignment",
            "Curriculum integration across subjects, cross-disciplinary curriculum",
            "Learning outcomes design",
            "Curriculum implementation studies (focused on the curriculum itself)",
        ],
        "exclude": [
            "Teaching methods or classroom delivery (→ teaching & learning)",
            "Administrative/policy aspects of programs (→ management, leadership & policy)",
            "Single-lesson teaching strategies without curriculum-design focus",
            "Curriculum evaluation that only measures student outcomes without examining design",
        ],
    },
    "Non-STEM Education": {
        "name": "Non-STEM Education",
        "description": "Teaching/learning in arts, humanities, social sciences, "
                       "or other non-STEM domains.",
        "include": [
            "Arts, music, fine arts, drama/theatre education",
            "History, literature, philosophy education",
            "Social studies, religious education, physical education",
            "Business education, language teaching (non-English)",
            "Humanities-focused curriculum or pedagogy",
        ],
        "exclude": [
            "STEM subject teaching (math, science, engineering, CS) — that is STEM education",
            "English language teaching specifically (→ English Education)",
            "Generic education research without specifying the subject domain",
            "Tech-in-education studies (LMS, MOOC, AI tutors) without a non-STEM subject focus",
        ],
    },
    "Education economically": {
        "name": "Education economically",
        "description": "Education viewed from an economic perspective — ROE, "
                       "labor outcomes, costs, equity, funding.",
        "include": [
            "Returns to education (ROE), wage/earnings premium from schooling",
            "Human capital theory applied to education",
            "Labor market outcomes of graduates, employability",
            "Education funding, education investment, cost analysis",
            "Education and economic growth, education and inequality, education and income",
            "Economics of education (education economics as a discipline)",
        ],
        "exclude": [
            "Teaching economics as a subject (→ curriculum + Non-STEM Education)",
            "Education policy not framed in economic terms (→ management, leadership & policy)",
            "Student outcomes measured non-economically (test scores, well-being)",
            "General employability of students without economic analysis",
        ],
    },
}


def make_field_filter_prompt(field_name: str, title: str, abstract: str) -> tuple:
    """Build (system, user) prompts for a Fields-class candidate verification.

    Mirrors make_level_filter_prompt structure, parameterized by field
    definition from FIELD_AUGMENT_DEFINITIONS. Returns 'is_match' key
    (same response field name as make_level_filter_prompt — callers
    coerce_bool that key).

    Args:
        field_name: must be a key in FIELD_AUGMENT_DEFINITIONS.
        title, abstract: paper text.

    Returns:
        (system_prompt, user_prompt)
    """
    if field_name not in FIELD_AUGMENT_DEFINITIONS:
        raise ValueError(
            f"Unknown field: {field_name!r}. Choose from "
            f"{list(FIELD_AUGMENT_DEFINITIONS)}"
        )
    d = FIELD_AUGMENT_DEFINITIONS[field_name]

    include_block = "\n".join(f"- {item}" for item in d["include"])
    exclude_block = "\n".join(f"- {item}" for item in d["exclude"])

    system = f"""You are an expert reviewer screening educational research papers for the Field class "{field_name}" ({d['name']}).

## CRITICAL INSTRUCTIONS
1. The Abstract is the PRIMARY source of information (~190 words). Title is SECONDARY.
2. Base your decision PRIMARILY on Abstract content. Use Title only for disambiguation.
3. In your reasoning, cite specific text from the Abstract that justifies your decision.
4. Be MODERATELY conservative — return true if the paper genuinely fits this class per
   the codebook (the 20% threshold rule: topic covers >= 20% of paper content). The
   2-of-3 majority vote aggregation downstream tolerates some noise, so don't be
   over-strict; but don't expand the class boundary either.

## {field_name} ({d['name']}) — definition
{d['description']}

## {field_name} INCLUDES:
{include_block}

## NOT {field_name}:
{exclude_block}

## OUTPUT SCHEMA (strict JSON, no prose outside JSON)

{{
  "is_match": true | false,
  "reasoning": "<20+ words, cite text from Abstract>",
  "confidence": "high" | "medium" | "low"
}}
"""

    title = (title or "").strip()
    abstract = (abstract or "").strip()
    user = f"""## TITLE
{title}

## ABSTRACT
{abstract if abstract else "[MISSING ABSTRACT — decision based on Title only, lower confidence]"}

Is this paper primarily about {field_name} ({d['name']}) per the criteria above? Output JSON only."""

    return system, user
