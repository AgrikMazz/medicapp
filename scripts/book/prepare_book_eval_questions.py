"""Assemble an auditable, source-span QA set without rebuilding the graph.

New questions are authored against unused T passages. The old synthetic pilot
is retained in development only after dropping invalid or unsupported items.
"""

import hashlib
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "medical-CareForInsane" / "book.txt"
GRAPH = ROOT / "data" / "book-graph" / "graph.pkl"
LEGACY_QA = ROOT / "data" / "book-graph" / "eval_questions.generated.jsonl"
OUTPUT = ROOT / "data" / "book-graph" / "eval_questions.reviewed.jsonl"


# Each quote is looked up only inside the stated persisted T passage. The
# resulting absolute offsets are validated against the unchanged book text.
CURATED = [
    {
        "id": "cur-dev-001", "split": "dev", "question_category": "single_fact",
        "question": "Why does the manual say a new attendant should begin in a minor position?",
        "answers": ["To learn ward work gradually and acquire caution and watchfulness before taking on greater responsibility."],
        "evidence": [("T_8937ba1230e86993f2745cc0", "the new attendant must be willing to fill, at first, a minor position, to begin at the beginning and learn gradually all the details of ward work; he must acquire habits of caution and watchfulness")],
    },
    {
        "id": "cur-dev-002", "split": "dev", "question_category": "single_fact",
        "question": "What does the manual say attendants should avoid doing to change a patient's delusion?",
        "answers": ["They should not argue patients out of their delusions because it can fix the belief more firmly."],
        "evidence": [("T_a4e97d003189ecae22d1a13e", "It is useless for attendants to try to argue patients out of a belief in their delusions, and to do so often results in fixing them more firmly in the mind.")],
    },
    {
        "id": "cur-dev-003", "split": "dev", "question_category": "single_fact",
        "question": "What does the manual call a useful attendant's main strength?",
        "answers": ["Gain the patients' confidence and exert beneficial influence over them."],
        "evidence": [("T_363118afb7c29ca7818cfbc2", "is one who tempers these mentioned traits, by striving to gain the confidence of his patients, by exerting over them a beneficial influence")],
    },
    {
        "id": "cur-dev-004", "split": "dev", "question_category": "single_fact",
        "question": "Which methods does the manual recommend for influencing patients' conduct?",
        "answers": ["Sympathy, kindness, tact, reasons for requests, and when needed firm but kind authority."],
        "evidence": [("T_1cf2799c8afa279aef5130f2", "We can generally control our patients by the exercise of sympathy, kindness, and tact, joined with a reason for what is required, and where more is needed, a firm, kind authority and command will suffice.")],
    },
    {
        "id": "cur-dev-005", "split": "dev", "question_category": "single_fact",
        "question": "What rule should govern the work and occupation assigned to patients?",
        "answers": ["The work should be for the patients' own benefit."],
        "evidence": [("T_2679bbd5ed82631813fd1ad2", "The only rule to go by is, that the work and occupation shall be for their own good")],
    },
    {
        "id": "cur-dev-006", "split": "dev", "question_category": "single_fact",
        "question": "What does the manual say about patients who are too weak to work?",
        "answers": ["They may need rest, quiet, and nursing, with their care and occupation directed by a physician."],
        "evidence": [("T_31770bd9d885820fb835609c", "many need rest, quiet, and nursing, and directions for the care and occupation of such patients should come from the physician.")],
    },
    {
        "id": "cur-dev-007", "split": "dev", "question_category": "single_fact",
        "question": "What should patients be allowed to do to support self-respect and individuality?",
        "answers": ["Whatever helps maintain self-respect, self-care, and a feeling of individuality."],
        "evidence": [("T_2f6e0c1347e8ccd14f123e51", "and should be allowed to do whatever will help maintain their self-respect, self-care, and a feeling of individuality.")],
    },
    {
        "id": "cur-dev-008", "split": "dev", "question_category": "single_fact",
        "question": "When should a physician be told that a patient is not getting enough food?",
        "answers": ["Before the patient loses flesh or strength from insufficient or improper food."],
        "evidence": [("T_8b9875a875af0a45f4900e76", "No patient should be allowed to lose in flesh and strength on account of failure to take sufficient, or proper food; before these things happen it should be reported to the physician.")],
    },
    {
        "id": "cur-dev-009", "split": "dev", "question_category": "single_fact",
        "question": "How does the manual prioritize patients when they are put to bed?",
        "answers": ["The helpless go first; violent and homicidal patients are watched and put to bed early; suicidal patients remain supervised and go at a convenient time."],
        "evidence": [("T_2a0b2bd73d2cce532a2b6011", "The helpless should be dressed and undressed, and put to bed first: the violent and homicidal need to be watched, and should be put to bed early, while the suicidal should be kept under supervision, and put to bed at the most convenient time.")],
    },
    {
        "id": "cur-dev-010", "split": "dev", "question_category": "single_fact",
        "question": "What can attendants do to prevent some outbursts of violence?",
        "answers": ["Learn what causes violence and remove the cause where possible."],
        "evidence": [("T_c5314a8b44fa0e4a3c4dc546", "Having learned what will cause violence, it can often be avoided by removing the cause;")],
    },
    {
        "id": "cur-dev-011", "split": "dev", "question_category": "single_fact",
        "question": "How gently should attendants use force with violent patients?",
        "answers": ["As gently as possible."],
        "evidence": [("T_430fd1a2ade40c791d810561", "In using force in the care of violent patients, it should always be done as gently as possible")],
    },
    {
        "id": "cur-dev-012", "split": "dev", "question_category": "single_fact",
        "question": "What should attendants do when a patient begins a violent paroxysm?",
        "answers": ["Give immediate care; temporary control may be needed."],
        "evidence": [("T_ebae2ccb8e8d109c3ac92f0d", "require immediate care and often temporary control at the hands of attendants.")],
    },
    {
        "id": "cur-dev-013", "split": "dev", "question_category": "multi_evidence_reasoning",
        "question": "How does the manual connect preventing violent outbursts with the way attendants should intervene when force is needed?",
        "answers": ["Learn and remove causes where possible; if force is necessary, use it gently and avoid struggling."],
        "evidence": [
            ("T_c5314a8b44fa0e4a3c4dc546", "Having learned what will cause violence, it can often be avoided by removing the cause;"),
            ("T_430fd1a2ade40c791d810561", "In using force in the care of violent patients, it should always be done as gently as possible")
        ],
    },
    {
        "id": "cur-dev-014", "split": "dev", "question_category": "multi_evidence_reasoning",
        "question": "How should patient work be balanced with other forms of occupation?",
        "answers": ["Work should benefit the patient, but recreation and diversion are also needed; work alone is not enough."],
        "evidence": [
            ("T_2679bbd5ed82631813fd1ad2", "The only rule to go by is, that the work and occupation shall be for their own good"),
            ("T_31770bd9d885820fb835609c", "nothing but work is as bad as no work, and that they should have diversion, enjoyment, and entertainment.")
        ],
    },
    {
        "id": "cur-dev-015", "split": "dev", "question_category": "multi_evidence_reasoning",
        "question": "What two features does the manual associate with an effective attendant's care of individual patients?",
        "answers": ["The attendant should understand each patient's needs and gain confidence through beneficial influence."],
        "evidence": [
            ("T_363118afb7c29ca7818cfbc2", "striving to gain the confidence of his patients, by exerting over them a beneficial influence"),
            ("T_1cf2799c8afa279aef5130f2", "it is their duty to know thoroughly the wants, and condition of each case, and how best to care for and control it.")
        ],
    },
    {
        "id": "cur-test-001", "split": "heldout", "question_category": "single_fact",
        "question": "Why are patients with delusions of suspicion treated as potentially dangerous?",
        "answers": ["Their delusions may fully control them and make them dangerous and difficult to manage."],
        "evidence": [("T_4f1cdc732e81c85ce2c1b4a1", "because they are frequently fully under the control of delusions, which make them dangerous and difficult to manage.")],
    },
    {
        "id": "cur-test-002", "split": "heldout", "question_category": "single_fact",
        "question": "What kinds of fears may patients with suspicions of bodily harm express?",
        "answers": ["They may believe they will be tortured, burned alive, or killed."],
        "evidence": [("T_de6cf88c6bc142065552f76c", "They may think they are to be tortured, that they are to be burned alive, or that some one is trying to kill them.")],
    },
    {
        "id": "cur-test-003", "split": "heldout", "question_category": "single_fact",
        "question": "What motive may lead a patient to become homicidal, according to the manual?",
        "answers": ["A desire to protect themselves from supposed assaults."],
        "evidence": [("T_b01522eb635d715a82f81844", "Other patients become homicidal under the desire to protect themselves from supposed assaults.")],
    },
    {
        "id": "cur-test-004", "split": "heldout", "question_category": "single_fact",
        "question": "What level of observation does the manual prescribe for suicidal patients?",
        "answers": ["Constant watchfulness day and night."],
        "evidence": [("T_ea7f0eeec461c1b86ee92980", "The only way to care for patients who are suicidal, is by constant watchfulness day and night.")],
    },
    {
        "id": "cur-test-005", "split": "heldout", "question_category": "single_fact",
        "question": "What does the manual say may motivate patients to set fires?",
        "answers": ["Delusions or hallucinations, along with suspicions and feelings of being treated wrongly."],
        "evidence": [("T_c1d1e021dd0058cfca2629c0", "Patients with these tendencies generally desire to commit incendiary acts under the influence of delusions or hallucinations; added to these there are frequently suspicions and feelings of wrong treatment")],
    },
    {
        "id": "cur-test-006", "split": "heldout", "question_category": "multi_evidence_reasoning",
        "question": "What may happen when early-stage patients are left unoccupied and distressed by confinement?",
        "answers": ["Confinement may make them fretful and lead to violent efforts to escape; without occupation, their distress may also provoke violence."],
        "evidence": [
            ("T_a5051e3476bd1977ac65243c", "they may, and often do, become fretful, are irritated by their confinement, sleep poorly, eat little, and may make violent efforts to escape."),
            ("T_a5051e3476bd1977ac65243c", "These conditions, if nothing is done to occupy the patient’s time and mind, and so relieve them, will often be sufficient to provoke violence.")
        ],
    },
    {
        "id": "cur-test-007", "split": "heldout", "question_category": "single_fact",
        "question": "How should food be offered to a patient who alternates between refusing it and accepting it?",
        "answers": ["It should be offered frequently."],
        "evidence": [("T_4f34f5769fc76bc537fea4d3", "Some patients, for reasons not always known, will refuse food one hour and take it freely the next; it[Pg 63] should, therefore, be frequently offered.")],
    },
    {
        "id": "cur-test-008", "split": "heldout", "question_category": "single_fact",
        "question": "Which symptoms does the manual list as warnings of danger that should be reported?",
        "answers": ["Unconsciousness; slow, labored, rapid, shallow, or irregular breathing; increased weakness and a rapid heart or pulse; cold hands or feet; picking at bedclothes or imaginary objects; or vacant staring."],
        "evidence": [("T_6c3ed0f0583234686a14c3c6", "There are certain symptoms which should warn the attendant of danger, and which often precede death. When any of these are present they should be reported to the physician. They are: partial or complete unconsciousness, slow and labored, rapid, shallow, or irregular breathing, increased weakness and rapidity of heart or pulse, cold hands and feet. Picking at the bedclothes, or at imaginary objects in the air, or vacant staring, are bad symptoms.")],
    },
    {
        "id": "cur-test-009", "split": "heldout", "question_category": "single_fact",
        "question": "What care does the manual recommend for patients recovering from dementia?",
        "answers": ["Feed them well, take them out for exercise, and encourage them to work as they are able."],
        "evidence": [("T_fffa1fccdca51c2d85fe434b", "They must be well fed, regularly taken out for exercise, and, as they are able, encouraged to employ themselves.")],
    },
    {
        "id": "cur-test-010", "split": "heldout", "question_category": "single_fact",
        "question": "During an epileptic fit, should attendants hold the patient to stop the convulsions?",
        "answers": ["No. They should prevent injury, not hold the patient to stop convulsions."],
        "evidence": [("T_46ee6edab6639ac05028a759", "Patients should not be held to prevent the convulsions, but so that they shall not injure themselves.")],
    },
    {
        "id": "cur-test-011", "split": "heldout", "question_category": "single_fact",
        "question": "What is required of attendants after an epileptic fit?",
        "answers": ["Guarded care and watching, because patients may be dangerous after a fit."],
        "evidence": [("T_1ad70ede2f937b917043b2d5", "After a fit they are frequently dangerous and always require guarded care and watching.")],
    },
    {
        "id": "cur-test-012", "split": "heldout", "question_category": "single_fact",
        "question": "How does the manual recommend preventing bed-sores in bedridden patients?",
        "answers": ["Keep them clean; if possible, sit them up for several hours daily or reposition them from side to side and back."],
        "evidence": [
            ("T_8d1e97514702b67a3537ff12", "They will always appear in a bedridden paretic in a few days if not kept perfectly clean."),
            ("T_8d1e97514702b67a3537ff12", "Such patients, should, if possible, be made to sit up several hours every day, or placed first on one side, then on the back, and then on the other side.")
        ],
    },
    {
        "id": "cur-test-013", "split": "heldout", "question_category": "single_fact",
        "question": "What should attendants do when a patient has been injured?",
        "answers": ["Report every injury to a physician immediately."],
        "evidence": [("T_8d61fe72e8e28762a26614a4", "Every injury received by a patient should be immediately reported to a physician.")],
    },
    {
        "id": "cur-test-014", "split": "heldout", "question_category": "single_fact",
        "question": "Why must attendants check a splint after a fracture?",
        "answers": ["Swelling can make a splint restrict circulation and cause gangrene."],
        "evidence": [("T_8d61fe72e8e28762a26614a4", "If temporary splints are put on do not make them too tight, and loosen them from time to time as needed. The extremities sometimes swell rapidly after a fracture, and the splints may so stop the circulation that, in a few hours, gangrene may be caused by them.")],
    },
    {
        "id": "cur-test-015", "split": "heldout", "question_category": "single_fact",
        "question": "What may happen if a choking patient does not receive relief?",
        "answers": ["The patient may die within a few minutes."],
        "evidence": [("T_63a9608de60cfbf5e4d801ed", "if no relief is obtained, the sufferer will die in a few minutes.")],
    },
    {
        "id": "cur-test-016", "split": "heldout", "question_category": "single_fact",
        "question": "What does the manual advise when a patient has swallowed glass?",
        "answers": ["Do not give an emetic or a cathartic."],
        "evidence": [("T_9d0c289ecba2157c3a9ac856", "In the treatment do not give an emetic or a cathartic.")],
    },
    {
        "id": "cur-test-017", "split": "heldout", "question_category": "single_fact",
        "question": "Which symptoms from prolonged or large medicine doses should be reported?",
        "answers": ["Report rashes, puffiness or irritation around the eyes, metallic taste, bleeding gums or sore teeth, and other listed effects such as nausea, vomiting, bowel changes, dizziness, drowsiness, or convulsions."],
        "evidence": [("T_754561b88ab9dba6a367959e", "Sometimes medicines, given in large or long-continued doses, cause symptoms that an attendant should notice and report to the physician; some of these are, eruptions on the face and body, puffiness about the eyes, irritation and running of the eyes, a metallic taste in the mouth, bleeding of the gums or soreness of the teeth and profuse flow of saliva, nausea, vomiting, diarrhœa, constipation, indigestion, ringing of the ears, feeling of fulness in the head, headache, dizziness, drowsiness, coma, convulsions, or convulsive movements of muscles.")],
    },
    {
        "id": "cur-test-018", "split": "heldout", "question_category": "single_fact",
        "question": "What does the manual say about the use of alcohol in emergencies?",
        "answers": ["It should not be given without a good reason because it may cause serious or fatal harm."],
        "evidence": [("T_4f2b770699243fb89ff66fb2", "Do not give them in emergencies, without a good reason for so doing, and not simply because you feel you must do something, for in some emergencies they may do a great deal of harm, and perhaps, a fatal injury.")],
    },
    {
        "id": "cur-test-019", "split": "heldout", "question_category": "multi_evidence_reasoning",
        "question": "How do the manual's instructions differ during an epileptic fit and after it ends?",
        "answers": ["During the fit, prevent injury without restraining convulsions; afterward, provide guarded care and continued watching."],
        "evidence": [
            ("T_46ee6edab6639ac05028a759", "Patients should not be held to prevent the convulsions, but so that they shall not injure themselves."),
            ("T_1ad70ede2f937b917043b2d5", "After a fit they are frequently dangerous and always require guarded care and watching.")
        ],
    },
    {
        "id": "cur-test-020", "split": "heldout", "question_category": "multi_evidence_reasoning",
        "question": "What anatomical explanation and warning signs does the manual give for choking?",
        "answers": ["The mouth and nose open into the pharynx. Choking causes immediate distress, ineffective attempts to breathe, and a dark-blue face; without relief, the patient may die within minutes."],
        "evidence": [
            ("T_32e3b403139914b440cfe24e", "They open into a common passage, the pharynx"),
            ("T_63a9608de60cfbf5e4d801ed", "The symptoms of choking are immediate, and if no relief is obtained, the sufferer will die in a few minutes. If the patient knows any thing, he will show immediate signs of distress, violent but ineffectual attempts to breathe, and the face quickly becomes a dark blue color, from the accumulation of carbonic acid in the blood.")
        ],
    },
    {
        "id": "cur-test-021", "split": "heldout", "question_category": "multi_evidence_lookup",
        "question": "Which routes can introduce medicine into the body, and why might patients refuse it?",
        "answers": ["Routes include the stomach, lungs, rectum, skin, and injection under the skin; a patient may refuse because they believe they are well or that the medicine will injure them."],
        "evidence": [
            ("T_2e8667b3bad13ea334ce6ca6", "Medicines are introduced into the system through the stomach, the lungs, the rectum, the skin, or by being injected into the tissues, under the skin."),
            ("T_63a88ceae524db2d0646fb98", "some say they are perfectly well and need no medical treatment, others think the medicine injures them")
        ],
    },
    {
        "id": "cur-dev-016", "split": "dev", "question_category": "multi_evidence_lookup",
        "question_source": "assistant-authored; evidence overlaps a legacy development passage",
        "question": "Which supplies should be ready before feeding, and when should an attendant stop a tube-feeding attempt?",
        "answers": ["Have the feeding apparatus, food and medicine, spoons, cups, pitcher, water, soap, towels, and a basin ready. Stop for unusual difficulty, severe struggling, or any sign of danger, and report it to the physician."],
        "evidence": [
            ("T_23754d598e633b74bd00513f", "Upon a tray should be all the feeding apparatus—the food and medicine, several spoons, and cups, and a pitcher. Near at hand should be plenty of water, soap, and towels, and a tin basin."),
            ("T_96cdd8185b4fad63b719ad04", "If there seem to be unusual difficulty in so doing, severe and unusual struggling, or the slightest symptom of danger, cease the effort to feed, and report to the physician.")
        ],
    },
    {
        "id": "cur-test-023", "split": "heldout", "question_category": "multi_evidence_reasoning",
        "question": "What care does the manual recommend for severe exhaustion, and which warning signs require reporting?",
        "answers": ["Provide constant nursing day and night, and report unconsciousness, abnormal breathing, a weak or rapid pulse, or cold hands and feet."],
        "evidence": [
            ("T_4f34f5769fc76bc537fea4d3", "Such patients must never be left alone and need constant nursing day and night."),
            ("T_6c3ed0f0583234686a14c3c6", "partial or complete unconsciousness, slow and labored, rapid, shallow, or irregular breathing, increased weakness and rapidity of heart or pulse, cold hands and feet.")
        ],
    },
    {
        "id": "cur-test-024", "split": "heldout", "question_category": "single_fact",
        "question": "What changes in a recovering patient's condition should prompt attendants to report a possible setback?",
        "answers": ["A return of violent symptoms, failure to sleep, or any change in health should be reported promptly."],
        "evidence": [("T_fffa1fccdca51c2d85fe434b", "Any symptoms of a return of their more violent condition, any failure to sleep, or change noticed in the health of the patient, should be at once reported.")],
    },
]


def exact_source_match(source, text, lower, upper):
    at = source.find(text, lower, upper)
    if at >= 0:
        return at, source[at:at + len(text)]
    # Passage serialization occasionally normalizes whitespace at sentence
    # boundaries. Locate equivalent whitespace, then store the source-exact quote.
    words = re.findall(r"\S+", text.strip())
    if not words:
        raise ValueError("Empty evidence quote")
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, source[lower:upper])
    if not match:
        raise ValueError(f"Evidence quote not found in source window: {text!r}")
    start = lower + match.start()
    return start, source[start:lower + match.end()]


def main():
    source = SOURCE.read_text(encoding="utf-8-sig")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with open(GRAPH, "rb") as handle:
        graph = pickle.load(handle)

    legacy = []
    with open(LEGACY_QA, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["id"] in {"synthetic-001", "synthetic-002", "synthetic-019"}:
                continue
            item["split"] = "dev"
            item["question_category"] = (
                "multi_evidence_lookup" if len(item.get("evidence_spans", [])) > 1
                else "single_fact"
            )
            item["review_status"] = "assistant source-span and answer audit; not externally human-annotated"
            item["question_source"] = "legacy LLM-generated pilot, development-only"
            item["source_sha256"] = source_hash
            if item["id"] == "synthetic-003":
                item["answers"] = [
                    "The manual calls the brain the organ of the mind, but says their true relation is not known."
                ]
            legacy.append(item)

    curated = []
    for definition in CURATED:
        evidence_spans = []
        for passage_id, requested_quote in definition["evidence"]:
            if passage_id not in graph:
                raise ValueError(f"Missing passage in saved graph: {passage_id}")
            passage = graph.nodes[passage_id]
            start_base = int(passage["char_start"])
            end_base = int(passage["char_end"])
            start, quote = exact_source_match(
                source, requested_quote, start_base, end_base
            )
            end = start + len(quote)
            if source[start:end] != quote:
                raise ValueError(f"Quote/offset mismatch for {definition['id']}")
            evidence_spans.append({
                "char_start": start,
                "char_end": end,
                "quote": quote,
                "passage_id": passage_id,
            })
        curated.append({
            "id": definition["id"],
            "question": definition["question"],
            "answers": definition["answers"],
            "answerable": True,
            "question_category": definition["question_category"],
            "evidence_spans": evidence_spans,
            "split": definition["split"],
            "question_source": definition.get(
                "question_source",
                "assistant-authored from a passage not used by the legacy QA spans",
            ),
            "review_status": "source quote and offset mechanically verified; answer reviewed against cited text",
            "source_sha256": source_hash,
        })

    # Keep legacy QA only in dev. New held-out questions use passage IDs absent
    # from all legacy evidence annotations, avoiding passage-level split leakage.
    legacy_passages = set()
    for item in legacy:
        for span in item.get("evidence_spans", []):
            for node_id, data in graph.nodes(data=True):
                if data.get("type") != "T":
                    continue
                if int(data.get("char_start", -1)) < int(span["char_end"]) and int(data.get("char_end", -1)) > int(span["char_start"]):
                    legacy_passages.add(node_id)
    for item in curated:
        if item["split"] != "heldout":
            continue
        overlap = {
            span["passage_id"] for span in item["evidence_spans"]
        } & legacy_passages
        if overlap:
            raise ValueError(
                f"Held-out evidence overlaps legacy development passages in {item['id']}: {sorted(overlap)}"
            )

    dev_passages = {
        span["passage_id"]
        for item in legacy + curated
        if item["split"] == "dev"
        for span in item.get("evidence_spans", [])
        if span.get("passage_id")
    }
    heldout_passages = {
        span["passage_id"]
        for item in curated
        if item["split"] == "heldout"
        for span in item.get("evidence_spans", [])
    }
    overlap = dev_passages & heldout_passages
    if overlap:
        raise ValueError(
            "Development and held-out questions share evidence passages: "
            + ", ".join(sorted(overlap))
        )

    combined = legacy + curated
    ids = [item["id"] for item in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs detected.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        for item in combined:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(combined)} questions to {OUTPUT}")
    print(f"Development: {sum(item['split'] == 'dev' for item in combined)}")
    print(f"Held-out: {sum(item['split'] == 'heldout' for item in combined)}")
    print(f"Legacy evidence passages: {len(legacy_passages)}")


if __name__ == "__main__":
    main()
