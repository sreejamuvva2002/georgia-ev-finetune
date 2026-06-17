# v2 (7B) Diagnostic — failure map by question shape

Probe set: **36 questions** generated deterministically from the KB (never from the 50 human Q&A). Each scored against the KB-computed gold.

**Overall probe accuracy: 56%** (20/36)

## Accuracy by shape (worst first)

| shape | correct | n | acc |
|---|---|---|---|
| superlative | 1 | 4 | 25% |
| aggregation | 1 | 3 | 33% |
| count | 3 | 8 | 38% |
| multi_filter | 3 | 7 | 43% |
| distribution | 1 | 2 | 50% |
| list_enum | 3 | 4 | 75% |
| refusal | 3 | 3 | 100% |
| setop_negation | 2 | 2 | 100% |
| single_fact | 3 | 3 | 100% |

## Failing examples (concrete)

- **[count]** How many companies are classified under each Category in the Georgia EV KB?
  - expected: category breakdown + total
  - score detail: `{"expected_numbers": ["77", "73", "18", "17", "12", "5", "3", "205"], "present": ["5", "3"], "frac": 0.25, "model_count": null}`
  - model said: According to the Georgia EV KB, companies are categorized as follows: Category 1: Tier 2/3 Count: 74 Category 2: Tier 1 Count: 68 Category 3: OEM Supply Chain Count: 60 Category 4: OEM Count: 19 Category 5: OEM (Footprint) Count: 13 Total: 234 entries.…

- **[count]** How many companies are marked Yes, No, and Indirect for EV/Battery relevance?
  - expected: Yes/No/Indirect
  - score detail: `{"expected_numbers": ["41", "87", "76"], "present": ["41"], "frac": 0.33, "model_count": null}`
  - model said: According to the Georgia EV KB, company EV/battery relevance is marked as: Yes: 41 No: 48 Indirect: 73…

- **[count]** How many companies are classified as Tier 1 in the Georgia EV KB?
  - expected: Tier 1=77
  - score detail: `{"expected_numbers": ["77"], "present": [], "frac": 0.0, "model_count": "76"}`
  - model said: According to the Georgia EV KB, there are 76 companies classified as Tier 1.…

- **[count]** How many companies are classified as Tier 2/3 in the Georgia EV KB?
  - expected: Tier 2/3=73
  - score detail: `{"expected_numbers": ["73"], "present": [], "frac": 0.0, "model_count": "77"}`
  - model said: According to the Georgia EV KB, there are 77 companies classified as Tier 2/3.…

- **[count]** How many companies are classified as OEM in the Georgia EV KB?
  - expected: OEM=12
  - score detail: `{"expected_numbers": ["12"], "present": [], "frac": 0.0, "model_count": "10"}`
  - model said: According to the Georgia EV KB, there are 10 companies classified as OEM.…

- **[distribution]** How are Tier 1 Georgia companies distributed across the EV/Battery Relevance classifications (Yes, No, Indirect)?
  - expected: Tier1 across relevance: {'No': np.int64(43), 'Indirect': np.int64(33), 'Yes': np.int64(1)}
  - score detail: `{"expected_numbers": ["1", "43", "33"], "present": [], "frac": 0.0, "model_count": null}`
  - model said: According to the Georgia EV KB, Tier 1 companies are classified as follows: - Yes: 57 - No: 28 - Indirect: 39…

- **[multi_filter]** Identify Tier 1 Georgia companies classified under the Materials EV Supply Chain Role and list their products.
  - expected: Tier1 ∩ Materials
  - score detail: `{"name_f1": 0.0, "name_p": 0.0, "name_r": 0.0, "exp_n": 3, "model_count": "5", "count_ok": false, "listed": 5, "consistency": true, "hallucinated": ["Ascend Elements", "Aspen Aerogels", "Aurubis", "Beaver Manufacturing Co. Inc.", "Blue Bird Corp."], "missing": ["PHA Body Systems LLC", "Panasonic Automotive Systems Co.", "SK Battery America"]}`
  - model said: According to the Georgia EV KB, there are 5 Tier 1 companies in the Materials role: Ascend Elements | Product: Interior and exterior plastic parts Aspen Aerogels | Product: Gaskets, plastic precision moldings, oil seals and valve steam seals Aurubis | Product: Automotive interior…

- **[multi_filter]** Which Tier 2/3 Georgia companies are directly EV/battery relevant (marked 'Yes'), and what are their roles?
  - expected: Tier2/3 ∩ Yes
  - score detail: `{"name_f1": 0.5, "name_p": 0.33, "name_r": 1.0, "exp_n": 3, "model_count": "8", "count_ok": false, "listed": 9, "consistency": false, "hallucinated": ["F&P Georgia Manufacturing", "Hitachi Astemo", "Hitachi Astemo Americas Inc.", "Honda Development & Manufacturing", "Hyundai Transys Georgia Powertrain", "Hyundai Transys Georgia Seating Systems"], "missing": []}`
  - model said: According to the Georgia EV KB, there are 8 Tier 2/3 companies marked 'Yes' (directly EV/battery relevant): Duckyang | Role: General Automotive Enchem America Inc. | Role: General Automotive F&P Georgia Manufacturing | Role: Battery Pack GSC Steel Stamping LLC | Role: Power Elect…

- **[multi_filter]** Which Tier 2/3 Georgia companies are in the Materials role?
  - expected: Tier2/3 ∩ Materials
  - score detail: `{"name_f1": 0.0, "name_p": 0.0, "name_r": 0.0, "exp_n": 19, "model_count": "10", "count_ok": false, "listed": 10, "consistency": true, "hallucinated": ["Dinex Emissions Inc.", "Dongwon Autopart Technology Georgia LLC", "Dorsett Industries Inc.", "Down 2 Earth Trailers", "Eaton Corp.", "Erdrich USA Inc."], "missing": ["Ascend Elements", "Aspen Aerogels", "Aurubis", "Beaver Manufacturing Co. Inc.", "Blue Bird Corp.", "Blue Ridge Manufacturing"]}`
  - model said: According to the Georgia EV KB, there are 10 Tier 2/3 Materials companies in Georgia: Dinex Emissions Inc. | Category: Tier 2/3 | Product: Aluminum sheet for automotive bodies Dongwon Autopart Technology Georgia LLC | Category: Tier 2/3 | Product: Aluminum extrusions Dorsett Indu…

- **[multi_filter]** Which Tier 1 Georgia suppliers are linked to Hyundai or Kia through their Primary OEMs?
  - expected: Tier1 ∩ Hyundai/Kia
  - score detail: `{"name_f1": 0.13, "name_p": 0.07, "name_r": 1.0, "exp_n": 1, "model_count": "13", "count_ok": false, "listed": 14, "consistency": false, "hallucinated": ["Duckyang", "Enchem America Inc.", "F&P Georgia Manufacturing", "GSC Steel Stamping LLC", "Hitachi Astemo", "Hitachi Astemo Americas Inc."], "missing": []}`
  - model said: According to the Georgia EV KB, 13 Tier 1 companies in Georgia have Hyundai/Kia vehicles in their Primary OEMs: Duckyang Enchem America Inc. F&P Georgia Manufacturing GSC Steel Stamping LLC Hitachi Astemo Americas Inc. Hollingsworth & Vose Co. Honda Development & Manufacturing Hy…

- **[superlative]** Which company in the Georgia EV KB has the highest employment, and how many employees is that?
  - expected: global argmax employment
  - score detail: `{"expected_company": "WIKA USA", "comp_ok": false, "expected_numbers": ["250000"], "num_ok": false}`
  - model said: According to the Georgia EV KB, Toyota Industries Group (TACG-TICA) has the highest employment among supplier companies, with 18,100 employees.…

- **[superlative]** In Bryan County, which company has the highest employment, and what is its EV supply chain role?
  - expected: argmax in Bryan County
  - score detail: `{"expected_company": "Hyundai Motor Group", "comp_ok": false, "expected_numbers": ["164"], "num_ok": false}`
  - model said: According to the Georgia EV KB, TI Fluid Systems has the highest employment in Bryan County (Employment: 1900; Role: Vehicle Assembly).…

- **[superlative]** In Troup County, which company has the highest employment, and what is its EV supply chain role?
  - expected: argmax in Troup County
  - score detail: `{"expected_company": "Kia Georgia Inc.", "comp_ok": false, "expected_numbers": ["777"], "num_ok": false}`
  - model said: According to the Georgia EV KB, TI Fluid Systems has the highest employment in Troup County (Employment: 1900; Role: OEM corporate footprint influencing EV strategy).…

- **[aggregation]** What is the total combined employment across all companies in the Georgia EV KB?
  - expected: grand total emp
  - score detail: `{"expected_numbers": ["1448330"], "num_ok": false, "pred_nums": ["2437"]}`
  - model said: According to the Georgia EV KB, the combined employment across all companies is 2437.…

- **[aggregation]** What is the total employment of all Tier 1 companies in the Georgia EV KB?
  - expected: tier1 total emp
  - score detail: `{"expected_numbers": ["21690"], "num_ok": false, "pred_nums": ["837"]}`
  - model said: According to the Georgia EV KB, the combined employment across all Tier 1 companies is 837.…

- **[list_enum]** List every Georgia company with the EV Supply Chain Role 'Materials'.
  - expected: role=Materials
  - score detail: `{"name_f1": 0.67, "name_p": 1.0, "name_r": 0.5, "exp_n": 22, "model_count": "12", "count_ok": false, "listed": 11, "consistency": false, "hallucinated": [], "missing": ["Bosch (Automotive Division)", "Bridgestone Bandag", "DAS Corp.", "EVCO Plastics", "Fanello Industries Inc.", "First American Resources"]}`
  - model said: According to the Georgia EV KB, 12 companies have the role 'Materials': Ascend Elements | Category: Tier 2/3 | Primary OEMs: Multiple OEMs | Employment: 456 Aspen Aerogels | Category: Tier 2/3 | Primary OEMs: Multiple OEMs | Employment: 414 Aurubis | Category: Tier 2/3 | Primary …

