# 14B (MLX) Evaluation — held-out human Q&A + KB diagnostic

## A. 50 held-out human-validated Q&A (structured score >= 0.6)

| model | accuracy | mean score |
|---|---|---|
| Base 7B (Qwen2.5-Coder) | 4% (2/50) | 0.201 |
| Fine-tuned v2 (7B LoRA) | 54% (27/50) | 0.584 |
| **Fine-tuned 14B (MLX LoRA)** | 46% (23/50) | 0.605 |

### By use-case category (accuracy)

| category | n | v2 7B | 14B |
|---|---|---|---|
| Product & Technology Trends | 11 | 5/11 | 4/11 |
| Site Selection & Expansion Planning | 5 | 4/5 | 2/5 |
| Supplier Discovery & Matchmaking | 11 | 5/11 | 4/11 |
| Supply Chain Mapping & Visibility | 16 | 8/16 | 10/16 |
| Supply Chain Risk & Resilience | 7 | 5/7 | 3/7 |

## B. KB diagnostic probe set (36 Q, by shape)

**Overall: v2 20/36 → 14B 21/36**

| shape | n | v2 7B | 14B |
|---|---|---|---|
| distribution | 2 | 1 | 0 |
| refusal | 3 | 3 | 1 |
| setop_negation | 2 | 2 | 1 |
| aggregation | 3 | 1 | 2 |
| list_enum | 4 | 3 | 2 |
| count | 8 | 3 | 3 |
| single_fact | 3 | 3 | 3 |
| superlative | 4 | 1 | 4 |
| multi_filter | 7 | 3 | 5 |

## C. Remaining 14B failures on the probe set

- **[count]** How many companies are classified under each Category in the Georgia EV KB?  
  detail: `{"expected_numbers": ["77", "73", "18", "17", "12", "5", "3", "205"], "present": ["18", "5"], "frac": 0.25, "model_count": "193"}`
- **[count]** How many companies are marked Yes, No, and Indirect for EV/Battery relevance?  
  detail: `{"expected_numbers": ["41", "87", "76"], "present": [], "frac": 0.0, "model_count": null}`
- **[count]** How many companies are classified as Tier 1 in the Georgia EV KB?  
  detail: `{"expected_numbers": ["77"], "present": [], "frac": 0.0, "model_count": "71"}`
- **[count]** How many companies are classified as Tier 2/3 in the Georgia EV KB?  
  detail: `{"expected_numbers": ["73"], "present": [], "frac": 0.0, "model_count": "70"}`
- **[count]** How many companies are classified as OEM in the Georgia EV KB?  
  detail: `{"expected_numbers": ["12"], "present": [], "frac": 0.0, "model_count": "11"}`
- **[distribution]** How are Tier 1 Georgia companies distributed across the EV/Battery Relevance classifications (Yes, No, Indirect)?  
  detail: `{"expected_numbers": ["1", "43", "33"], "present": ["1"], "frac": 0.33, "model_count": "71"}`
- **[distribution]** Break down the Vehicle Assembly companies in Georgia by their Primary OEM.  
  detail: `{"name_f1": 0.45, "name_p": 0.29, "name_r": 1.0, "exp_n": 10, "model_count": null, "count_ok": false, "listed": 34, "consistency": null, "hallucinated": ["Archer Aviation Inc.", "Blue Bird Corp.", "Club Car LLC", "Hyundai Motor Group", "Mercedes-Benz USA LLC", "Rivian Automotive"], "missing": []}`
- **[multi_filter]** Which Georgia Thermal Management companies have fewer than 200 employees?  
  detail: `{"name_f1": 0.57, "name_p": 0.5, "name_r": 0.67, "exp_n": 3, "model_count": "3", "count_ok": true, "listed": 4, "consistency": false, "hallucinated": ["Hyundai Transys Georgia Seating Systems", "Novelis Inc."], "missing": ["Peerless-Winsmith Inc."]}`
- **[multi_filter]** Which Tier 2/3 Georgia companies are in the Materials role?  
  detail: `{"name_f1": 0.54, "name_p": 1.0, "name_r": 0.37, "exp_n": 19, "model_count": "7", "count_ok": false, "listed": 7, "consistency": true, "hallucinated": [], "missing": ["Bosch (Automotive Division)", "Bridgestone Bandag", "Club Car LLC", "Constellium Automotive USA", "DAEHAN Solution Georgia LLC", "DAS Corp."]}`
- **[aggregation]** What is the total employment of all Tier 1 companies in the Georgia EV KB?  
  detail: `{"expected_numbers": ["21690"], "num_ok": false, "pred_nums": ["27930"]}`
- **[setop_negation]** Which EV supply chain roles in Georgia are served by only a single company (single-point-of-failure risk)?  
  detail: `{"expected_numbers": ["27"], "present": [], "frac": 0.0, "model_count": null}`
- **[list_enum]** List every Georgia company with the EV Supply Chain Role 'Materials'.  
  detail: `{"name_f1": 0.84, "name_p": 1.0, "name_r": 0.73, "exp_n": 22, "model_count": "18", "count_ok": false, "listed": 16, "consistency": false, "hallucinated": [], "missing": ["Bosch (Automotive Division)", "Haering Precision USA LP", "PHA Body Systems LLC", "Panasonic Automotive Systems Co.", "SK Battery America", "SungEel Recycling Park Georgia"]}`
- **[list_enum]** List every Georgia company with the EV Supply Chain Role 'Thermal Management'.  
  detail: `{"name_f1": 0.8, "name_p": 0.67, "name_r": 1.0, "exp_n": 4, "model_count": "7", "count_ok": false, "listed": 6, "consistency": false, "hallucinated": ["Hyundai Transys Georgia Seating Systems", "ZF Gainesville LLC"], "missing": []}`
- **[refusal]** Which Georgia companies manufacture solid-state batteries?  
  detail: `{"declined": false, "named_companies": 1}`
- **[refusal]** List the EV supply chain companies located in Florida.  
  detail: `{"declined": false, "named_companies": 2}`
