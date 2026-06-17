# Dataset Report (v4)

- KB source: `/Users/surya/Downloads/GNEM - Auto Landscape Lat Long Updated (1).xlsx` (205 rows, 191 unique companies)
- Convention: **unique companies** everywhere; RoleNorm buckets: ['General Automotive', 'Materials', 'Vehicle Assembly', 'Battery Cell', 'Battery Pack', 'Thermal Management', 'Power Electronics', 'Charging Infrastructure', 'Wiring Harness', 'OEM Corporate Footprint']
- Held-out (never trained): 50 human Q&A + probe benchmark
- **Leakage guard:** 40 generated questions dropped (exact or >=0.85 Jaccard).

| split | examples |
|---|---|
| train | 5635 |
| valid | 248 |
| test | 50 |

- refusal examples in train: 192  (3.4%)
- none_match examples in train: 6

## Examples by generator tag (post-oversample)

- list_county: 813
- list_city: 471
- kw_search: 330
- abstain_county: 292
- list_role: 231
- list_oem: 222
- cross_cat_role: 210
- refusal: 192
- company_summary: 178
- company_facility: 177
- company_location: 177
- company_employment: 175
- company_role: 175
- company_oems: 174
- company_products: 173
- company_category: 173
- abstain_cross: 148
- list_industry: 144
- list_category: 120
- cross_cat_relevance: 117
- agg_county_argmax: 99
- agg_county_total_emp: 96
- cross_role_relevance: 93
- list_exact_role: 84
- agg_role_count: 84
- cross_3way: 69
- cross_cat_oem: 66
- agg_category_count: 54
- list_relevance: 45
- kw_none: 33
- list_multi_oem: 27
- company_multientry: 22
- list_facility: 21
- agg_cat_total_emp: 18
- agg_topk_emp: 12
- agg_category_dist: 12
- cross_emp_big_indirect: 9
- cross_emp_t23ga300: 9
- agg_gap_counties: 9
- agg_relevance_dist: 9
- agg_argmax_emp: 9
- agg_single_source: 9
- agg_top_emp_county: 9
- agg_oem_footprint: 9
- agg_total_emp: 6
- agg_role_dist: 6
- none_match: 6
- agg_oem_affiliation: 6
- agg_top_emp_county_t1: 6
- cross_emp_tmpe200: 6
