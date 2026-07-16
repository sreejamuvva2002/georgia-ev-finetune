# GNEM-Bench-v1 — system comparison

Sections reported **independently** (no combined score). Accuracy = primary per-question correctness (list Qs: F1≥0.5; value Qs: exact); [..] = 95% Wilson CI.

## GNEM-KB-42 — structured-KB reasoning & retention

| System | KB-42 | Deterministic | Judgment |
|---|---|---|---|
| Base | 5.0% [1–16] | 6.5% | 0.0% |
| KB-only | 2.5% [0–13] | 3.2% | 0.0% |
| KB+web | 7.5% [3–20] | 9.7% | 0.0% |

### KB-42 by category

| Category | Base | KB-only | KB+web |
|---|---|---|---|
| count | 0.0% | 0.0% | 0.0% |
| direct_lookup | 100.0% | 50.0% | 100.0% |
| geographic_aggregation | 0.0% | 0.0% | 50.0% |
| judgment | 0.0% | 0.0% | 0.0% |
| max_min | 0.0% | 0.0% | 0.0% |
| multi_filter_list | 0.0% | 0.0% | 0.0% |
| top_k_ranking | 0.0% | 0.0% | 0.0% |

### KB-42 by operation count (complexity)

| #ops | Base | KB-only | KB+web |
|---|---|---|---|
| 1 | 14.6% | 4.9% | 21.9% |
| 2 | 0.0% | 0.0% | 0.0% |
| 3 | 0.0% | 0.0% | 0.0% |
| 4 | 0.0% | 0.0% | 100.0% |

## GNEM-Web-18 — training-corpus web-fact absorption

| System | Web-18 | Δ vs Base | Δ vs KB-only |
|---|---|---|---|
| Base | 22.2% [9–45] | +0.0 | +16.7 |
| KB-only | 5.6% [1–26] | -16.7 | +0.0 |
| KB+web | 38.9% [20–61] | +16.7 | +33.3 |

## Per-question detail

### kb_q01 (multi_filter_list/deterministic)
**Gold:** There are 18 Tier 1/2 companies in Georgia.
F&P Georgia Manufacturing [Tier 1/2] | Role: Battery Pack | Product: Lithium-ion battery recycler and raw materials provider
Fouts Brothers Fire Equipment [Tier 1/2] | Role: General Automotive | Product: Automotive body parts and electronics Cylinder-head 
- **Base** ✗: 'Sort by EV Supply Chain Role. Show only the first 10 results.\nHere is a list of the first 10 Tier 1/2 suppliers in Georgia, sorted by their EV Supply Chain Role and Product / Service:\n\n| Company Name | EV Supply Chain Ro'
- **KB-only** ✗: 'Sort by Company.\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>'
- **KB+web** ✗: '## Facts - Georgia has 100 Tier 1/2 suppliers in the EV supply chain. - Suppliers are categorized by their role in the supply chain: Battery Pack, Battery Cell, General Automotive, or Materials. - SK Battery America is l'
- **RAG** ·: _pending_

### kb_q02 (multi_filter_list/deterministic)
**Gold:** There are 6 companies found with Battery Cell or Battery Pack roles.    

F&P Georgia Manufacturing [Tier 1/2] | Role: Battery Pack 
Hitachi Astemo Americas Inc. [Tier 1/2] | Role: Battery Cell 
Hollingsworth & Vose Co. [Tier 1/2] | Role: Battery Pack 
Honda Development & Manufacturing [Tier 1/2] | 
- **Base** ✗: 'Please provide the company name, role, and tier in the following format: Company Name, Role, Tier. Here is a list of Georgia-based companies that are classified under Battery Cell or Battery Pack roles, along with their '
- **KB-only** ✗: 'Battery cells are the heart of electric mobility, and companies like Hyundai Kia are leading the pack.\nCompany: Lark United Manufacturing Inc.\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Dublin, J'
- **KB+web** ✗: '- Ascend Elements is a Tier 2/3 company located in Covington, GA. - Ascend Elements is classified under Battery Cell and Battery Pack roles. - Ascend Elements is a battery recycling company that processes spent lithium-i'
- **RAG** ·: _pending_

### kb_q03 (multi_filter_list/deterministic)
**Gold:** There are 5 Thermal Management companies in Georgia.
ZF Gainesville LLC | OEMs: Not specified
Freudenberg-NOK | OEMs: Multiple OEMs
Hyundai Transys Georgia Powertrain | OEMs: Hyundai Kia Rivian
Novelis Inc. | OEMs: Multiple OEMs
Peerless-Winsmith Inc. | OEMs: Multiple OEMs
- **Base** ✗: 'The map should include the names of the suppliers and the Primary OEMs they are linked to. Additionally, the map should be color-coded to indicate the number of Primary OEMs each supplier is linked to. The map should als'
- **KB-only** ✗: 'Indicate which suppliers are automotive supply chain participants.\nSupplier or Affiliation Type: Automotive supply chain participant\nEmployment: 170\nProduct or Service: Stamped blanks Flexible metal hoses for exhaust sys'
- **KB+web** ✗: '- Map all Thermal Management suppliers in Georgia and show which Tier 1 suppliers they are linked to. - Map all Thermal Management suppliers in Georgia and show which Tier 2/3 suppliers they are linked to. - Map all Ther'
- **RAG** ·: _pending_

### kb_q04 (multi_filter_list/deterministic)
**Gold:** There are 4 companies classified under Power Electronics or Charging Infrastructure:
GSC Steel Stamping LLC | Role: Power Electronics | Employment: 350
Morgan Corp. | Role: Charging Infrastructure | Employment: 320
Yazaki North America | Role: Power electronics, sensors, and EV systems | Employment:
- **Base** ✗: 'The list should be sorted by the number of employees in descending order. Additionally, the list should only include companies that have been in operation for at least 5 years and have a minimum of 10 employees. The outp'
- **KB-only** ✗: 'This list is part of the 2021 Electric Vehicle Supply Chain Report.\nRow ID: 123\nCompany: NIFCO KTW America Corp.\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Toccoa, Franklin County\nAddress: 100 Ni'
- **KB+web** ✗: '- The page lists 10 companies in Georgia under Power Electronics or Charging Infrastructure. - The page lists 10 companies in Georgia under Battery Cell. - The page lists 10 companies in Georgia under Battery Pack. - The'
- **RAG** ·: _pending_

### kb_q05 (multi_filter_list/deterministic)
**Gold:** Original Equipment Manufacturer companies (11):
Kia Georgia Inc. | EV Supply Chain Role: Vehicle Assembly
Minebea AccessSolutions USA Inc. | EV Supply Chain Role: General Automotive
Superior Essex Inc. | EV Supply Chain Role: Vehicle Assembly
Suzuki Manufacturing of America Corp. | EV Supply Chain R
- **Base** ✗: "Please provide a list of companies and their corresponding EV Supply Chain Roles.\nI'm sorry, but I cannot provide a list of companies and their corresponding EV Supply Chain Roles as it is beyond the scope of my training"
- **KB-only** ✗: 'To answer these questions, we will analyze the companies listed in the table below:\n\nCompany: Haering Precision USA LP\nCategory: Tier 2/3\nIndustry Group: Machinery Except Electrical\nLocation: Buford, Hall County\nAddress:'
- **KB+web** ✗: '- Aspen Aerogels is classified as a Direct Manufacturer and covers the Materials role. - Aspen Aerogels is classified as a Direct Manufacturer and covers the Thermal Management role. - Aspen Aerogels is classified as a D'
- **RAG** ·: _pending_

### kb_q06 (direct_lookup/deterministic)
**Gold:** Novelis Inc. has three operating entries in Georgia, and all three are located in Atlanta, Fulton County, with each one classified as a Manufacturing Plant as its primary facility type.
- **Base** ✓: 'Answer according to: Novelis Inc. is a global leader in aluminum rolled products and the world’s largest recycler of aluminum. The company operates in 11 countries, has 14 rolling mills in 9 countries, 7 can recycling fa'
- **KB-only** ✓: 'Additionally, what EV Supply Chain Role does each location play?\nCompany: Novelis Inc.\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Atlanta, Fulton County\nAddress: 3560 Lenox Rd NE, Suite 2000, Atl'
- **KB+web** ✓: '- Novelis Inc. operates in 10 locations across the United States, Canada, and the United Kingdom. - The primary facility types associated with each location include Manufacturing Plant, R&D Center, and Headquarters. - No'
- **RAG** ·: _pending_

### kb_q07 (max_min/deterministic)
**Gold:** WIKA USA has the highest employment in Gwinnett County (Employment: 250000; Role: HV and LV wiring harnesses for EVs and ICE vehicles).
- **Base** ✗: 'Answer according to: The Gwinnett County, GA market analytics report provides key economic, demographic and business statistics and trends,Showing current data and forecasts for 2018 through 2023, including industry over'
- **KB-only** ✗: 'Company: ZF Gainesville LLC\nCategory: OEM Supply Chain\nIndustry Group: Automotive systems and components\nLocation: Gainesville, Hall County\nAddress: 1261 Palmour Dr SW, Gainesville, GA 30501\nLatitude: 34.2647050791808\nLo'
- **KB+web** ✗: '- In Gwinnett County, Hyundai Mobis has the highest Employment with 1,700 employees. - Hyundai Mobis is a Tier 1 supplier for Hyundai Motor Group Metaplant America (HMGMA). - Hyundai Mobis manufactures electric powertrai'
- **RAG** ·: _pending_

### kb_q08 (geographic_aggregation/deterministic)
**Gold:** Troup County has the highest total Employment among Tier 1 suppliers with a total of 2,435 employees
- **Base** ✗: 'To determine which county has the highest total employment among Tier 1 suppliers, we need to follow these steps:\n\n1. **Identify Tier 1 Suppliers**: First, we need to identify the Tier 1 suppliers. These are typically th'
- **KB-only** ✗: 'Supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 400\nProduct or Service: All terrain vehicles\nEV or Battery Relevant: Indirect\nClassification Method: Supplier\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n'
- **KB+web** ✓: '- The county with the highest total employment among Tier 1 suppliers only is Troup County with 1,700 employees. - Troup County is located in the Central region of Georgia. - Troup County is home to the Kia Georgia plant'
- **RAG** ·: _pending_

### kb_q09 (geographic_aggregation/deterministic)
**Gold:** Hall County has the highest total employment across all companies: 314452.
- **Base** ✗: 'To determine which county has the highest total employment across all companies and what the combined employment in that county is, we need to follow these steps:\n\n1. Identify the total employment for each county.\n2. Com'
- **KB-only** ✗: 'To determine the county with the highest total employment across all companies, we need to look at the combined employment numbers. Here are the top 10 counties:\n\n1. Fulton County: 1000+ employees\n2. DeKalb County: 800+ '
- **KB+web** ✗: '## Facts - The top county by total employment is Fulton County with 1,000,000 employees. - The second highest county by total employment is Gwinnett County with 500,000 employees. - The third highest county by total empl'
- **RAG** ·: _pending_

### kb_q10 (multi_filter_list/deterministic)
**Gold:** There are 10 Vehicle Assembly OEMs along with their Primary OEMs are:
Kia Georgia Inc. | Primary OEMs: Club Car LLC
Superior Essex Inc. | Primary OEMs: Hyundai Motor Group
Suzuki Manufacturing of America Corp. | Primary OEMs: Rivian Automotive
TCI Powder Coatings | Primary OEMs: Kia Georgia Inc.
TDK
- **Base** ✗: 'Additionally, provide the total number of vehicles produced by each facility in the past year and the percentage of vehicles produced by each Primary OEM at each facility. Finally, calculate the average production rate p'
- **KB-only** ✗: 'Also, provide the address and Latitude/Longitude for each location.\nRow ID: 12\nCompany: Aspen Aerogels\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Statesboro, Bulloch County\nAdd'
- **KB+web** ✗: '- Identify all Vehicle Assembly facilities in Georgia and list the corresponding Primary OEM associated with each facility. - Identify all Vehicle Assembly facilities in Georgia and list the corresponding Primary OEM ass'
- **RAG** ·: _pending_

### kb_q11 (direct_lookup/deterministic)
**Gold:** Sewon America Inc. has three operating entries in Georgia and all three are located in LaGrange, Troup County, Georgia.
Sewon America Inc. | Product: Motor vehicle engines and parts
Sewon America Inc. | Product: Fire truck bodies
Sewon America Inc. | Product: Automotive aftermarket parts
- **Base** ✓: 'Sewon America Inc. is a company that specializes in the production and distribution of various products and services. Here are the primary products and services associated with Sewon America Inc. across its different ope'
- **KB-only** ✗: 'Include the LaGrange, Troup County locations.\nPrimary Facility Type: Manufacturing Plant\nEV Supply Chain Role: General Automotive\nPrimary OEMs: Multiple OEMs\nSupplier or Affiliation Type: Automotive supply chain particip'
- **KB+web** ✓: '- Sewon America Inc. operates a facility located at 100 Sewon Dr, LaGrange, GA 30240. - The LaGrange, GA facility primarily manufactures automotive body parts and components. - The LaGrange, GA facility also manufactures'
- **RAG** ·: _pending_

### kb_q12 (multi_filter_list/deterministic)
**Gold:** There are 3 companies with primary involvement in the electric vehicle or battery supply chain. 

Duckyang | Role: General Automotive 
GSC Steel Stamping LLC | Role: Power Electronics 
Enchem America Inc. | Role: General Automotive
- **Base** ✗: "Additionally, provide a detailed analysis of the potential impact of these companies on the state's economy, including job creation, investment, and technological innovation. Finally, compare and contrast the strategies "
- **KB-only** ✗: 'Include company names, categories, and locations.\nCompany: Flambeau Inc.\nCategory: Tier 2/3\nIndustry Group: Fabricated Metal Products\nLocation: Madison, Morgan County\nAddress: 1330 Atlanta Highway, Madison, GA 30650\nLati'
- **KB+web** ✗: '- The list includes companies like Ascend Elements, Aurubis AG, Aurubis Richmond, Inc., Battery Resourcers, Inc., Blue Bird Corporation, Bonnell Aluminum, Carcoustics USA, Celgard, LLC, Ceres Nanosciences, Inc., Cirba So'
- **RAG** ·: _pending_

### kb_q13 (multi_filter_list/deterministic)
**Gold:** Rivian-linked suppliers in GA (by Primary OEMs column):
Duckyang [Tier 2/3] | Role: General Automotive | Primary OEMs: Hyundai Kia Rivian
Enchem America Inc. [Tier 2/3] | Role: General Automotive | Primary OEMs: Hyundai Kia Rivian
GSC Steel Stamping LLC [Tier 2/3] | Role: Power Electronics | Primary
- **Base** ✗: 'Include the following information for each supplier: Company Name, Country, Tier, EV Supply Chain Role, and Website. The EV Supply Chain Role should be categorized into the following: Battery, Battery Cells, Battery Mana'
- **KB-only** ✗: 'The network includes OEMs, suppliers, and employment figures.\nRow ID: 173\nCompany: TCI Powder Coatings\nCategory: OEM\nIndustry Group: Transportation Equipment\nLocation: Ellaville, Schley County\nAddress: 610 Dixon Dr, Ella'
- **KB+web** ✗: '- Rivian Automotive is a Tier 1 supplier to itself. - Rivian Automotive is a Tier 1 supplier to Hyundai Motor Group. - Rivian Automotive is a Tier 1 supplier to Kia Georgia. - Rivian Automotive is a Tier 1 supplier to Me'
- **RAG** ·: _pending_

### kb_q14 (multi_filter_list/deterministic)
**Gold:** There are 2 Georgia companies with an EV Supply Chain Role related to wiring harnesses:
WIKA USA | Primary OEMs: n/a
Woodbridge Foam Corp. | Primary OEMs: n/a
- **Base** ✗: 'The EV Supply Chain Role is defined as a company that is involved in the production of EV components, such as batteries, motors, and charging systems, or in the distribution of these components to EV manufacturers. The P'
- **KB-only** ✗: 'Also, list the supplier or Affiliation Type: Automotive supply chain participant.\nEmployment: 165700\nProduct or Service: Vehicle power & data solutions; wiring harnesses and connectors\nEV or Battery Relevant: Indirect\nCl'
- **KB+web** ✗: "## Links - https://www.georgia.org/georgia-electric-vehicle-ecosystem — Georgia EV ecosystem information\n## Facts - Georgia is home to more than 500 automotive-related facilities. - Georgia's automotive industry employs "
- **RAG** ·: _pending_

### kb_q15 (judgment/judgment)
**Gold:** There are no Georgia Tier 2/3 companies listed under the Electronic and Other Electrical Equipment and Components industry group. So none can be upgraded to supply EV power electronics based on the provided evidence.
- **Base** ·: 'The companies should have a minimum of 5 years of experience in the industry and a minimum of 10 employees. The companies should also have a minimum of 50% of their revenue coming from the Electronic and Electrical Equip'
- **KB-only** ·: 'List the companies, their locations, and the primary Facility Type: Manufacturing Plant.\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n<'
- **KB+web** ·: '- Identify Georgia Tier 2/3 companies in the Electronic and Other Electrical Equipment and Components industry group that could be upgraded to supply EV power electronics. - Identify Georgia Tier 2/3 companies in the Ele'
- **RAG** ·: _pending_

### kb_q16 (judgment/judgment)
**Gold:** There is 1 company in Georgia specifically identified for the manufacture of materials suitable for EV battery current collectors:
Duckyang | Produce: High-quality electrodeposited (ED) copper foil for electric vehicles
- **Base** ✗: 'Provide a list of at least 5 companies, including their contact information and a brief description of their products. Additionally, provide a comparison of the different types of copper foil and electrodeposited materia'
- **KB-only** ✗: 'These companies are part of the supply chain for electric mobility.\nCompany: Flambeau Inc.\nCategory: Tier 2/3\nIndustry Group: Fabricated Metal Products\nLocation: Madison, Morgan County\nAddress: 1330 Atlanta Highway, Madi'
- **KB+web** ✗: '- The search should include companies that produce copper foil for printed circuit board (PCB) applications. - The search should include companies that produce copper foil for lithium-ion battery applications. - The sear'
- **RAG** ·: _pending_

### kb_q17 (judgment/judgment)
**Gold:** No Georgia Tier 1/2 companies are explicitly identified as producing engineered plastics, polymers, or composite materials applicable to EV structural or thermal components.
- **Base** ·: "Please provide a list of companies and their specific products or services. Additionally, can you provide any information on the companies' research and development efforts in this area? Please include any relevant paten"
- **KB-only** ·: 'If so, what are the specific OEMs and supplier or Affiliation Type: Automotive supply chain participant?\nEmployment: 200\nProduct or Service: Automotive plastic parts\nEV or Battery Relevant: Indirect\nClassification Method'
- **KB+web** ·: '- The Georgia Department of Economic Development (GDEcD) maintains a list of Tier 1/2 companies in Georgia. - The list includes companies that produce engineered plastics, polymers, or composite materials applicable to E'
- **RAG** ·: _pending_

### kb_q18 (multi_filter_list/deterministic)
**Gold:** There are 5 Georgia companies producing DC-to-DC converters, capacitors, or power electronics components relevant to EV drivetrains. 
Yazaki North America [OEM Footprint]
ZF Gainesville LLC [OEM Supply Chain]
GSC Steel Stamping LLC [Tier 2/3] 
Hyundai MOBIS (Georgia) [Tier 1/2] 
Hyundai Transys Geor
- **Base** ✗: 'Georgia is home to several companies that produce DC-to-DC converters, capacitors, and power electronics components. Here are some notable companies and their respective tiers:\n\n1. **Georgia Power** - This is a subsidiar'
- **KB-only** ✗: '1: OEM; 2: Tier 1; 3: Automotive supply chain participant\nCompany: Continental Automotive\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Fairburn, Fayette County\nAddress: 7310 Oakl'
- **KB+web** ✗: '- Georgia is home to a diverse range of companies producing DC-to-DC converters, capacitors, and power electronics components. - These companies are categorized into Tier 1, Tier 2/3, and Tier 4/5 based on their role in '
- **RAG** ·: _pending_

### kb_q19 (multi_filter_list/deterministic)
**Gold:** There is only 1 Georgia company that provides Powder coating-related products/services:
Archer Aviation Inc. [Tier 2/3] | Produces: Powder coatings for automotive and other industries
- **Base** ✗: 'Answer according to: The Powder Coating Institute (PCI) is pleased to announce the release of the 2019 PCI Powder Coating Directory. The directory is a comprehensive listing of powder coating-related products and service'
- **KB-only** ✗: 'Company: DAS Corp.\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Metter, Emanuel County\nAddress: 730 Fortner Rd, Metter, GA 30439\nLatitude: 32.3838336\nLongitude: -82.0753462\nPrima'
- **KB+web** ✗: '- The Georgia Powder Coating Alliance (GPCA) is a non-profit organization that promotes the use of powder coating in the state of Georgia. - The GPCA maintains a list of member companies that provide powder coating-relat'
- **RAG** ·: _pending_

### kb_q20 (multi_filter_list/deterministic)
**Gold:** There are 6 Georgia companies classified under Battery Cell/Battery Pack roles or battery-related products and Tier 1/2, making them ready for direct OEM engagement. 

F&P Georgia Manufacturing | OEMs: Hyundai Kia 
Hitachi Astemo Americas Inc. | OEMs: Hyundai Kia 
Hollingsworth & Vose Co. | OEMs: Hy
- **Base** ✗: 'Please provide a list of companies that are classified under Battery Cell/Battery Pack roles or battery-related products and are Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are '
- **KB-only** ✗: 'The supplier or Affiliation Type: Automotive supply chain participant.\nEmployment: 400\nProduct or Service: Battery cells for electric mobility\nEV or Battery Relevant: Yes\nClassification Method: Supplier\n</record>\n>\n>\n</>'
- **KB+web** ✗: '- The table lists companies by role, location, and primary OEMs. - The table includes columns for Battery Cell/Battery Pack, Battery Pack Assembly, Battery Pack Materials, Battery Pack Testing, Battery Pack Recycling, Ba'
- **RAG** ·: _pending_

### kb_q21 (judgment/judgment)
**Gold:** There are 5 Tier 2/3 Georgia-based suppliers with employment over 300 that are classified as General Automotive but produce components transferable to EV platforms. 

Arising Industries Inc. | Employment: 582 | EV Relevant: Indirect
Dinex Emissions Inc. | Employment: 500 | EV Relevant: Indirect
ACM 
- **Base** ✗: 'The suppliers must have a minimum of 5 years of experience in the industry and have a history of successful partnerships with major automotive manufacturers. Additionally, the suppliers must have a strong focus on sustai'
- **KB-only** ✗: 'These suppliers are part of the automotive supply chain and play a role in the transition to electric vehicles.\nSupplier or Affiliation Type: Automotive supply chain participant\nEmployment: 350\nProduct or Service: Automa'
- **KB+web** ✗: '- Suppliers must be located in Georgia and have employment over 300. - Suppliers must be classified as General Automotive but produce components transferable to EV platforms. - Suppliers must be located in the United Sta'
- **RAG** ·: _pending_

### kb_q22 (multi_filter_list/deterministic)
**Gold:** There are 2 Georgia Tier 2/3 companies in Chemicals and Allied Products industry group.

Archer Aviation Inc. | Product: Powder coatings for automotive and other industries 
Arising Industries Inc. | Product: Rubber powders for tires
- **Base** ✗: 'Answer according to: The Chemicals and Allied Products industry group includes companies that manufacture chemicals and chemical products. The industry group includes companies that manufacture chemicals and chemical pro'
- **KB-only** ✗: 'Sort the companies by location.\nCompany: Fanello Industries Inc.\nCategory: Tier 2/3\nIndustry Group: Primary Metal Industries\nLocation: Lavonia, Franklin County\nAddress: 50 E Main Street, Lavonia, GA 30553\nLatitude: 34.42'
- **KB+web** ✗: '- Identify Georgia Tier 2/3 companies in the Rubber and Miscellaneous Plastic Products industry group and list their products. - Identify Georgia Tier 2/3 companies in the Primary Metal Industries industry group and list'
- **RAG** ·: _pending_

### kb_q23 (count/deterministic)
**Gold:** There are 28 Supply Chain Roles in Georgia served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem  

Role: Advanced electrical architecture for EVs | Only company: Yamaha Motor Manufacturing Corp.
Role: Body and chassis components for EV and ICE OEMs | 
- **Base** ✗: 'The EV supply chain in Georgia is served by multiple companies, and there is no single company that serves all the roles in the supply chain. However, there are some roles that are served by only a few companies, which c'
- **KB-only** ✗: 'If so, which company is it?\nYes, there is a single company that serves multiple roles in the EV supply chain. This company is not specified.\n</record>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>'
- **KB+web** ✗: "- The state of Georgia has attracted a significant number of electric vehicle (EV) supply chain companies, including battery manufacturers, EV component suppliers, and EV recyclers. - The state's EV ecosystem is heavily "
- **RAG** ·: _pending_

### kb_q24 (judgment/judgment)
**Gold:** There are 6 Georgia Battery Cell or Battery Pack suppliers that are sole-sourced by a specific OEM, indicating high dependency risk.
  • F&P Georgia Manufacturing | Role: Battery Pack | Primary OEM: Hyundai Kia
  • Hitachi Astemo Americas Inc. | Role: Battery Cell | Primary OEM: Hyundai Kia
  • Holl
- **Base** ✗: 'To determine which Georgia Battery Cell or Battery Pack suppliers are sole-sourced by a specific OEM, indicating high dependency risk, you would need to follow these steps:\n\n1. Identify the specific OEM (Original Equipme'
- **KB-only** ✗: 'Yes\nRow ID: 123\nCompany: NIFCO KTW America Corp.\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Toccoa, Franklin County\nAddress: 100 Nifco Way, Toccoa, GA 30577\nLatitude: 34.5179964289089\nLongitude: '
- **KB+web** ✗: '- Ascend Elements - Aurubis - Aurubis Richmond - Aurubis Richmond Recycling - Aurubis Richmond Recycling (Georgia) - Aurubis Richmond Recycling (Georgia) - Aurubis Richmond Recycling (Georgia) - Aurubis Richmond Recyclin'
- **RAG** ·: _pending_

### kb_q25 (count/deterministic)
**Gold:** For Hyundai Metaplant, there are 6 Georgia-based EV component suppliers with less than 200 employees, flagging potential capacity fragility.
Enchem America Inc. | Employment: 155
F&P Georgia Manufacturing | Employment: 104
Hyundai Motor Group | Employment: 164
Hyundai Transys Georgia Powertrain | Em
- **Base** ✗: "To determine how many of Hyundai Metaplant's Georgia-based EV component suppliers have fewer than 200 employees, we need to follow these steps:\n\n1. **Identify the total number of EV component suppliers in Georgia**: Let'"
- **KB-only** ✗: 'What is the total number of employees in the supplier or Affiliation Type: Automotive supply chain participant\nProduct or Service: Automotive components\nEV or Battery Relevant: Indirect\nClassification Method: Supplier\n</'
- **KB+web** ✗: "- What is the average annual wage for Hyundai Metaplant's suppliers compared to the state average? - How many Hyundai Metaplant suppliers are located in counties outside the Savannah MSA, and what does this imply for reg"
- **RAG** ·: _pending_

### kb_q26 (multi_filter_list/deterministic)
**Gold:** There are 2 Georgia Tier 2/3 suppliers that are EV Relevant and classified as General Automotive:
Duckyang | OEMs: Hyundai Kia Rivian
Enchem America Inc. | OEMs: Hyundai Kia Rivian
- **Base** ✗: 'The list should include the following information: Company Name, Company Website, Company Description, Company Address, Company Phone Number, Company Email, Company Contact Name, Company Contact Title, Company Contact Ph'
- **KB-only** ✗: 'List: 139\nCompany: Peerless-Winsmith Inc.\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Flowery Branch, Hall County\nAddress: 4510 Cantrell Road, Flowery Branch, GA 30542\nLatitude: 34.1933557356756\nL'
- **KB+web** ✗: '- The supplier list includes companies like Adient, which is a Tier 1 supplier of automotive seating systems. - The supplier list includes companies like Arising Industries, which is a Tier 2/3 supplier of specialty chem'
- **RAG** ·: _pending_

### kb_q27 (multi_filter_list/deterministic)
**Gold:** There are 11 Georgia-based Tier 1/2 automotive suppliers that maintain a diversified customer base.  
Fouts Brothers Fire Equipment 
Hitachi Astemo
Hwashin
Hyundai & LG Energy Solution (LGES)
IMS Gear Georgia Inc.
JAC Products Inc.
Hyundai Industrial Co.
Hyundai MOBIS (Georgia)
Hyundai Transys Georg
- **Base** ✗: 'Additionally, provide a comprehensive analysis of their financial performance, including revenue, profit margins, and market share. Furthermore, evaluate their supply chain management strategies, including their ability '
- **KB-only** ✗: 'Company: Continental Automotive\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Fairburn, Fayette County\nAddress: 7310 Oakley Industrial Blvd, Fairburn, GA 30213\nLatitude: 33.549612'
- **KB+web** ✗: '- Include suppliers with a focus on battery components, charging infrastructure, and energy storage systems. - Include suppliers with a focus on electric vehicle (EV) thermal management solutions. - Include suppliers wit'
- **RAG** ·: _pending_

### kb_q28 (multi_filter_list/deterministic)
**Gold:** There are 3 Georgia companies in the Thermal Management or Power Electronics role with fewer than 200 employees.
 Freudenberg-NOK | Employment: 160 | Role: Thermal Management
 Hyundai Transys Georgia Powertrain | Employment: 130 | Role: Thermal Management
 Peerless-Winsmith Inc. | Employment: 160 | 
- **Base** ✗: '1. Thermal Management:\n   - Thermacore Inc. (150 employees)\n   - Thermotek Inc. (100 employees)\n   - Thermal Solutions Inc. (120 employees)\n   - Thermcraft Inc. (180 employees)\n   - Thermcraft Thermal Solutions (150 empl'
- **KB-only** ✗: 'List: 165\nCompany: Jefferson Southern Corp.\nCategory: Tier 1/2\nIndustry Group: Electronic and Other Electrical Equipment and Components\nLocation: Commerce, Forsyth County\nAddress: 1 Jefferson Southern Dr, Commerce, GA 30'
- **KB+web** ✗: '- Identify Georgia companies in the Thermal Management or Power Electronics role with more than 200 employees - their larger scale may limit surge production capacity. - Identify Georgia companies in the Thermal Manageme'
- **RAG** ·: _pending_

### kb_q29 (multi_filter_list/deterministic)
**Gold:** There are 23 distinct Georgia‑based companies that are classified as OEM Footprint or OEM Supply Chain

OEM Footprint

TI Fluid Systems
Tie Down Engineering
TN Americas Holding Inc.
Toyota Industries Group (TACG-TICA)
Trenton Pressing
Yazaki North America
YKK USA Inc.
ZF Gainesville LLC

OEM Supply 
- **Base** ✗: 'Please provide a brief description of each company and their role in the EV industry. Additionally, please include any relevant information about their EV-related products or services. Please provide your response in a t'
- **KB-only** ✗: 'If so, list them.\nCompany: GSC Steel Stamping LLC\nCategory: Tier 2/3\nIndustry Group: Machinery Except Electrical\nLocation: Adairsville, Bartow County\nAddress: 104 International Pkwy, Adairsville, GA 30103\nLatitude: 34.38'
- **KB+web** ✗: '- Identify any EV-relevant Georgia companies classified as Battery Pack? - Identify any EV-relevant Georgia companies classified as Battery Pack Materials? - Identify any EV-relevant Georgia companies classified as Batte'
- **RAG** ·: _pending_

### kb_q30 (top_k_ranking/deterministic)
**Gold:** Top 10 Georgia companies based on the employment size supply both General Automotive and have some EV relevance (Yes or Indirect): 

JTEKT North America Corp. | Employment: 860 
Kautex Inc. | Employment: 800 
Lark United Manufacturing Inc. | Employment: 700 
Fouts Brothers Fire Equipment | Employmen
- **Base** ✗: '1. General Motors\n2. Toyota Motor Manufacturing\n3. Honda Manufacturing of America\n4. Nissan North America\n5. Hyundai Motor Manufacturing\n6. Kia Motors Manufacturing\n7. BMW Manufacturing\n8. Mercedes-Benz U.S. Internationa'
- **KB-only** ✗: '?\nRow ID: 100\nCompany: Lear Corporation\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Hammond Industrial Park, Troup County\nAddress: 100 Lear Dr, Hammond Industrial Park, GA 30240\nLatitude: 33.05186'
- **KB+web** ✗: '- Top 10 Georgia companies based on employment size that supply both General Automotive and EV-specific components, indicating transition readiness? - Top 10 Georgia companies based on employment size that supply both Ge'
- **RAG** ·: _pending_

### kb_q31 (judgment/judgment)
**Gold:** There are 7 Tier 2/3 companies producing lightweight aluminum or composite materials and marked EV-relevant or indirectly EV-relevant.
Bridgestone Bandag | Product: Composite materials and specialty polymers for automotive components | EV Relevant: Indirect
Dinex Emissions Inc. | Product: Aluminum s
- **Base** ✗: 'Please provide a list of suppliers and their respective products.\nAs an AI language model, I do not have access to the latest information on Georgia Tier 2/3 suppliers. However, I can suggest some ways to find this infor'
- **KB-only** ✗: 'If so, what is the supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 170\nProduct or Service: Lightweight structural automotive parts\nEV or Battery Relevant: Indirect\nClassification Method: Sup'
- **KB+web** ✗: '- Novelis Inc. is a Tier 1 supplier that produces lightweight aluminum products for the automotive industry. - Novelis Inc. is located in Atlanta, Georgia. - Novelis Inc. is indirectly EV relevant. - Novelis Inc. is a su'
- **RAG** ·: _pending_

### kb_q32 (multi_filter_list/deterministic)
**Gold:** There is only 1 Georgia company with 'high-voltage', 'DC-to-DC', 'inverter', or 'motor controller' product signals.
GSC Steel Stamping LLC | Product: DC-to-DC converters
- **Base** ✗: "The company's product descriptions include 'high-voltage', 'DC-to-DC', 'inverter', and 'motor controller', which are all components of an electric vehicle's powertrain. This indicates that the company is likely experienc"
- **KB-only** ✗: ''
- **KB+web** ✗: "- Identify Georgia companies whose product descriptions include 'battery pack', 'battery management system', 'cell-to-pack', or 'thermal management' — these signal EV battery pack assembly and management growth. - Identi"
- **RAG** ·: _pending_

### kb_q33 (multi_filter_list/deterministic)
**Gold:** There are 3 Georgia automotive companies employ over 1,000 workers but are currently categorized as only "Indirectly Relevant" to the EV sector
TCI Powder Coatings | Employment: 3000 | Role: Vehicle Assembly
Teklas USA | Employment: 1800 | Role: Vehicle Assembly
Textron Specialized Vehicles | Employ
- **Base** ✗: 'Additionally, what are the potential benefits and challenges of transitioning these companies to the EV industry, and how can the state incentivize this shift to ensure a smooth and successful transition? Please provide '
- **KB-only** ✗: 'Yes, these companies are part of the broader automotive supply chain and could play a significant role in the future of EVs.\nAnswer: Yes\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>'
- **KB+web** ✗: '- The Georgia Department of Economic Development (GDEcD) maintains a comprehensive database of automotive companies operating in the state. - The GDEcD database categorizes companies based on their direct or indirect rel'
- **RAG** ·: _pending_

### kb_q34 (top_k_ranking/deterministic)
**Gold:** The 4 companies with the largest thermal-management-related employment in Georgia are:
ZF Gainesville LLC: 170000 employees
Novelis Inc.: 200 employees
Freudenberg-NOK: 160 employees
Peerless-Winsmith Inc.: 160 employees
Freudenberg-NOK and Peerless-Winsmith Inc. tie at 160 employees, so both are in
- **Base** ✗: 'Answer according to: The Georgia Department of Economic Development (GDEcD) has released a new report, “Electric Vehicle Thermal Management in Georgia,” which highlights the state’s strengths in the EV thermal management'
- **KB-only** ✗: '1. Daesol Ausys 2. First American Resources 3. Ajin Georgia 4. Neaton Rome Inc.\nCompany: Daesol Ausys\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Smyrna, Cobb County\nAddress: 21'
- **KB+web** ✗: '- Aspen Aerogels has 100 employees. - Freudenberg-NOK has 100 employees. - Hanon Systems has 100 employees. - Magna International has 100 employees. - These four companies have the largest employment in EV thermal manage'
- **RAG** ·: _pending_

### kb_q35 (multi_filter_list/deterministic)
**Gold:** There are 5 companies involved in thermal-related products or services, along with their associated EV supply chain roles and primary facility types are:
Freudenberg-NOK | Role: Thermal Management | Facility Type: Manufacturing Plant
Hyundai Transys Georgia Powertrain |Role: Thermal Management | Fac
- **Base** ✗: 'Answer according to: The thermal industry is a diverse and complex sector that encompasses a wide range of products and services related to heat transfer, energy efficiency, and thermal management. Some of the companies '
- **KB-only** ✗: '163\nCompany: SK Battery America\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Ellabell, Bryan County\nAddress: 700 Hyundai Blvd, Ellabell, GA 31308\nLatitude: 32.159233\nLongitude: -81.502963\nPrimary F'
- **KB+web** ✗: '- Aspen Aerogels is a manufacturer of aerogel insulation products for thermal management applications. - Aspen Aerogels operates a manufacturing facility in East Providence, RI. - Aspen Aerogels is associated with the Th'
- **RAG** ·: _pending_

### kb_q36 (multi_filter_list/deterministic)
**Gold:** There are 11 Tier 1/2 Georgia companies listed under General Automotive:
Fouts Brothers Fire Equipment [Tier 1/2] | Role: General Automotive
Hitachi Astemo [Tier 1/2] | Role: General Automotive
Hwashin [Tier 1/2] | Role: General Automotive
Hyundai & LG Energy Solution (LGES) [Tier 1/2] | Role: Gener
- **Base** ✗: 'General Motors, Ford Motor Company, and Toyota Motor North America are Tier 1/2 automotive companies in Georgia.'
- **KB-only** ✗: 'Yes\nIndustry Group: Electronic and Other Electrical Equipment and Components\nLocation: Carrollton, Carroll County\nAddress: 109 Carcoustics Dr, Carrollton, GA 30117\nLatitude: 33.601430192463\nLongitude: -85.1053708775212\nP'
- **KB+web** ✗: '- Aurubis AG - Aurubis Richmond - Aurubis Richmond Recycling - Aurubis Richmond Smelter - Aurubis Richmond Technical Center - Aurubis Richmond Warehouse - Aurubis Richmond Warehouse 2 - Aurubis Richmond Warehouse 3 - Aur'
- **RAG** ·: _pending_

### kb_q37 (judgment/judgment)
**Gold:** There are 5 thermal-management-related companies in Georgia.

ZF Gainesville LLC | Employment: 170000
Novelis Inc. | Employment: 200
Freudenberg-NOK | Employment: 160
Peerless-Winsmith Inc. | Employment: 160
Hyundai Transys Georgia Powertrain | Employment: 130

Total employment across the 5 distinct
- **Base** ✗: 'Answer according to: Georgia Thermal Management Suppliers - Request for Quote. Georgia Thermal Management Suppliers - Request for Quote. Georgia Thermal Management Suppliers - Request for Quote. Georgia Thermal Managemen'
- **KB-only** ✗: 'What is the supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 200\nProduct or Service: Stamped blanks Flexible metal hoses for exhaust systems, thermal compensators, among other automotive\nEV o'
- **KB+web** ✗: '- Georgia has 100 Thermal Management suppliers. - The employment size of Georgia Thermal Management suppliers ranges from 1 to 100. - The employment size of Georgia Thermal Management suppliers ranges from 100 to 500. - '
- **RAG** ·: _pending_

### kb_q38 (multi_filter_list/deterministic)
**Gold:** There are 2 Georgia companies involved in battery recycling or end-of-life processing:
Enplas USA Inc. | Product: Recycler of lithium ion batteries
F&P Georgia Manufacturing | Product: Lithium-ion battery recycler and raw materials provider
- **Base** ✗: 'Please provide a comprehensive list of companies that are actively engaged in these activities, including their specific roles and contributions to the circular economy. Additionally, please provide any relevant data or '
- **KB-only** ✗: 'What are their roles and how do they contribute to sustainability?\nCompany: Dinex Emissions Inc.\nCategory: Tier 2/3\nIndustry Group: Primary Metal Industries\nLocation: Dublin, Johnson County\nAddress: 2020 Waldrep Industri'
- **KB+web** ✗: '- Ascend Elements is a battery recycling company that has a facility in Covington, Georgia. - Ascend Elements is building a second facility in Hopkinsville, Kentucky. - Ascend Elements is building a third facility in Hop'
- **RAG** ·: _pending_

### kb_q39 (judgment/judgment)
**Gold:** There is only 1 Georgia company that clearly appears to play an innovation-stage role through research, development, or prototyping activity.
Racemark International LLC | Product: Manufacturing and R&D engine parts for EV
- **Base** ✗: 'A. Georgia-Pacific B. Georgia Tech C. Georgia Power D. Georgia Department of Transportation\nGeorgia Tech\n\nWhich of the following is not a type of innovation? A. Incremental B. Disruptive C. Radical D. Incremental\nIncreme'
- **KB-only** ✗: 'If so, what is the supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 807\nProduct or Service: Automotive seating and E‑systems\nEV or Battery Relevant: Indirect\nClassification Method: Supplier\n<'
- **KB+web** ✗: '- Novelis Inc. - Hyundai Mobis - Hyundai Transys - Hyundai Steel - Hyundai Glovis - Hyundai Motor Group Metaplant America - Hyundai Motor Group Innovation Center - Hyundai Motor Group Innovation Center - Hyundai Motor Gr'
- **RAG** ·: _pending_

### kb_q40 (judgment/judgment)
**Gold:** There are 5 Georgia-based suppliers that exhibit dual-platform supply capability, as they are linked to both traditional OEMs such as Hyundai/Kia and the EV-native OEM Rivian.
Duckyang
GSC Steel Stamping LLC
Hyundai Transys Georgia Powertrain
Enchem America Inc.
Racemark International LLC
- **Base** ✗: 'Answer according to: The Georgia EV Supply Chain Map is a tool to help you understand the supply chain for electric vehicles (EVs) in Georgia. The map shows the locations of EV-related companies in Georgia, and the conne'
- **KB-only** ✗: 'Yes\nEmployment: 300\nProduct or Service: Automotive seating and interior electronics\nEV or Battery Relevant: Indirect\nClassification Method: Supplier\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>'
- **KB+web** ✗: '- The Georgia automotive industry is a $20 billion industry with more than 500 suppliers. - The state has a diverse mix of traditional OEMs and EV-native OEMs. - Georgia’s automotive industry is positioned to support the'
- **RAG** ·: _pending_

### kb_q41 (judgment/judgment)
**Gold:** There are 2 Georgia areas with existing chemical manufacturing infrastructure:
Covington, Morgan County: Archer Aviation Inc.
Norcross, Gwinnett County: Arising Industries Inc.
- **Base** ✗: 'The company is looking for a location with a skilled workforce, access to major transportation routes, and a supportive business climate. Can you provide a list of potential sites that meet these criteria? Please include'
- **KB-only** ✗: 'Company: ZF Gainesville LLC\nCategory: OEM Supply Chain\nIndustry Group: Automotive thermal and electronics systems\nLocation: Gainesville, Hall County\nAddress: 1261 Palmour Dr SW, Gainesville, GA 30501\nLatitude: 34.2647050'
- **KB+web** ✗: '- The company is seeking a location in Georgia for its new manufacturing facility. - The company is seeking a location in Georgia for its new R&D facility. - The company is seeking a location in Georgia for its new headq'
- **RAG** ·: _pending_

### kb_q42 (judgment/judgment)
**Gold:** There is only 1 Georgia area with an R&D facility type in the automotive sector:
Gray, Jones County | Company: Racemark International LLC | Product: Manufacturing and R&D engine parts for EV
- **Base** ✗: 'The following Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers:\n\n1. Atlanta: Atlanta is home to several automotive compan'
- **KB-only** ✗: '(EV Supply Chain)\nCompany: Bosch (Automotive Division)\nCategory: Tier 2/3\nIndustry Group: Rubber and Miscellaneous Plastic Products\nLocation: Toccoa, Franklin County\nAddress: 1241 Meadowbrook Dr, Toccoa, GA 30577\nLatitud'
- **KB+web** ✗: '- The automotive sector in Georgia includes R&D facility types in the following areas: Atlanta, Columbus, Dalton, Gainesville, Macon, Savannah, and Statesboro. - These areas are suitable for establishing EV technology de'
- **RAG** ·: _pending_

### web_q01 (web_fact/web_absorption)
**Gold:** Hyundai
- **Base** ✗: 'SK On, a subsidiary of South Korean conglomerate SK Innovation, is working with Ford Motor Company on a battery joint-venture plant in Cartersville, Georgia. The plant, which is expected to be operational by 2023, will p'
- **KB-only** ✗: 'Company: SK Battery America\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Cartersville, Bartow County\nAddress: 2157 Centennial Pkwy NE, Cartersville, GA 30120\nLatitude: 34.181471\nLongitude: -84.7383'
- **KB+web** ✓: '- SK On is working with Hyundai Motor Group on a battery joint-venture plant in Cartersville, Georgia. - The plant is located in Bartow County, Georgia. - The plant is located in Bartow County, Georgia. - The plant is lo'
- **RAG** ·: _pending_

### web_q02 (web_fact/web_absorption)
**Gold:** synthetic graphite anode materials
- **Base** ✗: 'Answer according to: Anovion Technologies, a manufacturer of lithium-ion battery materials, has opened a new facility in Georgia, US.\nThe company has opened a new facility in Georgia, US, to manufacture lithium-ion batte'
- **KB-only** ✗: 'What is the primary OEM footprint of this supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 82\nProduct or Service: Molded injection auto parts\nEV or Battery Relevant: No\nClassification Method:'
- **KB+web** ✓: "- Anovion Technologies manufactures synthetic graphite anode material at its Georgia facility. - The facility is located in Bainbridge, Georgia. - The facility is Anovion's first commercial-scale manufacturing plant. - T"
- **RAG** ·: _pending_

### web_q03 (web_fact/web_absorption)
**Gold:** SK On
- **Base** ✗: 'The parent company of SK Battery America is SK Innovation.'
- **KB-only** ✗: 'What is the location of its U.S. manufacturing plant?\nCompany: SK Battery America\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Ellabell, Bryan County\nAddress: 700 Hyundai Blvd, Ellabell, GA 31308\nL'
- **KB+web** ✓: '- SK Battery America is a subsidiary of SK On. - SK On is a subsidiary of SK Group. - SK Group is a South Korean conglomerate with interests in energy, chemicals, and information technology. - SK Battery America is a lea'
- **RAG** ·: _pending_

### web_q04 (web_fact/web_absorption)
**Gold:** automotive battery modules and energy storage systems
- **Base** ✓: 'Duckyang supplies SK Battery America with various products, including but not limited to:\n\n1. Lithium-ion battery cells\n2. Battery modules\n3. Battery packs\n4. Battery management systems (BMS)\n5. Battery chargers\n6. Batte'
- **KB-only** ✗: 'How does this compare to the supplier or Affiliation Type: Automotive supply chain participant\nEmployment: 250\nProduct or Service: High-quality electrodeposited (ED) copper foil for electric vehicles\nEV or Battery Releva'
- **KB+web** ✗: '- Duckyang supplies battery modules and battery packs to SK Battery America. - Duckyang supplies battery modules and battery packs to Hyundai Motor Group. - Duckyang supplies battery modules and battery packs to LG Energ'
- **RAG** ·: _pending_

### web_q05 (web_fact/web_absorption)
**Gold:** Ford F-150 Lightning and Volkswagen ID.4
- **Base** ✗: 'Answer the above question based on the context below:  SK Innovation, a Korean company, will build a $1.4 billion lithium-ion battery plant in western Georgia, the company announced Thursday. The plant, which will be bui'
- **KB-only** ✗: 'Company: Mercedes-Benz USA LLC\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Sandy Springs, Fulton County\nAddress: 1 Mercedes-Benz Dr, Sandy Springs, GA 30328\nLatitude: 33.9345221674192\nLongitude: -'
- **KB+web** ✗: '- Hyundai IONIQ 5 - Hyundai IONIQ 6 - Kia EV6 - Kia EV9 - Hyundai IONIQ 9 - Hyundai IONIQ 7 - Hyundai IONIQ 8 - Hyundai IONIQ 3 - Hyundai IONIQ 4 - Hyundai IONIQ 10 - Hyundai IONIQ 11 - Hyundai IONIQ 12 - Hyundai IONIQ 1'
- **RAG** ·: _pending_

### web_q06 (web_fact/web_absorption)
**Gold:** over $184 million
- **Base** ✗: 'Answer according to: GF Casting Solutions AG, a Swiss-based manufacturer of high-performance castings, has announced plans to invest $100 million in a new facility in Augusta, Georgia. The company will create 150 new job'
- **KB-only** ✗: 'What is the expected completion date for the project?\nA: $400,000\nB: $800,000\nC: $1,200,000\nD: $1,600,000\nE: $2,000,000\nAnswer: C\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n'
- **KB+web** ✗: '- GF Casting Solutions AG will invest $100 million in a new facility in Augusta, Georgia. - The new facility will create 100 jobs. - The facility will produce aluminum components for the automotive industry. - The facili'
- **RAG** ·: _pending_

### web_q07 (web_fact/web_absorption)
**Gold:** more than $40 million
- **Base** ✗: 'Answer according to: Hanon Systems, a South Korean company that makes automotive parts, will invest $100 million in a new manufacturing facility in Bulloch County, Gov. Nathan Deal announced Thursday.\nThe company will cr'
- **KB-only** ✗: 'What is the projected completion date for this investment?\nA: $100\nB: $150\nC: $200\nD: $250\nE: $300\nAnswer: C\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>'
- **KB+web** ✗: '- Hanon Systems will invest $200 million in a new manufacturing facility in Bulloch County. - The facility will create 160 new jobs. - The facility will produce electric compressors for automotive HVAC systems. - The fac'
- **RAG** ·: _pending_

### web_q08 (web_fact/web_absorption)
**Gold:** $25 million
- **Base** ✗: 'Answer according to: Duracell Manufacturing, a subsidiary of Duracell Inc., has announced plans to expand its battery component manufacturing operations in the city of Greenville, South Carolina. The company will invest '
- **KB-only** ✗: 'What is the timeline for this expansion?\nAnswer: $400,000\nCompany: Duracell Manufacturing\nCategory: Tier 2/3\nIndustry Group: Primary Metal Industries\nLocation: Augusta, Richmond County\nAddress: 1614 Douglas Rd, Augusta, '
- **KB+web** ✗: '- Duracell Manufacturing will invest $100 million to expand its battery component manufacturing operations. - The expansion will create 100 new jobs. - The expansion will add 100,000 square feet to the company’s existing'
- **RAG** ·: _pending_

### web_q09 (web_fact/web_absorption)
**Gold:** $5.54 billion
- **Base** ✗: 'Answer the above question based on the context below:  In 2006, Kia Motors announced plans to build a new US$500 million (US$682 million in 2016 dollars), 650,000-unit-per-year automotive manufacturing facility on a 3,00'
- **KB-only** ✗: "What is the address' address?\nAnswer: $800\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>"
- **KB+web** ✓: '- Hyundai confirmed it would invest $5.54 billion to open its facility at the Bryan County Megasite. - The facility will be the largest economic development project in Georgia’s history. - The facility will create 8,100 '
- **RAG** ·: _pending_

### web_q10 (web_fact/web_absorption)
**Gold:** $7.6 billion
- **Base** ✗: 'Answer according to: The battery plant is part of a $1.7 billion complex that includes a battery research center and a battery recycling plant. The plant will produce batteries for the Chevrolet Volt, a plug-in hybrid ca'
- **KB-only** ✗: 'The complex includes manufacturing, R&D, and corporate operations.\nAnswer: $1.9 billion\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n'
- **KB+web** ✓: '- The HL-GA battery plant is part of a $7.6 billion complex. - The complex includes a Hyundai Motor Group Metaplant America (HMGMA) assembly plant. - The complex includes a Hyundai Mobis plant. - The complex includes a H'
- **RAG** ·: _pending_

### web_q11 (web_fact/web_absorption)
**Gold:** $5 billion
- **Base** ✓: 'According to the collected web corpus, Rivian will invest $5 billion into building its Georgia facility.'
- **KB-only** ✗: 'What is the timeline for construction and when is the planned opening date?\nCompany: Rivian Automotive\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Atlanta, Fulton County\nAddress: 1090 Roberts Dr N'
- **KB+web** ✓: '- Rivian will invest $5 billion into building its Georgia facility. - The facility is expected to create 7,500 jobs. - The facility is expected to produce up to 400,000 vehicles per year. - The facility is expected to pr'
- **RAG** ·: _pending_

### web_q12 (web_fact/web_absorption)
**Gold:** some $1.7 billion
- **Base** ✗: 'Answer according to: FREYR Battery, a Norwegian company that produces sustainable battery cells, has announced that it will build a new battery cell factory in the United States. The factory will be located in the state '
- **KB-only** ✗: 'What is the timeline for this investment?\nAnswer:\n$110 million\nTimeline: 2021-2022\nCompany: FREYR Battery\nCategory: Tier 2/3\nIndustry Group: Fabricated Metal Products\nLocation: Atlanta, Cobb County\nAddress: 601 10th St N'
- **KB+web** ✓: '- FREYR plans to invest $1.7 billion in the first phase of Giga America. - The company plans to invest $2.6 billion in the second phase of Giga America. - The company plans to invest $3.8 billion in the third phase of Gi'
- **RAG** ·: _pending_

### web_q13 (web_fact/web_absorption)
**Gold:** 350 new jobs
- **Base** ✗: 'Answer according to: GF Casting Solutions, a German company that specializes in producing aluminum castings for the automotive industry, is planning to build a $100 million plant in Richmond County, North Carolina. The c'
- **KB-only** ✗: "What is the estimated investment for this project?\nAnswer: 100\nQuestion: How many new jobs is the GF Metalsmiths' Rome, Floyd County project expected to create? What is the estimated investment for this project?\nAnswer: "
- **KB+web** ✗: '- The Richmond County project is expected to create 100 new jobs. - The project is expected to bring $100 million in investment to the area. - The project is expected to be completed by 2026. - The project is expected to'
- **RAG** ·: _pending_

### web_q14 (web_fact/web_absorption)
**Gold:** 600 new jobs
- **Base** ✗: 'A. 100 B. 200 C. 300 D. 400\nAnswer: C\n\nWhich of the following statements about the characteristics of the Internet of Things (IoT) is incorrect?\nA. The IoT is a network that connects objects to the internet through infor'
- **KB-only** ✗: 'What is the product or service? Tools and dies for automotive industry\nEV Supply Chain Role: General Automotive\nClassification Method: Supplier\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n'
- **KB+web** ✗: '- The project will create 150 new jobs. - The project will bring $60 million in investment to the region. - The project will bring 150 new jobs to the region. - The project will bring 150 new jobs to the region. - The pr'
- **RAG** ·: _pending_

### web_q15 (web_fact/web_absorption)
**Gold:** about 8,100 new jobs
- **Base** ✓: "Answer according to: Hyundai Motor Group’s Bryan County project is expected to create about 8,100 new jobs, according to the collected web corpus.\nThe collected web corpus indicates that Hyundai Motor Group's Bryan Count"
- **KB-only** ✓: '8100\nCompany: Hyundai Motor Group\nCategory: Tier 1/2\nIndustry Group: Electronic and Other Electrical Equipment and Components\nLocation: Ellabell, Bryan County\nAddress: 700 Hyundai Blvd, Ellabell, GA 31308\nLatitude: 32.15'
- **KB+web** ✗: '- The Bryan County project is expected to create 8,500 new jobs. - The project is expected to generate $7.6 billion in investment. - The project is expected to generate $4.5 billion in wages. - The project is expected to'
- **RAG** ·: _pending_

### web_q16 (web_fact/web_absorption)
**Gold:** nearly 1,000 jobs
- **Base** ✓: 'Answer according to: SK Battery, the battery unit of South Korea’s SK Innovation, is cutting 1,000 jobs at its Commerce, Georgia plant, according to a report from the Atlanta Journal-Constitution.\nThe company is cutting '
- **KB-only** ✗: 'A: 300\nB: 723\nC: 7500\nD: 8100\nAnswer: C\n</record>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n</>\n>\n>\n>\n</>\n>\n>\n'
- **KB+web** ✗: '- SK Battery is cutting 958 jobs at its Commerce plant. - The job cuts are part of a broader restructuring plan to reduce costs and improve profitability. - The company is shifting its focus to more profitable business a'
- **RAG** ·: _pending_

### web_q17 (web_fact/web_absorption)
**Gold:** approximately 34 GWh
- **Base** ✗: 'Answer according to: Giga America is Tesla’s new factory in Austin, Texas. It is the second Gigafactory in the United States, after Gigafactory 1 in Nevada. The factory is expected to produce the Tesla Model Y and the Te'
- **KB-only** ✗: 'The approximate annual capacity is 250,000 cells.'
- **KB+web** ✗: '- The first-phase cell production module at Giga America has an annual capacity of 30 GWh. - The first-phase cell production module at Giga America has an annual capacity of 30 GWh. - The first-phase cell production modu'
- **RAG** ·: _pending_

### web_q18 (web_fact/web_absorption)
**Gold:** 22 GWh a year
- **Base** ✗: 'Answer according to: SK Battery America, a subsidiary of South Korea’s SK Innovation, has announced plans to build a $1.7bn lithium-ion battery plant in Georgia, US.\nThe plant will be built in Commerce, Georgia, and will'
- **KB-only** ✗: 'Company: SK Battery America\nCategory: Tier 1\nIndustry Group: Transportation Equipment\nLocation: Ellabell, Bryan County\nAddress: 700 Hyundai Blvd, Ellabell, GA 31308\nLatitude: 32.159233\nLongitude: -81.502963\nPrimary Facil'
- **KB+web** ✗: '- SK Battery America will have a combined annual battery-cell capacity of 22 gigawatt-hours in Georgia. - The company is investing $2.6 billion in two plants in Commerce, Georgia. - The company is investing $940 million '
- **RAG** ·: _pending_
