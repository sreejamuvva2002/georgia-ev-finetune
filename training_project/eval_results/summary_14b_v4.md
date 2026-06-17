# 14B v4 Evaluation

## A. 50 held-out human Q&A (structured score >= 0.6)

**Accuracy: 27/50 = 54%  |  mean score 0.598**

_The 50-question gold is internally inconsistent; each FAIL below is dumped for manual attribution (real model error vs gold inconsistency)._


### By use-case category
| category | n | pass |
|---|---|---|
| Product & Technology Trends | 11 | 4/11 |
| Site Selection & Expansion Planning | 5 | 2/5 |
| Supplier Discovery & Matchmaking | 11 | 4/11 |
| Supply Chain Mapping & Visibility | 16 | 13/16 |
| Supply Chain Risk & Resilience | 7 | 4/7 |

### FAILs (for attribution)

**Q2.** Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?

- gold: There are 6 companies found with Battery Cell or Battery Pack roles.    

F&P Georgia Manufacturing [Tier 1/2] | Role: Battery Pack 
Hitachi Astemo Americas Inc. [Tier 1/2] | Role: Battery Cell 
Hollingsworth & Vose Co. [Tier 1/2] | Role: Battery Pack 
Honda Development & Manufacturing [Tier 1/2] | 
- model: According to the Georgia EV KB, there are 3 companies in the Battery Cell/Pack role:
Hitachi Astemo Americas Inc. | Category: Tier 1/2 | Primary OEMs: Hyundai Kia | Employment: 723
Honda Development & Manufacturing | Category: Tier 1/2 | Primary OEMs: Hyundai Kia | Employment: 400
Hyundai Motor Grou
- missing companies: ['F&P Georgia Manufacturing', 'IMMI'] | hallucinated: []

**Q14.** Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil, and what tier are they classified as?

- gold: There are 4 Georgia companies producing battery materials such as anodes, cathodes, electrolytes, or copper foil. 
Duckyang [Tier 2/3] | Produce: High-quality electrodeposited (ED) copper foil for electric vehicles
F&P Georgia Manufacturing [Tier 1/2] | Produce: Lithium-ion battery recycler and raw 
- model: According to the Georgia EV KB, there are 4 Georgia companies matching battery materials:
Hitachi Astemo Americas Inc. | Category: Tier 1/2 | Product: Battery cells for electric mobility
Hollingsworth & Vose Co. | Category: Tier 1/2 | Product: Lithium-ion battery materials
Honda Development & Manufa
- missing companies: ['Duckyang', 'F&P Georgia Manufacturing', 'IMMI'] | hallucinated: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing', 'Hyundai Motor Group']

**Q17.** Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms?

- gold: There are 2 Georgia companies manufacturing high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms

WIKA USA | Role: HV and LV wiring harnesses for EVs and ICE vehicles | Product: Vehicle power & data solutions; wiring harnesses and connectors | Ev Relevant
- model: According to the Georgia EV KB, there is 1 Georgia company matching high-voltage wiring harnesses or EV electrical distribution components:
Yamaha Motor Manufacturing Corp. | Category: OEM Supply Chain | Product: Automotive electrical architecture, connectivity, and advanced safety components
- missing companies: ['WIKA USA', 'Woodbridge Foam Corp.'] | hallucinated: ['Yamaha Motor Manufacturing Corp.']

**Q18.** Identify Georgia Tier 2/3 companies in the Electronic and Electrical Equipment industry group that could be upgraded to supply EV power electronics.

- gold: There are no Georgia Tier 2/3 companies listed under the Electronic and Other Electrical Equipment and Components industry group. So none can be upgraded to supply EV power electronics based on the provided evidence.
- model: According to the Georgia EV KB, there are 3 Tier 2/3 companies in the Electronic and Electrical Equipment industry group that could be upgraded to supply EV power electronics:
GSC Steel Stamping LLC | Product: Deep grove ball bearings Toyota Industries Electric Systems North 255 DC-to-DC converters 
- missing companies: [] | hallucinated: ['GSC Steel Stamping LLC', 'Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Toyota Industries Group (TACG-TICA)']

**Q20.** Which Georgia Tier 1/2 companies produce engineered plastics, polymers, or composite materials applicable to EV structural or thermal components?

- gold: No Georgia Tier 1/2 companies are explicitly identified as producing engineered plastics, polymers, or composite materials applicable to EV structural or thermal components.
- model: According to the Georgia EV KB, there are 2 Tier 1/2 Georgia companies matching engineered plastics, polymers, or composite materials:
Bridgestone Bandag | Product: Composite materials and specialty polymers for automotive components | Role: General Automotive
Die-Tech Industries Inc. | Product: Gla
- missing companies: [] | hallucinated: ['Bridgestone Bandag', 'Die-Tech Industries Inc.']

**Q21.** Find Georgia suppliers with existing Hyundai Kia contracts that could be expanded to support Hyundai Metaplant's EV battery production ramp-up.

- gold: There are 11 companies with existing Hyundai Kia contracts that could be expanded to support Hyundai Metaplant's EV battery production ramp-up.  

F&P Georgia Manufacturing 
Duckyang 
GSC Steel Stamping LLC 
Hitachi Astemo Americas Inc. 
Hollingsworth & Vose Co. 
Honda Development & Manufacturing 
H
- model: According to the Georgia EV KB, there are 12 Georgia companies with existing contracts to supply components for Hyundai Kia vehicles:
Duckyang | Category: Tier 2/3 | Product: High-quality electrodeposited (ED) copper foil for electric vehicles
Enchem America Inc. | Category: Tier 2/3 | Product: Elec
- missing companies: ['Racemark International LLC'] | hallucinated: ['Kia Georgia Inc.', 'Superior Essex Inc.', 'Toyota Industries Group (TACG-TICA)']

**Q22.** Identify Georgia companies producing DC-to-DC converters, capacitors, or power electronics components relevant to EV drivetrains and what tier is each assigned?

- gold: There are 3 Georgia companies producing DC-to-DC converters, capacitors, or power electronics components relevant to EV drivetrains. 

GSC Steel Stamping LLC [Tier 2/3] 
Hyundai MOBIS (Georgia) [Tier 1/2] 
Hyundai Transys Georgia Seating Systems [Tier 1/2]
- model: According to the Georgia EV KB, there is 1 Georgia company matching DC-to-DC converters, capacitors, or power electronics components:
GSC Steel Stamping LLC | Category: Tier 2/3 | Product: Deep grove ball bearings Toyota Industries Electric Systems North 255 DC-to-DC converters Jackson America, Inc.
- missing companies: ['Hyundai MOBIS (Georgia)', 'Hyundai Transys Georgia Powertrain', 'Hyundai Transys Georgia Seating Systems'] | hallucinated: ['Toyota Industries Group (TACG-TICA)']

**Q24.** Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are linked to.

- gold: There are 6 Georgia companies manufacturing battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement. 

F&P Georgia Manufacturing | OEMs: Hyundai Kia 
Hitachi Astemo Americas Inc. | OEMs: Hyundai Kia 
Hollingsworth & Vose Co. | OEMs: Hyundai Kia 
- model: According to the Georgia EV KB, there are 4 Tier 1/2 Georgia companies that manufacture battery parts or enclosure systems:
F&P Georgia Manufacturing | Product: Lithium-ion battery recycler and raw materials provider | Primary OEMs: Hyundai Kia
Hollingsworth & Vose Co. | Product: Lithium-ion battery
- missing companies: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing'] | hallucinated: []

**Q25.** Find Tier 2/3 Georgia-based suppliers with employment over 300 that are classified as General Automotive but produce components transferable to EV platforms.

- gold: There are 8 Tier 2/3 Georgia-based suppliers with employment over 300 that are classified as General Automotive but produce components transferable to EV platforms. 

Arising Industries Inc. | Employment: 582 
Dinex Emissions Inc. | Employment: 500 
ACM Georgia LLC | Employment: 400 
Dongwon Autopar
- model: According to the Georgia EV KB, there are 3 Tier 2/3 General Automotive suppliers with employment over 300:
ACM Georgia LLC | Employment: 400 | Product: Automotive floor mats made of purchased wire
Adient | Employment: 180 | Product: Backing for automotive carpets and other auto related applications
- missing companies: ['AVS', 'Arising Industries Inc.', 'Dinex Emissions Inc.', 'Dongwon Autopart Technology Georgia LLC', 'FOX Factory', 'Flambeau Inc.', 'Grudem'] | hallucinated: ['ADVICS Manufacturing Georgia LLC', 'Adient', 'F&P Georgia Manufacturing']

**Q28.** Which Georgia Battery Cell or Battery Pack suppliers are sole-sourced by a specific OEM, indicating high dependency risk?

- gold: There are 6 Georgia Battery Cell or Battery Pack suppliers that are sole-sourced by a specific OEM, indicating high dependency risk.
  • F&P Georgia Manufacturing | Role: Battery Pack | Primary OEM: Hyundai Kia
  • Hitachi Astemo Americas Inc. | Role: Battery Cell | Primary OEM: Hyundai Kia
  • Holl
- model: According to the Georgia EV KB, there are 3 Battery Cell/Pack suppliers with sole OEMs (OEMs):
F&P Georgia Manufacturing | Product: Lithium-ion battery recycler and raw materials provider | Primary OEMs: Hyundai Kia
Hollingsworth & Vose Co. | Product: Lithium-ion battery materials | Primary OEMs: Hy
- missing companies: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing', 'IMMI'] | hallucinated: []

**Q29.** For Hyundai Metaplant, how many of its Georgia-based EV component suppliers have fewer than 200 employees, flagging potential capacity fragility?

- gold: For Hyundai Metaplant, there are 6 Georgia-based EV component suppliers with less than 200 employees, flagging potential capacity fragility.
Enchem America Inc. | Employment: 155
F&P Georgia Manufacturing | Employment: 104
Hyundai Motor Group | Employment: 164
Hyundai Transys Georgia Powertrain | Em
- model: According to the Georgia EV KB, there are 7 Georgia-based EV component suppliers with fewer than 200 employees linked to Hyundai Metaplant:
F&P Georgia Manufacturing | Employment: 164 | Role: Battery Pack
Hitachi Astemo Americas Inc. | Employment: 175 | Role: Battery Cell
Hollingsworth & Vose Co. | 
- missing companies: ['Enchem America Inc.', 'Racemark International LLC'] | hallucinated: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing']

**Q30.** Identify Georgia Tier 2/3 suppliers that are EV Relevant, classified as General Automotive, and provide materials to Battery Cell or Battery Pack companies, suggesting supply chain misalignment risk.

- gold: There are 2 Georgia Tier 2/3 suppliers that are EV Relevant, classified as General Automotive, and provide materials to Battery Cell or Battery Pack companies, suggesting supply chain misalignment risk.
Duckyang | OEMs: Hyundai Kia Rivian
Enchem America Inc. | OEMs: Hyundai Kia Rivian
- model: According to the Georgia EV KB, there are 4 Tier 2/3 General Automotive companies that supply materials to Battery Cell/Pack companies:
Ascend Elements | Product: Interior and exterior plastic parts | Primary OEMs: Multiple OEMs
Aspen Aerogels | Product: Gaskets, plastic precision moldings, oil seal
- missing companies: ['Duckyang', 'Enchem America Inc.'] | hallucinated: ['Ascend Elements', 'Aspen Aerogels', 'Aurubis', 'Beaver Manufacturing Co. Inc.']

**Q34.** Top 10 Georgia companies based on employment size that supply both General Automotive and EV-specific components, indicating transition readiness?

- gold: Top 10 Georgia companies based on the employment size supply both General Automotive and have some EV relevance (Yes or Indirect): 

JTEKT North America Corp. | Employment: 860 
Kautex Inc. | Employment: 800 
Lark United Manufacturing Inc. | Employment: 700 
Fouts Brothers Fire Equipment | Employmen
- model: According to the Georgia EV KB, there are 10 companies that supply both General Automotive and EV-specific components (marked 'Yes' in the 'Supports EV' column):
WIKA USA | Employment: 250,000 | Role: HV and LV wiring harnesses for EVs and ICE vehicles
Yazaki North America | Employment: 230,000 | Ro
- missing companies: ['ACM Georgia LLC', 'Arising Industries Inc.', 'Dinex Emissions Inc.', 'Fouts Brothers Fire Equipment', 'JTEKT North America Corp.', 'Kautex Inc.', 'Lark United Manufacturing Inc.', 'Lund International Inc.'] | hallucinated: ['TI Fluid Systems', 'Toyota Industries Group (TACG-TICA)', 'WIKA USA', 'Woory Industrial Co.', 'YKK USA Inc.', 'Yachiyo Manufacturing of America LLC', 'Yamaha Motor Manufacturing Corp.', 'Yazaki North America']

**Q35.** How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes?

- gold: There are 5 Georgia‑based companies currently producing lithium‑ion battery materials, battery cells, or battery electrolytes:
  F&P Georgia Manufacturing [Tier 1/2] | Lithium-ion battery recycler & raw materials
  Hitachi Astemo Americas Inc. [Tier 1/2] | Battery cells for electric mobility
  Holli
- model: According to the Georgia EV KB, there are 4 Georgia companies matching lithium-ion battery materials, cells, or electrolytes:
Down 2 Earth Trailers | Category: Tier 2/3 | Product: Standard and custom aluminum-lithium products for automotive customers
Enplas USA Inc. | Category: Tier 2/3 | Product: R
- missing companies: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing', 'IMMI'] | hallucinated: ['Down 2 Earth Trailers', 'Enplas USA Inc.']

**Q36.** Which Georgia Tier 2/3 suppliers currently produce lightweight aluminum or composite materials and are growing their EV-specific customer base?

- gold: There are 7 Tier 2/3 companies producing lightweight aluminum or composite materials and are growing their EV-specific customer base
Bridgestone Bandag | Product: Composite materials and specialty polymers for automotive components | EV Relevant: Indirect
Dinex Emissions Inc. | Product: Aluminum she
- model: According to the Georgia EV KB, there are 3 Tier 2/3 companies that match your criteria:
Dinex Emissions Inc. | Role: General Automotive | Product: Aluminum sheet for automotive bodies
Dongwon Autopart Technology Georgia LLC | Role: General Automotive | Product: Aluminum extrusions
Dorsett Industrie
- missing companies: ['Bridgestone Bandag', 'Down 2 Earth Trailers', 'Eaton Corp.', 'Erdrich USA Inc.', 'Global Powertrain Systems LLC'] | hallucinated: ['Dongwon Autopart Technology Georgia LLC', 'Dorsett Industries Inc.']

**Q37.** Identify Georgia companies whose product descriptions include 'high-voltage', 'DC-to-DC', 'inverter', or 'motor controller' — these signal EV powertrain electronics growth.

- gold: There is only 1 Georgia company with 'high-voltage', 'DC-to-DC', 'inverter', or 'motor controller' product signals.
GSC Steel Stamping LLC | Product: DC-to-DC converters
- model: According to the Georgia EV KB, there are 4 Georgia companies matching EV powertrain electronics growth:
GSC Steel Stamping LLC | Category: Tier 2/3 | Product: Deep grove ball bearings Toyota Industries Electric Systems North 255 DC-to-DC converters Jackson America, Inc.*
Hyundai MOBIS (Georgia) | C
- missing companies: [] | hallucinated: ['DAS Corp.', 'Hyundai MOBIS (Georgia)', 'Hyundai Transys Georgia Powertrain', 'Hyundai Transys Georgia Seating Systems', 'Toyota Industries Group (TACG-TICA)', 'ZF Gainesville LLC']

**Q41.** Which Tier 1/2 Georgia companies listed under General Automotive suggest a gradual evolution toward EV-related products or markets?

- gold: There are 6 Tier 1/2 Georgia companies listed under General Automotive:
Fouts Brothers Fire Equipment [Tier 1/2] | Role: General Automotive
Hyundai Industrial Co. [Tier 1/2] | Role: General Automotive
Hyundai MOBIS (Georgia) [Tier 1/2] | Role: General Automotive
Hyundai Transys Georgia Seating Syste
- model: According to the Georgia EV KB, there are 5 Tier 1/2 General Automotive suppliers that show a gradual EV evolution:
Fouts Brothers Fire Equipment | Role: General Automotive
Hitachi Astemo | Role: General Automotive
Hwashin | Role: General Automotive
Hyundai & LG Energy Solution (LGES) | Role: Genera
- missing companies: ['Hyundai Industrial Co.', 'Hyundai MOBIS (Georgia)', 'Hyundai Transys Georgia Powertrain', 'Hyundai Transys Georgia Seating Systems', 'Inalfa Roof Systems Inc.', 'Jefferson Southern Corp.', 'Jefferson Southern Corporation'] | hallucinated: ['Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Hwashin', 'IMS Gear Georgia Inc.']

**Q42.** How is demand for thermal management solutions reflected in the number and employment size of Georgia Thermal Management suppliers?

- gold: There are 4 companies for Thermal Management suppliers in Georgia
Employment Size: 650
Freudenberg-NOK | Employment: 160
Hyundai Transys Georgia Powertrain | Employment: 130
Novelis Inc. | Employment: 200
Peerless-Winsmith Inc. | Employment: 160
- model: According to the Georgia EV KB, there are 5 companies in the Thermal Management role. Their total employment is 180. This suggests that while thermal management is important, it may not be as labor-intensive as other EV roles.
- missing companies: ['Freudenberg-NOK', 'Hyundai Transys Georgia Powertrain', 'Hyundai Transys Georgia Seating Systems', 'Novelis Inc.', 'Peerless-Winsmith Inc.'] | hallucinated: []

**Q44.** Which Georgia suppliers appear to play innovation-stage roles through research, development, or prototyping activity?

- gold: There is only 1 Georgia company that clearly appears to be an innovation-stage roles through research, development, or prototyping activity.
Racemark International LLC | Product: Manufacturing and R&D engine parts for EV
- model: According to the Georgia EV KB, there are 5 companies in the innovation stage of the EV supply chain:
Racemark International LLC | Role: General Automotive | Activity: Manufacturing and R&D engine parts for EV
Superior Essex Inc. | Role: Vehicle Assembly | Activity: Vehicle assembly plant
Suzuki Man
- missing companies: [] | hallucinated: ['Superior Essex Inc.', 'Suzuki Manufacturing of America Corp.', 'Thermal Ceramics Inc.', 'Thomson Plastics Inc.']

**Q45.** Which Georgia suppliers currently serving traditional OEMs are also linked to EV-native OEMs, showing dual-platform supply capability?

- gold: There are 5 Georgia-based suppliers that exhibit dual-platform supply capability, as they are linked to both traditional OEMs such as Hyundai/Kia and the EV-native OEM Rivian.
Duckyang
GSC Steel Stamping LLC
Hyundai Transys Georgia Powertrain
Enchem America Inc.
Racemark International LLC
- model: According to the Georgia EV KB, there are 7 Georgia companies serving both traditional OEMs and EV-native OEMs (marked 'Yes' for both columns in the 'OEMs' row):
Duckyang | Role: General Automotive
Enchem America Inc. | Role: General Automotive
F&P Georgia Manufacturing | Role: Battery Pack
GSC Stee
- missing companies: ['Hyundai Transys Georgia Powertrain', 'Hyundai Transys Georgia Seating Systems', 'Racemark International LLC'] | hallucinated: ['F&P Georgia Manufacturing', 'Hitachi Astemo', 'Hitachi Astemo Americas Inc.', 'Honda Development & Manufacturing']

**Q47.** For a new Tier 1 battery thermal management company looking to locate in Georgia, which areas have the highest concentration of Materials-category suppliers that could support thermal management production?

- gold: Covington, Henry County has 2 Materials‑category suppliers that could support battery thermal management production for a new Tier 1 company in Georgia.
- model: According to the Georgia EV KB, the Materials category has the highest number of suppliers in Gwinnett County (11), followed by Cobb County (9) and Fulton County (8).
- missing companies: [] | hallucinated: []

**Q48.** How many Georgia areas have concentrated Manufacturing Plant facilities but no EV-specific production presence, indicating potential conversion-ready industrial sites?

- gold: There are 100 Georgia areas that currently host Manufacturing Plant facilities but have no EV‑specific production presence, indicating a large pool of potentially conversion‑ready industrial sites.
Top concentrations include:
LaGrange, Troup County | 7 plants
Atlanta, Fulton County | 6 plants
West P
- model: According to the Georgia EV KB, there are 3 Manufacturing Plant facilities in areas marked only Manufacturing Plant (no EV-specific production):
Boogook Industries | Location: Macon, Bibb County | Product: Plastic injection gear wheels
Kumho Tire USA Inc. | Location: Macon, Bibb County | Product: St
- missing companies: ['F&P Georgia Manufacturing', 'Morgan Corp.'] | hallucinated: ['Boogook Industries', 'Kumho Tire USA Inc.', 'Vista Metals Corp.']

**Q49.** For an international battery materials company seeking a Georgia location, which areas have existing chemical manufacturing infrastructure?

- gold: There are 2 Georgia areas with existing chemical manufacturing infrastructure:
Covington, Morgan County: Archer Aviation Inc.
Norcross, Gwinnett County: Arising Industries Inc.
- model: According to the Georgia EV KB, there are 3 Georgia counties with chemical manufacturing:
Bibb County | Product: Plastic and metal fasteners and precision strings for automotive industry | Primary OEMs: Multiple OEMs
Bullough County | Product: Rubber powders for tires | Primary OEMs: Multiple OEMs
C
- missing companies: ['Archer Aviation Inc.', 'Arising Industries Inc.', 'Morgan Corp.'] | hallucinated: []

