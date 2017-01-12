clear
set more off
use plans2013.dta
keep DUPERSID PANEL RN CMJINS BY* PRIVCAT OOPPREMX PHOLDER STATUS* 

keep if OOPPREMX >= 0
keep if PHOLDER == 1
bysort DUPERSID: egen premR3 = sum(OOPPREMX) if RN == 3
bysort DUPERSID: egen premR1 = sum(OOPPREMX) if RN == 1
gen totalprem = 0
replace totalprem = premR3 if PANEL == 17
replace totalprem = premR1 if PANEL == 18
by DUPERSID: keep if _n == 1
keep DUPERSID totalprem
sort DUPERSID
save tempprem2013.dta, replace

use raw2013.dta
keep PANEL PID DUID DUPERSID FAMIDYR REFPRS31 REFPRS42 REFPRS53 REFRL31X REFRL42X REFRL53X RUSIZE31 RUSIZE42 RUSIZE53 RULETR31 RULETR42 RULETR53 AGE31X AGE42X AGE53X SEX RACEV1X RACETHX MARRY31X MARRY42X MARRY53X SPOUID31 SPOUID42 SPOUID53 EDUYRDG RTHLTH31 RTHLTH42 RTHLTH53 EMPST31 EMPST42 EMPST53 HRWG31X HRWG42X HRWG53X HOUR31 HOUR42 HOUR53 SELFCM31 SELFCM42 SELFCM53 OFFER31X OFFER42X OFFER53X CHOIC31 CHOIC42 CHOIC53 HELD31X HELD42X HELD53X OFREMP31 OFREMP42 OFREMP53 NUMEMP31 NUMEMP42 NUMEMP53 CHGJ3142 CHGJ4253 TTLP13X FAMINC13 POVLEV13 WAGEP13X BUSNP13X TRIAT13X MCDAT13X PUBAT13X PRIVAT13 UNINS13 TOTTCH13 TOTEXP13 TOTSLF13 TOTMCR13 TOTMCD13 TOTPRV13 TOTVA13 TOTTRI13 TOTOFD13 TOTSTL13 TOTWCP13 TOTOPR13 TOTOPU13 TOTOSR13 

rename PANEL PANEL
rename PID PID
rename DUID DUID
rename DUPERSID DUPERSID
rename FAMIDYR FAMIDYR
rename REFPRS31 refper1
rename REFPRS42 refper2
rename REFPRS53 refper3
rename REFRL31X reltype1
rename REFRL42X reltype2
rename REFRL53X reltype3
rename RUSIZE31 HHsize1
rename RUSIZE42 HHsize2
rename RUSIZE53 HHsize3
rename RULETR31 HHlet1
rename RULETR42 HHlet2
rename RULETR53 HHlet3
rename AGE31X age1
rename AGE42X age2
rename AGE53X age3
rename SEX sex
rename RACEV1X race
rename RACETHX ethnicity
rename MARRY31X marry1
rename MARRY42X marry2
rename MARRY53X marry3
rename SPOUID31 spouseID1
rename SPOUID42 spouseID2
rename SPOUID53 spouseID3
rename EDUYRDG educ
rename RTHLTH31 health1
rename RTHLTH42 health2
rename RTHLTH53 health3
rename EMPST31 employ1
rename EMPST42 employ2
rename EMPST53 employ3
rename HRWG31X wage1
rename HRWG42X wage2
rename HRWG53X wage3
rename HOUR31 hours1
rename HOUR42 hours2
rename HOUR53 hours3
rename SELFCM31 selfemp1
rename SELFCM42 selfemp2
rename SELFCM53 selfemp3
rename OFFER31X offered1
rename OFFER42X offered2
rename OFFER53X offered3
rename CHOIC31 choice1
rename CHOIC42 choice2
rename CHOIC53 choice3
rename HELD31X hasESI1
rename HELD42X hasESI2
rename HELD53X hasESI3
rename OFREMP31 offerothers1
rename OFREMP42 offerothers2
rename OFREMP53 offerothers3
rename NUMEMP31 numemp1
rename NUMEMP42 numemp2
rename NUMEMP53 numemp3
rename CHGJ3142 jobchange1
rename CHGJ4253 jobchange2
rename TTLP13X persinc
rename FAMINC13 faminc
rename POVLEV13 povertypct
rename WAGEP13X wageinc
rename BUSNP13X businc
rename TRIAT13X tricareAT
rename MCDAT13X medicaidAT
rename PUBAT13X publicAT
rename PRIVAT13 privateAT
rename UNINS13 uninsured
rename TOTTCH13 nonRXmed
rename TOTEXP13 totalmed
rename TOTSLF13 OOPmed
rename TOTMCR13 mcrmed
rename TOTMCD13 mcdmed
rename TOTPRV13 privmed
rename TOTVA13 vetmed
rename TOTTRI13 trimed
rename TOTOFD13 fedmed
rename TOTSTL13 statemed
rename TOTWCP13 wcmed
rename TOTOPR13 oprivmed
rename TOTOPU13 opubmed
rename TOTOSR13 osrcmed
gen year = 2013
order year

sort DUPERSID
merge 1:1 DUPERSID using tempprem2013.dta
save ind2013.dta, replace
