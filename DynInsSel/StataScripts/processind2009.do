clear
set more off
use plans2009.dta
keep DUPERSID PANEL RN CMJINS BY* PRIVCAT OOPPREMX PHOLDER STATUS* 

keep if OOPPREMX >= 0
keep if PHOLDER == 1
bysort DUPERSID: egen premR3 = sum(OOPPREMX) if RN == 3
bysort DUPERSID: egen premR1 = sum(OOPPREMX) if RN == 1
gen totalprem = 0
replace totalprem = premR3 if PANEL == 13
replace totalprem = premR1 if PANEL == 14
by DUPERSID: keep if _n == 1
keep DUPERSID totalprem
sort DUPERSID
save tempprem2009.dta, replace

use raw2009.dta
keep PANEL PID DUID DUPERSID FAMIDYR REFPRS31 REFPRS42 REFPRS53 RFREL31X RFREL42X RFREL53X RUSIZE31 RUSIZE42 RUSIZE53 RULETR31 RULETR42 RULETR53 AGE31X AGE42X AGE53X SEX RACEX RACETHNX MARRY31X MARRY42X MARRY53X SPOUID31 SPOUID42 SPOUID53 HIDEG RTHLTH31 RTHLTH42 RTHLTH53 EMPST31 EMPST42 EMPST53 HRWG31X HRWG42X HRWG53X HOUR31 HOUR42 HOUR53 SELFCM31 SELFCM42 SELFCM53 OFFER31X OFFER42X OFFER53X CHOIC31 CHOIC42 CHOIC53 HELD31X HELD42X HELD53X OFREMP31 OFREMP42 OFREMP53 NUMEMP31 NUMEMP42 NUMEMP53 CHGJ3142 CHGJ4253 TTLP09X FAMINC09 POVLEV09 WAGEP09X BUSNP09X TRIAT09X MCDAT09X PUBAT09X PRIVAT09 UNINS09 TOTTCH09 TOTEXP09 TOTSLF09 TOTMCR09 TOTMCD09 TOTPRV09 TOTVA09 TOTTRI09 TOTOFD09 TOTSTL09 TOTWCP09 TOTOPR09 TOTOPU09 TOTOSR09 

rename PANEL PANEL
rename PID PID
rename DUID DUID
rename DUPERSID DUPERSID
rename FAMIDYR FAMIDYR
rename REFPRS31 refper1
rename REFPRS42 refper2
rename REFPRS53 refper3
rename RFREL31X reltype1
rename RFREL42X reltype2
rename RFREL53X reltype3
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
rename RACEX race
rename RACETHNX ethnicity
rename MARRY31X marry1
rename MARRY42X marry2
rename MARRY53X marry3
rename SPOUID31 spouseID1
rename SPOUID42 spouseID2
rename SPOUID53 spouseID3
rename HIDEG educ
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
rename TTLP09X persinc
rename FAMINC09 faminc
rename POVLEV09 povertypct
rename WAGEP09X wageinc
rename BUSNP09X businc
rename TRIAT09X tricareAT
rename MCDAT09X medicaidAT
rename PUBAT09X publicAT
rename PRIVAT09 privateAT
rename UNINS09 uninsured
rename TOTTCH09 nonRXmed
rename TOTEXP09 totalmed
rename TOTSLF09 OOPmed
rename TOTMCR09 mcrmed
rename TOTMCD09 mcdmed
rename TOTPRV09 privmed
rename TOTVA09 vetmed
rename TOTTRI09 trimed
rename TOTOFD09 fedmed
rename TOTSTL09 statemed
rename TOTWCP09 wcmed
rename TOTOPR09 oprivmed
rename TOTOPU09 opubmed
rename TOTOSR09 osrcmed
gen year = 2009
order year

sort DUPERSID
merge 1:1 DUPERSID using tempprem2009.dta
save ind2009.dta, replace
