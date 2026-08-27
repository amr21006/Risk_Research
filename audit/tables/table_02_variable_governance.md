| dataset   | column                                      | decision                              | reason                                                         |
|:----------|:--------------------------------------------|:--------------------------------------|:---------------------------------------------------------------|
| raw       | persistent_id                               | exclude from model features           | identifier or contact field                                    |
| raw       | tender_id                                   | exclude from model features           | identifier or contact field                                    |
| raw       | tender_title                                | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_proceduretype                        | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_nationalproceduretype                | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_isawarded                            | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_supplytype                           | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_biddeadline                          | exclude from model features           | identifier or contact field; temporal variable                 |
| raw       | tender_isjointprocurement                   | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_lotscount                            | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_recordedbidscount                    | exclude from model features           | identifier or contact field                                    |
| raw       | tender_isframeworkagreement                 | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_isdps                                | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_contractsignaturedate                | exclude from model features           | temporal variable                                              |
| raw       | tender_cpvs                                 | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_maincpv                              | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_iseufunded                           | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_selectionmethod                      | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_awardcriteria_count                  | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_cancellationdate                     | exclude from model features           | temporal variable                                              |
| raw       | cancellation_reason                         | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_awarddecisiondate                    | exclude from model features           | temporal variable                                              |
| raw       | tender_estimatedprice                       | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_finalprice                           | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_estimatedprice                          | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | bid_price                                   | exclude from model features           | identifier or contact field                                    |
| raw       | tender_corrections_count                    | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_row_nr                                  | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_title                                   | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_status                                  | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_bidscount                               | exclude from model features           | identifier or contact field                                    |
| raw       | lot_validbidscount                          | exclude from model features           | identifier or contact field                                    |
| raw       | lot_electronicbidscount                     | exclude from model features           | identifier or contact field                                    |
| raw       | lot_smebidscount                            | exclude from model features           | identifier or contact field                                    |
| raw       | lot_updateddurationdays                     | exclude from model features           | temporal variable                                              |
| raw       | buyer_id                                    | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_masterid                              | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_name                                  | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_nuts                                  | exclude from model features           | geographic variable                                            |
| raw       | buyer_city                                  | exclude from model features           | geographic variable                                            |
| raw       | buyer_country                               | exclude from model features           | geographic variable                                            |
| raw       | buyer_mainactivities                        | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | buyer_buyertype                             | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | buyer_postcode                              | exclude from model features           | geographic variable                                            |
| raw       | buyer_postcode_e                            | exclude from model features           | geographic variable                                            |
| raw       | buyer_city_e                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_nuts_e_clean                          | exclude from model features           | geographic variable                                            |
| raw       | buyer_country_e                             | exclude from model features           | geographic variable                                            |
| raw       | buyer_nuts_3                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_nuts_2                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_nuts_1                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_nuts_0                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_street                                | exclude from model features           | geographic variable                                            |
| raw       | buyer_type                                  | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | buyer_url                                   | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_email                                 | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_phone                                 | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_contactName                           | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_extra_source_id                       | exclude from model features           | identifier or contact field                                    |
| raw       | buyer_sourceid_type                         | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_id                                   | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_masterid                             | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_name                                 | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_nuts                                 | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_city                                 | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_country                              | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_postcode                             | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_street                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_type                                 | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_email                                | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_phone                                | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_extra_source_id                      | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_sourceid_type                        | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_url                                  | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_contactName                          | exclude from model features           | identifier or contact field                                    |
| raw       | bidder_postcode_e                           | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_city_e                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_nuts_e_clean                         | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_country_e                            | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_nuts_3                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_nuts_2                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_nuts_1                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bidder_nuts_0                               | exclude from model features           | geographic variable; identifier or contact field               |
| raw       | bid_iswinning                               | exclude from model features           | identifier or contact field                                    |
| raw       | bid_issubcontracted                         | exclude from model features           | identifier or contact field                                    |
| raw       | bid_subcontractedproportion                 | exclude from model features           | identifier or contact field                                    |
| raw       | bid_isconsortium                            | exclude from model features           | identifier or contact field                                    |
| raw       | source                                      | exclude from model features           | identifier or contact field                                    |
| raw       | tender_publications_lastcontractawardurl    | exclude from model features           | identifier or contact field                                    |
| raw       | tender_publications_firstdcontractawarddate | exclude from model features           | temporal variable                                              |
| raw       | notice_url                                  | exclude from model features           | identifier or contact field                                    |
| raw       | tender_publications_firstcallfortenderdate  | exclude from model features           | temporal variable                                              |
| raw       | tender_year                                 | exclude from model features           | temporal variable                                              |
| raw       | tender_addressofimplementation_nuts         | exclude from model features           | geographic variable                                            |
| raw       | tender_description_length                   | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | lot_description_length                      | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_personalrequirements_length          | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_technicalrequirements_length         | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_economicrequirements_length          | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | currency                                    | exclude from model features           | currency proxy                                                 |
| raw       | tender_digiwhist_price                      | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | bid_digiwhist_price                         | exclude from model features           | identifier or contact field                                    |
| raw       | lot_id                                      | exclude from model features           | identifier or contact field                                    |
| raw       | bid_id                                      | exclude from model features           | identifier or contact field                                    |
| raw       | bid_priceUsd                                | exclude from model features           | identifier or contact field                                    |
| raw       | lot_estimatedpriceUsd                       | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_estimatedpriceUsd                    | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | tender_finalpriceUsd                        | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | filter_framework                            | exclude from model features           | filter or derived screening field                              |
| raw       | filter_buyer                                | exclude from model features           | filter or derived screening field                              |
| raw       | filter_bidder                               | exclude from model features           | filter or derived screening field; identifier or contact field |
| raw       | filter_cancelled                            | exclude from model features           | filter or derived screening field                              |
| raw       | filter_opentender                           | exclude from model features           | filter or derived screening field                              |
| raw       | filter_year                                 | exclude from model features           | filter or derived screening field; temporal variable           |
| raw       | filter_losingbids                           | exclude from model features           | filter or derived screening field; identifier or contact field |
| raw       | filter_ok                                   | exclude from model features           | filter or derived screening field                              |
| raw       | corr_singleb                                | exclude from model features           | target or validation label                                     |
| raw       | corr_proc                                   | exclude from model features           | target or validation label                                     |
| raw       | submission_period                           | exclude from model features           | temporal variable                                              |
| raw       | corr_subm                                   | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | corr_nocft                                  | exclude from model features           | target or validation label                                     |
| raw       | decision_period                             | exclude from model features           | temporal variable                                              |
| raw       | corr_decp                                   | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | corr_tax_haven                              | candidate feature after preprocessing | no automatic exclusion                                         |
| raw       | corr_buyer_concentration                    | exclude from model features           | target or validation label                                     |
| raw       | cri                                         | exclude from model features           | target or validation label                                     |
| raw       | cpv_str_cleaned                             | candidate feature after preprocessing | no automatic exclusion                                         |
| processed | trace_group_id                              | exclude from model features           | trace or identifier field                                      |
| processed | federated_client_id                         | exclude from model features           | federated grouping field; trace or identifier field            |
| processed | client_available                            | exclude from model features           | federated grouping field                                       |
| processed | tender_proceduretype                        | candidate model feature               | no automatic exclusion                                         |
| processed | tender_nationalproceduretype                | candidate model feature               | no automatic exclusion                                         |
| processed | tender_supplytype                           | candidate model feature               | no automatic exclusion                                         |
| processed | tender_isjointprocurement                   | candidate model feature               | no automatic exclusion                                         |
| processed | tender_isframeworkagreement                 | candidate model feature               | no automatic exclusion                                         |
| processed | tender_isdps                                | candidate model feature               | no automatic exclusion                                         |
| processed | tender_maincpv                              | candidate model feature               | no automatic exclusion                                         |
| processed | tender_iseufunded                           | candidate model feature               | no automatic exclusion                                         |
| processed | tender_selectionmethod                      | candidate model feature               | no automatic exclusion                                         |
| processed | buyer_mainactivities                        | candidate model feature               | no automatic exclusion                                         |
| processed | buyer_buyertype                             | candidate model feature               | no automatic exclusion                                         |
| processed | buyer_type                                  | candidate model feature               | no automatic exclusion                                         |
| processed | bidder_type                                 | candidate model feature               | no automatic exclusion                                         |
| processed | bid_issubcontracted                         | candidate model feature               | no automatic exclusion                                         |
| processed | bid_isconsortium                            | candidate model feature               | no automatic exclusion                                         |
| processed | cpv_division                                | candidate model feature               | no automatic exclusion                                         |
| processed | cpv_group                                   | candidate model feature               | no automatic exclusion                                         |
| processed | tender_lotscount                            | candidate model feature               | no automatic exclusion                                         |
| processed | tender_recordedbidscount                    | candidate model feature               | no automatic exclusion                                         |
| processed | tender_awardcriteria_count                  | candidate model feature               | no automatic exclusion                                         |
| processed | tender_corrections_count                    | candidate model feature               | no automatic exclusion                                         |
| processed | lot_row_nr                                  | candidate model feature               | no automatic exclusion                                         |
| processed | lot_bidscount                               | candidate model feature               | no automatic exclusion                                         |
| processed | lot_validbidscount                          | candidate model feature               | no automatic exclusion                                         |
| processed | lot_electronicbidscount                     | candidate model feature               | no automatic exclusion                                         |
| processed | lot_smebidscount                            | candidate model feature               | no automatic exclusion                                         |
| processed | bid_subcontractedproportion                 | candidate model feature               | no automatic exclusion                                         |
| processed | tender_description_length                   | candidate model feature               | no automatic exclusion                                         |
| processed | lot_description_length                      | candidate model feature               | no automatic exclusion                                         |
| processed | tender_personalrequirements_length          | candidate model feature               | no automatic exclusion                                         |
| processed | tender_technicalrequirements_length         | candidate model feature               | no automatic exclusion                                         |
| processed | tender_economicrequirements_length          | candidate model feature               | no automatic exclusion                                         |
| processed | rows_aggregated                             | candidate model feature               | no automatic exclusion                                         |
| processed | reported_bid_count                          | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_count                         | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_min                           | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_max                           | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_mean                          | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_std                           | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_cv                            | candidate model feature               | no automatic exclusion                                         |
| processed | bid_price_usd_spread_ratio                  | candidate model feature               | no automatic exclusion                                         |
| processed | log_bid_price_usd_min                       | candidate model feature               | no automatic exclusion                                         |
| processed | log_bid_price_usd_mean                      | candidate model feature               | no automatic exclusion                                         |
| processed | lot_estimated_price_usd                     | candidate model feature               | no automatic exclusion                                         |
| processed | tender_estimated_price_usd                  | candidate model feature               | no automatic exclusion                                         |
| processed | tender_final_price_usd                      | candidate model feature               | no automatic exclusion                                         |
| processed | log_lot_estimated_price_usd                 | candidate model feature               | no automatic exclusion                                         |
| processed | log_tender_estimated_price_usd              | candidate model feature               | no automatic exclusion                                         |
| processed | log_tender_final_price_usd                  | candidate model feature               | no automatic exclusion                                         |
| processed | final_to_estimated_ratio                    | candidate model feature               | no automatic exclusion                                         |
| processed | lot_to_tender_estimated_ratio               | candidate model feature               | no automatic exclusion                                         |
| processed | final_to_min_bid_ratio                      | candidate model feature               | no automatic exclusion                                         |
| processed | estimated_to_min_bid_ratio                  | candidate model feature               | no automatic exclusion                                         |
| processed | missing_bid_price_usd                       | candidate model feature               | no automatic exclusion                                         |
| processed | missing_tender_estimated_price_usd          | candidate model feature               | no automatic exclusion                                         |
| processed | missing_tender_final_price_usd              | candidate model feature               | no automatic exclusion                                         |
| processed | label_corr_singleb                          | exclude from model features           | label or evaluation target                                     |
| processed | label_corr_proc                             | exclude from model features           | label or evaluation target                                     |
| processed | label_corr_nocft                            | exclude from model features           | label or evaluation target                                     |
| processed | label_corr_buyer_concentration              | exclude from model features           | label or evaluation target                                     |
| processed | label_cri                                   | exclude from model features           | label or evaluation target                                     |
| processed | y_cri_high                                  | exclude from model features           | label or evaluation target                                     |
| processed | y_proc_high                                 | exclude from model features           | label or evaluation target                                     |
| processed | y_buyer_concentration_high                  | exclude from model features           | label or evaluation target                                     |
| processed | y_single_bid                                | exclude from model features           | label or evaluation target                                     |
| processed | y_no_call_for_tender                        | exclude from model features           | label or evaluation target                                     |