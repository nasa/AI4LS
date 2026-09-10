# Project abstract (verbatim)

**Status:** canonical conference abstract, reproduced verbatim from the authors' submission. Two of its early figures are superseded by verified values used everywhere else in this repository: the abstract's "4,834 data files" is the LSDA catalog-listed expectation, while the verified on-disk inventory is 4,809 files (discrepancy disclosed in `DATA_PROVENANCE.md`); the abstract's "32% filled" is the rounded value, the computed value is 32.1% (`build/C11_data_dictionary.md`).

---

A FEASIBILITY STUDY USING TABULAR FOUNDATION MODELS FOR HEALTH PREDICTIONS ON NASA/UTMB BED REST DATA

Ryan T. Scott1, James Casaletto2, Amanda Saravia-Butler1, Walter Alvarado3, Jonathan Galazka3, Samrawit Gebre3

1Amentum, Moffett Field, CA; 2Blue Marble Space, Moffett Field, CA, 3Space Biosciences Division, NASA Ames Research Center, Moffett Field, CA

Spaceflight alters most physiological systems, and individuals differ widely in how they respond. NASA cannot yet accurately predict individually who is most at risk. There is evidence in tabular health data collected from crews and ground analogs, in studies too small (n = 25 to 80) for classical machine learning to learn from. In 2025, tabular foundation models (TFMs) such as TabPFN [1] arrived and spread across terrestrial biomedicine. Pre-trained on synthetic tables, TFMs predict after one forward pass: no training, no fine-tuning, no weight update, no subject data retained for training. This makes them ideal for NASA's crew data privacy protections. Predicting individual risk before it materializes could help guide countermeasure allocation. Here we develop a pipeline TFM method on analog bed rest data across all measures, which later can be applied to spaceflight crew data.

In 2025/26, NASA's Life Sciences Data Archive publicly released the data from the University of Texas Medical Branch bed rest campaigns (C1, C3, C11). We used a custom Python downloader to retrieve all 4,834 data files from these campaigns. TFMs need one row per subject and one column per measurement. We normalized identifiers and flattened every file into one master table (C11: 43 subjects, 7,545 columns, 32% filled), encoding measure, source file, and timepoint in each header.

We started with C11 (70-day head-down-tilt bed rest, control plus three countermeasure groups) and change in total hip bone mineral density (BMD, n = 38). From 628 pre-bed-rest features, TabPFN predicted individual BMD responses, leaving one subject out (Spearman ρ = 0.38; pvalue = 0.007 [compared to 1,000 outcome-shuffled null models]). Grouped permutation importance ranked screening fitness and cardiovascular measures as the two highest contributors, above regional bone mineral density (ranked 10 of 16). Regression performance was best at full width of 628, not in random subsets of 250, 100, 25, or 5 features. A follow-on experiment then looked at predicting pre-to-post bed rest fat-mass change and tandem walk performance. Regression performance was poor, suggesting not every measure is a viable target. All pipelines were designed, developed, and run through the Biomni AI agent (A10G GPU).

This is step one of four: establish whether TFMs predict stably at small n, and document where they break, since failure modes matter to NASA as much as successes. Step two: pool data between C11 and C3 experiments, and benchmark TabPFN against TabICL, Google's TabFM, random forest, and XGBoost. Step three: design and develop a standard pipeline with which to process standard measures. Step four: apply the pipeline to ISS crew data from the analog-validated experiences. The goal: knowing before launch which crew member needs which countermeasure, so lunar crews launch with per-person risk characterized across every measured system.

REFERENCES

[1] Hollmann N. et al (2025) Nature 637, 319-326.
