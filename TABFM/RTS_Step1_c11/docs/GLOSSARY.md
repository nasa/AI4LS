# GLOSSARY.md — Every technical term, in plain language

**Status:** complete for all terms used in this folder's documents. Alphabetical. If a term appears in a README, PIPELINE, DATA_PROVENANCE, or figure and is not defined here, that is a documentation bug — please report it.

**Arm.** See *treatment group*. File names and column names in `results/` use the shorter word "arm" because the analysis scripts wrote them; prose and figures say "treatment group". They mean the same thing.

**Balanced accuracy.** A classification score that averages the per-group success rates, so a model that always guesses the largest group does not look good. Used because the three treatment groups are small and unequal (11, 19, 8).

**Baseline.** Anything measured before bed rest began. Only baseline measurements are allowed as predictors, so the model can never see the answer.

**Bed rest campaign (C1, C3, C11).** NASA/UTMB studies in which healthy volunteers lie in bed, head tilted 6 degrees down, for weeks to months to mimic the unloading of spaceflight. C11 is the 70-day campaign analyzed here; C1 and C3 are earlier, smaller campaigns used in Step 2.

**BMD / BMC.** Bone mineral density (grams per square centimeter) and bone mineral content (grams), measured by DXA scan. The outcome is the change in total hip BMD.

**Byte-verified.** A number that was recomputed from a saved file and matched exactly, not copied from a document. The opposite is *prose-sourced*.

**Checksum (MD5, SHA-256).** A short fingerprint computed from a file's bytes. If the file changes by one bit, the fingerprint changes. Used to prove which exact table a run consumed.

**CONTROL / EXERCISE / FLY.** The three treatment groups after pooling: no countermeasure (11 people), exercise with or without testosterone (19 people, two design sub-groups pooled), and flywheel resistive exercise (8 people).

**Countermeasure.** Anything given to volunteers to prevent bed-rest deconditioning — here, the exercise programs.

**Coverage filter.** The rule that kept only features measured in at least 80% of subjects. Sparse features were dropped before modeling.

**DXA / iDXA.** The bone-and-body-composition scanner (GE Lunar iDXA) and the measurement technique. Produces the hip scans the outcome is built from.

**Fill rate.** The fraction of table cells that contain a value. The master table is 32.1% filled; the modeling table is 64.9% filled (35.1% empty). Empty cells are never filled in.

**Floor p-value.** The smallest p-value a shuffle test can possibly report: with 500 shuffles, nothing smaller than 1/501 exists. Reported as "p ≤ 0.002" and always annotated as a resolution limit, not a precise value.

**Forward pass.** Running a trained model once to get predictions, with no learning happening. TabPFN needs only this: the training data is shown as context, not used to update weights.

**Ground analog.** An Earth-based experiment that mimics a spaceflight stressor. Bed rest is the standard analog for microgravity unloading.

**Group importance (permutation importance).** Shuffling one group of related features and measuring how much the prediction worsens. The drop estimates how much that group contributed.

**Head-down tilt.** The 6-degree head-down bed angle that shifts body fluids toward the head, as in spaceflight.

**Hip leakage family.** The six columns derived from the hip DXA file that the outcome itself is computed from. All six are removed from the predictors; keeping any of them would let the model peek at the answer.

**iRAT.** The integrated resistance-and-aerobic exercise prescription used in the exercise treatment groups.

**Leakage.** Any path by which information about the outcome (or the treatment group) reaches the predictors. Guarded against by removing the hip source file, using baseline-only features, and keeping group labels out of the table.

**LOOCV (leave-one-person-out cross-validation).** Train on 37 people, predict the 1 held-out person, repeat 38 times, then score all 38 held-out predictions together. Nobody's prediction ever comes from a model that saw their data.

**Long form / wide form.** Two table shapes. Long: one row per measurement (subject, variable, timepoint, value). Wide: one row per subject, one column per measurement. Models need wide; the archive builds long first.

**LSDA.** NASA's Life Sciences Data Archive, the public repository the raw data was downloaded from.

**Master table.** The wide table built from every C11 file: 43 subjects × 7,545 columns, 32.1% filled. The modeling table is a filtered subset of it.

**Missingness fingerprint.** The pattern of which cells are empty. In this cohort the pattern alone reveals the treatment group (357 of 628 features are entirely absent in the flywheel group), so any group classifier must prove it is not just reading the fingerprint.

**Negative control.** A target that should NOT be predictable (here: fat-mass change, tandem-walk performance). If the pipeline "predicts" those too, the pipeline is broken, not brilliant.

**Null distribution / outcome-shuffled null.** The answer to "what would the score be if the features knew nothing about the outcome?" Computed by shuffling the outcome values, rerunning the whole model, and repeating (here: 1,000 times for the headline). The real score must beat the shuffled scores.

**p-value.** The fraction of shuffled runs that matched or beat the real score. p = 0.007 means 7 of the 1,000 shuffled runs did as well as the truth.

**Permutation / shuffle.** Randomly reordering the outcome (or labels) to destroy any real relationship, used to build the null distribution. Each shuffle has a recorded seed so it can be replayed exactly.

**Pre-registration / registered battery.** The practice of writing down the tests before running them. `analysis/` holds the registered battery; `analysis/exploratory/` holds the earlier try-things-first scripts.

**Prose-sourced.** A number known only from notes or text, not recomputed from a saved file. Always labeled as such; never mixed with byte-verified numbers.

**Reading A / B / C.** The three explanations the bias battery adjudicates for the bone-density signal: A, the model ranks individuals within their own group (real biology); B, the model only detects group membership (artifact); C, the model ranks measurement noise (artifact).

**Residualization.** Subtracting group means from the outcome so group membership cannot drive the score. Must be done inside each training fold; doing it on everyone at once leaks (see README corrections).

**ROI (region of interest).** The named anatomical region a DXA scan reports (for example "Femoral Left Total Hip"). The outcome averages the left and right total-hip ROIs.

**Seed.** The starting number for a random generator. Fixing it (here: 42 for the model) makes a random-looking process exactly repeatable.

**Spearman ρ (rho).** A correlation between rankings: do the people the model ranks highest actually lose the most bone? Ranges from −1 (perfectly backwards) through 0 (no relationship) to +1 (perfect). Chosen because it is robust to outliers at small n.

**Stratified permutation.** Shuffling outcomes only within each treatment group, so the null itself keeps any group structure. The strongest test that the signal is individual, not group, membership.

**TabPFN / TabPFN-3.** The tabular foundation model used here (PriorLabs). Pre-trained on synthetic tables; predicts in one forward pass with no training on subject data.

**Tabular foundation model (TFM).** A model pre-trained on many tables that can make predictions on a brand-new table without being trained on it. The class of tool this project tests for NASA.

**Tandem walk.** The heel-to-toe walking test used as one of the negative-control outcomes.

**Timepoint.** When a measurement was taken relative to bed rest (for example Pre, Post). Encoded in every master-table column header.

**Width sweep.** Rerunning the model with random subsets of 5, 25, 100, 250, or all 628 features. The bone-density signal appears only at full width, meaning it is spread across many weak features rather than a few strong ones.
