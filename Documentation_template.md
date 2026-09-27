# Amazon ML Business Entity Resolution Challenge

## 1. Approach

The solution treats Source 1 as the deduplicated reference dataset and identifies matching entities from Source 2 and Source 3.

The pipeline consists of:

1. Candidate generation using country-aware character 4-gram fuzzy-name blocking.
2. Deterministic pairwise feature generation.
3. Fuzzy name and address similarity features.
4. LightGBM binary entity-matching model.
5. Threshold-based pair prediction.
6. Aggregation into the required one-row-per-Source-1 submission format.

## 2. Candidate Generation

The blocker uses character 4-grams from normalized business names.

Country is included in the blocking key so records are compared within the same country. The blocker:

- removes extremely frequent n-grams;
- removes singleton n-grams;
- retains informative n-grams;
- selects the most informative n-grams for each Source-1 entity;
- scores candidate pairs using shared n-grams and n-gram rarity;
- retains the top candidate records per Source-1 entity.

The blocker is a candidate-generation stage only. Final match decisions are made by the trained matcher.

## 3. Matching Features

The deterministic feature stage calculates:

- exact business-name match;
- exact address match;
- exact country match;
- missing-value indicators;
- name lengths;
- name length difference;
- address lengths;
- address length difference.

The fuzzy stage calculates RapidFuzz similarities for:

- name ratio;
- name partial ratio;
- name token-sort ratio;
- name token-set ratio;
- address ratio;
- address partial ratio;
- address token-sort ratio;
- address token-set ratio.

## 4. Matching Model

A LightGBM binary classifier is trained on candidate-level positive and negative pairs.

Positive pairs are derived from the supplied training ground truth.

Negative examples are sampled from blocked candidate pairs that are not ground-truth matches.

The trained model outputs a match probability for every final candidate pair.

The decision threshold is selected using F0.5-oriented validation.

## 5. Test Inference

The test pipeline applies the same feature schema and trained model to the test candidate pairs.

No external entity lookup or external business information is used.

The country field is treated as an open-set categorical/text attribute; no country-specific hard-coded entity list is used.

## 6. Final Outputs

The final submission contains:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Both contain exactly one row for every Source-1 test entity.

`matching_results.tsv` contains:

- `source1_entity_id`
- `matched_entity_ids`

`candidate_pairs.tsv` contains:

- `source1_entity_id`
- `candidate_entity_ids`

Comma-separated IDs are used for multiple matches. An empty string represents no predicted match / no candidate.

## 7. Reproducibility

The implementation is contained under:

`code/business_entity_resolution/`

The trained model artifacts used for inference are stored in the project's output/model directory during development.

Python dependencies are listed in `requirements.txt`.

## 8. Important Evaluation Property

Every predicted match is required to belong to the final candidate set. The final submission-generation stage explicitly validates this invariant before producing the submission files.
