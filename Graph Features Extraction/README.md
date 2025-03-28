# Graph Features Extraction
Code to enrich OpTC, Pivoting and LANL with graph features. LANL followed the original extraction provided by the authors, OpTc and Pivoting uses simpler features to the lack of user info.

## Extract maps
To extract the host/user maps used for calculating features, use files extract_maps_{dataset}.py. Update dataset paths in the scripts with your dataset locations.

## Add Features
To add features to the dataset, use files add_features_{dataset}.py. Update dataset paths in the scripts with your dataset locations.