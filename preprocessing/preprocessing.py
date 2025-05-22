
# preprocessing/preprocessing.py

import numpy as np
import pandas as pd
from nilearn import input_data, connectome
from neurocombat_sklearn import CombatModel
import os
import argparse
import joblib

def fisher_z_transform(corrs):
    return np.arctanh(np.clip(corrs, -0.999999, 0.999999))

def compute_fc_matrices(fmri_files, masker):
    time_series = [masker.fit_transform(f) for f in fmri_files]
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    fc_matrices = correlation_measure.fit_transform(time_series)
    return fisher_z_transform(fc_matrices)

def apply_combat(fc_matrices, site_labels):
    n_subjects, n_rois, _ = fc_matrices.shape
    fc_reshaped = fc_matrices.reshape(n_subjects, -1)
    combat = CombatModel()
    fc_harmonized = combat.fit_transform(fc_reshaped, site_labels)
    return fc_harmonized.reshape(n_subjects, n_rois, n_rois)

def main(data_csv, atlas_path, output_path):
    df = pd.read_csv(data_csv)
    masker = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    fc_matrices = compute_fc_matrices(df['fmri_path'].tolist(), masker)
    harmonized = apply_combat(fc_matrices, df['site'].values)

    np.save(os.path.join(output_path, 'fc_matrices_harmonized.npy'), harmonized)
    joblib.dump(masker, os.path.join(output_path, 'masker.pkl'))
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_csv, args.atlas_path, args.output_path)
