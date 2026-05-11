import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
DATA_DIR  = r'C:\Users\HP\Desktop\eel208 project\test'
OUT_DIR   = r'C:\Users\HP\Desktop\eel208 project\csv1_data'
MODE      = 'one_per_subject'  

TASK_NAMES = [
    'T1_Reach_Fwd', 'T2_Reach_Side', 'T3_Grasp_Cup',
    'T4_Pour_Water', 'T5_Open_Jar',  'T6_Fold_Towel',
]

MUSCLE_NAMES = [
    'ch0_DeltAnt',  'ch1_DeltPost',
    'ch2_Biceps',   'ch3_Triceps',
    'ch4_FlexCarpi','ch5_ExtCarpi',
    'ch6_FlexDig',  'ch7_ExtDig',
    'ch8_UpperTrap','ch9_LowerTrap',
    'ch10_SerrAnt', 'ch11_PecMaj',
]

CANDIDATE_FIELDS = [
    'DataULpleg', 'DataULdom', 'DataULaff',
    'DataULndom', 'DataULaffected', 'DataULnonpleg', 'DataUL'
]


def load_subject_raw(filepath):
    mat     = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    s       = mat['s']
    fs      = int(s.EmgFreq)
    fname   = os.path.basename(filepath)
    subj_id = fname.replace('.mat', '')
    prefix  = subj_id[:2].upper()
    group   = 'Stroke' if prefix in ('PS', 'ST') else 'Healthy'

    dul = None
    for field in CANDIDATE_FIELDS:
        if hasattr(s, field):
            dul = getattr(s, field)
            break

    if dul is None:
        raise AttributeError(f"No data field found in {fname}")

    n_tasks = len(dul) if hasattr(dul, '__len__') else 1
    records = []
    for i in range(min(n_tasks, len(TASK_NAMES))):
        task    = dul[i]
        emg_raw = task.EMG
        emg     = emg_raw.T if emg_raw.ndim == 2 else emg_raw
        # emg shape: (n_samples, n_channels)
        records.append({
            'subject_id': subj_id,
            'group':      group,
            'task':       TASK_NAMES[i],
            'emg':        emg,
            'fs':         fs,
        })
    return records
def emg_to_dataframe(rec):
    emg  = rec['emg']                          
    n_samples, n_ch = emg.shape
    t    = np.arange(n_samples) / rec['fs']    

    ch_names = MUSCLE_NAMES[:n_ch]

    df = pd.DataFrame(emg, columns=ch_names)
    df.insert(0, 'time_s',     t)
    df.insert(0, 'task',       rec['task'])
    df.insert(0, 'group',      rec['group'])
    df.insert(0, 'subject_id', rec['subject_id'])
    return df


def convert_all(data_dir, out_dir, mode):
    os.makedirs(out_dir, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))

    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        return

    print(f"Found {len(mat_files)} .mat files")
    print(f"Mode   : {mode}")
    print(f"Output : {out_dir}\n")

    all_dfs = []

    for fp in mat_files:
        fname = os.path.basename(fp)
        try:
            records = load_subject_raw(fp)
            subj_id = records[0]['subject_id']
            group   = records[0]['group']

            if mode == 'one_per_task':
                for rec in records:
                    df   = emg_to_dataframe(rec)
                    name = f"{subj_id}_{rec['task']}.csv"
                    df.to_csv(os.path.join(out_dir, name), index=False)
                print(f"  [OK] {fname} → {len(records)} CSV files")

            elif mode == 'one_per_subject':
                dfs  = [emg_to_dataframe(r) for r in records]
                df   = pd.concat(dfs, ignore_index=True)
                name = f"{subj_id}.csv"
                df.to_csv(os.path.join(out_dir, name), index=False)
                rows = f"{len(df):,}"
                print(f"  [OK] {fname} → {name}  ({rows} rows)")

            elif mode == 'combined':
                dfs = [emg_to_dataframe(r) for r in records]
                all_dfs.extend(dfs)
                print(f"  [OK] {fname} ({group}) buffered")

        except Exception as e:
            print(f"  [!] Skipped {fname}: {e}")

    if mode == 'combined' and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(out_dir, 'all_subjects_combined.csv')
        combined.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"\n  [OK] Saved: all_subjects_combined.csv")
        print(f"       Rows   : {len(combined):,}")
        print(f"       Columns: {len(combined.columns)}")
        print(f"       Size   : {size_mb:.1f} MB")

    print(f"\nDone. Files saved to: {out_dir}")


if __name__ == '__main__':
    convert_all(DATA_DIR, OUT_DIR, MODE)