import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import joblib
warnings.filterwarnings('ignore')

# SECTION 1: DATA LOADING (from CSV files)
TASK_NAMES = [
    'T1_Reach_Fwd', 'T2_Reach_Side', 'T3_Grasp_Cup',
    'T4_Pour_Water', 'T5_Open_Jar',  'T6_Fold_Towel',
]

MUSCLE_NAMES = [
    'Deltoid_Ant',    'Deltoid_Post',
    'Biceps_Brachii', 'Triceps_Brachii',
    'Flex_Carpi_Rad', 'Ext_Carpi_Rad',
    'Flex_Digit_Sup', 'Ext_Digitorum',
    'Upper_Trap',     'Lower_Trap',
    'Serratus_Ant',   'Pectoralis_Maj',
]

EMG_FREQ = 1000   


def _csv_to_recordings(df_subj, subj_id):
    """
    Convert a per-subject DataFrame (loaded from CSV) into a list of
    recording dicts — one per task — matching the format expected by
    preprocess_emg() and build_feature_dataset().

    CSV must have columns: subject_id, group, task, time_s, ch0_*, ch1_*, ...
    """
    prefix = subj_id[:2].upper()
    group  = 'Stroke' if prefix in ('PS', 'ST') else 'Healthy'
    label  = 1 if group == 'Stroke' else 0

    ch_cols = [c for c in df_subj.columns if c.startswith('ch')]

    recordings = []
    for task_name in TASK_NAMES:
        task_df = df_subj[df_subj['task'] == task_name]
        if task_df.empty:
            continue

        emg = task_df[ch_cols].values.astype(np.float64)   # (n_samples, n_ch)
        n_ch = emg.shape[1]

        recordings.append({
            'subject_id':   subj_id,
            'group':        group,
            'label':        label,
            'task':         task_name,
            'emg':          emg,
            'fs':           EMG_FREQ,
            'muscle_names': MUSCLE_NAMES[:n_ch],
        })
    return recordings


def load_csv_subject(filepath):
    """
    Load one subject CSV file (e.g. HS01.csv or ST02.csv) produced by
    mat_to_csv.py and return a list of recording dicts (one per task).
    """
    df      = pd.read_csv(filepath)
    fname   = os.path.basename(filepath)
    subj_id = fname.replace('.csv', '')
    return _csv_to_recordings(df, subj_id)


def load_all_subjects(csv_dir):
    """
    Scan csv_dir for all HS*.csv and ST*.csv files and load them.
    Also accepts a single combined CSV (all_subjects_combined.csv).
    Returns a flat list of recording dicts (subjects x tasks).
    """
    # Check for combined file first
    combined_path = os.path.join(csv_dir, 'all_subjects_combined.csv')
    per_subject_files = sorted(
        glob.glob(os.path.join(csv_dir, 'HS*.csv')) +
        glob.glob(os.path.join(csv_dir, 'ST*.csv')) +
        glob.glob(os.path.join(csv_dir, 'PS*.csv'))
    )

    if not per_subject_files and not os.path.exists(combined_path):
        raise FileNotFoundError(
            f"No CSV files found in '{csv_dir}'.\n"
            "Run mat_to_csv.py first to convert your .mat files."
        )

    all_recordings = []
    n_healthy = 0
    n_stroke  = 0

    if per_subject_files:
        for fp in per_subject_files:
            try:
                recs = load_csv_subject(fp)
                if not recs:
                    print(f"  [!] No task data found in {os.path.basename(fp)}")
                    continue
                all_recordings.extend(recs)
                if recs[0]['group'] == 'Healthy':
                    n_healthy += 1
                else:
                    n_stroke += 1
                print(f"  [OK] Loaded {os.path.basename(fp)}  "
                      f"({recs[0]['group']}, {len(recs)} tasks)")
            except Exception as e:
                print(f"  [!] Skipped {os.path.basename(fp)}: {e}")

    else:
        # Combined CSV mode
        print(f"  Loading combined CSV: {combined_path}")
        df_all = pd.read_csv(combined_path)
        for subj_id in sorted(df_all['subject_id'].unique()):
            try:
                df_subj = df_all[df_all['subject_id'] == subj_id]
                recs    = _csv_to_recordings(df_subj, subj_id)
                if not recs:
                    continue
                all_recordings.extend(recs)
                if recs[0]['group'] == 'Healthy':
                    n_healthy += 1
                else:
                    n_stroke += 1
                print(f"  [OK] {subj_id}  ({recs[0]['group']}, {len(recs)} tasks)")
            except Exception as e:
                print(f"  [!] Skipped {subj_id}: {e}")

    print(f"\n  Subjects loaded : {n_healthy} healthy, {n_stroke} stroke")
    print(f"  Total recordings: {len(all_recordings)}")
    return all_recordings


def load_csv_single_subject(filepath):
    """
    Load one CSV file for prediction (test subject).
    Wrapper around load_csv_subject for clarity.
    """
    return load_csv_subject(filepath)
# SECTION 2: SIGNAL FILTERING
def preprocess_emg(emg, fs):
    """
    Standard EMG preprocessing pipeline:
      1. Bandpass 20–450 Hz  — removes motion artifact & high-freq noise
      2. Notch 50 Hz         — removes power-line interference (India grid = 50 Hz)
      3. Full-wave rectification
      4. Low-pass envelope 6 Hz — smooth activation envelope

    Args:
        emg : ndarray, shape (n_samples, n_channels)
        fs  : int, sampling frequency in Hz

    Returns dict with keys: 'raw', 'bandpassed', 'notched', 'rectified', 'envelope'
    """
    nyq = fs / 2.0

    # 1. Bandpass 20–450 Hz
    b_bp, a_bp = signal.butter(4, [20 / nyq, 450 / nyq], btype='bandpass')
    bandpassed = signal.filtfilt(b_bp, a_bp, emg, axis=0)

    # 2. Notch at 50 Hz
    b_n, a_n = signal.iirnotch(50.0, Q=30.0, fs=fs)
    notched = signal.filtfilt(b_n, a_n, bandpassed, axis=0)

    # 3. Full-wave rectification
    rectified = np.abs(notched)

    # 4. Linear envelope: low-pass at 6 Hz
    b_lp, a_lp = signal.butter(4, 6.0 / nyq, btype='low')
    envelope = signal.filtfilt(b_lp, a_lp, rectified, axis=0)

    return {
        'raw':        emg,
        'bandpassed': bandpassed,
        'notched':    notched,
        'rectified':  rectified,
        'envelope':   envelope,
    }
# SECTION 3: FEATURE EXTRACTION
def extract_features(emg_processed, fs):
    """
    Extract time-domain and frequency-domain features per channel.

    Time domain : RMS, MAV, WL, ZC, SSC, VAR
    Frequency   : MNF, MDF, PKF, TTP
    Envelope    : ENV_MAX, ENV_MEAN

    Returns:
        feat_vec   : 1-D ndarray of length (n_channels × 12)
        feat_labels: list of str, e.g. ['ch0_RMS', 'ch0_MAV', ...]
    """
    raw      = emg_processed['raw']
    envelope = emg_processed['envelope']
    n_samples, n_ch = raw.shape

    feature_names = ['RMS', 'MAV', 'WL', 'ZC', 'SSC', 'VAR',
                     'MNF', 'MDF', 'PKF', 'TTP', 'ENV_MAX', 'ENV_MEAN']
    features    = {}
    feat_labels = []

    for ch in range(n_ch):
        sig = raw[:, ch]
        env = envelope[:, ch]

        # Time domain
        rms = np.sqrt(np.mean(sig ** 2))
        mav = np.mean(np.abs(sig))
        wl  = np.sum(np.abs(np.diff(sig)))
        zc  = np.sum(np.diff(np.sign(sig)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(sig))) != 0)
        var = np.var(sig)

        # Frequency domain
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, n_samples))
        total_power = np.sum(psd)
        mnf = np.sum(freqs * psd) / (total_power + 1e-10)
        cumsum = np.cumsum(psd)
        mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        mdf = freqs[min(mdf_idx, len(freqs) - 1)]
        pkf = freqs[np.argmax(psd)]

        # Envelope
        env_max  = np.max(env)
        env_mean = np.mean(env)

        features[ch] = [rms, mav, wl, zc, ssc, var, mnf, mdf, pkf, total_power, env_max, env_mean]

    feat_vec = []
    for ch in range(n_ch):
        for fname in feature_names:
            feat_labels.append(f'ch{ch}_{fname}')
        feat_vec.extend(features[ch])

    return np.array(feat_vec), feat_labels
# SECTION 4: BUILD FEATURE DATASET
def build_feature_dataset(subjects):
    """
    Run preprocessing + feature extraction on every recording.
    Returns:
        df         : DataFrame with metadata + one column per feature
        feat_labels: list of feature column names
    """
    rows       = []
    feat_labels = None

    for i, subj in enumerate(subjects):
        print(f"  Processing {subj['subject_id']} / {subj['task']}  ({i+1}/{len(subjects)})", end='\r')
        emg_proc = preprocess_emg(subj['emg'], subj['fs'])
        feat_vec, feat_labels = extract_features(emg_proc, subj['fs'])

        row = {
            'subject_id': subj['subject_id'],
            'group':      subj['group'],
            'label':      subj['label'],
            'task':       subj['task'],
        }
        for name, val in zip(feat_labels, feat_vec):
            row[name] = val
        rows.append(row)

    print()  # newline after carriage-return progress
    df = pd.DataFrame(rows)
    return df, feat_labels

# SECTION 5: DELIVERABLE 1
def plot_raw_vs_filtered(subjects, save_path='deliverable1_raw_vs_filtered.png'):
    """
    Deliverable 1: Raw vs Filtered Signal Comparison.
    Plots all 5 pipeline stages for one healthy + one stroke subject
    on channel 2 (Biceps Brachii), Task T3 (Grasp Cup).
    """
    task_target = 'T3_Grasp_Cup'

    # Find a healthy and a stroke recording for the same task
    h_subj = next((s for s in subjects if s['group'] == 'Healthy' and s['task'] == task_target), None)
    s_subj = next((s for s in subjects if s['group'] == 'Stroke'  and s['task'] == task_target), None)

    if h_subj is None or s_subj is None:
        print("  [!] Could not find both groups for T3_Grasp_Cup — skipping Deliverable 1")
        return

    channel     = min(2, h_subj['emg'].shape[1] - 1)   # Biceps Brachii (ch index 2)
    muscle_name = h_subj['muscle_names'][channel]

    h_proc = preprocess_emg(h_subj['emg'], h_subj['fs'])
    s_proc = preprocess_emg(s_subj['emg'], s_subj['fs'])

    fs  = h_subj['fs']
    t_h = np.arange(h_subj['emg'].shape[0]) / fs
    t_s = np.arange(s_subj['emg'].shape[0]) / fs

    stages       = ['raw', 'bandpassed', 'notched', 'rectified', 'envelope']
    stage_labels = ['Raw EMG', 'Bandpass\n(20–450 Hz)', 'Notch\n(50 Hz removed)',
                    'Rectified', 'Envelope\n(6 Hz LP)']
    colors_h = ['#2196F3', '#1976D2', '#0D47A1', '#64B5F6', '#1565C0']
    colors_s = ['#F44336', '#D32F2F', '#B71C1C', '#EF9A9A', '#C62828']

    fig, axes = plt.subplots(5, 2, figsize=(16, 14))
    fig.suptitle(
        f'Raw vs Filtered EMG — {muscle_name}\nTask: Grasp Cup (T3)',
        fontsize=14, fontweight='bold', y=0.98
    )

    for i, (stage, lbl) in enumerate(zip(stages, stage_labels)):
        ax_h, ax_s = axes[i, 0], axes[i, 1]

        sig_h = h_proc[stage][:, channel]
        sig_s = s_proc[stage][:, channel]

        ax_h.plot(t_h, sig_h, color=colors_h[i], linewidth=0.6, alpha=0.85)
        ax_s.plot(t_s, sig_s, color=colors_s[i], linewidth=0.6, alpha=0.85)

        ax_h.set_ylabel(lbl, fontsize=9)
        ax_h.set_xlim([0, t_h[-1]])
        ax_s.set_xlim([0, t_s[-1]])

        if i == 0:
            ax_h.set_title(f'Healthy Subject ({h_subj["subject_id"]})',
                           fontsize=12, fontweight='bold', color='#1565C0')
            ax_s.set_title(f'Post-Stroke Subject ({s_subj["subject_id"]})',
                           fontsize=12, fontweight='bold', color='#C62828')

        if i == len(stages) - 1:
            ax_h.set_xlabel('Time (s)')
            ax_s.set_xlabel('Time (s)')

        for ax in [ax_h, ax_s]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {save_path}")

# SECTION 6: DELIVERABLE 2

def plot_feature_table(df, feat_labels, save_path='deliverable2_feature_table.png'):
    """
    Deliverable 2: Extracted Feature Table.
    Boxplots + scatter for 8 key features, with Mann-Whitney U p-values.
    """
    n_ch = len([c for c in df.columns if c.endswith('_RMS')])
    candidates = [
        ('ch2_RMS',     'Biceps RMS'),
        ('ch2_MAV',     'Biceps MAV'),
        ('ch2_MNF',     'Biceps MNF'),
        ('ch2_MDF',     'Biceps MDF'),
        ('ch0_RMS',     'DeltAnt RMS'),
        ('ch0_ENV_MAX', 'DeltAnt Peak'),
        ('ch3_RMS',     'Triceps RMS'),
        ('ch2_WL',      'Biceps WL'),
    ]
    key_features = [(col, name) for col, name in candidates if col in df.columns]

    summary = df.groupby(['subject_id', 'group', 'label'])\
                [[col for col, _ in key_features]].mean().reset_index()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Extracted Feature Comparison: Healthy vs Post-Stroke\n'
                 '(Mean across all tasks per subject)',
                 fontsize=13, fontweight='bold')

    for ax, (feat, name) in zip(axes.flat, key_features):
        h_vals = summary[summary['group'] == 'Healthy'][feat].values
        s_vals = summary[summary['group'] == 'Stroke'][feat].values

        bp = ax.boxplot([h_vals, s_vals],
                        labels=['Healthy', 'Stroke'],
                        patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        widths=0.5)
        bp['boxes'][0].set_facecolor('#BBDEFB')
        bp['boxes'][1].set_facecolor('#FFCDD2')

        for i, (vals, color) in enumerate([(h_vals, '#1565C0'), (s_vals, '#B71C1C')], 1):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([i] * len(vals) + jitter, vals,
                       color=color, s=40, alpha=0.7, zorder=5)

        if len(h_vals) >= 2 and len(s_vals) >= 2:
            _, pval = mannwhitneyu(h_vals, s_vals, alternative='two-sided')
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
            ax.set_title(f'{name}\np={pval:.3f} {sig}', fontsize=9, fontweight='bold')
        else:
            ax.set_title(name, fontsize=9, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {save_path}")

# SECTION 7: DELIVERABLE 3
def plot_comparative_analysis(df, save_path='deliverable3_comparative_analysis.png'):
    """
    Deliverable 3: Healthy vs Patient Comparative Analysis.
    Six sub-plots: RMS bar chart, MNF heatmap, PCA scatter,
    RMS violin, frequency features, Random Forest feature importance.
    """
    feature_cols = [c for c in df.columns if c.startswith('ch')]

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Healthy vs Post-Stroke Comparative Analysis',
                 fontsize=14, fontweight='bold')

    # --- (A) Mean RMS per muscle 
    ax1 = fig.add_subplot(gs[0, 0])
    rms_cols   = [c for c in feature_cols if c.endswith('_RMS')]
    n_muscles  = len(rms_cols)
    ch_labels  = [f'Ch{i}' for i in range(n_muscles)]

    h_rms = df[df['group'] == 'Healthy'][rms_cols].mean().values
    s_rms = df[df['group'] == 'Stroke'][rms_cols].mean().values

    x = np.arange(n_muscles)
    w = 0.35
    ax1.bar(x - w/2, h_rms, w, label='Healthy', color='#2196F3', alpha=0.8)
    ax1.bar(x + w/2, s_rms, w, label='Stroke',  color='#F44336', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ch_labels, fontsize=7, rotation=45)
    ax1.set_title('(A) Mean RMS per Channel', fontweight='bold')
    ax1.set_ylabel('RMS Amplitude (mV)')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, axis='y')

    # --- (B) Mean Frequency heatmap per task 
    ax2 = fig.add_subplot(gs[0, 1])
    mnf_col = 'ch2_MNF' if 'ch2_MNF' in df.columns else feature_cols[6]
    tasks_sorted = sorted(df['task'].unique())
    h_mnf = df[df['group'] == 'Healthy'].groupby('task')[mnf_col].mean()
    s_mnf = df[df['group'] == 'Stroke'].groupby('task')[mnf_col].mean()

    mnf_matrix = np.array([[h_mnf.get(t, np.nan) for t in tasks_sorted],
                            [s_mnf.get(t, np.nan) for t in tasks_sorted]])
    im = ax2.imshow(mnf_matrix, aspect='auto', cmap='RdYlGn_r')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Healthy', 'Stroke'], fontsize=9)
    ax2.set_xticks(range(len(tasks_sorted)))
    ax2.set_xticklabels([t[:10] for t in tasks_sorted], rotation=30, fontsize=7)
    ax2.set_title('(B) Mean Frequency by Task\n(darker = lower / more fatigue)',
                  fontweight='bold', fontsize=9)
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # (C) PCA scatter 
    ax3 = fig.add_subplot(gs[0, 2])
    X    = df[feature_cols].fillna(0).values
    y    = df['label'].values
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)

    for lbl, name, color in [(0, 'Healthy', '#2196F3'), (1, 'Stroke', '#F44336')]:
        mask = y == lbl
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=name, s=40, alpha=0.7,
                    edgecolors='white', linewidth=0.5)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
    ax3.set_title('(C) PCA Feature Space', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- (D) RMS violin 
    ax4 = fig.add_subplot(gs[1, 0])
    h_all_rms = df[df['group'] == 'Healthy'][rms_cols].values.flatten()
    s_all_rms = df[df['group'] == 'Stroke'][rms_cols].values.flatten()
    h_all_rms = h_all_rms[~np.isnan(h_all_rms)]
    s_all_rms = s_all_rms[~np.isnan(s_all_rms)]

    if len(h_all_rms) > 0 and len(s_all_rms) > 0:
        vp = ax4.violinplot([h_all_rms, s_all_rms], positions=[1, 2],
                            showmedians=True, showmeans=False)
        vp['bodies'][0].set_facecolor('#BBDEFB'); vp['bodies'][0].set_alpha(0.8)
        vp['bodies'][1].set_facecolor('#FFCDD2'); vp['bodies'][1].set_alpha(0.8)
        for pc in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            vp[pc].set_color('black')
        _, pval = mannwhitneyu(h_all_rms, s_all_rms, alternative='two-sided')
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['Healthy', 'Stroke'])
        ax4.set_title(f'(D) RMS Distribution (all channels)\np={pval:.4f}',
                      fontweight='bold', fontsize=9)
    else:
        data_avail = h_all_rms if len(h_all_rms) > 0 else s_all_rms
        label_avail = 'Healthy' if len(h_all_rms) > 0 else 'Stroke'
        color_avail = '#BBDEFB' if label_avail == 'Healthy' else '#FFCDD2'
        vp = ax4.violinplot([data_avail], positions=[1], showmedians=True)
        vp['bodies'][0].set_facecolor(color_avail); vp['bodies'][0].set_alpha(0.8)
        ax4.set_xticks([1])
        ax4.set_xticklabels([label_avail])
        ax4.set_title('(D) RMS Distribution\n(only one group loaded)',
                      fontweight='bold', fontsize=9)
    ax4.set_ylabel('RMS Amplitude (mV)')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # --- (E) Frequency features bar 
    ax5 = fig.add_subplot(gs[1, 1])
    freq_feats = [c for c in ['ch2_MNF', 'ch2_MDF', 'ch2_PKF'] if c in df.columns]
    freq_names = {'ch2_MNF': 'Mean Freq', 'ch2_MDF': 'Median Freq', 'ch2_PKF': 'Peak Freq'}

    h_means = [df[df['group'] == 'Healthy'][f].mean() for f in freq_feats]
    s_means = [df[df['group'] == 'Stroke'][f].mean()  for f in freq_feats]
    h_stds  = [df[df['group'] == 'Healthy'][f].std()  for f in freq_feats]
    s_stds  = [df[df['group'] == 'Stroke'][f].std()   for f in freq_feats]
    names   = [freq_names[f] for f in freq_feats]

    x5 = np.arange(len(freq_feats))
    ax5.bar(x5 - 0.2, h_means, 0.35, yerr=h_stds, label='Healthy',
            color='#2196F3', alpha=0.8, capsize=5)
    ax5.bar(x5 + 0.2, s_means, 0.35, yerr=s_stds, label='Stroke',
            color='#F44336', alpha=0.8, capsize=5)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(names, fontsize=9)
    ax5.set_title('(E) Frequency Features\n(Ch2, mean ± SD)', fontweight='bold', fontsize=9)
    ax5.set_ylabel('Frequency (Hz)')
    ax5.legend(fontsize=8)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(True, alpha=0.2, axis='y')

    # --- (F) Random Forest feature importance ---
    ax6 = fig.add_subplot(gs[1, 2])
    subj_df = df.groupby(['subject_id', 'group', 'label'])[feature_cols].mean().reset_index()
    X_sub   = subj_df[feature_cols].fillna(0).values
    y_sub   = subj_df['label'].values

    if len(np.unique(y_sub)) >= 2:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(StandardScaler().fit_transform(X_sub), y_sub)

        importances = rf.feature_importances_
        top_n   = min(12, len(feature_cols))
        top_idx = np.argsort(importances)[-top_n:]
        top_names = [feature_cols[i] for i in top_idx]

        bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(top_idx)))
        ax6.barh(range(len(top_idx)), importances[top_idx], color=bar_colors, alpha=0.9)
        ax6.set_yticks(range(len(top_idx)))
        ax6.set_yticklabels(top_names, fontsize=7)
        ax6.set_title('(F) Top Feature Importances\n(Random Forest)', fontweight='bold', fontsize=9)
        ax6.set_xlabel('Importance Score')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
    else:
        ax6.text(0.5, 0.5, 'Need ≥2 groups\nfor importance',
                 ha='center', va='center', transform=ax6.transAxes)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {save_path}")

# SECTION 8: DELIVERABLE 4

def plot_classification_results(df, save_path='deliverable4_classification_results.png'):
    """
    Deliverable 4: Classification Results.
    Trains SVM, Random Forest, and Logistic Regression.
    Evaluates with Leave-One-Out CV and 5-Fold CV.
    Outputs confusion matrices, accuracy bar chart, per-task accuracy, and summary table.
    """
    feature_cols = [c for c in df.columns if c.startswith('ch')]

    # Subject-level features: mean across tasks
    subj_df = df.groupby(['subject_id', 'group', 'label'])[feature_cols].mean().reset_index()
    X = subj_df[feature_cols].fillna(0).values
    y = subj_df['label'].values

    if len(np.unique(y)) < 2:
        print("  [!] Need both Healthy and Stroke subjects for classification.")
        return {}

    X_sc = StandardScaler().fit_transform(X)
    n_subjects = len(y)

    classifiers = {
        'SVM (RBF)':           SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }

    loo = LeaveOneOut()
    cv_folds = min(5, n_subjects)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle('Classification Results: Healthy vs Post-Stroke EMG\n'
                 f'(Subject-Level, LOO + {cv_folds}-Fold CV, n={n_subjects} subjects)',
                 fontsize=13, fontweight='bold')

    results = {}

    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        # Leave-One-Out
        loo_preds, loo_true = [], []
        for train_idx, test_idx in loo.split(X_sc):
            clf.fit(X_sc[train_idx], y[train_idx])
            loo_preds.append(clf.predict(X_sc[test_idx])[0])
            loo_true.append(y[test_idx[0]])

        loo_acc = np.mean(np.array(loo_preds) == np.array(loo_true))

        # k-Fold CV
        cv_scores = cross_val_score(clf, X_sc, y, cv=skf, scoring='accuracy')

        results[clf_name] = {
            'loo_acc': loo_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std':  cv_scores.std(),
            'preds':   loo_preds,
            'true':    loo_true,
        }

        #  (top row)
        ax_cm = axes[0, idx]
        cm = confusion_matrix(loo_true, loo_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Stroke'])
        disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
        ax_cm.set_title(f'{clf_name}\nLOO Acc: {loo_acc*100:.1f}%',
                        fontweight='bold', fontsize=10)
        ax_cm.grid(False)

    #  (bottom left)
    ax_bar = axes[1, 0]
    names  = list(results.keys())
    means  = [results[n]['cv_mean'] for n in names]
    stds   = [results[n]['cv_std']  for n in names]
    bar_colors = ['#5C6BC0', '#26A69A', '#EF5350']

    bars = ax_bar.bar(range(len(names)), [m * 100 for m in means],
                      yerr=[s * 100 for s in stds],
                      color=bar_colors, alpha=0.85, capsize=8,
                      error_kw={'linewidth': 2})
    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax_bar.set_ylabel('Accuracy (%)')
    ax_bar.set_ylim([0, 115])
    ax_bar.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax_bar.set_title(f'{cv_folds}-Fold CV Accuracy (mean ± SD)', fontweight='bold')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.2, axis='y')
    for bar, m in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{m*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    #  (bottom middle)
    ax_task = axes[1, 1]
    best_clf_name = max(results, key=lambda n: results[n]['loo_acc'])
    best_clf = classifiers[best_clf_name]

    task_accs = {}
    for task in df['task'].unique():
        task_df = df[df['task'] == task]\
                    .groupby(['subject_id', 'group', 'label'])[feature_cols].mean().reset_index()
        if len(task_df) < 4 or len(task_df['label'].unique()) < 2:
            continue
        Xt = StandardScaler().fit_transform(task_df[feature_cols].fillna(0).values)
        yt = task_df['label'].values
        sc = cross_val_score(best_clf, Xt, yt, cv=min(5, len(yt)), scoring='accuracy')
        task_accs[task.replace('_', '\n')] = sc.mean() * 100

    if task_accs:
        task_names = list(task_accs.keys())
        task_vals  = list(task_accs.values())
        bar_c = plt.cm.viridis(np.linspace(0.2, 0.9, len(task_names)))
        ax_task.barh(task_names, task_vals, color=bar_c, alpha=0.85)
        ax_task.axvline(50, color='gray', linestyle='--', alpha=0.5)
        ax_task.set_xlabel('Accuracy (%)')
        ax_task.set_xlim([0, 110])
    ax_task.set_title(f'Per-Task Accuracy\n({best_clf_name})', fontweight='bold', fontsize=10)
    ax_task.spines['top'].set_visible(False)
    ax_task.spines['right'].set_visible(False)
    ax_task.grid(True, alpha=0.2, axis='x')

    #  (bottom right)
    ax_tbl = axes[1, 2]
    ax_tbl.axis('off')
    table_data = [
        [n, f"{results[n]['loo_acc']*100:.1f}%",
         f"{results[n]['cv_mean']*100:.1f} ± {results[n]['cv_std']*100:.1f}%"]
        for n in names
    ]
    table = ax_tbl.table(
        cellText=table_data,
        colLabels=['Classifier', 'LOO Acc', f'{cv_folds}-Fold CV'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BDBDBD')
    ax_tbl.set_title('Classification Summary', fontweight='bold', fontsize=10, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {save_path}")
    return results

# MAIN PIPELINE

def main(data_dir):
    """
    Full pipeline.
    data_dir : path to folder containing all HS*.mat and ST*.mat files.
    """
    print("=" * 65)
    print("  EMG ANALYSIS: Healthy vs Post-Stroke Classification")
    print("  Dataset: Lucchetti et al. (2025), FigShare 7720187")
    print("=" * 65)

    # Save outputs to an 'outputs' folder inside the project directory
    project_dir = os.path.dirname(os.path.abspath(data_dir))
    out_dir = os.path.join(project_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Output folder: {out_dir}")

    def out(filename):
        return os.path.join(out_dir, filename)

    print(f"\n[1] Loading .mat files from: {data_dir}")
    subjects = load_all_subjects(data_dir)

    print("\n[2] Preprocessing EMG signals & extracting features...")
    df, feat_labels = build_feature_dataset(subjects)
    print(f"  Feature matrix: {df.shape[0]} recordings x {len(feat_labels)} features")

    df.to_csv(out('feature_table_full.csv'), index=False)
    print(f"  [OK] Saved: feature_table_full.csv")

    rms_cols = [c for c in df.columns if c.endswith('_RMS')][:5]
    if rms_cols:
        summary = df.groupby('group')[rms_cols].agg(['mean', 'std']).round(5)
        summary.to_csv(out('feature_summary_stats.csv'))
        print(f"  [OK] Saved: feature_summary_stats.csv")
    print("\n[3] Generating deliverables...")

    print("\n  Deliverable 1: Raw vs Filtered Signal Comparison")
    plot_raw_vs_filtered(subjects, save_path=out('deliverable1_raw_vs_filtered.png'))

    print("  Deliverable 2: Extracted Feature Table")
    plot_feature_table(df, feat_labels, save_path=out('deliverable2_feature_table.png'))

    print("  Deliverable 3: Healthy vs Patient Comparative Analysis")
    plot_comparative_analysis(df, save_path=out('deliverable3_comparative_analysis.png'))

    print("  Deliverable 4: Classification Results")
    results = plot_classification_results(df, save_path=out('deliverable4_classification_results.png'))
    if results:
        print("\n" + "=" * 65)
        print("  CLASSIFICATION SUMMARY")
        print("=" * 65)
        for clf_name, r in results.items():
            print(f"  {clf_name:<25}  LOO: {r['loo_acc']*100:.1f}%  |  "
                  f"CV: {r['cv_mean']*100:.1f} +/- {r['cv_std']*100:.1f}%")

    print(f"\n  All outputs saved to: {out_dir}")

# SECTION 9: PREDICT A SINGLE 

def predict_subject(test_mat_path, train_data_dir, out_dir=None):
    """
    Load one new .mat file, train classifiers on all training data,
    and produce a full report with plots + predicted label.

    Args:
        test_mat_path  : full path to the new subject's .mat file
        train_data_dir : folder containing all HS*.mat / ST*.mat training files
        out_dir        : where to save output (default: 'outputs' next to train_data_dir)
    """
    print("=" * 65)
    print("  EMG — SINGLE SUBJECT PREDICTION")
    print("=" * 65)

    # Output folder
    if out_dir is None:
        project_dir = os.path.dirname(os.path.abspath(train_data_dir))
        out_dir = os.path.join(project_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    def out(filename):
        return os.path.join(out_dir, filename)

    test_fname  = os.path.basename(test_mat_path)
    test_id     = test_fname.replace('.mat', '')
    print(f"\n  Test subject : {test_id}")
    print(f"  Output folder: {out_dir}")

    # Load & process TRAINING data
    print(f"\n[1] Loading training data from: {train_data_dir}")
    train_subjects = load_all_subjects(train_data_dir)

    print("\n[2] Building training feature matrix...")
    train_df, feat_labels = build_feature_dataset(train_subjects)
    feature_cols = [c for c in train_df.columns if c.startswith('ch')]

    # Subject-level features 
    train_subj = train_df.groupby(['subject_id', 'group', 'label'])[feature_cols].mean().reset_index()
    X_train = train_subj[feature_cols].fillna(0).values
    y_train = train_subj['label'].values

    if len(np.unique(y_train)) < 2:
        print("\n  [!] Training data must contain BOTH Healthy and Stroke subjects.")
        print("      Please make sure ST*.mat files are loading correctly first.")
        return

    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # Train all three classifiers on full training set
    classifiers = {
        'SVM (RBF)':           SVC(kernel='rbf', C=1.0, gamma='scale',
                                   probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }
    for clf in classifiers.values():
        clf.fit(X_train_sc, y_train)
    print("  Classifiers trained on training data.")

    # Load & process TEST subject 
    print(f"\n[3] Loading test subject: {test_fname}")
    test_subjects = load_csv_single_subject(test_mat_path)

    print("[4] Preprocessing & extracting features for test subject...")
    test_df_rows = []
    for subj in test_subjects:
        emg_proc = preprocess_emg(subj['emg'], subj['fs'])
        feat_vec, _ = extract_features(emg_proc, subj['fs'])
        row = {'subject_id': subj['subject_id'],
               'group': subj['group'], 'label': subj['label'],
               'task': subj['task']}
        for name, val in zip(feat_labels, feat_vec):
            row[name] = val
        test_df_rows.append(row)

    test_df = pd.DataFrame(test_df_rows)
    test_subj_feat = test_df[feature_cols].mean().fillna(0).values.reshape(1, -1)
    test_subj_sc   = scaler.transform(test_subj_feat)

    # Predict 
    print("\n[5] Running predictions...")
    predictions = {}
    for clf_name, clf in classifiers.items():
        pred  = clf.predict(test_subj_sc)[0]
        proba = clf.predict_proba(test_subj_sc)[0]
        label = 'Stroke' if pred == 1 else 'Healthy'
        confidence = proba[pred] * 100
        predictions[clf_name] = {
            'pred': pred, 'label': label,
            'confidence': confidence, 'proba': proba
        }
        print(f"  {clf_name:<25} → {label}  (confidence: {confidence:.1f}%)")

    # Majority vote
    votes = [v['pred'] for v in predictions.values()]
    final_pred  = int(np.round(np.mean(votes)))
    final_label = 'Stroke' if final_pred == 1 else 'Healthy'
    print(f"\n  MAJORITY VOTE PREDICTION → {final_label}")

    # Generate full report
    print("\n[6] Generating report plots...")
    _plot_test_signal(test_subjects, test_id, out)   
    _plot_test_vs_population(test_df, train_df, feat_labels, test_id, out)
    _plot_test_classification(predictions, final_label, test_id,
                              X_train_sc, y_train, test_subj_sc, out)

    print(f"\n  All outputs saved to: {out_dir}")
    print("=" * 65)
    return final_label, predictions


# Helper: Signal plot for test subject 
def _plot_test_signal(test_subjects, test_id, out):
    """Plot raw vs filtered for the test subject (Task T3 if available)."""
    task_target = 'T3_Grasp_Cup'
    subj = next((s for s in test_subjects if s['task'] == task_target), test_subjects[0])
    task_used = subj['task']

    channel = min(2, subj['emg'].shape[1] - 1)
    proc    = preprocess_emg(subj['emg'], subj['fs'])
    fs      = subj['fs']
    t       = np.arange(subj['emg'].shape[0]) / fs

    stages       = ['raw', 'bandpassed', 'notched', 'rectified', 'envelope']
    stage_labels = ['Raw EMG', 'Bandpass (20-450 Hz)', 'Notch (50 Hz)',
                    'Rectified', 'Envelope (6 Hz LP)']
    colors = ['#37474F', '#1565C0', '#283593', '#0277BD', '#01579B']

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Signal Processing Pipeline — {test_id}\n'
                 f'Channel: {subj["muscle_names"][channel]}  |  Task: {task_used}',
                 fontsize=13, fontweight='bold')

    for i, (stage, lbl, color) in enumerate(zip(stages, stage_labels, colors)):
        sig = proc[stage][:, channel]
        axes[i].plot(t, sig, color=color, linewidth=0.6)
        axes[i].set_ylabel(lbl, fontsize=9)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(True, alpha=0.25, linestyle='--')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = out(f'test_{test_id}_signal_processing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: test_{test_id}_signal_processing.png")


# Feature comparison vs population 

def _plot_test_vs_population(test_df, train_df, feat_labels, test_id, out):
    """Bar chart comparing test subject's features to healthy/stroke means."""
    feature_cols = [c for c in train_df.columns if c.startswith('ch')]
    key_feats = [f for f in ['ch2_RMS','ch2_MAV','ch2_MNF','ch2_MDF',
                              'ch0_RMS','ch0_ENV_MAX','ch3_RMS','ch2_WL']
                 if f in feature_cols][:8]

    h_means = train_df[train_df['group']=='Healthy'][key_feats].mean()
    s_means = train_df[train_df['group']=='Stroke'][key_feats].mean()
    t_vals  = test_df[key_feats].mean()

    x = np.arange(len(key_feats))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w,   h_means.values, w, label='Healthy (train mean)',
           color='#2196F3', alpha=0.75)
    ax.bar(x,       s_means.values, w, label='Stroke (train mean)',
           color='#F44336', alpha=0.75)
    ax.bar(x + w,   t_vals.values,  w, label=f'Test subject ({test_id})',
           color='#FF9800', alpha=0.9, edgecolor='black', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('ch2_','Biceps\n').replace('ch0_','DeltAnt\n')
                         .replace('ch3_','Triceps\n') for f in key_feats], fontsize=8)
    ax.set_ylabel('Feature Value')
    ax.set_title(f'Test Subject Feature Comparison vs Training Population — {test_id}',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = out(f'test_{test_id}_feature_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: test_{test_id}_feature_comparison.png")


# Helper: Classification result plot 

def _plot_test_classification(predictions, final_label, test_id,
                               X_train_sc, y_train, test_subj_sc, out):
    """Confidence bar + PCA plot showing where test subject falls."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Classification Report — {test_id}  |  Prediction: {final_label}',
                 fontsize=13, fontweight='bold',
                 color='#C62828' if final_label == 'Stroke' else '#1565C0')

    clf_names = list(predictions.keys())
    colors_bar = ['#5C6BC0', '#26A69A', '#EF5350']

    # Confidence bars 
    ax1 = axes[0]
    healthy_conf = [predictions[n]['proba'][0] * 100 for n in clf_names]
    stroke_conf  = [predictions[n]['proba'][1] * 100 for n in clf_names]
    x = np.arange(len(clf_names))
    w = 0.35
    ax1.bar(x - w/2, healthy_conf, w, label='P(Healthy)', color='#2196F3', alpha=0.8)
    ax1.bar(x + w/2, stroke_conf,  w, label='P(Stroke)',  color='#F44336', alpha=0.8)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(' ', '\n') for n in clf_names], fontsize=8)
    ax1.set_ylabel('Probability (%)')
    ax1.set_ylim([0, 110])
    ax1.set_title('Classifier Probabilities', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, axis='y')

    # PCA — test subject vs training population
    ax2 = axes[1]
    pca = PCA(n_components=2)
    X_pca_train = pca.fit_transform(X_train_sc)
    X_pca_test  = pca.transform(test_subj_sc)

    for lbl, name, color in [(0,'Healthy','#2196F3'), (1,'Stroke','#F44336')]:
        mask = y_train == lbl
        ax2.scatter(X_pca_train[mask, 0], X_pca_train[mask, 1],
                    c=color, label=name, s=40, alpha=0.5,
                    edgecolors='white', linewidth=0.4)

    pred_color = '#C62828' if final_label == 'Stroke' else '#1565C0'
    ax2.scatter(X_pca_test[0, 0], X_pca_test[0, 1],
                c=pred_color, s=250, marker='*',
                edgecolors='black', linewidth=1.2,
                label=f'{test_id} (predicted: {final_label})', zorder=10)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
    ax2.set_title('PCA: Test Subject vs Training Population', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Summary verdict 
    ax3 = axes[2]
    ax3.axis('off')
    verdict_color = '#C62828' if final_label == 'Stroke' else '#1565C0'
    verdict_bg    = '#FFEBEE' if final_label == 'Stroke' else '#E3F2FD'

    ax3.add_patch(plt.Rectangle((0.05, 0.3), 0.9, 0.5,
                                 facecolor=verdict_bg, edgecolor=verdict_color,
                                 linewidth=2, transform=ax3.transAxes))
    ax3.text(0.5, 0.72, 'PREDICTION', ha='center', va='center',
             transform=ax3.transAxes, fontsize=11, color='gray')
    ax3.text(0.5, 0.58, final_label.upper(), ha='center', va='center',
             transform=ax3.transAxes, fontsize=26, fontweight='bold',
             color=verdict_color)

    vote_counts = sum(1 for v in predictions.values() if v['label'] == final_label)
    ax3.text(0.5, 0.42, f'{vote_counts}/3 classifiers agree',
             ha='center', va='center', transform=ax3.transAxes,
             fontsize=10, color='gray')

    # Individual classifier results table
    table_y = 0.28
    for i, (name, res) in enumerate(predictions.items()):
        c = '#C62828' if res['label'] == 'Stroke' else '#1565C0'
        ax3.text(0.5, table_y - i * 0.08,
                 f"{name.split()[0]}: {res['label']} ({res['confidence']:.0f}%)",
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=9, color=c)

    ax3.set_title('Final Verdict', fontweight='bold')

    plt.tight_layout()
    path = out(f'test_{test_id}_classification_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: test_{test_id}_classification_report.png")

# SECTION 10: SAVE & LOAD TRAINED MODEL

def save_model(classifiers, scaler, feat_labels, train_df, model_dir):
    """
    Save trained classifiers + scaler + metadata to disk.
    Only needs to be done ONCE after training.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save each 
    for name, clf in classifiers.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(model_dir, f'{safe_name}.joblib')
        joblib.dump(clf, path)

    # Save scaler too
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

    # Save feature label list + training summary as metadata
    meta = {
        'feat_labels':   feat_labels,
        'feature_cols':  [c for c in train_df.columns if c.startswith('ch')],
        'n_healthy':     int((train_df['group'] == 'Healthy').sum()),
        'n_stroke':      int((train_df['group'] == 'Stroke').sum()),
        'classifier_names': list(classifiers.keys()),
    }
    joblib.dump(meta, os.path.join(model_dir, 'metadata.joblib'))

    print(f"\n  [OK] Model saved to: {model_dir}")
    print(f"       Files: scaler.joblib, metadata.joblib, "
          f"{len(classifiers)} classifier .joblib files")


def load_model(model_dir):
    """
    Load previously saved classifiers + scaler from disk.
    Returns (classifiers dict, scaler, meta dict).
    """
    meta_path = os.path.join(model_dir, 'metadata.joblib')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No saved model found in '{model_dir}'.\n"
            "Run with MODE = 'train' first to train and save the model."
        )

    meta    = joblib.load(meta_path)
    scaler  = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

    classifiers = {}
    for name in meta['classifier_names']:
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(model_dir, f'{safe_name}.joblib')
        classifiers[name] = joblib.load(path)

    print(f"  [OK] Model loaded from: {model_dir}")
    print(f"       Trained on: {meta['n_healthy']} healthy, "
          f"{meta['n_stroke']} stroke subjects")
    return classifiers, scaler, meta


def train_and_save(csv_dir, model_dir, out_dir):
    print("=" * 65)
    print("  MODE: TRAIN  —  Training classifiers & saving model")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)
    def out(f): return os.path.join(out_dir, f)

    # Load all CSV data
    print(f"\n[1] Loading training data from: {csv_dir}")
    subjects = load_all_subjects(csv_dir)

    # Feature extraction
    print("\n[2] Extracting features...")
    train_df, feat_labels = build_feature_dataset(subjects)
    feature_cols = [c for c in train_df.columns if c.startswith('ch')]
    print(f"  Feature matrix: {train_df.shape[0]} recordings x {len(feat_labels)} features")

    train_df.to_csv(out('feature_table_full.csv'), index=False)
    print(f"  [OK] Saved: feature_table_full.csv")

    # Train classifiers
    print("\n[3] Training classifiers...")
    train_subj = train_df.groupby(['subject_id', 'group', 'label'])[feature_cols].mean().reset_index()
    X = train_subj[feature_cols].fillna(0).values
    y = train_subj['label'].values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    classifiers = {
        'SVM (RBF)':           SVC(kernel='rbf', C=1.0, gamma='scale',
                                   probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }
    for name, clf in classifiers.items():
        clf.fit(X_sc, y)
        print(f"  [OK] Trained: {name}")

    # Save model
    print("\n[4] Saving model to disk...")
    save_model(classifiers, scaler, feat_labels, train_df, model_dir)

    # Generate deliverable plots
    print("\n[5] Generating deliverable plots...")

    print("  Deliverable 1: Raw vs Filtered")
    plot_raw_vs_filtered(subjects, save_path=out('deliverable1_raw_vs_filtered.png'))

    print("  Deliverable 2: Feature Table")
    plot_feature_table(train_df, feat_labels, save_path=out('deliverable2_feature_table.png'))

    print("  Deliverable 3: Comparative Analysis")
    plot_comparative_analysis(train_df, save_path=out('deliverable3_comparative_analysis.png'))

    print("  Deliverable 4: Classification Results")
    results = plot_classification_results(train_df, save_path=out('deliverable4_classification_results.png'))

    if results:
        print("\n" + "=" * 65)
        print("  CLASSIFICATION SUMMARY")
        print("=" * 65)
        for clf_name, r in results.items():
            print(f"  {clf_name:<25}  LOO: {r['loo_acc']*100:.1f}%  |  "
                  f"CV: {r['cv_mean']*100:.1f} +/- {r['cv_std']*100:.1f}%")

    print(f"\n  Training complete. Model saved to: {model_dir}")
    print(f"  Plots saved to : {out_dir}")
    print("=" * 65)


def predict_from_saved(test_csv_path, model_dir, out_dir):
    """
    Load a previously saved model and predict on a new subject CSV.
    NO retraining — instant prediction.
    """
    print("=" * 65)
    print("  MODE: PREDICT  —  Loading saved model (no retraining)")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)
    def out(f): return os.path.join(out_dir, f)

    test_fname = os.path.basename(test_csv_path)
    test_id    = test_fname.replace('.csv', '')
    print(f"\n  Test subject : {test_id}")

    # Load saved model
    print(f"\n[1] Loading saved model from: {model_dir}")
    classifiers, scaler, meta = load_model(model_dir)
    feat_labels  = meta['feat_labels']
    feature_cols = meta['feature_cols']

    # Load & process test subject
    print(f"\n[2] Loading test subject: {test_fname}")
    test_subjects = load_csv_single_subject(test_csv_path)

    print("[3] Extracting features...")
    test_rows = []
    for subj in test_subjects:
        emg_proc = preprocess_emg(subj['emg'], subj['fs'])
        feat_vec, _ = extract_features(emg_proc, subj['fs'])
        row = {'subject_id': subj['subject_id'],
               'group': subj['group'], 'label': subj['label'],
               'task': subj['task']}
        for name, val in zip(feat_labels, feat_vec):
            row[name] = val
        test_rows.append(row)

    test_df       = pd.DataFrame(test_rows)
    test_feat     = test_df[feature_cols].mean().fillna(0).values.reshape(1, -1)
    test_feat_sc  = scaler.transform(test_feat)

    # Predict
    print("\n[4] Predicting...")
    predictions = {}
    for clf_name, clf in classifiers.items():
        pred  = clf.predict(test_feat_sc)[0]
        proba = clf.predict_proba(test_feat_sc)[0]
        label = 'Stroke' if pred == 1 else 'Healthy'
        conf  = proba[pred] * 100
        predictions[clf_name] = {
            'pred': pred, 'label': label,
            'confidence': conf, 'proba': proba
        }
        print(f"  {clf_name:<25} -> {label}  ({conf:.1f}% confidence)")

    votes       = [v['pred'] for v in predictions.values()]
    final_pred  = int(np.round(np.mean(votes)))
    final_label = 'Stroke' if final_pred == 1 else 'Healthy'
    print(f"\n  MAJORITY VOTE  ->  {final_label}")

    #  Generate report plots
    print("\n[5] Generating report...")
    _plot_test_signal(test_subjects, test_id, out)

    # For feature comparison we need training population data
    feat_csv = os.path.join(out_dir, 'feature_table_full.csv')
    if os.path.exists(feat_csv):
        train_df = pd.read_csv(feat_csv)
        _plot_test_vs_population(test_df, train_df, feat_labels, test_id, out)
    else:
        print("  [!] feature_table_full.csv not found — skipping population comparison plot")

    # For PCA we need the training feature matrix
    # Reconstruct from scaler 
    if os.path.exists(feat_csv):
        train_df   = pd.read_csv(feat_csv)
        train_subj = train_df.groupby(['subject_id','group','label'])[feature_cols].mean().reset_index()
        X_train    = scaler.transform(train_subj[feature_cols].fillna(0).values)
        y_train    = train_subj['label'].values
        _plot_test_classification(predictions, final_label, test_id,
                                  X_train, y_train, test_feat_sc, out)

    print(f"\n  Report saved to: {out_dir}")
    print("=" * 65)
    return final_label, predictions

# ENTRY POINT
if __name__ == '__main__':

    CSV_DIR   = r'C:\Users\HP\Desktop\eel208 project\csv_data'
    MODEL_DIR = r'C:\Users\HP\Desktop\eel208 project\saved_model'
    OUT_DIR   = r'C:\Users\HP\Desktop\eel208 project\outputs'

    MODE = 'predict'
    TEST_FILE = r'C:\Users\HP\Desktop\eel208 project\csv1_data\ST10.csv'
    if MODE == 'train':
        train_and_save(
            csv_dir   = CSV_DIR,
            model_dir = MODEL_DIR,
            out_dir   = OUT_DIR,
        )
    elif MODE == 'predict':
        predict_from_saved(
            test_csv_path = TEST_FILE,
            model_dir     = MODEL_DIR,
            out_dir       = OUT_DIR,
        )
    else:
        print(f"Unknown MODE '{MODE}'. Set MODE to 'train' or 'predict'.")