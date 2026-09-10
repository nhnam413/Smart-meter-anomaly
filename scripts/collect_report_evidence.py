"""Recompute report facts from existing artifacts; never fit or overwrite data."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from generate_report_figures import load_inputs, validate_inputs, extract_features

PROJECT = Path(__file__).resolve().parents[1]


def fingerprints():
    paths = [PROJECT / 'data/train_hourly.csv', PROJECT / 'data/demo_stream.csv',
             PROJECT / 'models/model_bundle.pkl']
    paths += sorted((PROJECT / 'data').glob('alert_history.sqlite3*'))
    paths += sorted(p for p in (PROJECT / 'src').iterdir() if p.is_file())
    return {str(p.relative_to(PROJECT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def environment():
    result = {'python': platform.python_version(), 'os': platform.platform(),
              'logical_cpus': os.cpu_count(), 'packages': {n: importlib.metadata.version(n)
              for n in ['pandas', 'numpy', 'scikit-learn', 'joblib', 'matplotlib', 'streamlit', 'plotly']}}
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ['total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended']]
        mem = Memory()
        mem.length = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            result['ram_gib'] = round(mem.total_phys / 2**30, 2)
    return result


def collect():
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import pandas as pd
    before = fingerprints()
    train, demo, bundle, features, scores, predictions, labels, types = load_inputs()
    train_features = extract_features(train)
    validate_inputs(train, demo, bundle, train_features, features, scores, predictions, labels, types)
    expected_scaled = (features[bundle['features']].values - bundle['scaler'].center_) / bundle['scaler'].scale_
    np.testing.assert_allclose(bundle['scaler'].transform(features[bundle['features']].values), expected_scaled, rtol=1e-12, atol=1e-12)
    result = {'environment': environment(), 'input_sha256': before, 'datasets': {},
              'feature_order': bundle['features'], 'scaler': {}, 'cases': {}}
    for name, data, feat, expected in [('train', train, train_features, (182, 120)), ('demo', demo, features, (239, 72))]:
        stamps = data.index.to_series()
        gaps = stamps.diff()
        missing = int((stamps.iloc[-1] - stamps.iloc[0]) / pd.Timedelta(hours=1)) + 1 - len(data)
        mismatch = int(((stamps - stamps.shift(24)).loc[feat.index] != pd.Timedelta(hours=24)).sum())
        assert (missing, mismatch) == expected, (name, missing, mismatch)
        result['datasets'][name] = {'rows': len(data), 'valid': len(feat), 'start': str(data.index[0]),
            'end': str(data.index[-1]), 'missing_hours': missing, 'lag24_not_24h': mismatch,
            'gaps': {str(k): float(v.total_seconds()/3600) for k,v in gaps[gaps > pd.Timedelta(hours=1)].items()}}
    for i, name in enumerate(bundle['features']):
        q1, q3 = np.percentile(train_features[name], [25, 75])
        result['scaler'][name] = {'center': float(bundle['scaler'].center_[i]),
            'scale': float(bundle['scaler'].scale_[i]), 'q1': float(q1), 'q3': float(q3), 'iqr': float(q3-q1)}
    assert result['scaler']['is_night']['iqr'] == 0 and result['scaler']['is_night']['scale'] == 1
    result['model'] = {'parameters': bundle['model'].get_params(), 'offset': float(bundle['model'].offset_)}
    result['metrics'] = {'auc': roc_auc_score(labels, -scores), 'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions), 'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions), 'confusion_matrix': confusion_matrix(labels,predictions).tolist(),
        'alerts': int(predictions.sum()), 'injected_before_warmup': int(demo['is_anomaly'].sum()),
        'injected_valid': int(labels.sum()), 'types': {}}
    for name, expected in [('power_surge',(127,205)), ('voltage_drop',(193,205)), ('night_spike',(77,135))]:
        mask = (types == name).to_numpy()
        caught, total = int(predictions[mask].sum()), int(mask.sum())
        assert (caught, total) == expected
        result['metrics']['types'][name] = {'caught': caught, 'total': total, 'recall': caught/total}
    for name, mask, expected in [
        ('TP', (labels.to_numpy()==1)&(predictions==1), '2010-02-06 09:00:00'),
        ('FN', (labels.to_numpy()==1)&(predictions==0), '2010-02-08 03:00:00'),
        ('FP', (labels.to_numpy()==0)&(predictions==1), '2010-02-06 10:00:00')]:
        i = int(np.flatnonzero(mask)[0]); timestamp = features.index[i]
        assert str(timestamp) == expected
        result['cases'][name] = {'time': str(timestamp), 'synthetic_type': str(types.iloc[i]),
            'decision': float(scores[i]), 'power': float(demo.loc[timestamp, 'Global_active_power']),
            'voltage': float(demo.loc[timestamp, 'Voltage']), 'features': features.iloc[i].to_dict()}
    result['correlations'] = train_features.corr().to_dict()
    assert before == fingerprints(), 'Input artifacts changed while collecting evidence'
    path = PROJECT / 'reports/report_evidence.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Validated evidence: {path}')


if __name__ == '__main__':
    collect()
