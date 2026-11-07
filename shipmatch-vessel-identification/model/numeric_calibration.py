"""Use digit-preserving augmentation only where a row has numeric evidence."""
import numpy as np
from shipmatch_v4 import numeric
from phonetic_calibration import CalibratedRough

class NumericCalibratedRough:
    def __init__(self,original_path,numeric_path,index,weight=1.,blend=.5,ipa=None):
        self.original=CalibratedRough(original_path,index,weight=weight,blend=blend,ipa=ipa)
        self.numeric=CalibratedRough(numeric_path,index,weight=weight,blend=blend,ipa=ipa)
        self.evidence=self.original.evidence
        self.numeric.evidence=self.evidence
        self.ipa_enabled=ipa is not None

    def apply(self,row,candidates,features,scores,order,rough):
        available=any(np.isfinite(numeric(row.get(field))) for field in ['gross_tonnage','international_gross_tonnage'])
        model=self.numeric if available else self.original
        return model.apply(row,candidates,features,scores,order,rough)


def build_calibration(model_dir,index,config,ipa_model_dir=None):
    """Opt-in IPA bundle; absent option preserves the original V40 models."""
    from pathlib import Path
    model_dir=Path(model_dir)
    kwargs=dict(weight=config['calibration_weight'],blend=config['rough_blend'])
    if ipa_model_dir is not None:
        import json
        from ipa_features import IPAFeatures,IPA_FEATURE_NAMES,digest
        bundle=Path(ipa_model_dir)
        manifest=json.loads((bundle/'training_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('ipa_feature_names')!=IPA_FEATURE_NAMES:raise ValueError('IPA training schema mismatch')
        if manifest.get('ipa_feature_source_sha256')!=digest(Path(__file__).with_name('ipa_features.py')):raise ValueError('IPA training/inference feature source differs')
        for name in ['rough_phonetic_ipa.cbm','rough_phonetic_numeric_ipa.cbm']:
            if digest(bundle/name)!=manifest['model_sha256'][name]:raise ValueError('IPA model checksum mismatch: '+name)
        ipa=IPAFeatures(bundle/'ipa_index',index.catalog)
        return NumericCalibratedRough(bundle/'rough_phonetic_ipa.cbm',bundle/'rough_phonetic_numeric_ipa.cbm',index,ipa=ipa,**kwargs)
    if config.get('numeric_calibration',False):
        return NumericCalibratedRough(model_dir/'rough_phonetic.cbm',model_dir/'rough_phonetic_numeric.cbm',index,**kwargs)
    return CalibratedRough(model_dir/'rough_phonetic.cbm',index,**kwargs)
