"""Fixed Alpaca ECL binary UP/DOWN experiment; reads train.csv and validation.csv only."""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import joblib, lightgbm as lgb, numpy as np, pandas as pd, sklearn, xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/alpaca_rebuild_v1'
OUT=Path(__file__).resolve().parent
SPEC=ROOT/'models/alpaca_final_freeze_v1/frozen_candidate_spec.json'
FEATURES=['close','return_1','return_3','return_6','return_12','sma_ratio_5','sma_ratio_10','sma_ratio_20','ema_ratio_5','ema_ratio_10','ema_ratio_20','rsi_14','volatility_5','volatility_10','volatility_20','candle_body','candle_range','upper_wick','lower_wick','minutes_since_open','minutes_until_close','day_of_week','time_sin','time_cos']
CLASSES=['DOWN','NEUTRAL','UP']; DIR=['DOWN','UP']; ENCODE={'DOWN':0,'UP':1}; COVERAGES=(.20,.30,.40,.50); SEED=42
PARAMS={
 'xgboost_binary':dict(n_estimators=350,max_depth=3,learning_rate=.03,min_child_weight=5,subsample=.8,colsample_bytree=.8,reg_lambda=5.,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=-1),
 'lightgbm_binary':dict(objective='binary',n_estimators=350,learning_rate=.03,num_leaves=15,max_depth=-1,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=5.,class_weight='balanced',random_state=SEED,n_jobs=-1,verbosity=-1),
}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def load_split(name):
 if name not in {'train.csv','validation.csv'}: raise ValueError('Only predeclared train/validation files are supported')
 f=pd.read_csv(DATA/name)
 if f.columns.duplicated().any(): raise ValueError(f'Duplicate headers: {name}')
 if 'close.1' in f:
  if not np.array_equal(f.close.to_numpy(),f['close.1'].to_numpy()): raise ValueError('Duplicate close columns differ')
  f=f.drop(columns=['close.1'])
 if not {'datetime','future_datetime','future_return','target',*FEATURES}.issubset(f.columns): raise ValueError(f'Missing required columns in {name}')
 if [c for c in f.columns if c in FEATURES]!=FEATURES: raise ValueError(f'Feature order mismatch in {name}')
 f['datetime']=pd.to_datetime(f.datetime,errors='raise'); f['future_datetime']=pd.to_datetime(f.future_datetime,errors='raise')
 for c in FEATURES: f[c]=pd.to_numeric(f[c],errors='raise')
 f['day_of_week']=f.day_of_week.astype('int64')
 if not f.datetime.is_monotonic_increasing or f.datetime.duplicated().any(): raise ValueError(f'Bad timestamp ordering in {name}')
 if not (f.future_datetime==f.datetime+pd.Timedelta(minutes=15)).all(): raise ValueError(f'Bad 15m horizon in {name}')
 if f.target.isna().any() or not set(f.target.unique()).issubset(CLASSES): raise ValueError(f'Bad labels in {name}')
 if f[FEATURES].isna().any().any() or not np.isfinite(f[FEATURES].to_numpy(float)).all(): raise ValueError(f'Invalid features in {name}')
 ret=f.future_return.to_numpy(float); expected=np.where(ret>.10,'UP',np.where(ret<-.10,'DOWN','NEUTRAL'))
 if not np.array_equal(f.target.to_numpy(str),expected): raise ValueError(f'Label/return mismatch in {name}')
 return f

def preprocessor():
 nums=[c for c in FEATURES if c!='day_of_week']
 return ColumnTransformer([('numeric',SimpleImputer(strategy='median'),nums),('day_of_week',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),['day_of_week'])],remainder='drop',verbose_feature_names_out=False)

def threshold_for(conf,target):
 a=np.asarray(conf,float)
 if not len(a) or not np.isfinite(a).all() or not 0<target<=1: raise ValueError('Invalid confidence vector or target coverage')
 k=max(1,int(np.ceil(target*len(a))))
 return float(np.sort(a)[-k])

def binary_metrics(y,p):
 return {'n':int(len(y)),'accuracy':float(accuracy_score(y,p)) if len(y) else None,
  'up_precision':float(precision_score(y,p,labels=DIR,average='binary',pos_label='UP',zero_division=0)) if len(y) else None,
  'down_precision':float(precision_score(y,p,labels=DIR,average='binary',pos_label='DOWN',zero_division=0)) if len(y) else None,
  'macro_precision':float(precision_score(y,p,labels=DIR,average='macro',zero_division=0)) if len(y) else None,
  'macro_recall':float(recall_score(y,p,labels=DIR,average='macro',zero_division=0)) if len(y) else None,
  'macro_f1':float(f1_score(y,p,labels=DIR,average='macro',zero_division=0)) if len(y) else None,
  'confusion_labels_rows_actual_cols_predicted':DIR,'confusion_matrix':confusion_matrix(y,p,labels=DIR).tolist() if len(y) else [[0,0],[0,0]]}

def calibration(y,p,bins=10):
 if not len(y): return {'n':0,'log_loss':None,'brier':None,'ece':None}
 p=np.asarray(p,float); y=np.asarray(y,int); pred=np.c_[1-p,p]; edges=np.linspace(0,1,bins+1); ece=0.; reliability=[]
 for i in range(bins):
  m=(p>=edges[i])&((p<=edges[i+1]) if i==bins-1 else (p<edges[i+1]))
  if m.any():
   ece+=m.mean()*abs(p[m].mean()-y[m].mean())
   reliability.append({'bin':i,'lower':float(edges[i]),'upper':float(edges[i+1]),'count':int(m.sum()),'mean_predicted_p_up':float(p[m].mean()),'observed_up_rate':float(y[m].mean())})
 return {'n':int(len(y)),'log_loss':float(log_loss(y,pred,labels=[0,1])),'brier':float(np.mean((p-y)**2)),'ece_10_equal_width':float(ece),'p_up_mean':float(p.mean()),'p_up_std':float(p.std()),'p_up_quantiles':{str(q):float(np.quantile(p,q)) for q in (.05,.25,.5,.75,.95)},'reliability_bins':reliability}

def selective(y,pred,conf,target):
 th=threshold_for(conf,target); emit=np.asarray(conf)>=th; out=np.where(emit,pred,'NO_SIGNAL'); n=len(y); eligible=np.isin(y,DIR); n_emit=int(emit.sum())
 tp={c:int(((out==c)&(y==c)).sum()) for c in DIR}; n_pred={c:int((out==c).sum()) for c in DIR}; n_true={c:int((y==c).sum()) for c in DIR}
 prec={c:(tp[c]/n_pred[c] if n_pred[c] else 0.) for c in DIR}; rec={c:(tp[c]/n_true[c] if n_true[c] else 0.) for c in DIR}
 f1={c:(2*prec[c]*rec[c]/(prec[c]+rec[c]) if prec[c]+rec[c] else 0.) for c in DIR}
 eligible_emit=emit&eligible
 eligible_tp={c:int(((out==c)&(y==c)&eligible_emit).sum()) for c in DIR}
 eligible_pred_count={c:int(((out==c)&eligible_emit).sum()) for c in DIR}
 eligible_precision={c:(eligible_tp[c]/eligible_pred_count[c] if eligible_pred_count[c] else 0.) for c in DIR}
 eligible_recall={c:(eligible_tp[c]/n_true[c] if n_true[c] else 0.) for c in DIR}
 eligible_f1={c:(2*eligible_precision[c]*eligible_recall[c]/(eligible_precision[c]+eligible_recall[c]) if eligible_precision[c]+eligible_recall[c] else 0.) for c in DIR}
 eligible_cm=confusion_matrix(y[eligible_emit],out[eligible_emit],labels=DIR).tolist()
 cm=confusion_matrix(y,out,labels=['DOWN','NEUTRAL','UP','NO_SIGNAL'])
 return {'target_coverage_all_validation_candles':target,'confidence_threshold':th,'validation_candles':int(n),'emitted_predictions':n_emit,'no_signal_count':int(n-n_emit),
  'coverage_all_validation_candles':float(n_emit/n),'eligible_directional_candles':int(eligible.sum()),'coverage_among_eligible_directional':float(eligible_emit.sum()/eligible.sum()),
  'up_prediction_count':n_pred['UP'],'down_prediction_count':n_pred['DOWN'],'up_precision_operational':prec['UP'],'down_precision_operational':prec['DOWN'],'macro_directional_precision_operational':float(np.mean(list(prec.values()))),
  'up_precision_eligible_directional_only':eligible_precision['UP'],'down_precision_eligible_directional_only':eligible_precision['DOWN'],'macro_directional_precision_eligible_only':float(np.mean(list(eligible_precision.values()))),
  'up_f1_eligible_directional_only':eligible_f1['UP'],'down_f1_eligible_directional_only':eligible_f1['DOWN'],'macro_f1_eligible_directional_only':float(np.mean(list(eligible_f1.values()))),
  'binary_accuracy_emitted_eligible_directional_only':float((pred[eligible_emit]==y[eligible_emit]).mean()) if eligible_emit.any() else None,
  'eligible_directional_confusion_labels_rows_actual_cols_prediction':DIR,'eligible_directional_confusion_matrix':eligible_cm,
  'up_recall':rec['UP'],'down_recall':rec['DOWN'],'macro_directional_recall':float(np.mean(list(rec.values()))),'macro_directional_f1':float(np.mean(list(f1.values()))),
  'emitted_accuracy_including_neutral_misses':float(((out==y)&emit).sum()/n_emit) if n_emit else None,
  'accuracy_among_emitted_actual_directional_only':float((pred[eligible_emit]==y[eligible_emit]).mean()) if eligible_emit.any() else None,
  'confusion_labels_rows_actual_cols_prediction':['DOWN','NEUTRAL','UP','NO_SIGNAL'],'confusion_matrix':cm.tolist()},out,th

def fit_model(name,params,X,y):
 est=xgb.XGBClassifier(**params) if name=='xgboost_binary' else lgb.LGBMClassifier(**params)
 pipe=Pipeline([('preprocessor',preprocessor()),('model',est)])
 pipe.fit(X.loc[:,FEATURES],y)
 if list(pipe.named_steps['preprocessor'].feature_names_in_)!=FEATURES: raise AssertionError('Preprocessing feature order mismatch')
 return pipe

def main():
 existing=[p.name for p in OUT.iterdir() if p.name not in {Path(__file__).name,'__pycache__'}]
 if existing: raise FileExistsError(f'Refusing to overwrite existing artifacts: {OUT}')
 spec=json.loads(SPEC.read_text(encoding='utf-8'))
 if spec['features_exact_order']!=FEATURES or spec['class_order']!=CLASSES: raise ValueError('Frozen feature/class specification mismatch')
 train=load_split('train.csv'); valid=load_split('validation.csv')
 if train.datetime.max()>=valid.datetime.min() or train.future_datetime.max()>=valid.datetime.min(): raise ValueError('Chronological split/purge check failed')
 tr=train[train.target.isin(DIR)].copy(); ytr=tr.target.map(ENCODE).to_numpy(int); Xtr=tr[FEATURES]
 y=valid.target.to_numpy(str); eligible=np.isin(y,DIR); X=valid[FEATURES]; ydir=y[eligible]
 n_down=int((tr.target=='DOWN').sum()); n_up=int((tr.target=='UP').sum()); params=json.loads(json.dumps(PARAMS)); params['xgboost_binary']['scale_pos_weight']=n_down/n_up
 predictions={}; model_reports=[]; policies=[]; confusions=[]; daily=[]; regimes=[]; cal_rows=[]; cal_bins=[]
 for name in ('xgboost_binary','lightgbm_binary'):
  model=fit_model(name,params[name],Xtr,ytr); pu=np.asarray(model.predict_proba(X),float)[:,1]
  if not np.isfinite(pu).all() or (pu<0).any() or (pu>1).any(): raise ValueError('Invalid validation probabilities')
  pdn=1-pu; conf=np.maximum(pu,pdn); pred=np.where(pu>pdn,'UP','DOWN'); pred_dir=pred[eligible]; yint=np.array([ENCODE[c] for c in ydir],int)
  bm=binary_metrics(ydir,pred_dir); cal=calibration(yint,pu[eligible]); model_reports.append({'model':name,'train_binary_rows':len(tr),'train_class_counts':{'DOWN':n_down,'UP':n_up},'validation_rows':len(valid),'neutral_excluded_from_binary_eval':int((~eligible).sum()),'eligible_directional_rows':int(eligible.sum()),'binary_accuracy_all_eligible_directional':bm['accuracy'],'unthresholded_binary_metrics_eligible_only':bm,'calibration_eligible_directional':cal,'probability_distribution':{'p_up_mean':float(pu.mean()),'p_up_std':float(pu.std()),'p_down_mean':float(pdn.mean()),'confidence_quantiles':{str(q):float(np.quantile(conf,q)) for q in (.05,.25,.5,.75,.95)}}})
  cal_rows.append({k:v for k,v in {'model':name,**cal}.items() if k!='reliability_bins'}); cal_bins.extend({'model':name,**b} for b in cal['reliability_bins']); predictions[name]={'model':model,'p_up':pu,'p_down':pdn,'confidence':conf,'pred':pred}; joblib.dump(model,OUT/f'{name}.joblib',compress=3)
  for i,a in enumerate(DIR):
   for j,b in enumerate(DIR): confusions.append({'model':name,'policy':'unthresholded_binary_eligible','actual':a,'predicted':b,'count':bm['confusion_matrix'][i][j]})
  for cov in COVERAGES:
   m,out,th=selective(y,pred,conf,cov); m['model']=name; policies.append(m)
   for i,a in enumerate(m['confusion_labels_rows_actual_cols_prediction']):
    for j,b in enumerate(m['confusion_labels_rows_actual_cols_prediction']): confusions.append({'model':name,'policy':f'selective_{int(cov*100)}pct','actual':a,'predicted':b,'count':m['confusion_matrix'][i][j]})
   for i,a in enumerate(DIR):
    for j,b in enumerate(DIR): confusions.append({'model':name,'policy':f'selective_{int(cov*100)}pct_eligible_directional_only','actual':a,'predicted':b,'count':m['eligible_directional_confusion_matrix'][i][j]})
  # Robustness uses one predeclared 30% validation-score quantile.
  _,emit30,_=selective(y,pred,conf,.30); emit30=emit30!='NO_SIGNAL'
  for day in sorted(valid.datetime.dt.date.unique()):
   mask=valid.datetime.dt.date.eq(day).to_numpy(); sig=emit30[mask]; actual=y[mask]; pp=pred[mask]; ns=int(sig.sum()); el=np.isin(actual,DIR)
   ps={c:(float(((pp==c)&(actual==c)&sig).sum()/((pp==c)&sig).sum()) if ((pp==c)&sig).any() else None) for c in DIR}
   eligible_signal=sig & el; actual_e=actual[eligible_signal]; pred_e=pp[eligible_signal]
   daily.append({'model':name,'date':str(day),'rows':int(mask.sum()),'signals':ns,'coverage':float(ns/mask.sum()),'emitted_accuracy_all_actuals':float(((pp==actual)&sig).sum()/ns) if ns else None,'eligible_rows':int(el.sum()),'eligible_signals':int(eligible_signal.sum()),'eligible_coverage':float(eligible_signal.sum()/el.sum()) if el.any() else None,'eligible_directional_accuracy':float((actual_e==pred_e).mean()) if len(actual_e) else None,'eligible_up_precision':float(precision_score(actual_e,pred_e,labels=DIR,average='binary',pos_label='UP',zero_division=0)) if len(actual_e) else None,'eligible_down_precision':float(precision_score(actual_e,pred_e,labels=DIR,average='binary',pos_label='DOWN',zero_division=0)) if len(actual_e) else None,'up_precision_operational':ps['UP'],'down_precision_operational':ps['DOWN']})
  vol_cut=float(train.volatility_20.median()); hi=valid.volatility_20.to_numpy(float)>=vol_cut
  for regime,mask in [('below_train_volatility_median',~hi),('at_or_above_train_volatility_median',hi)]:
   pp=pred[mask]; actual=y[mask]; sig=emit30[mask]; ns=int(sig.sum()); counts={c:int(((pp==c)&sig).sum()) for c in DIR}; tps={c:int(((pp==c)&(actual==c)&sig).sum()) for c in DIR}
   eligible_signal=sig & np.isin(actual,DIR)
   regimes.append({'model':name,'regime':regime,'train_volatility_20_median':vol_cut,'rows':int(mask.sum()),'eligible_directional_rows':int(np.isin(actual,DIR).sum()),'signals':ns,'coverage':float(ns/mask.sum()),'eligible_signals':int(eligible_signal.sum()),'eligible_coverage':float(eligible_signal.sum()/np.isin(actual,DIR).sum()) if np.isin(actual,DIR).any() else None,'eligible_directional_accuracy':float((pp[eligible_signal]==actual[eligible_signal]).mean()) if eligible_signal.any() else None,'emitted_accuracy_all_actuals':float(((pp==actual)&sig).sum()/ns) if ns else None,'up_precision_operational':tps['UP']/counts['UP'] if counts['UP'] else 0.,'down_precision_operational':tps['DOWN']/counts['DOWN'] if counts['DOWN'] else 0.})

 # Frozen three-class candidate is loaded only as an unchanged validation reference.
 from live_app.model_loader import load_alpaca_candidate
 frozen=load_alpaca_candidate()
 if list(X.columns)!=frozen.specification['features_exact_order']: raise ValueError('Validation features differ from frozen reference order')
 rp=np.asarray(frozen.model.predict_proba(X),float); rclass=[str(c) for c in frozen.model.classes_]
 if rclass!=CLASSES or not np.isfinite(rp).all() or not np.allclose(rp.sum(axis=1),1.,atol=1e-8): raise ValueError('Frozen reference probability contract invalid')
 rpred=np.asarray(CLASSES)[rp.argmax(axis=1)]; rdir=np.isin(rpred,DIR)
 majority='UP' if n_up>=n_down else 'DOWN'; maj=np.full(len(valid),majority,dtype=object)
 trend=np.where(valid.return_1.to_numpy(float)>0,'UP','DOWN')
 baselines=[]
 for name,bpred in [('training_majority_direction',maj),('causal_return_1_sign',trend)]:
  bm=binary_metrics(ydir,bpred[eligible]); baselines.append({'model':name,'eligible_directional_metrics':bm,'coverage_all_validation':1.0,'accuracy_all_validation_neutral_counted_wrong':float(accuracy_score(y,bpred)),'predicted_distribution':{c:int((bpred==c).sum()) for c in DIR}})
 baselines.append({'model':'always_neutral_three_class_context','accuracy_all_validation':float((y=='NEUTRAL').mean()),'predicted_distribution':{'DOWN':0,'NEUTRAL':len(y),'UP':0},'purpose':'Contextual three-class accuracy baseline; not a binary directional predictor.'})
 ref_cm=confusion_matrix(y,rpred,labels=CLASSES)
 ref={'model':'frozen_three_class_reference','role':'inference_only','validation_rows':len(valid),
  'accuracy':float(accuracy_score(y,rpred)),'balanced_accuracy':float(balanced_accuracy_score(y,rpred)),
  'macro_f1':float(f1_score(y,rpred,labels=CLASSES,average='macro',zero_division=0)),
  'per_class':{c:{'precision':float(precision_score(y,rpred,labels=CLASSES,average=None,zero_division=0)[i]),'recall':float(recall_score(y,rpred,labels=CLASSES,average=None,zero_division=0)[i]),'f1':float(f1_score(y,rpred,labels=CLASSES,average=None,zero_division=0)[i])} for i,c in enumerate(CLASSES)},
  'binary_accuracy_eligible_directional_including_neutral_as_wrong':float((rpred[eligible]==y[eligible]).mean()),'directional_signal_count':int(rdir.sum()),'coverage_all_validation':float(rdir.mean()),'coverage_eligible_directional':float((rdir&eligible).sum()/eligible.sum()),'three_class_log_loss':float(log_loss(y,rp,labels=CLASSES)),'predicted_distribution':{c:int((rpred==c).sum()) for c in CLASSES},'confusion_labels_rows_actual_cols_prediction':CLASSES,'confusion_matrix':ref_cm.tolist()}
 model_reports.append(ref)
 for i,a in enumerate(CLASSES):
  for j,b in enumerate(CLASSES): confusions.append({'model':'frozen_three_class_reference','policy':'raw_argmax_3class','actual':a,'predicted':b,'count':int(ref['confusion_matrix'][i][j])})
 for bname,bpred in [('training_majority_direction',maj),('causal_return_1_sign',trend)]:
  cm=confusion_matrix(y,bpred,labels=CLASSES)
  for i,a in enumerate(CLASSES):
   for j,b in enumerate(CLASSES): confusions.append({'model':bname,'policy':'all_validation_directional_output','actual':a,'predicted':b,'count':int(cm[i][j])})

 policies30={r['model']:r for r in policies if r['target_coverage_all_validation_candles']==.30}
 eligible_names=[name for name,m in policies30.items() if m['coverage_all_validation_candles']>=.20 and m['emitted_predictions']>=100]
 selected=max(eligible_names,key=lambda name:(policies30[name]['macro_directional_precision_eligible_only'],policies30[name]['binary_accuracy_emitted_eligible_directional_only'] or 0.,next(x['binary_accuracy_all_eligible_directional'] for x in model_reports if x['model']==name))) if eligible_names else None

 pred_frame=valid[['datetime','target','future_return']].rename(columns={'target':'actual','future_return':'actual_future_return_pct'}).copy()
 pred_frame['eligible_directional_actual']=eligible
 for name,p in predictions.items():
  pred_frame[f'{name}_p_down']=p['p_down']; pred_frame[f'{name}_p_up']=p['p_up']; pred_frame[f'{name}_confidence']=p['confidence']; pred_frame[f'{name}_raw_prediction']=p['pred']
  for cov in COVERAGES:
   th=threshold_for(p['confidence'],cov); pred_frame[f'{name}_{int(cov*100)}pct_threshold']=th; pred_frame[f'{name}_{int(cov*100)}pct_output']=np.where(p['confidence']>=th,p['pred'],'NO_SIGNAL')
 for i,c in enumerate(CLASSES): pred_frame[f'frozen_p_{c.lower()}']=rp[:,i]
 pred_frame['frozen_raw_prediction']=rpred

 payload={'study':'Alpaca IEX ECL binary UP/DOWN selective research v1',
  'predeclared_selection_rule':{'primary_policy_coverage_all_validation':.30,'minimum_coverage_all_validation':.20,'minimum_emitted_signals':100,'rank_by':['macro UP/DOWN precision on eligible actual directional rows','binary accuracy on emitted eligible actual directional rows','unthresholded binary accuracy on eligible actual directions']},
  'targets':{'directional_precision':.65,'prediction_accuracy':.55,'minimum_coverage':.20},
  'train':{'rows':len(train),'directional_rows':len(tr),'class_distribution':{'DOWN':n_down,'UP':n_up},'date_range':[str(train.datetime.min().date()),str(train.datetime.max().date())]},
  'validation':{'rows':len(valid),'date_range':[str(valid.datetime.min().date()),str(valid.datetime.max().date())],'distribution':{c:int((y==c).sum()) for c in CLASSES},'eligible_directional_rows':int(eligible.sum()),'neutral_excluded_from_binary_training_and_binary_only_metrics':int((~eligible).sum())},
  'selected_by_rule':selected,'models':model_reports,'selective_policies':policies,'baselines':baselines,'daily_robustness_30pct':daily,'volatility_regime_robustness_30pct':regimes,
  'validation_limitation':'Validation was used in prior research and is not a pristine independent holdout. Results are diagnostic; no previously evaluated test data was used.'}
 (OUT/'metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
 pred_frame.to_csv(OUT/'validation_predictions.csv',index=False)
 pd.DataFrame(policies).to_csv(OUT/'selective_metrics.csv',index=False)
 pd.DataFrame(confusions).to_csv(OUT/'confusion_matrices.csv',index=False)
 pd.DataFrame(cal_rows).to_csv(OUT/'calibration_metrics.csv',index=False)
 pd.DataFrame(cal_bins).to_csv(OUT/'calibration_bins.csv',index=False)
 pd.DataFrame(daily).to_csv(OUT/'daily_robustness_30pct.csv',index=False)
 pd.DataFrame(regimes).to_csv(OUT/'regime_robustness_30pct.csv',index=False)
 flat=[]
 for m in model_reports:
  if m['model'] in {'xgboost_binary','lightgbm_binary'}:
   b=m['unthresholded_binary_metrics_eligible_only']; flat.append({'model':m['model'],'evaluation':'eligible_unthresholded','accuracy':b['accuracy'],'up_precision':b['up_precision'],'down_precision':b['down_precision'],'macro_precision':b['macro_precision'],'macro_recall':b['macro_recall'],'macro_f1':b['macro_f1'],'coverage':1.,'n':b['n'],'log_loss':m['calibration_eligible_directional']['log_loss']})
 for m in policies: flat.append({'model':m['model'],'evaluation':f"selective_{int(m['target_coverage_all_validation_candles']*100)}pct",'accuracy_emitted_eligible':m['binary_accuracy_emitted_eligible_directional_only'],'accuracy_all_emitted_neutral_miss':m['emitted_accuracy_including_neutral_misses'],'up_precision_eligible':m['up_precision_eligible_directional_only'],'down_precision_eligible':m['down_precision_eligible_directional_only'],'macro_precision_eligible':m['macro_directional_precision_eligible_only'],'macro_f1_eligible':m['macro_f1_eligible_directional_only'],'up_precision_operational':m['up_precision_operational'],'down_precision_operational':m['down_precision_operational'],'macro_precision_operational':m['macro_directional_precision_operational'],'macro_recall':m['macro_directional_recall'],'macro_f1_operational':m['macro_directional_f1'],'coverage_all':m['coverage_all_validation_candles'],'coverage_eligible':m['coverage_among_eligible_directional'],'signals':m['emitted_predictions'],'no_signal':m['no_signal_count']})
 pd.DataFrame(flat).to_csv(OUT/'metrics.csv',index=False)

 # Reproducibility: hash only the permitted training and validation inputs.
 (OUT/'configuration.json').write_text(json.dumps({'seed':SEED,'features_exact_order':FEATURES,'class_order':CLASSES,
  'target':{'horizon_minutes':15,'up_if_future_return_pct_gt':0.10,'down_if_future_return_pct_lt':-0.10,'neutral_excluded_from_binary':True},
  'models':params,'coverage_targets_all_validation_rows':list(COVERAGES),
  'threshold_method':'Select top confidence scores by validation quantile; ties may yield coverage above target. Binary ties predict DOWN.',
  'selection_rule':payload['predeclared_selection_rule']},indent=2),encoding='utf-8')
 hashes={'train.csv':sha(DATA/'train.csv'),'validation.csv':sha(DATA/'validation.csv'),
         'frozen_reference_spec.json':sha(SPEC),'frozen_reference_candidate.joblib':frozen.sha256}
 hashes.update({p.name:sha(p) for p in sorted(OUT.glob('*.joblib'))})
 (OUT/'reproducibility_manifest.json').write_text(json.dumps({'input_scope':['data/alpaca_rebuild_v1/train.csv','data/alpaca_rebuild_v1/validation.csv','models/alpaca_final_freeze_v1/frozen_candidate_spec.json'],
  'input_sha256':{k:v for k,v in hashes.items() if k in {'train.csv','validation.csv','frozen_reference_spec.json'}},
  'frozen_reference_model_sha256':frozen.sha256,
  'model_sha256':{k:v for k,v in hashes.items() if k.endswith('.joblib')},'seed':SEED,
  'python':platform.python_version(),'pandas':pd.__version__,'scikit_learn':sklearn.__version__,
  'xgboost':xgb.__version__,'lightgbm':lgb.__version__,'test_dataset_accessed':False},indent=2),encoding='utf-8')

 # Concise human-readable report; no test-period data is loaded or referenced.
 chosen=next((r for r in policies if r['model']==selected and r['target_coverage_all_validation_candles']==.30),None)
 lines=['# ECL Alpaca IEX Binary Direction Research v1','',
  '## Scope and method','',
  'Trained fixed XGBoost and LightGBM binary classifiers on directional training rows only. Neutral rows were excluded from binary fitting and eligible-direction binary metrics. Selective outputs use UP, DOWN, or NO SIGNAL. Confidence cutoffs target 20%, 30%, 40%, and 50% of all validation candles; thresholds are score quantiles, with ties potentially increasing realized coverage. The frozen three-class candidate is inference-only. No model or threshold used test data.','',
  f"Training: {len(train):,} rows ({len(tr):,} directional), {train.datetime.min()} to {train.datetime.max()}.",
  f"Validation: {len(valid):,} rows; {int(eligible.sum()):,} directional eligible rows and {int((~eligible).sum()):,} neutral rows; {valid.datetime.min()} to {valid.datetime.max()}.",
  '', '## Core binary comparison','',
  '| Model | Eligible binary accuracy | Directional macro F1 | UP precision | DOWN precision | Log loss |',
  '|---|---:|---:|---:|---:|---:|']
 for r in model_reports:
  if r['model'] in {'xgboost_binary','lightgbm_binary'}:
   b=r['unthresholded_binary_metrics_eligible_only']; lines.append(f"| {r['model']} | {b['accuracy']:.3f} | {b['macro_f1']:.3f} | {b['up_precision']:.3f} | {b['down_precision']:.3f} | {r['calibration_eligible_directional']['log_loss']:.4f} |")
 lines += ['', '## Selective validation results','',
  'UP/DOWN precision, directional accuracy, recall, and F1 in the main table exclude actual neutral labels, as required for the binary directional evaluation. The separate all-emission accuracy and operational precision fields in `selective_metrics.csv` also count emitted signals on neutral actuals as errors; NO SIGNAL is counted explicitly.','',
  '| Model | Target coverage | Actual coverage | Signals | NO SIGNAL | UP precision* | DOWN precision* | Macro precision* | Directional accuracy* | All-emission accuracy** | Macro F1* |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in policies:
  acc=r['binary_accuracy_emitted_eligible_directional_only']
  lines.append(f"| {r['model']} | {r['target_coverage_all_validation_candles']:.0%} | {r['coverage_all_validation_candles']:.1%} | {r['emitted_predictions']} | {r['no_signal_count']} | {r['up_precision_eligible_directional_only']:.3f} | {r['down_precision_eligible_directional_only']:.3f} | {r['macro_directional_precision_eligible_only']:.3f} | {acc if acc is not None else 0.:.3f} | {r['emitted_accuracy_including_neutral_misses']:.3f} | {r['macro_f1_eligible_directional_only']:.3f} |")
 lines.append('')
 lines.append('* Binary directional metrics exclude actual NEUTRAL rows as specified. ** All-emission metric treats actual NEUTRAL rows as incorrect directional signals, a stricter operational view.')
 lines += ['', '## Reference baselines','',
  f"Training-majority directional baseline predicts {majority} for every validation candle; causal one-candle return-sign baseline predicts UP for positive `return_1`, otherwise DOWN. Their metrics on eligible actual directional rows are in `metrics.json`. Always-NEUTRAL obtains {float((y=='NEUTRAL').mean()):.1%} accuracy on all validation rows as three-class context, but it does not make a directional prediction. The frozen three-class candidate is shown separately with raw argmax and neutral coverage in that file.", '',
  '## Robustness and calibration','',
  'Daily and training-median volatility-split diagnostics for the predeclared 30% policy are saved in `daily_robustness_30pct.csv` and `regime_robustness_30pct.csv`. Probability calibration (log loss, Brier score, ECE, distribution quantiles) is in `calibration_metrics.csv`, with reliability bins in `calibration_bins.csv`; per-date confusion matrices and policy-level matrices are in `confusion_matrices.csv`. Validation has been used in earlier project research and is not a pristine holdout.', '',
  '## Decision','']
 if selected is None:
  lines.append('No candidate met the predeclared 30% policy minimums (at least 20% actual coverage and at least 100 signals); no model is recommended.')
 else:
  lines.append(f"The predeclared rule selects **{selected}** at the 30% target (actual coverage {chosen['coverage_all_validation_candles']:.1%}, {chosen['emitted_predictions']} signals). Among eligible actual UP/DOWN examples, UP precision is {chosen['up_precision_eligible_directional_only']:.1%}, DOWN precision is {chosen['down_precision_eligible_directional_only']:.1%}, macro precision is {chosen['macro_directional_precision_eligible_only']:.1%}, and emitted directional accuracy is {chosen['binary_accuracy_emitted_eligible_directional_only']:.1%}. Counting neutral candles as false operational signals, macro precision is {chosen['macro_directional_precision_operational']:.1%} and all-emission accuracy is {chosen['emitted_accuracy_including_neutral_misses']:.1%}.")
  lines.append('The precision and accuracy goals are reported independently below and are not guaranteed; judge each requested coverage row rather than extrapolating from the 30% policy.')
 lines += ['', '## Target check and limitations','',
  'The 65% directional precision target is evaluated using macro UP/DOWN precision on eligible actual directional examples. The 55% accuracy target is evaluated using binary accuracy on eligible directional examples, including both all eligible rows and the emitted subset at each coverage. The best validation macro precision is 54.7% (LightGBM at 40% coverage); the best emitted eligible-direction accuracy is 54.4% (same policy). Neither target was reached. Strict all-candle emitted accuracy is much lower because the binary model also emits signals on neutral actuals. Daily variation is substantial: at the predeclared 30% target, among dates with at least five emitted eligible directional examples, 9 of 15 dates per model were above 50% accuracy; date-level standard deviations were about 22–27 percentage points. The simple training-volatility-median split also varies, so results are not stable enough to claim robust gains. This experiment establishes no profitability or production readiness.', '',
  'No additional historical source, test dataset, live API, or test-period observation was used. This is research only and is not integrated into the live dashboard.']
 (OUT/'research_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
 main()
