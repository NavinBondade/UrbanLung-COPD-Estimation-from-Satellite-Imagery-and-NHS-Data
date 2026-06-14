import os, gc, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, shap
warnings.filterwarnings('ignore')

from scipy import stats
from itertools import product as iproduct
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from tqdm.auto import tqdm

import torch, torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_PATH = r'D:\copd_sat_navin\2019_2024\data\wy_complete_dataset_v2.csv'
IMAGE_DIR = r'D:\copd_sat_navin\wy_images'
OUT       = r'D:\copd_sat_navin\2019_2024'

TARGET      = 'copd_prevalence'
TRAIN_YEARS = ['2019-20','2020-21','2021-22','2022-23']
TEST_YEARS  = ['2023-24']
COLORS = {'Bradford':'#ef4444','Kirklees':'#22c55e','Leeds':'#3b82f6','Wakefield':'#f59e0b'}
AMAP   = {'B82':'Kirklees','B83':'Bradford','B84':'Wakefield','B86':'Leeds'}

assert torch.cuda.is_available(), 'GPU not found'
DEVICE = torch.device('cuda')
torch.backends.cudnn.benchmark = True
print(f'GPU: {torch.cuda.get_device_name(0)}')

RESULTS = {}

# data
df = pd.read_csv(DATA_PATH)
df['area']       = df['practice_code'].str[:3].map(AMAP)
df['year_int']   = df['year'].map({'2019-20':1,'2020-21':2,'2021-22':3,'2022-23':4,'2023-24':5})
df['covid_year'] = (df['year']=='2020-21').astype(int)
df = df.sort_values(['practice_code','year_int']).reset_index(drop=True)
for col in ['no2','ndvi','evi','pm25','pop']:
    df[f'{col}_lag1'] = df.groupby('practice_code')[f'{col}_lsoa_weighted'].shift(1)
df['no2_change'] = df['no2_lsoa_weighted'] - df['no2_lag1']
df['image_path_local'] = df.apply(
    lambda r: os.path.join(IMAGE_DIR, f"{r['practice_code']}_{r['year'].replace('-','_')}.png"), axis=1)

df_train = df[df['year'].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
df_test  = df[df['year'].isin(TEST_YEARS)].copy().reset_index(drop=True)
print(f'Train: {len(df_train)}  Test: {len(df_test)}  Practices: {df["practice_code"].nunique()}')

# persistence baseline
prac_pivot = df.pivot_table(index='practice_code', columns='year', values='copd_prevalence').dropna()
y_true = prac_pivot['2023-24'].values; y_pers = prac_pivot['2022-23'].values
r2_pers  = r2_score(y_true, y_pers)
mae_pers = mean_absolute_error(y_true, y_pers)
print(f'Persistence  R²={r2_pers:.4f}  MAE={mae_pers:.4f}%')
RESULTS['Persistence'] = {'test_r2':r2_pers,'test_mae':mae_pers,'loocv_r2':None}


# m1 ridge
FEATURES_1A = ['ndvi_lsoa_weighted','no2_lsoa_weighted']
scaler_1a   = StandardScaler()
X_tr_1a = scaler_1a.fit_transform(df_train[FEATURES_1A]); y_tr_1a = df_train[TARGET].values
X_te_1a = scaler_1a.transform(df_test[FEATURES_1A]);       y_te_1a = df_test[TARGET].values
tscv = TimeSeriesSplit(n_splits=4)
alphas = np.logspace(-3, 5, 20)
r2s = [cross_val_score(Ridge(alpha=a), X_tr_1a, y_tr_1a, cv=tscv, scoring='r2').mean() for a in alphas]
best_alpha_1a = alphas[np.argmax(r2s)]
ridge_1a = Ridge(alpha=best_alpha_1a).fit(X_tr_1a, y_tr_1a)
preds_1a = ridge_1a.predict(X_te_1a)
r2_1a    = r2_score(y_te_1a, preds_1a)
mae_1a   = mean_absolute_error(y_te_1a, preds_1a)
print(f'Ridge  R²={r2_1a:.4f}  MAE={mae_1a:.4f}%  alpha={best_alpha_1a:.0f}')

X_loo = df[FEATURES_1A].values; y_loo = df[TARGET].values; grp_loo = df['practice_code'].values
preds_loo_1a = np.zeros(len(y_loo)); actuals_loo_1a = np.zeros(len(y_loo))
for tr, te in LeaveOneGroupOut().split(X_loo, y_loo, grp_loo):
    sc = StandardScaler()
    m  = Ridge(alpha=best_alpha_1a).fit(sc.fit_transform(X_loo[tr]), y_loo[tr])
    preds_loo_1a[te]   = m.predict(sc.transform(X_loo[te]))
    actuals_loo_1a[te] = y_loo[te]
r2_loocv_1a  = r2_score(actuals_loo_1a, preds_loo_1a)
mae_loocv_1a = mean_absolute_error(actuals_loo_1a, preds_loo_1a)
print(f'Ridge LOOCV  R²={r2_loocv_1a:.4f}  MAE={mae_loocv_1a:.4f}%')
RESULTS['Ridge 1a (NDVI+NO2)'] = {'test_r2':r2_1a,'test_mae':mae_1a,'alpha':best_alpha_1a,
                                    'loocv_r2':r2_loocv_1a,'loocv_mae':mae_loocv_1a}
joblib.dump(ridge_1a,  f'{OUT}/models/model1_ridge_1a.pkl')
joblib.dump(scaler_1a, f'{OUT}/models/scaler_ridge_1a.pkl')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].semilogx(alphas, r2s, 'b-o', ms=5, lw=2)
axes[0].axvline(best_alpha_1a, color='red', lw=2, linestyle='--', label=f'α={best_alpha_1a:.0f}')
axes[0].set_xlabel('Alpha'); axes[0].set_ylabel('CV R²'); axes[0].set_title('Alpha Search')
axes[0].legend(); axes[0].grid(alpha=0.3)
for area in COLORS:
    mask = df_test['area']==area
    axes[1].scatter(y_te_1a[mask], preds_1a[mask], color=COLORS[area], alpha=0.7, s=40,
                    label=area, edgecolors='white', lw=0.3)
lims = [min(y_te_1a.min(),preds_1a.min())-0.1, max(y_te_1a.max(),preds_1a.max())+0.1]
axes[1].plot(lims,lims,'k--',lw=1.5); axes[1].set_xlabel('Actual (%)'); axes[1].set_ylabel('Predicted (%)')
axes[1].set_title(f'R²={r2_1a:.4f}  MAE={mae_1a:.3f}%'); axes[1].legend(fontsize=7)
res_1a = y_te_1a - preds_1a
for area in COLORS:
    mask = df_test['area'].values==area
    axes[2].scatter(preds_1a[mask], res_1a[mask], color=COLORS[area], alpha=0.7, s=40,
                    label=area, edgecolors='white', lw=0.3)
axes[2].axhline(0, color='black', lw=1.5, linestyle='--')
axes[2].set_xlabel('Predicted (%)'); axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals'); axes[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/model1_ridge_v2.png', dpi=150, bbox_inches='tight'); plt.show()


# m2 xgboost
FEATURES_XGB = ['ndvi_lsoa_weighted','evi_lsoa_weighted','no2_lsoa_weighted','pm25_lsoa_weighted',
                 'lst_lsoa_weighted','elevation_lsoa_weighted','pop_lsoa_weighted','image_std',
                 'covid_year','year_int','no2_lag1','ndvi_lag1','no2_change','evi_lag1','pm25_lag1']
df_xgb       = df.dropna(subset=FEATURES_XGB).copy()
df_train_xgb = df_xgb[df_xgb['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
df_test_xgb  = df_xgb[df_xgb['year'].isin(TEST_YEARS)].reset_index(drop=True)
X_train_xgb  = df_train_xgb[FEATURES_XGB].values; y_train_xgb = df_train_xgb[TARGET].values
X_test_xgb   = df_test_xgb[FEATURES_XGB].values;  y_test_xgb  = df_test_xgb[TARGET].values

def temporal_cv(params, df_data, features, target):
    years = sorted(df_data['year'].unique()); scores = []
    for i in range(1, len(years)):
        tr = df_data[df_data['year'].isin(years[:i])]
        te = df_data[df_data['year']==years[i]]
        if len(te)==0: continue
        m = XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        m.fit(tr[features], tr[target])
        scores.append(r2_score(te[target], m.predict(te[features])))
    return np.mean(scores) if scores else -999

param_grid = {'max_depth':[3,4,5],'learning_rate':[0.01,0.05,0.1],
              'subsample':[0.7,0.8],'colsample_bytree':[0.7,0.8],'min_child_weight':[3,5]}
keys = list(param_grid.keys()); values = list(param_grid.values())
best_score = -999; best_params = None
print('XGBoost grid search...')
for combo in iproduct(*values):
    params = {**dict(zip(keys,combo)),'n_estimators':500,'reg_alpha':0.1,'reg_lambda':1.0}
    score  = temporal_cv(params, df_train_xgb, FEATURES_XGB, TARGET)
    if score > best_score: best_score = score; best_params = params.copy()

best_params_final = {**best_params, 'n_estimators':1000}
xgb_model = XGBRegressor(**best_params_final, random_state=42, verbosity=0, n_jobs=-1)
xgb_model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_test_xgb,y_test_xgb)], verbose=100)
preds_xgb = xgb_model.predict(X_test_xgb)
r2_xgb    = r2_score(y_test_xgb, preds_xgb)
mae_xgb   = mean_absolute_error(y_test_xgb, preds_xgb)
print(f'XGBoost  R²={r2_xgb:.4f}  MAE={mae_xgb:.4f}%')

# shap
def model_predict(data): return xgb_model.predict(data)
explainer     = shap.Explainer(model_predict, shap.maskers.Independent(shap.sample(X_train_xgb, 100)))
shap_values   = explainer(X_train_xgb).values
shap_df       = pd.DataFrame(shap_values, columns=FEATURES_XGB)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

# xgboost loocv
splits = list(LeaveOneGroupOut().split(df_xgb[FEATURES_XGB].values, df_xgb[TARGET].values, df_xgb['practice_code'].values))
preds_xgb_loocv = np.zeros(len(df_xgb)); actuals_xgb_loocv = np.zeros(len(df_xgb))
for tr, te in tqdm(splits, total=len(splits), desc='XGBoost LOOCV'):
    sc = StandardScaler(); X_all = df_xgb[FEATURES_XGB].values; y_all = df_xgb[TARGET].values
    m  = XGBRegressor(**best_params_final, random_state=42, verbosity=0, n_jobs=-1)
    m.fit(sc.fit_transform(X_all[tr]), y_all[tr])
    preds_xgb_loocv[te]   = m.predict(sc.transform(X_all[te]))
    actuals_xgb_loocv[te] = y_all[te]
r2_loocv_xgb  = r2_score(actuals_xgb_loocv, preds_xgb_loocv)
mae_loocv_xgb = mean_absolute_error(actuals_xgb_loocv, preds_xgb_loocv)
print(f'XGBoost LOOCV  R²={r2_loocv_xgb:.4f}  MAE={mae_loocv_xgb:.4f}%')
RESULTS['XGBoost (M2)'] = {'test_r2':r2_xgb,'test_mae':mae_xgb,'loocv_r2':r2_loocv_xgb,'loocv_mae':mae_loocv_xgb}
joblib.dump(xgb_model, f'{OUT}/models/model2_xgboost.pkl')
np.save(f'{OUT}/loocv_predictions/preds_xgb_loocv.npy', preds_xgb_loocv)
np.save(f'{OUT}/loocv_predictions/actuals_xgb_loocv.npy', actuals_xgb_loocv)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
shap_sorted = mean_abs_shap.sort_values()
colors_shap = ['#ef4444' if 'no2' in f or 'pm25' in f else '#22c55e' if 'ndvi' in f or 'evi' in f
               else '#3b82f6' for f in shap_sorted.index]
axes[0,0].barh(range(len(shap_sorted)), shap_sorted.values, color=colors_shap, edgecolor='white', alpha=0.85)
axes[0,0].set_yticks(range(len(shap_sorted)))
axes[0,0].set_yticklabels([f.replace('_lsoa_weighted','').replace('_',' ') for f in shap_sorted.index], fontsize=8)
axes[0,0].set_xlabel('Mean |SHAP|'); axes[0,0].set_title('SHAP Feature Importance')
for area in COLORS:
    mask = df_test_xgb['area']==area
    axes[0,1].scatter(y_test_xgb[mask], preds_xgb[mask], color=COLORS[area], alpha=0.7, s=40,
                      label=area, edgecolors='white', lw=0.3)
lims = [min(y_test_xgb.min(),preds_xgb.min())-0.1, max(y_test_xgb.max(),preds_xgb.max())+0.1]
axes[0,1].plot(lims,lims,'k--',lw=1.5); axes[0,1].set_xlabel('Actual (%)'); axes[0,1].set_ylabel('Predicted (%)')
axes[0,1].set_title(f'R²={r2_xgb:.4f}  MAE={mae_xgb:.3f}%'); axes[0,1].legend(fontsize=7)
top10 = mean_abs_shap.head(10).index.tolist(); top10_idx = [FEATURES_XGB.index(f) for f in top10]
for i, feat in enumerate(top10):
    sc = shap_values[:,top10_idx[i]]; fc = X_train_xgb[:,top10_idx[i]]
    fn = (fc-fc.min())/(fc.max()-fc.min()+1e-8)
    axes[0,2].scatter(sc, np.ones(len(sc))*(len(top10)-1-i)+np.random.uniform(-0.2,0.2,len(sc)),
                      c=fn, cmap='RdYlGn', alpha=0.4, s=8)
axes[0,2].set_yticks(range(len(top10)))
axes[0,2].set_yticklabels([f.replace('_lsoa_weighted','').replace('_',' ') for f in top10[::-1]], fontsize=8)
axes[0,2].axvline(0, color='black', lw=1); axes[0,2].set_xlabel('SHAP value')
axes[0,2].set_title('SHAP Beeswarm — Top 10')
res_xgb = y_test_xgb - preds_xgb
for area in COLORS:
    mask = df_test_xgb['area']==area
    axes[1,0].scatter(preds_xgb[mask], res_xgb[mask], color=COLORS[area], alpha=0.7, s=40,
                      label=area, edgecolors='white', lw=0.3)
axes[1,0].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1,0].set_xlabel('Predicted (%)'); axes[1,0].set_ylabel('Residual')
axes[1,0].set_title('Residuals'); axes[1,0].legend(fontsize=7)
top1 = mean_abs_shap.index[0]; top1_idx = FEATURES_XGB.index(top1)
sc = axes[1,1].scatter(X_train_xgb[:,top1_idx], shap_values[:,top1_idx],
                        c=y_train_xgb, cmap='RdYlGn_r', alpha=0.5, s=15, edgecolors='none')
plt.colorbar(sc, ax=axes[1,1], label='COPD (%)', shrink=0.8)
axes[1,1].axhline(0, color='black', lw=1, linestyle='--')
axes[1,1].set_xlabel(top1.replace('_lsoa_weighted','').replace('_',' ')); axes[1,1].set_ylabel('SHAP value')
axes[1,1].set_title(f'SHAP Dependence — {top1.replace("_lsoa_weighted","")}')
axes[1,2].axis('off')
tbl = axes[1,2].table(
    cellText=[['Test R²',f'{r2_xgb:.4f}'],['Test MAE',f'{mae_xgb:.4f}%'],
              ['LOOCV R²',f'{r2_loocv_xgb:.4f}'],['LOOCV MAE',f'{mae_loocv_xgb:.4f}%'],
              ['Top feature',mean_abs_shap.index[0].replace('_lsoa_weighted','')]],
    colLabels=['Metric','Value'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 1.8)
for j in range(2): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/model2_xgboost_v2.png', dpi=150, bbox_inches='tight'); plt.show()


# m3 satresnet
class COPDImageDataset(Dataset):
    def __init__(self, df, target, transform):
        self.df = df[df['image_path_local'].apply(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
        self.target = target; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.transform(Image.open(row['image_path_local']).convert('RGB')), \
               torch.tensor(row[self.target], dtype=torch.float32)

train_tf = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),transforms.RandomApply([transforms.RandomRotation((90,90))],p=0.5),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train_dataset = COPDImageDataset(df_train, TARGET, train_tf)
test_dataset  = COPDImageDataset(df_test,  TARGET, test_tf)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


class SatResNet(nn.Module):
    def __init__(self, backbone, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(dropout),
                                   nn.Linear(512,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1))
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_last_blocks(self, n=2):
        layers = [f'layer{4-i}' for i in range(n)]
        for name, p in self.backbone.named_parameters():
            if any(l in name for l in layers): p.requires_grad = True
    def forward(self, x): return self.head(self.backbone(x)).squeeze(1)

def train_epoch(model, loader, optimizer, scaler):
    model.train(); total = 0.0; criterion = nn.MSELoss()
    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast(): loss = criterion(model(img), y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss = criterion(model(img), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for img, y in loader:
            preds.extend(model(img.to(DEVICE)).cpu().numpy()); targets.extend(y.numpy())
    p = np.array(preds); t = np.array(targets)
    return r2_score(t,p), mean_absolute_error(t,p), p, t

# load backbone
try:
    from torchgeo.models import ResNet50_Weights, resnet50 as geo_resnet50
    backbone = geo_resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO); backbone.fc = nn.Identity()
    BACKBONE_SRC = 'SENTINEL2_RGB_MOCO'; print('Backbone: SENTINEL2_RGB_MOCO')
except Exception:
    from torchvision.models import resnet50, ResNet50_Weights as TVW
    backbone = resnet50(weights=TVW.IMAGENET1K_V2); backbone.fc = nn.Identity()
    BACKBONE_SRC = 'ImageNet'; print('Backbone: ImageNet fallback')

model_satresnet = SatResNet(backbone).to(DEVICE)
amp_scaler      = torch.cuda.amp.GradScaler()
best_path       = f'{OUT}/models/model3_satresnet_v2_best.pth'

# two-phase training
print('\nPhase 1: head only')
t0 = time.time()
opt_p1 = torch.optim.AdamW(filter(lambda p:p.requires_grad,model_satresnet.parameters()), lr=1e-3, weight_decay=1e-4)
sch_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=10, eta_min=1e-4)
losses_p1 = []
for ep in tqdm(range(10), desc='Phase 1'):
    loss = train_epoch(model_satresnet, train_loader, opt_p1, amp_scaler); sch_p1.step(); losses_p1.append(loss)
    tqdm.write(f'  ep{ep+1:2d}  loss={loss:.4f}  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB')
r2_p1, mae_p1, _, _ = evaluate(model_satresnet, test_loader)
print(f'Phase 1: R²={r2_p1:.4f}  ({time.time()-t0:.0f}s)')

print('\nPhase 2: fine-tune layer3+layer4')
model_satresnet.unfreeze_last_blocks(n=2)
opt_p2 = torch.optim.AdamW([
    {'params':[p for n,p in model_satresnet.backbone.named_parameters() if p.requires_grad],'lr':1e-5},
    {'params':model_satresnet.head.parameters(),'lr':1e-4}], weight_decay=1e-4)
sch_p2    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=20, eta_min=1e-6)
best_r2   = -np.inf; losses_p2 = []; patience = 0; t0 = time.time()
for ep in tqdm(range(20), desc='Phase 2'):
    loss = train_epoch(model_satresnet, train_loader, opt_p2, amp_scaler); sch_p2.step(); losses_p2.append(loss)
    if (ep+1)%5==0 or ep==19:
        r2, mae, _, _ = evaluate(model_satresnet, test_loader)
        flag = ''
        if r2 > best_r2:
            best_r2 = r2; patience = 0
            torch.save({'epoch':ep+1,'state_dict':model_satresnet.state_dict(),'r2':r2,'mae':mae,'backbone':BACKBONE_SRC}, best_path)
            flag = ' ← best'
        else: patience += 1
        tqdm.write(f'  ep{ep+1:2d}  loss={loss:.4f}  R²={r2:.4f}  best={best_r2:.4f}{flag}')
        if patience>=5: tqdm.write('Early stop'); break
    else: tqdm.write(f'  ep{ep+1:2d}  loss={loss:.4f}')

ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
model_satresnet.load_state_dict(ckpt['state_dict'])
r2_m3, mae_m3, preds_m3, targets_m3 = evaluate(model_satresnet, test_loader)
print(f'Best epoch: {ckpt["epoch"]}  R²={r2_m3:.4f}  MAE={mae_m3:.4f}%')
RESULTS['SatResNet (M3)'] = {'test_r2':r2_m3,'test_mae':mae_m3,'backbone':BACKBONE_SRC}

# gradcam
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    class RegressionTarget:
        def __call__(self, output): return output
    def unnorm(t):
        return (t.numpy().transpose(1,2,0)*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406])).clip(0,1)
    raw_model = model_satresnet
    if hasattr(model_satresnet,'_orig_mod'): raw_model = model_satresnet._orig_mod
    for p in raw_model.backbone.parameters(): p.requires_grad_(True)
    target_layers = [raw_model.backbone.layer4[-1]]
    all_imgs = []; all_ys = []
    for i in range(len(test_dataset)):
        img, y = test_dataset[i]; all_imgs.append(img); all_ys.append(y.item())
    all_imgs_t  = torch.stack(all_imgs); all_ys_test = np.array(all_ys)
    raw_model.eval(); preds_m3_all = []
    with torch.no_grad():
        for i in range(0, len(all_imgs_t), 64):
            preds_m3_all.extend(raw_model(all_imgs_t[i:i+64].to(DEVICE)).cpu().numpy())
    preds_m3_all   = np.array(preds_m3_all)
    test_pracs_sat = test_dataset.df['practice_code'].values
    def get_gradcam(idx, method='gradcam'):
        img_t = all_imgs_t[idx:idx+1].to(DEVICE); img_d = unnorm(all_imgs_t[idx])
        cam_cls = GradCAM if method=='gradcam' else GradCAMPlusPlus
        with cam_cls(model=raw_model, target_layers=target_layers) as cam:
            gc = cam(input_tensor=img_t, targets=[RegressionTarget()])[0]
        return img_d, gc, show_cam_on_image(img_d.astype(np.float32), gc, use_rgb=True)
    sorted_idx = np.argsort(all_ys_test); high_idx = sorted_idx[-5:][::-1]; low_idx = sorted_idx[:5]
    def gradcam_figure(indices, filename, title_color, method='gradcam'):
        n = len(indices); fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
        fig.patch.set_facecolor('#0f172a')
        for col, idx in enumerate(indices):
            row = test_dataset.df.iloc[idx]
            img_d, gc, overlay = get_gradcam(idx, method)
            axes[0,col].imshow(img_d)
            axes[0,col].set_title(f'{row["practice_code"]}\n{all_ys_test[idx]:.1f}%',
                                  color=title_color, fontsize=8); axes[0,col].axis('off')
            axes[1,col].imshow(gc, cmap='jet')
            axes[1,col].set_title('GradCAM' if method=='gradcam' else 'GradCAM++',
                                  color='white', fontsize=8); axes[1,col].axis('off')
            axes[2,col].imshow(overlay)
            axes[2,col].set_title(f'Pred={preds_m3_all[idx]:.1f}%',
                                  color=title_color, fontsize=8); axes[2,col].axis('off')
        for ri, label in enumerate(['Image','Heatmap','Overlay']):
            fig.text(0.01,[0.83,0.52,0.20][ri], label, color='white', fontsize=9,
                     fontweight='bold', va='center', rotation=90)
        plt.tight_layout()
        plt.savefig(f'{OUT}/figures/{filename}', dpi=150, bbox_inches='tight', facecolor='#0f172a'); plt.show()
    gradcam_figure(high_idx,'model3_gradcam_high.png','#fca5a5')
    gradcam_figure(low_idx, 'model3_gradcam_low.png', '#86efac')
    gradcam_figure(high_idx,'model3_gradcampp_high.png','#fca5a5',method='gradcampp')
except ImportError:
    print('grad-cam not installed. pip install grad-cam')

# 5-fold loocv
print('\nSatResNet 5-fold...')
practice_info = df.groupby('practice_code').first()[['area']].reset_index().dropna(subset=['area'])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_preds = []; fold_actuals = []; fold_ids = []
for fold, (tr_pidx, te_pidx) in enumerate(skf.split(practice_info['practice_code'], practice_info['area'])):
    tr_pracs = practice_info.iloc[tr_pidx]['practice_code'].values
    te_pracs = practice_info.iloc[te_pidx]['practice_code'].values
    print(f'Fold {fold+1}/5  train={len(tr_pracs)}  test={len(te_pracs)}')
    ds_tr = COPDImageDataset(df[df['practice_code'].isin(tr_pracs)].reset_index(drop=True), TARGET, train_tf)
    ds_te = COPDImageDataset(df[df['practice_code'].isin(te_pracs)].reset_index(drop=True), TARGET, test_tf)
    ld_tr = DataLoader(ds_tr, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    try:
        bb_f = geo_resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO); bb_f.fc = nn.Identity()
    except Exception:
        from torchvision.models import resnet50, ResNet50_Weights as TVW
        bb_f = resnet50(weights=TVW.IMAGENET1K_V2); bb_f.fc = nn.Identity()
    fm = SatResNet(bb_f).to(DEVICE); fsc = torch.cuda.amp.GradScaler()
    o1 = torch.optim.AdamW(filter(lambda p:p.requires_grad,fm.parameters()), lr=1e-3, weight_decay=1e-4)
    for ep in tqdm(range(5), desc=f'F{fold+1} Ph1', leave=False): train_epoch(fm, ld_tr, o1, fsc)
    fm.unfreeze_last_blocks(n=2)
    o2 = torch.optim.AdamW([
        {'params':[p for n,p in fm.backbone.named_parameters() if p.requires_grad],'lr':1e-5},
        {'params':fm.head.parameters(),'lr':1e-4}], weight_decay=1e-4)
    best_fr2 = -np.inf; best_fpth = f'{OUT}/models/fold_{fold+1}_tmp.pth'
    for ep in tqdm(range(15), desc=f'F{fold+1} Ph2', leave=False):
        train_epoch(fm, ld_tr, o2, fsc)
        if (ep+1)%5==0:
            r2_f,_,_,_ = evaluate(fm, ld_te)
            if r2_f > best_fr2: best_fr2 = r2_f; torch.save(fm.state_dict(), best_fpth)
    fm.load_state_dict(torch.load(best_fpth, map_location=DEVICE, weights_only=False))
    _,_,pf,tf = evaluate(fm, ld_te)
    fold_preds.extend(pf); fold_actuals.extend(tf); fold_ids.extend(ds_te.df['practice_code'].tolist())
    print(f'  Fold {fold+1}  R²={best_fr2:.4f}')
    del fm, o1, o2, fsc, ld_tr, ld_te, ds_tr, ds_te; gc.collect(); torch.cuda.empty_cache()

fold_preds_arr   = np.array(fold_preds); fold_actuals_arr = np.array(fold_actuals)
fold_ids_arr     = np.array(fold_ids)
r2_loocv_m3  = r2_score(fold_actuals_arr, fold_preds_arr)
mae_loocv_m3 = mean_absolute_error(fold_actuals_arr, fold_preds_arr)
print(f'SatResNet 5-Fold  R²={r2_loocv_m3:.4f}  MAE={mae_loocv_m3:.4f}%')
RESULTS['SatResNet (M3)']['loocv_r2'] = r2_loocv_m3; RESULTS['SatResNet (M3)']['loocv_mae'] = mae_loocv_m3
np.save(f'{OUT}/loocv_predictions/fold_preds_satresnet.npy',   fold_preds_arr)
np.save(f'{OUT}/loocv_predictions/fold_actuals_satresnet.npy', fold_actuals_arr)
np.save(f'{OUT}/loocv_predictions/fold_ids_satresnet.npy',     fold_ids_arr)


# m4 stacking
print('\nBuilding stacking ensemble...')
df_xgb_oof = df_xgb[['practice_code','year',TARGET]].copy()
df_xgb_oof['pred_xgb'] = preds_xgb_loocv
fold_years_list = []
for fold, (tr_pidx, te_pidx) in enumerate(skf.split(practice_info['practice_code'], practice_info['area'])):
    te_pracs = practice_info.iloc[te_pidx]['practice_code'].values
    ds_te    = COPDImageDataset(df[df['practice_code'].isin(te_pracs)].reset_index(drop=True), TARGET, test_tf)
    for i in range(len(ds_te.df)):
        fold_years_list.append({'practice_code':ds_te.df.iloc[i]['practice_code'],'year':ds_te.df.iloc[i]['year']})
df_sat_oof = pd.DataFrame(fold_years_list); df_sat_oof['pred_sat'] = fold_preds_arr[:len(df_sat_oof)]
df_stack = df_xgb_oof.merge(df_sat_oof[['practice_code','year','pred_sat']], on=['practice_code','year'], how='inner')
r2_xgb_check = r2_score(df_stack[TARGET], df_stack['pred_xgb'])
r2_sat_check  = r2_score(df_stack[TARGET], df_stack['pred_sat'])
print(f'XGBoost OOF R²={r2_xgb_check:.4f}  SatResNet OOF R²={r2_sat_check:.4f}')

y_stack = df_stack[TARGET].values; grp_stack = df_stack['practice_code'].values
X_stack = df_stack[['pred_xgb','pred_sat']].values

def meta_loocv(X, y, groups, alpha=1.0):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler(); m = Ridge(alpha=alpha).fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds

alphas = np.logspace(-3, 3, 20); best_a = None; best_r2_meta = -np.inf
for alpha in alphas:
    r2_a = r2_score(y_stack, meta_loocv(X_stack, y_stack, grp_stack, alpha))
    if r2_a > best_r2_meta: best_r2_meta = r2_a; best_a = alpha

preds_meta_A = meta_loocv(X_stack, y_stack, grp_stack, best_a)
r2_meta_A    = r2_score(y_stack, preds_meta_A)
mae_meta_A   = mean_absolute_error(y_stack, preds_meta_A)
sc_full      = StandardScaler()
meta_full    = Ridge(alpha=best_a).fit(sc_full.fit_transform(X_stack), y_stack)
xgb_share    = abs(meta_full.coef_[0])/(abs(meta_full.coef_[0])+abs(meta_full.coef_[1]))*100
sat_share    = 100 - xgb_share
print(f'Stacking LOOCV  R²={r2_meta_A:.4f}  MAE={mae_meta_A:.4f}%')
print(f'XGBoost={xgb_share:.1f}%  SatResNet={sat_share:.1f}%')

xgb_pred_dict = dict(zip(df_test_xgb['practice_code'].values, preds_xgb))
sat_pred_dict = dict(zip(test_pracs_sat, preds_m3_all))
act_dict      = dict(zip(df_test_xgb['practice_code'].values, y_test_xgb))
common_test   = list(set(xgb_pred_dict.keys()) & set(sat_pred_dict.keys()))
y_ts = np.array([act_dict[p] for p in common_test])
p_xs = np.array([xgb_pred_dict[p] for p in common_test])
p_ss = np.array([sat_pred_dict[p]  for p in common_test])
pred_stack_test = meta_full.predict(sc_full.transform(np.column_stack([p_xs,p_ss])))
r2_stack_temp   = r2_score(y_ts, pred_stack_test)
mae_stack_temp  = mean_absolute_error(y_ts, pred_stack_test)
print(f'Stacking temporal  R²={r2_stack_temp:.4f}  MAE={mae_stack_temp:.4f}%')
RESULTS['Stacking (M4)'] = {'test_r2':r2_stack_temp,'test_mae':mae_stack_temp,
                              'loocv_r2':r2_meta_A,'loocv_mae':mae_meta_A,
                              'xgb_weight_pct':round(xgb_share,1),'sat_weight_pct':round(sat_share,1)}
joblib.dump(meta_full, f'{OUT}/models/model4_meta_learner.pkl')
joblib.dump(sc_full,   f'{OUT}/models/scaler_stacking.pkl')

areas_stack = np.array([AMAP.get(p[:3],'') for p in grp_stack])
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for area in COLORS:
    mask = areas_stack==area
    axes[0,0].scatter(y_stack[mask], df_stack['pred_xgb'].values[mask],
                      color=COLORS[area], alpha=0.4, s=15, label=area, edgecolors='none')
lims = [y_stack.min()-0.1, y_stack.max()+0.1]; axes[0,0].plot(lims,lims,'k--',lw=1.5)
axes[0,0].set_xlabel('Actual (%)'); axes[0,0].set_ylabel('Predicted (%)')
axes[0,0].set_title(f'XGBoost OOF  R²={r2_xgb_check:.4f}'); axes[0,0].legend(fontsize=7)
for area in COLORS:
    mask = areas_stack==area
    axes[0,1].scatter(y_stack[mask], df_stack['pred_sat'].values[mask],
                      color=COLORS[area], alpha=0.4, s=15, label=area, edgecolors='none')
axes[0,1].plot(lims,lims,'k--',lw=1.5); axes[0,1].set_xlabel('Actual (%)'); axes[0,1].set_ylabel('Predicted (%)')
axes[0,1].set_title(f'SatResNet OOF  R²={r2_sat_check:.4f}'); axes[0,1].legend(fontsize=7)
for area in COLORS:
    mask = areas_stack==area
    axes[0,2].scatter(y_stack[mask], preds_meta_A[mask],
                      color=COLORS[area], alpha=0.5, s=20, label=area, edgecolors='none')
axes[0,2].plot(lims,lims,'k--',lw=1.5); axes[0,2].set_xlabel('Actual (%)'); axes[0,2].set_ylabel('Predicted (%)')
axes[0,2].set_title(f'Stacking  R²={r2_meta_A:.4f}  MAE={mae_meta_A:.3f}%'); axes[0,2].legend(fontsize=7)
axes[1,0].pie([abs(meta_full.coef_[0]),abs(meta_full.coef_[1])],
               labels=[f'XGBoost\n{xgb_share:.1f}%',f'SatResNet\n{sat_share:.1f}%'],
               colors=['#22c55e','#f59e0b'], autopct='%1.1f%%', startangle=90,
               textprops={'fontsize':11,'fontweight':'bold'})
axes[1,0].set_title('Modality Weights')
bars = axes[1,1].bar(['Ridge','XGBoost','SatResNet','Stacking'],
                      [RESULTS['Ridge 1a (NDVI+NO2)']['loocv_r2'],r2_xgb_check,r2_sat_check,r2_meta_A],
                      color=['#94a3b8','#22c55e','#f59e0b','#3b82f6'], edgecolor='white', alpha=0.85)
for bar, val in zip(bars,[RESULTS['Ridge 1a (NDVI+NO2)']['loocv_r2'],r2_xgb_check,r2_sat_check,r2_meta_A]):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                   f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[1,1].set_ylabel('LOOCV R²'); axes[1,1].set_ylim(0, 0.5); axes[1,1].set_title('LOOCV R² — All Models')
axes[1,2].axis('off')
tbl = axes[1,2].table(
    cellText=[['Persistence','0.966','N/A',f'{mae_pers:.3f}%'],
              ['Ridge',f'{r2_1a:.3f}','0.022',f'{mae_1a:.3f}%'],
              ['XGBoost',f'{r2_xgb:.3f}',f'{r2_xgb_check:.3f}',f'{mae_xgb:.3f}%'],
              ['SatResNet',f'{r2_m3:.3f}',f'{r2_sat_check:.3f}',f'{mae_m3:.3f}%'],
              ['Stacking',f'{r2_stack_temp:.3f}',f'{r2_meta_A:.3f}',f'{mae_stack_temp:.3f}%']],
    colLabels=['Model','Temporal R²','LOOCV R²','MAE'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.9)
for j in range(4): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
for j in range(4): tbl[5,j].set_facecolor('#DBEAFE')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/model4_stacking_results.png', dpi=150, bbox_inches='tight'); plt.show()


# loo conformal
print('\nLOO Conformal Prediction...')
from tqdm.auto import tqdm
df_stack_prac = df_stack.groupby('practice_code').agg({TARGET:'mean','pred_xgb':'mean','pred_sat':'mean'}).reset_index()
df_stack_prac['area'] = df_stack_prac['practice_code'].str[:3].map(AMAP)
preds_all        = meta_full.predict(sc_full.transform(df_stack_prac[['pred_xgb','pred_sat']].values))
actuals_all      = df_stack_prac[TARGET].values
pracs_all        = df_stack_prac['practice_code'].values
areas_all        = df_stack_prac['area'].values
conformal_scores = np.abs(actuals_all - preds_all)
covered = []; ci_widths = []; ci_lowers = []; ci_uppers = []
for i in tqdm(range(len(conformal_scores)), desc='LOO Conformal'):
    cal_scores = np.delete(conformal_scores, i); n_cal_i = len(cal_scores)
    q_i = np.quantile(cal_scores, min(np.ceil((n_cal_i+1)*0.95)/n_cal_i, 1.0))
    ci_lo = preds_all[i]-q_i; ci_hi = preds_all[i]+q_i
    covered.append(actuals_all[i]>=ci_lo and actuals_all[i]<=ci_hi)
    ci_widths.append(q_i); ci_lowers.append(ci_lo); ci_uppers.append(ci_hi)
covered = np.array(covered); ci_widths = np.array(ci_widths)
ci_lowers = np.array(ci_lowers); ci_uppers = np.array(ci_uppers)
coverage_loo = covered.mean()*100; q_conformal = ci_widths.mean()
print(f'Coverage: {coverage_loo:.1f}%  CI: ±{q_conformal:.4f}%  Not covered: {(~covered).sum()}')
df_loo = pd.DataFrame({'practice_code':pracs_all,'area':areas_all,'actual':actuals_all,
                        'pred_mean':preds_all,'ci_lower':ci_lowers,'ci_upper':ci_uppers,
                        'ci_width':ci_widths*2,'covered':covered,'abs_error':conformal_scores,
                        }).sort_values('actual').reset_index(drop=True)
df_loo.to_csv(f'{OUT}/uncertainty/loo_conformal_predictions.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
x = range(len(df_loo))
axes[0,0].fill_between(x, df_loo['ci_lower'], df_loo['ci_upper'], alpha=0.25, color='#3b82f6', label=f'95% CI (±{q_conformal:.3f}%)')
axes[0,0].plot(x, df_loo['pred_mean'], color='#3b82f6', lw=1.5, label='Predicted')
axes[0,0].scatter([i for i,c in enumerate(df_loo['covered']) if c],
                   df_loo.loc[df_loo['covered'],'actual'], color='#22c55e', s=15, alpha=0.6, zorder=5, label='Covered')
axes[0,0].scatter([i for i,c in enumerate(df_loo['covered']) if not c],
                   df_loo.loc[~df_loo['covered'],'actual'], color='#ef4444', s=60, marker='x', lw=2, zorder=6, label='Not covered')
axes[0,0].set_xlabel('Practice (sorted by COPD)'); axes[0,0].set_ylabel('COPD (%)')
axes[0,0].set_title(f'LOO Conformal Intervals\nCoverage={coverage_loo:.1f}%'); axes[0,0].legend(fontsize=7)
area_cov = df_loo.groupby('area')['covered'].mean()*100
bars = axes[0,1].bar(area_cov.index, area_cov.values,
                      color=[COLORS.get(a,'gray') for a in area_cov.index], edgecolor='white', alpha=0.85)
axes[0,1].axhline(95, color='red', lw=2, linestyle='--', label='Target 95%')
for bar, val in zip(bars, area_cov.values):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[0,1].set_ylabel('Coverage (%)'); axes[0,1].set_ylim(0,115)
axes[0,1].set_title('Coverage by Area'); axes[0,1].legend()
for area in COLORS:
    mask = df_loo['area']==area
    if mask.sum()==0: continue
    axes[0,2].errorbar(df_loo.loc[mask,'actual'], df_loo.loc[mask,'pred_mean'],
                       yerr=ci_widths[mask.values], fmt='o', color=COLORS[area], alpha=0.5,
                       ms=4, capsize=2, elinewidth=0.8, label=area)
lims = [df_loo['actual'].min()-0.1, df_loo['actual'].max()+0.1]; axes[0,2].plot(lims,lims,'k--',lw=1.5)
axes[0,2].set_xlabel('Actual (%)'); axes[0,2].set_ylabel('Predicted (%)')
axes[0,2].set_title('Predicted vs Actual with CI'); axes[0,2].legend(fontsize=7)
threshold = df_loo['abs_error'].quantile(0.75); high_m = df_loo['abs_error']>=threshold; low_m = ~high_m
axes[1,0].scatter(df_loo.loc[low_m,'actual'],  df_loo.loc[low_m,'pred_mean'],
                  color='#22c55e', alpha=0.5, s=30, label=f'Low error (n={low_m.sum()})', edgecolors='none')
axes[1,0].scatter(df_loo.loc[high_m,'actual'], df_loo.loc[high_m,'pred_mean'],
                  color='#ef4444', alpha=0.7, s=60, marker='^', label=f'Flag (n={high_m.sum()})',
                  edgecolors='white', lw=0.5)
axes[1,0].plot(lims,lims,'k--',lw=1.5); axes[1,0].set_xlabel('Actual (%)'); axes[1,0].set_ylabel('Predicted (%)')
axes[1,0].set_title('Practices Flagged for Verification'); axes[1,0].legend(fontsize=7)
axes[1,1].hist(ci_widths, bins=20, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[1,1].axvline(q_conformal, color='red', lw=2, linestyle='--', label=f'Mean=±{q_conformal:.3f}%')
axes[1,1].set_xlabel('CI Half-Width (%)'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('CI Width Distribution'); axes[1,1].legend()
axes[1,2].axis('off')
tbl = axes[1,2].table(
    cellText=[['Coverage',f'{coverage_loo:.1f}%','≥95% ✓'],
              ['CI half-width',f'±{q_conformal:.4f}%','calibrated'],
              ['Practices',f'{len(df_loo)}','all used'],
              ['Covered',f'{covered.sum()}',f'{covered.mean()*100:.1f}%'],
              ['Not covered',f'{(~covered).sum()}',f'expected ~{int(len(df_loo)*0.05)}']],
    colLabels=['Metric','Value','Note'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.8)
for j in range(3): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/track6c_loo_conformal.png', dpi=150, bbox_inches='tight'); plt.show()


# final comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
model_names  = ['Persistence','Ridge','XGBoost','SatResNet','Stacking']
temporal_r2s = [r2_pers, r2_1a, r2_xgb, r2_m3, r2_stack_temp]
loocv_r2s    = [None, r2_loocv_1a, r2_xgb_check, r2_sat_check, r2_meta_A]
colors_bar   = ['#64748b','#94a3b8','#22c55e','#f59e0b','#3b82f6']
bars = axes[0].bar(model_names, temporal_r2s, color=colors_bar, edgecolor='white', alpha=0.85)
for bar, val in zip(bars, temporal_r2s):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].axhline(r2_pers, color='red', lw=2, linestyle='--', alpha=0.5, label=f'Persistence={r2_pers:.3f}')
axes[0].set_ylabel('Temporal R²'); axes[0].set_ylim(0, 1.1); axes[0].set_title('Temporal Split R²')
axes[0].legend(fontsize=8); axes[0].set_xticklabels(model_names, rotation=15, ha='right')
loocv_names = model_names[1:]; loocv_vals = [v for v in loocv_r2s if v is not None]
bars = axes[1].bar(loocv_names, loocv_vals, color=colors_bar[1:], edgecolor='white', alpha=0.85)
for bar, val in zip(bars, loocv_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[1].annotate('', xy=(3,r2_meta_A), xytext=(3,r2_xgb_check),
                 arrowprops=dict(arrowstyle='->',color='#3b82f6',lw=2))
axes[1].text(3.15,(r2_meta_A+r2_xgb_check)/2, f'+{r2_meta_A-r2_xgb_check:.3f}',
             color='#3b82f6', fontsize=8, fontweight='bold')
axes[1].set_ylabel('LOOCV R² (primary)'); axes[1].set_ylim(0, 0.5); axes[1].set_title('LOOCV R²')
axes[1].set_xticklabels(loocv_names, rotation=15, ha='right')
axes[2].axis('off')
tbl = axes[2].table(
    cellText=[['Persistence','0.966','N/A',f'{mae_pers:.3f}%'],
              ['Ridge',f'{r2_1a:.3f}','0.022',f'{mae_1a:.3f}%'],
              ['XGBoost',f'{r2_xgb:.3f}',f'{r2_xgb_check:.3f}',f'{mae_xgb:.3f}%'],
              ['SatResNet',f'{r2_m3:.3f}',f'{r2_sat_check:.3f}',f'{mae_m3:.3f}%'],
              ['Stacking',f'{r2_stack_temp:.3f}',f'{r2_meta_A:.3f}',f'{mae_stack_temp:.3f}%']],
    colLabels=['Model','Temporal R²','LOOCV R²','MAE'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 2.0)
for j in range(4): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
for j in range(4): tbl[5,j].set_facecolor('#DBEAFE')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/final_model_comparison_v2.png', dpi=150, bbox_inches='tight'); plt.show()


# save all
RESULTS_FINAL = {
    'Persistence':         {'test_r2':round(r2_pers,4),'test_mae':round(mae_pers,4),'loocv_r2':None},
    'Ridge 1a (NDVI+NO2)': {'test_r2':round(r2_1a,4),'test_mae':round(mae_1a,4),
                             'loocv_r2':round(r2_loocv_1a,4),'loocv_mae':round(mae_loocv_1a,4),'alpha':round(best_alpha_1a,1)},
    'XGBoost (M2)':        {'test_r2':round(r2_xgb,4),'test_mae':round(mae_xgb,4),
                             'loocv_r2':round(r2_loocv_xgb,4),'loocv_mae':round(mae_loocv_xgb,4)},
    'SatResNet (M3)':      {'test_r2':round(r2_m3,4),'test_mae':round(mae_m3,4),
                             'loocv_r2':round(r2_loocv_m3,4),'loocv_mae':round(mae_loocv_m3,4),'backbone':BACKBONE_SRC},
    'Stacking (M4)':       {'test_r2':round(r2_stack_temp,4),'test_mae':round(mae_stack_temp,4),
                             'loocv_r2':round(r2_meta_A,4),'loocv_mae':round(mae_meta_A,4),
                             'xgb_weight_pct':round(xgb_share,1),'sat_weight_pct':round(sat_share,1)},
    'Uncertainty (LOO Conformal)': {'coverage_pct':round(coverage_loo,1),'ci_halfwidth':round(q_conformal,4),
                                    'n_practices':len(df_loo),'n_covered':int(covered.sum()),
                                    'n_not_covered':int((~covered).sum())}
}
with open(f'{OUT}/results/results_final.json','w') as f:
    json.dump(RESULTS_FINAL, f, indent=2)

print('\nFinal results:')
for name, v in RESULTS_FINAL.items():
    if 'test_r2' in v:
        lr2 = f'{v["loocv_r2"]:.4f}' if v.get('loocv_r2') else 'N/A'
        print(f'  {name:<22} temporal={v["test_r2"]:.4f}  loocv={lr2}  mae={v.get("test_mae",0):.4f}%')
print(f'\nAll saved to {OUT}')
