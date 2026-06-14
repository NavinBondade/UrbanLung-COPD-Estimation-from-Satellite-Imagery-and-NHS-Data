import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

DATA_PATH  = r'D:\copd_sat_navin\2019_2024\data\wy_complete_dataset_v2.csv'
IMAGE_DIR  = r'D:\copd_sat_navin\wy_images'
OUT        = r'D:\copd_sat_navin\2019_2024\figures\eda'
os.makedirs(OUT, exist_ok=True)

COLORS = {'Bradford':'#ef4444','Kirklees':'#22c55e','Leeds':'#3b82f6','Wakefield':'#f59e0b'}
AMAP   = {'B82':'Kirklees','B83':'Bradford','B84':'Wakefield','B86':'Leeds'}
AREAS  = ['Bradford','Kirklees','Leeds','Wakefield']
TARGET = 'copd_prevalence'

# load data
df = pd.read_csv(DATA_PATH)
df['area']     = df['practice_code'].str[:3].map(AMAP)
df['year_int'] = df['year'].map({'2019-20':1,'2020-21':2,'2021-22':3,'2022-23':4,'2023-24':5})
df = df.sort_values(['practice_code','year_int']).reset_index(drop=True)
df['no2_lag1']  = df.groupby('practice_code')['no2_lsoa_weighted'].shift(1)
df['ndvi_lag1'] = df.groupby('practice_code')['ndvi_lsoa_weighted'].shift(1)
df['no2_change']= df['no2_lsoa_weighted'] - df['no2_lag1']

# overview
print(f'{df.shape[0]} rows  {df["practice_code"].nunique()} practices')


# save helper
def save(name):
    plt.tight_layout()
    plt.savefig(f'{OUT}/{name}.png', dpi=150, bbox_inches='tight')
    plt.show()


# dataset overview
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
test_yr     = df[df['year']=='2023-24'][TARGET]
area_counts = df[df['year']=='2023-24']['area'].value_counts()
axes[0].bar(area_counts.index, area_counts.values,
            color=[COLORS.get(a,'gray') for a in area_counts.index],
            edgecolor='white', alpha=0.85)
for bar, val in zip(axes[0].patches, area_counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 str(val), ha='center', fontsize=10)
axes[0].set_ylabel('Practices'); axes[0].set_title('Practices per Area')
axes[1].hist(test_yr, bins=20, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[1].axvline(test_yr.mean(), color='red', lw=2, linestyle='--',
                label=f'Mean={test_yr.mean():.2f}%')
axes[1].set_xlabel('COPD (%)'); axes[1].set_ylabel('Count')
axes[1].set_title('COPD Distribution (2023-24)'); axes[1].legend(fontsize=8)
if 'list_size' in df.columns:
    for area in AREAS:
        mask = df['area']==area
        axes[2].scatter(df.loc[mask,'list_size'], df.loc[mask,TARGET],
                        color=COLORS[area], alpha=0.3, s=15, label=area, edgecolors='none')
    axes[2].set_xlabel('List Size'); axes[2].set_ylabel('COPD (%)'); axes[2].legend(fontsize=7)
    axes[2].set_title('List Size vs COPD')
save('s1_dataset_overview')


# copd distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].hist(df[TARGET], bins=30, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[0,0].axvline(df[TARGET].mean(), color='red', lw=2, linestyle='--',
                   label=f'Mean={df[TARGET].mean():.2f}%')
axes[0,0].axvline(df[TARGET].median(), color='orange', lw=2, linestyle=':',
                   label=f'Median={df[TARGET].median():.2f}%')
axes[0,0].set_xlabel('COPD (%)'); axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Overall Distribution'); axes[0,0].legend(fontsize=8)
data_by_area = [df[df['area']==a][TARGET].values for a in AREAS]
bp = axes[0,1].boxplot(data_by_area, labels=AREAS, patch_artist=True,
                        medianprops={'color':'black','lw':2})
for patch, area in zip(bp['boxes'], AREAS):
    patch.set_facecolor(COLORS[area]); patch.set_alpha(0.7)
axes[0,1].set_ylabel('COPD (%)'); axes[0,1].set_title('COPD by Area')
for area in AREAS:
    yearly = df[df['area']==area].groupby('year')[TARGET].mean()
    axes[0,2].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[0,2].set_xticks(range(1,6))
axes[0,2].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[0,2].set_ylabel('Mean COPD (%)'); axes[0,2].set_title('Year-on-Year Trend')
axes[0,2].legend(fontsize=8); axes[0,2].grid(alpha=0.3)
for i, area in enumerate(AREAS):
    parts = axes[1,0].violinplot(df[df['area']==area][TARGET].values,
                                  positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']: pc.set_facecolor(COLORS[area]); pc.set_alpha(0.7)
axes[1,0].set_xticks(range(4)); axes[1,0].set_xticklabels(AREAS)
axes[1,0].set_ylabel('COPD (%)'); axes[1,0].set_title('Distribution Shape by Area')
years = sorted(df['year'].unique())
colors_yr = plt.cm.Blues(np.linspace(0.3, 0.9, len(years)))
for yr, col in zip(years, colors_yr):
    axes[1,1].hist(df[df['year']==yr][TARGET], bins=20, alpha=0.5,
                   color=col, label=yr, edgecolor='none')
axes[1,1].set_xlabel('COPD (%)'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Distribution by Year'); axes[1,1].legend(fontsize=7)
f_stat, p_val = f_oneway(*data_by_area)
axes[1,2].axis('off')
tbl = axes[1,2].table(
    cellText=[[a, f'{df[df["area"]==a][TARGET].mean():.3f}%',
               f'{df[df["area"]==a][TARGET].std():.3f}%'] for a in AREAS],
    colLabels=['Area','Mean','SD'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
for j in range(3): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
axes[1,2].set_title(f'ANOVA: F={f_stat:.2f}, p={p_val:.4f}', pad=20)
save('s2_copd_distribution')


# environmental features
ENV = [f for f in ['ndvi_lsoa_weighted','no2_lsoa_weighted',
                    'lst_lsoa_weighted','elevation_lsoa_weighted'] if f in df.columns]
LABS = {'ndvi_lsoa_weighted':'NDVI','no2_lsoa_weighted':'NO2 (μmol/m²)',
        'lst_lsoa_weighted':'LST (°C)','elevation_lsoa_weighted':'Elevation (m)'}
# area comparison
fig, axes = plt.subplots(2, len(ENV), figsize=(5*len(ENV), 10))
for i, feat in enumerate(ENV):
    label = LABS[feat]
    for area in AREAS:
        axes[0,i].hist(df[df['area']==area][feat].dropna(), bins=15,
                       alpha=0.5, color=COLORS[area], label=area, edgecolor='none')
    axes[0,i].set_xlabel(label); axes[0,i].set_ylabel('Count')
    axes[0,i].set_title(f'{label} by Area'); axes[0,i].legend(fontsize=7)
    valid = df[[feat,TARGET]].dropna()
    m, b, r, p, _ = stats.linregress(valid[feat], valid[TARGET])
    for area in AREAS:
        mask = df['area']==area
        axes[1,i].scatter(df.loc[mask,feat], df.loc[mask,TARGET],
                          color=COLORS[area], alpha=0.3, s=10, label=area, edgecolors='none')
    x_line = np.linspace(valid[feat].min(), valid[feat].max(), 100)
    axes[1,i].plot(x_line, m*x_line+b, 'k--', lw=2, alpha=0.7)
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    axes[1,i].set_xlabel(label); axes[1,i].set_ylabel('COPD (%)')
    axes[1,i].set_title(f'r={r:.3f} {sig}'); axes[1,i].legend(fontsize=7)
save('s3_env_features')


# area comparison grouped
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x = np.arange(len(AREAS)); w = 0.15
for yi, yr in enumerate(sorted(df['year'].unique())):
    vals = [df[(df['area']==a)&(df['year']==yr)][TARGET].mean() for a in AREAS]
    axes[0,0].bar(x+yi*w, vals, w, label=yr, alpha=0.85, edgecolor='white')
axes[0,0].set_xticks(x+w*2); axes[0,0].set_xticklabels(AREAS)
axes[0,0].set_ylabel('Mean COPD (%)'); axes[0,0].set_title('COPD by Area by Year')
axes[0,0].legend(fontsize=7)

ax_radar = plt.subplot(2, 3, 2, polar=True)
features_r = ['copd_prevalence','no2_lsoa_weighted','ndvi_lsoa_weighted','elevation_lsoa_weighted']
labels_r   = ['COPD','NO2','NDVI','Elevation']
N = len(features_r)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
for area in AREAS:
    vals = df[df['area']==area][features_r].mean().values
    mins = df[features_r].min().values; maxs = df[features_r].max().values
    vals_n = ((vals-mins)/(maxs-mins+1e-8)).tolist(); vals_n += vals_n[:1]
    ax_radar.plot(angles, vals_n, 'o-', lw=2, color=COLORS[area], label=area)
    ax_radar.fill(angles, vals_n, alpha=0.1, color=COLORS[area])
ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(labels_r, fontsize=9)
ax_radar.set_title('Environmental Profile\n(normalised)', pad=20)
ax_radar.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.35,1.1))

for area in AREAS:
    yearly = df[df['area']==area].groupby('year')[TARGET].mean()
    axes[0,2].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[0,2].set_xticks(range(1,6))
axes[0,2].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[0,2].set_ylabel('Mean COPD (%)'); axes[0,2].set_title('COPD Trend by Area')
axes[0,2].legend(fontsize=8); axes[0,2].grid(alpha=0.3)

for area in AREAS:
    yearly = df[df['area']==area].groupby('year')['no2_lsoa_weighted'].mean()
    axes[1,0].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[1,0].set_xticks(range(1,6))
axes[1,0].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[1,0].set_ylabel('Mean NO2 (μmol/m²)'); axes[1,0].set_title('NO2 Trend by Area')
axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

for area in AREAS:
    yearly = df[df['area']==area].groupby('year')['ndvi_lsoa_weighted'].mean()
    axes[1,1].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[1,1].set_xticks(range(1,6))
axes[1,1].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[1,1].set_ylabel('Mean NDVI'); axes[1,1].set_title('NDVI Trend by Area')
axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)

axes[1,2].axis('off')
tbl = axes[1,2].table(
    cellText=[[a, f'{df[df["area"]==a][TARGET].mean():.2f}%',
               f'{df[df["area"]==a]["no2_lsoa_weighted"].mean():.1f}',
               f'{df[df["area"]==a]["ndvi_lsoa_weighted"].mean():.3f}',
               f'{df[df["area"]==a]["elevation_lsoa_weighted"].mean():.0f}m'] for a in AREAS],
    colLabels=['Area','COPD','NO2','NDVI','Elev'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 2.0)
for j in range(5): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
save('s4_area_comparison')


# covid lockdown
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for area in AREAS:
    yearly = df[df['area']==area].groupby('year')['no2_lsoa_weighted'].mean()
    axes[0].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[0].axvspan(1.5, 2.5, alpha=0.1, color='red', label='Lockdown')
axes[0].set_xticks(range(1,6))
axes[0].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[0].set_ylabel('Mean NO2 (μmol/m²)'); axes[0].set_title('NO2 Lockdown Dip')
axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

for area in AREAS:
    yearly = df[df['area']==area].groupby('year')[TARGET].mean()
    axes[1].plot(range(1,6), yearly.values, 'o-', color=COLORS[area], lw=2, ms=7, label=area)
axes[1].axvspan(1.5, 2.5, alpha=0.1, color='red')
axes[1].set_xticks(range(1,6))
axes[1].set_xticklabels(['19-20','20-21','21-22','22-23','23-24'], rotation=15)
axes[1].set_ylabel('Mean COPD (%)'); axes[1].set_title('COPD Registers — No Response')
axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

pre  = df[df['year']=='2019-20'].set_index('practice_code')[['no2_lsoa_weighted',TARGET]]
lock = df[df['year']=='2020-21'].set_index('practice_code')[['no2_lsoa_weighted',TARGET]]
common = pre.index.intersection(lock.index)
delta_no2  = lock.loc[common,'no2_lsoa_weighted'] - pre.loc[common,'no2_lsoa_weighted']
delta_copd = lock.loc[common,TARGET] - pre.loc[common,TARGET]
areas_c    = df[df['year']=='2020-21'].set_index('practice_code').loc[common,'area']
for area in AREAS:
    mask = areas_c==area
    axes[2].scatter(delta_no2[mask], delta_copd[mask], color=COLORS[area],
                    alpha=0.6, s=40, label=area, edgecolors='white', lw=0.3)
r_lock, p_lock = stats.pearsonr(delta_no2, delta_copd)
axes[2].axhline(0, color='black', lw=1, linestyle='--')
axes[2].axvline(0, color='black', lw=1, linestyle='--')
axes[2].set_xlabel('ΔNO2 (μmol/m²)'); axes[2].set_ylabel('ΔCOPD (pp)')
axes[2].set_title(f'ΔNO2 vs ΔCOPD\nr={r_lock:.3f}, p={p_lock:.3f}'); axes[2].legend(fontsize=7)
save('s5_covid_signal')


# valley trapping
r_elev_no2,_  = stats.pearsonr(df['elevation_lsoa_weighted'], df['no2_lsoa_weighted'])
r_elev_copd,_ = stats.pearsonr(df['elevation_lsoa_weighted'], df[TARGET])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for area in AREAS:
    mask = df['area']==area
    axes[0].scatter(df.loc[mask,'elevation_lsoa_weighted'], df.loc[mask,'no2_lsoa_weighted'],
                    color=COLORS[area], alpha=0.4, s=20, label=area, edgecolors='none')
m, b, *_ = stats.linregress(df['elevation_lsoa_weighted'], df['no2_lsoa_weighted'])
x_line = np.linspace(df['elevation_lsoa_weighted'].min(), df['elevation_lsoa_weighted'].max(), 100)
axes[0].plot(x_line, m*x_line+b, 'k--', lw=2)
axes[0].set_xlabel('Elevation (m)'); axes[0].set_ylabel('NO2 (μmol/m²)')
axes[0].set_title(f'Elevation vs NO2\nr={r_elev_no2:.3f}'); axes[0].legend(fontsize=7)

for area in AREAS:
    mask = df['area']==area
    axes[1].scatter(df.loc[mask,'elevation_lsoa_weighted'], df.loc[mask,TARGET],
                    color=COLORS[area], alpha=0.4, s=20, label=area, edgecolors='none')
m2, b2, *_ = stats.linregress(df['elevation_lsoa_weighted'], df[TARGET])
axes[1].plot(x_line, m2*x_line+b2, 'k--', lw=2)
axes[1].set_xlabel('Elevation (m)'); axes[1].set_ylabel('COPD (%)')
axes[1].set_title(f'Elevation vs COPD\nr={r_elev_copd:.3f}'); axes[1].legend(fontsize=7)

df['elev_q'] = pd.qcut(df['elevation_lsoa_weighted'], q=4, labels=['Q1','Q2','Q3','Q4'])
q_copd = df.groupby('elev_q')[TARGET].mean()
q_no2  = df.groupby('elev_q')['no2_lsoa_weighted'].mean()
x = np.arange(4)
axes[2].bar(x, q_copd.values, color='#3b82f6', edgecolor='white', alpha=0.85)
ax2 = axes[2].twinx()
ax2.plot(x, q_no2.values, 'ro-', ms=8, lw=2)
axes[2].set_xticks(x); axes[2].set_xticklabels(['Q1\n(low)','Q2','Q3','Q4\n(high)'])
axes[2].set_ylabel('Mean COPD (%)', color='#3b82f6')
ax2.set_ylabel('Mean NO2 (μmol/m²)', color='red')
axes[2].set_title('COPD and NO2 by Elevation Quartile')
save('s6_valley_trapping')


# wakefield anomaly
feats_s = ['no2_lsoa_weighted','ndvi_lsoa_weighted','elevation_lsoa_weighted']
df_c    = df.dropna(subset=feats_s).copy()
lr      = LinearRegression()
lr.fit(df_c[feats_s], df_c[TARGET])
df_c['copd_expected'] = lr.predict(df_c[feats_s])
df_c['residual']      = df_c[TARGET] - df_c['copd_expected']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, area in enumerate(AREAS):
    parts = axes[0].violinplot(df_c[df_c['area']==area]['residual'],
                                positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']: pc.set_facecolor(COLORS[area]); pc.set_alpha(0.7)
axes[0].axhline(0, color='black', lw=2, linestyle='--')
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(AREAS)
axes[0].set_ylabel('Residual (pp)'); axes[0].set_title('Actual minus Environment-Predicted')
for area in AREAS:
    mask = df_c['area']==area
    axes[1].scatter(df_c.loc[mask,'copd_expected'], df_c.loc[mask,TARGET],
                    color=COLORS[area], alpha=0.5, s=30, label=area, edgecolors='white', lw=0.3)
lims = [df_c['copd_expected'].min()-0.1, df_c['copd_expected'].max()+0.1]
axes[1].plot(lims,lims,'k--',lw=2); axes[1].set_xlabel('Environment-Predicted COPD (%)')
axes[1].set_ylabel('Actual COPD (%)'); axes[1].set_title('Actual vs Predicted'); axes[1].legend(fontsize=7)
for area in ['Bradford','Kirklees','Leeds']:
    mask = df_c['area']==area
    axes[2].scatter(df_c.loc[mask,'elevation_lsoa_weighted'], df_c.loc[mask,TARGET],
                    color=COLORS[area], alpha=0.3, s=20, label=area, edgecolors='none')
wake_mask = df_c['area']=='Wakefield'
axes[2].scatter(df_c.loc[wake_mask,'elevation_lsoa_weighted'], df_c.loc[wake_mask,TARGET],
                color=COLORS['Wakefield'], alpha=0.8, s=80, label='Wakefield',
                edgecolors='black', lw=1, zorder=5)
axes[2].set_xlabel('Elevation (m)'); axes[2].set_ylabel('COPD (%)')
axes[2].set_title('Wakefield: High COPD Despite High Elevation'); axes[2].legend(fontsize=7)
save('s7_wakefield_anomaly')


# feature correlations and lag justification
df_lag  = df.dropna(subset=['no2_lag1','ndvi_lag1']).copy()
ALL_F   = [f for f in ['ndvi_lsoa_weighted','no2_lsoa_weighted','lst_lsoa_weighted',
                        'elevation_lsoa_weighted','image_std','no2_lag1','ndvi_lag1',
                        'no2_change','year_int'] if f in df_lag.columns]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
feats_hm = ALL_F + [TARGET]
corr_mat = df_lag[feats_hm].corr()
labels_hm = [f.replace('_lsoa_weighted','').replace('_',' ') for f in feats_hm]
im = axes[0].imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=axes[0], shrink=0.8)
axes[0].set_xticks(range(len(feats_hm))); axes[0].set_yticks(range(len(feats_hm)))
axes[0].set_xticklabels(labels_hm, rotation=45, ha='right', fontsize=7)
axes[0].set_yticklabels(labels_hm, fontsize=7)
for i in range(len(feats_hm)):
    for j in range(len(feats_hm)):
        axes[0].text(j, i, f'{corr_mat.values[i,j]:.2f}', ha='center', va='center', fontsize=6)
axes[0].set_title('Correlation Matrix')

corr_results = [{'feature':f,'r':stats.pearsonr(df_lag[[f,TARGET]].dropna()[f],
                                                  df_lag[[f,TARGET]].dropna()[TARGET])[0]} for f in ALL_F]
corr_df  = pd.DataFrame(corr_results).sort_values('r', key=abs, ascending=True)
bar_cols = ['#ef4444' if r>0 else '#22c55e' for r in corr_df['r']]
axes[1].barh(range(len(corr_df)), corr_df['r'].values, color=bar_cols, edgecolor='white', alpha=0.85)
axes[1].set_yticks(range(len(corr_df)))
axes[1].set_yticklabels([f.replace('_lsoa_weighted','').replace('_',' ')
                         for f in corr_df['feature']], fontsize=8)
axes[1].axvline(0, color='black', lw=1); axes[1].set_xlabel('Pearson r with COPD')
axes[1].set_title('Feature Correlations with COPD')

r_ndvi_curr,_ = stats.pearsonr(df_lag['ndvi_lsoa_weighted'], df_lag[TARGET])
r_ndvi_lag1,_ = stats.pearsonr(df_lag['ndvi_lag1'],          df_lag[TARGET])
r_no2_curr,_  = stats.pearsonr(df_lag['no2_lsoa_weighted'],  df_lag[TARGET])
r_no2_lag1,_  = stats.pearsonr(df_lag['no2_lag1'],           df_lag[TARGET])
comparisons = [('NDVI\ncurrent',r_ndvi_curr,'#22c55e'),('NDVI\nlag-1',r_ndvi_lag1,'#16a34a'),
               ('NO2\ncurrent',r_no2_curr,'#ef4444'),('NO2\nlag-1',r_no2_lag1,'#dc2626')]
for i, (label, r_val, col) in enumerate(comparisons):
    axes[2].bar(i, abs(r_val), color=col, edgecolor='white', alpha=0.85)
    axes[2].text(i, abs(r_val)+0.003, f'r={r_val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[2].set_xticks(range(4)); axes[2].set_xticklabels([c[0] for c in comparisons])
axes[2].set_ylabel('|r| with COPD'); axes[2].set_title('Current vs Lag Correlation')
save('s8_correlations_lags')


# persistence baseline motivation
prac_pivot  = df.pivot_table(index='practice_code', columns='year', values=TARGET).dropna()
prac_stds   = df.groupby('practice_code')[TARGET].std()
within_std  = prac_stds.mean()
between_std = df.groupby('practice_code')[TARGET].mean().std()
y_true = prac_pivot['2023-24'].values; y_pers = prac_pivot['2022-23'].values
from sklearn.metrics import r2_score, mean_absolute_error
r2_pers  = r2_score(y_true, y_pers)
mae_pers = mean_absolute_error(y_true, y_pers)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].hist(prac_stds.values, bins=20, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[0].axvline(within_std,  color='blue', lw=2, linestyle='--', label=f'Within={within_std:.3f}%')
axes[0].axvline(between_std, color='red',  lw=2, linestyle='-',  label=f'Between={between_std:.3f}%')
axes[0].set_xlabel('SD of COPD (%)'); axes[0].set_ylabel('Count')
axes[0].set_title(f'Within vs Between Practice Variance ({between_std/within_std:.1f}x)')
axes[0].legend(fontsize=8)
areas_p = df[df['year']=='2023-24'].set_index('practice_code').loc[prac_pivot.index,'area'].values
for area in AREAS:
    mask = areas_p==area
    axes[1].scatter(y_pers[mask], y_true[mask], color=COLORS[area], alpha=0.7, s=40,
                    label=area, edgecolors='white', lw=0.3)
lims = [y_true.min()-0.1, y_true.max()+0.1]; axes[1].plot(lims,lims,'k--',lw=2)
axes[1].set_xlabel('2022-23 COPD (%)'); axes[1].set_ylabel('2023-24 COPD (%)')
axes[1].set_title(f'Persistence: R²={r2_pers:.4f}  MAE={mae_pers:.4f}%'); axes[1].legend(fontsize=7)
sample_pracs = prac_pivot.sample(30, random_state=42).index
for prac in sample_pracs:
    area = df[df['practice_code']==prac]['area'].iloc[0]
    axes[2].plot(range(1,len(prac_pivot.columns)+1), prac_pivot.loc[prac].values,
                 color=COLORS.get(area,'gray'), alpha=0.3, lw=1)
axes[2].set_xticks(range(1,len(prac_pivot.columns)+1))
axes[2].set_xticklabels(list(prac_pivot.columns), rotation=15, fontsize=8)
axes[2].set_ylabel('COPD (%)'); axes[2].set_title('Practice Trajectories (n=30)')
save('s9_persistence_motivation')


# geographic map
df_map = df[df['year']=='2023-24'].copy()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sc  = axes[0].scatter(df_map['lon'], df_map['lat'], c=df_map['copd_prevalence'],
                       cmap='RdYlGn_r', s=80, alpha=0.9, edgecolors='white', lw=0.5, vmin=1, vmax=4)
plt.colorbar(sc, ax=axes[0], label='COPD (%)', shrink=0.8)
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude'); axes[0].set_title('COPD Prevalence')
sc2 = axes[1].scatter(df_map['lon'], df_map['lat'], c=df_map['no2_lsoa_weighted'],
                       cmap='RdYlGn_r', s=80, alpha=0.9, edgecolors='white', lw=0.5)
plt.colorbar(sc2, ax=axes[1], label='NO2 (μmol/m²)', shrink=0.8)
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude'); axes[1].set_title('NO2 Distribution')
for area in AREAS:
    mask = df_map['area']==area
    axes[2].scatter(df_map.loc[mask,'lon'], df_map.loc[mask,'lat'],
                    color=COLORS[area], s=80, alpha=0.8, label=f'{area} (n={mask.sum()})',
                    edgecolors='white', lw=0.5)
axes[2].set_xlabel('Longitude'); axes[2].set_ylabel('Latitude')
axes[2].set_title('Practices by NHS Area'); axes[2].legend(fontsize=8)
save('s10_geographic_map')


# image quality
r_std, p_std = stats.pearsonr(df['image_std'], df[TARGET])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].hist(df['image_std'], bins=30, color='#3b82f6', edgecolor='white', alpha=0.85)
axes[0].axvline(df['image_std'].mean(), color='red', lw=2, linestyle='--',
                label=f'Mean={df["image_std"].mean():.1f}')
axes[0].set_xlabel('Image Std'); axes[0].set_ylabel('Count')
axes[0].set_title('Image Texture Distribution'); axes[0].legend(fontsize=8)
for area in AREAS:
    mask = df['area']==area
    axes[1].scatter(df.loc[mask,'image_std'], df.loc[mask,TARGET],
                    color=COLORS[area], alpha=0.4, s=15, label=area, edgecolors='none')
m3, b3, *_ = stats.linregress(df['image_std'], df[TARGET])
x3 = np.linspace(df['image_std'].min(), df['image_std'].max(), 100)
axes[1].plot(x3, m3*x3+b3, 'k--', lw=2)
sig_s = '***' if p_std<0.001 else '**'
axes[1].set_xlabel('Image Texture Std'); axes[1].set_ylabel('COPD (%)')
axes[1].set_title(f'Texture vs COPD\nr={r_std:.3f}{sig_s}'); axes[1].legend(fontsize=7)
for i, area in enumerate(AREAS):
    parts = axes[2].violinplot(df[df['area']==area]['image_std'].values,
                                positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']: pc.set_facecolor(COLORS[area]); pc.set_alpha(0.7)
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(AREAS)
axes[2].set_ylabel('Image Texture Std'); axes[2].set_title('Image Texture by Area')
save('s11_image_quality')

if os.path.exists(IMAGE_DIR):
    from PIL import Image as PILImage
    df_ty = df[df['year']=='2023-24'].copy()
    df_ty['image_path'] = df_ty.apply(
        lambda r: os.path.join(IMAGE_DIR, f"{r['practice_code']}_{r['year'].replace('-','_')}.png"), axis=1)
    df_ty = df_ty[df_ty['image_path'].apply(os.path.exists)].reset_index(drop=True)
    sorted_df = df_ty.sort_values(TARGET)
    n = 4; high_rows = sorted_df.tail(n); low_rows = sorted_df.head(n)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    fig.patch.set_facecolor('#0f172a')
    for col, (_, row) in enumerate(high_rows.iterrows()):
        try: axes[0,col].imshow(PILImage.open(row['image_path']).convert('RGB'))
        except: pass
        axes[0,col].set_title(f'{row["practice_code"]}\n{row[TARGET]:.2f}%',
                               color='#fca5a5', fontsize=8); axes[0,col].axis('off')
    for col, (_, row) in enumerate(low_rows.iterrows()):
        try: axes[1,col].imshow(PILImage.open(row['image_path']).convert('RGB'))
        except: pass
        axes[1,col].set_title(f'{row["practice_code"]}\n{row[TARGET]:.2f}%',
                               color='#86efac', fontsize=8); axes[1,col].axis('off')
    fig.text(0.01, 0.75, 'High COPD', color='#fca5a5', fontsize=10, fontweight='bold', va='center', rotation=90)
    fig.text(0.01, 0.27, 'Low COPD',  color='#86efac', fontsize=10, fontweight='bold', va='center', rotation=90)
    plt.tight_layout()
    plt.savefig(f'{OUT}/s11_image_samples.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


# temporal stability
prac_stats = df.groupby('practice_code').agg(
    mean_copd=(TARGET,'mean'), std_copd=(TARGET,'std'),
).reset_index()
prac_stats['area'] = prac_stats['practice_code'].str[:3].map(AMAP)
prac_stats = prac_stats.dropna(subset=['std_copd'])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for area in AREAS:
    mask = prac_stats['area']==area
    axes[0].scatter(prac_stats.loc[mask,'mean_copd'], prac_stats.loc[mask,'std_copd'],
                    color=COLORS[area], alpha=0.7, s=40, label=area, edgecolors='white', lw=0.3)
threshold_v = prac_stats['std_copd'].quantile(0.9)
axes[0].axhline(threshold_v, color='red', lw=2, linestyle='--', label=f'Top 10% ({threshold_v:.3f}%)')
axes[0].set_xlabel('Mean COPD (%)'); axes[0].set_ylabel('COPD Std')
axes[0].set_title('Practice Temporal Stability'); axes[0].legend(fontsize=7)
for i, area in enumerate(AREAS):
    parts = axes[1].violinplot(prac_stats[prac_stats['area']==area]['std_copd'].values,
                                positions=[i], widths=0.6, showmedians=True)
    for pc in parts['bodies']: pc.set_facecolor(COLORS[area]); pc.set_alpha(0.7)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(AREAS)
axes[1].set_ylabel('COPD Std (pp)'); axes[1].set_title('Volatility by Area')
axes[2].axis('off')
tbl = axes[2].table(
    cellText=[[a, f'{prac_stats[prac_stats["area"]==a]["mean_copd"].mean():.2f}%',
               f'{prac_stats[prac_stats["area"]==a]["std_copd"].mean():.4f}%',
               str(len(prac_stats[prac_stats["area"]==a]))] for a in AREAS],
    colLabels=['Area','Mean COPD','Mean Std','N'], loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
for j in range(4): tbl[0,j].set_facecolor('#1F4E79'); tbl[0,j].set_text_props(color='white', fontweight='bold')
axes[2].set_title('Stability Summary', pad=20)
save('s12_outliers_stability')

# done
print(f'Done. Figures saved to {OUT}')
