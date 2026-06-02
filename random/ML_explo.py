import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import argparse
import time
import shap
import optuna
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def getAllData(ds, lat, lon):
    """
    Get all data from the dataset.
    """
    output = {}
    varNames = extractVarNames(ds)
    for var in varNames:
        output[var],latVals, lonVals = extractData(ds, var, lat, lon)
    return output, varNames, latVals, lonVals

def extractVarNames(ds):
    ds_vars = ds.data_vars
    var_names = []
    for var in ds_vars:
        var_names.append(var)
    return var_names

def extractData(ds, var_name, lat, lon):
    """
    Extract data for a specific variable and region.
    """
    try:
        lat_nc = ds.latitude.values #21600
        lon_nc = ds.longitude.values #43200
        sel_lat, sel_lon  = loadRegion(lat_nc, lon_nc, lat, lon) # global
        ds_sel = ds[var_name].sel(latitude=lat_nc[sel_lat], longitude=lon_nc[sel_lon]).values
    except:
        print('latitude and longitude likely different name')
        lat_nc = ds.lat.values #21600
        lon_nc = ds.lon.values #43200
        sel_lat, sel_lon  = loadRegion(lat_nc, lon_nc, lat, lon) # global
        ds_sel = ds[var_name].sel(lat=lat_nc[sel_lat], lon=lon_nc[sel_lon]).values
    return ds_sel, lat_nc[sel_lat], lon_nc[sel_lon]

def loadRegion(lat_file, lon_file, lat, lon):
    """
    Load the region data for a given latitude and longitude.
    """
    ind_lat = np.where((lat_file >= lat[0]) & (lat_file <= lat[1]))[0]   
    ind_lon = np.where((lon_file >= lon[0]) & (lon_file <= lon[1]))[0]
    if lat_file[1] - lat_file[0] < 0:
        ind_lat = np.flip(ind_lat)
    return ind_lat, ind_lon


    
# # case area is India
lat = (6.0, 38.0)
lon = (68.0, 98.0)

train = 0.1


tar_var = ['average_recovery_rate']
dc_vars = ['average_recovery_rate', 'average_drought_duration', 'average_pre_rate', 'average_post_rate']
phys_vars = ['bed_conductance_used',
            'bottom_lowermost_layer',
            'bottom_uppermost_layer',
            'drain_conductance',
            'drain_elevation_lowermost_layer',
            'drain_elevation_uppermost_layer',
            'horizontal_conductivity_lowermost_layer',
            'horizontal_conductivity_uppermost_layer',
            'initial_head_lowermost_layer',
            'initial_head_uppermost_layer',
            'net_RCH',
            'primary_storage_coefficient_lowermost_layer',
            'primary_storage_coefficient_uppermost_layer',
            'surface_water_bed_elevation_used',
            'surface_water_elevation',
            'top_uppermost_layer',
            # 'vertical_conductivity_lowermost_layer',
            'vertical_conductivity_uppermost_layer']

wtd_vars = ['mean_wtd', 'std_wtd']
abs_vars = ['gwAbstraction']

figfolder = '/eejit/home/hausw001/droughtRecovery/figures/MLtest/no_vcdl_lat_%s_%s_lon_%s_%s_train_%s/'%(lat[0], lat[1], lon[0], lon[1], train)
modelfolder = '/archive/depfg/hausw001/droughtRecovery/ml/MLtest/no_vcdl_lat_%s_%s_lon_%s_%s_train_%s/'%(lat[0], lat[1], lon[0], lon[1], train)

if not os.path.exists(figfolder):
    os.makedirs(figfolder)
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)


# if case == 'base_phys':
ds_param = xr.open_zarr("/archive/depfg/hausw001/data/globgm/globgm_static.zarr/")
ds_param = ds_param[phys_vars]

ds_abstraction = xr.open_dataset("/archive/depfg/hausw001/data/globgm/gwAbstraction_mean_masked.nc")

ds = xr.open_dataset('/archive/depfg/hausw001/droughtRecovery/drought_characteristics_mapped_v7_cleaned.nc')
ds = ds[tar_var]
print('loaded data, variables:')
print(list(ds.data_vars))
print(list(ds_param.data_vars))
print(list(ds_abstraction.data_vars))
sys.stdout.flush()

print('extract data dc')
sys.stdout.flush()
output, varNames, latVals, lonVals = getAllData(ds, lat, lon)
print('output varnames', varNames)
del ds

print('extract data param')
sys.stdout.flush()
output_param, varNames_param, latVals_param, lonVals_param = getAllData(ds_param, lat, lon)
print('output_param varnames', varNames_param)
del ds_param

print('extract data abstraction')
sys.stdout.flush()
output_abstraction, varNames_abstraction, latVals_abstraction, lonVals_abstraction = getAllData(ds_abstraction, lat, lon)
print('output_abstraction varnames', varNames_abstraction)
del ds_abstraction


    
print('combining dc and params data')
sys.stdout.flush()
outputAll = {**output, **output_param, **output_abstraction}
varNamesAll = varNames + varNames_param + varNames_abstraction


'''per variable of interest: create an empty df and select the sampled points from the datasets (dc, params and wtd)'''
var_target = ['average_recovery_rate']
inp_var = [v for v in varNamesAll if v not in var_target]


print('replace inf with nan')
sys.stdout.flush()
for param in varNamesAll:
    arr = outputAll[param]
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    # print('variable %s has %s nan and %s inf values'%(param, n_nan, n_inf))
    arr = np.where(np.isinf(arr), np.nan, arr)  # replace inf with nan
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    # print('after replacing inf with nan, variable %s has %s nan and %s inf values'%(param, n_nan, n_inf))
    outputAll[param] = arr

print('prepare input and target data for xgb and keep arrays instead of dataframe for memory efficiency')
sys.stdout.flush()
# Flatten target
y_og = outputAll[var_target[0]].ravel()


print('create mask based on target variable to drop nan values')
sys.stdout.flush()
# Drop rows where target is NaN
mask = ~np.isnan(y_og)
#mask out lat and lon   
latData = np.tile(np.array(latVals).reshape(-1, 1), (1, len(lonVals)))
lonData = np.tile(np.array(lonVals), (len(latVals), 1))
latData = latData.ravel()
lonData = lonData.ravel()

latMasked = latData[mask]
lonMasked = lonData[mask]

print('flatten input variables and apply mask to drop rows with NaN target')
sys.stdout.flush()
# Flatten and stack features
X_og = np.column_stack([outputAll[p].ravel()[mask] for p in inp_var])
X_og_var_names = inp_var

# Apply mask to target
y_og = y_og[mask]

#TODO 20/10/2025 check what happens if target is log transformed
print('log transform target variable')
sys.stdout.flush()
y_og = np.log1p(y_og)

# print_save_stats(y_og, 'y_og_logtransformed','average_recovery_rate_logtransformed', 'Log Transformed Average Recovery Rate')

# #TODO add random variable only later for XGBoost modeling, not for PCA
# print('add random variable for baseline importance')
# sys.stdout.flush()
# # Add a random column (uniform between 0 and 1)
# rng = np.random.default_rng(seed=42)
# rand_col = rng.random((X_og.shape[0], 1), dtype=np.float32)
# X_og = np.hstack([X_og, rand_col])
# inp_var.append('random_var')
# np.save(os.path.join(figfolder, 'input_features.npy'), inp_var)

# print("X shape:", X_og.shape)
# print("y shape:", y_og.shape)
# sys.stdout.flush()

if region == 'global':
    print('perform pca on global input data and save pca model')
    sys.stdout.flush()
    '''new addition: first pca and then split training and testing data'''
    #first check how many nan values in X per variable
    n_nan_per_var = np.isnan(X_og).sum(axis=0)
    n_nan_percentage = n_nan_per_var / X_og.shape[0] * 100
    print('number of nan values per variable in X:, percentage of total samples:' )
    for var, n_nan in zip(inp_var, n_nan_per_var):
        print(f'{var}: {n_nan}, {n_nan / X_og.shape[0] * 100:.2f}%')

    #create mask of nan values per variable
    mask_nan = np.isnan(X_og).any(axis=1)
    print('number of samples with at least one nan value in X:', mask_nan.sum())

    #mask out samples with nan values in X
    X = X_og[~mask_nan]
    y = y_og[~mask_nan]
    latMasked_pca = latMasked[~mask_nan]
    lonMasked_pca = lonMasked[~mask_nan]
    #save lat and lon of samples used for pca
    np.save(os.path.join(modelfolder, 'lat_masked_pca.npy'), latMasked_pca)
    np.save(os.path.join(modelfolder, 'lon_masked_pca.npy'), lonMasked_pca)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA
    print('Running PCA')
    sys.stdout.flush()
    # pca_comp = 6
    # pca = PCA(n_components=pca_comp)
    pca = PCA()
    X_dev_pca = pca.fit_transform(X_scaled)
    corrs = [np.corrcoef(X_dev_pca[:, i], y)[0, 1] for i in range(X_dev_pca.shape[1])]
    pd.Series(corrs, index=[f'PC{i+1}' for i in range(len(corrs))]).to_frame('corr_with_recovery')
    print(corrs)
    sys.stdout.flush()
    np.save(os.path.join(figfolder, 'pca_pc_correlations.npy'), corrs)  


#TODO add random variable here for RFoost modeling, not for PCA
print('add random variable for baseline importance')
sys.stdout.flush()
# Add a random column (uniform between 0 and 1)
rng = np.random.default_rng(seed=42)
rand_col = rng.random((X_dev_pca.shape[0], 1), dtype=np.float32)
X_dev_pca_rv = np.hstack([X_dev_pca, rand_col])
xgb_inp_var = pc_df.index.tolist()
xgb_inp_var.append('random_var')
print('xgb input features:', xgb_inp_var)
sys.stdout.flush()
np.save(os.path.join(figfolder, 'input_features_afterpca.npy'), xgb_inp_var)

# print_save_stats(X_dev_pca_rv[:,-1], 'X_dev_pca_randomvar','PCA_Random_Variable', 'PCA Random Variable')

print("Xog shape:", X_og.shape)
print("yog shape:", y_og.shape)
print('pca X_dev_pca shape:', X_dev_pca.shape)
print('pca X_dev_pca_rv shape:', X_dev_pca_rv.shape)
print('y shape:', y.shape)  
sys.stdout.flush()

# Split the data into training and testing sets
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_pca_rv, y, test_size=1-train, random_state=random_state)
#save the X_train, X_test, y_train, y_test
np.save(os.path.join(modelfolder, 'X_train.npy'), X_dev)
np.save(os.path.join(modelfolder, 'X_test.npy'), X_test)
np.save(os.path.join(modelfolder, 'y_train.npy'), y_dev)
np.save(os.path.join(modelfolder, 'y_test.npy'), y_test)

#print_save_stats(y_dev, 'y_dev','y_development_set', 'Development Set Target Variable')
#print_save_stats(y_test, 'y_test','y_test_set', 'Test Set Target Variable')
# for i, var in enumerate(xgb_inp_var):
    #print_save_stats(X_dev[:,i], 'X_dev_%s' %var,'X_development_PC_%s'%var, 'Development Set PC %s'%var)
    #print_save_stats(X_test[:,i], 'X_test_%s' %var,'X_test_PC_%s'%var, 'Test Set PC %s'%var)


print('tune or build and train model')
sys.stdout.flush()
# Tune or use defaults
if tune == 'yes':
    print('tune model')
    sys.stdout.flush()

    # print('check what percentage of training data seems reasonable for tuning')
    # sys.stdout.flush()

    # #check if file 'learning_curve_hptuning_fraction_devdata.png' exists in figfolder, if so, skip this step
    # if os.path.exists(os.path.join(figfolder, 'learning_curve_hptuning_fraction_devdata.png')):
    #     print('learning curve figure already exists, skip this step')
    #     sys.stdout.flush()

    # else:
    #     fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]  # fractions of dev pool
    #     res = []
    #     xgb_default = dict(
    #         n_estimators=1764,
    #         learning_rate=0.08098144347654342,
    #         max_depth=10,
    #         subsample=0.5441890639352657,
    #         colsample_bytree=0.8752865584409767,
    #         gamma = 0.3571172235503476, 
    #         min_child_weight = 17,
    #         reg_alpha=9.244306105018245,
    #         reg_lambda=0.0474511289970302,
    #         n_jobs=-1,
    #         random_state=random_state,
    #         eval_metric="rmse"
    #         )

    #     for f in fractions:
    #         print('\nEvaluating fraction:', f)
    #         sys.stdout.flush()
    #         t0 = time.time()
    #         mean_rmse, std_rmse, raw = evaluate_subsample(X_dev, y_dev, train_fraction=f,
    #                                                     n_repeats=3, n_splits=3, xgb_params=xgb_default)
    #         t = time.time()-t0
    #         print(f"frac={f:.3f}, mean_rmse={mean_rmse:.4f}, std={std_rmse:.4f}, time={t:.1f}s")
    #         res.append({'fraction': f, 'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 'raw': raw})

    #     df = pd.DataFrame(res)
    #     df.to_pickle(os.path.join(figfolder, 'learning_curve_hptuning_fraction_devdata.pkl'))

    #     print('plot learning curve')    
    #     sys.stdout.flush()
    #     # Plot learning curve with error bars
    #     plt.fill_between(df['fraction'], df['mean_rmse']-df['std_rmse'], df['mean_rmse']+df['std_rmse'], alpha=0.2)
    #     plt.plot(df['fraction'], df['mean_rmse'], marker='o')
    #     # plt.xscale('log')
    #     plt.xlabel('Fraction of dev pool used for training (log scale)')
    #     plt.ylabel('Validation RMSE')
    #     plt.title('Learning curve (pilot)')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(figfolder, 'learning_curve_hptuning_fraction_devdata.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     print('check out figure, stop here for now')
    #     exit()

    # fraction = 0.2
    # print('run tuning with small fraction of  %s training data to save time' %fraction)
    # sys.stdout.flush()
    # best_params = tune_fraction_xgb(X_dev, y_dev, n_trials=30, n_splits=3, train_fraction=fraction) # use 5% of training data for tuning

    best_params = tune_xgb(X_dev, y_dev, n_trials=50, n_splits=3)
    #save best params
    joblib.dump(best_params, os.path.join(modelfolder, 'xgb_best_params.pkl'))
    print('best params saved')
    sys.stdout.flush()
    print('stop here for now')
    sys.exit()

else:
    print('use default or best known parameters')
    sys.stdout.flush()

    default_params = dict(
    n_estimators=50,#501
    n_jobs=-1,
    random_state=42,
    max_depth=38,
    min_samples_leaf=3,
    max_features=0.49889858496662565,
    bootstrap=True,
    max_samples=0.8378606694864698
    )


print('build and train model')
sys.stdout.flush()
model = RandomForestRegressor(**default_params, verbose=2)
start = time.time()
model.fit(X_dev, y_dev)
elapsed = time.time() - start
print(f"model training completed in {elapsed/60:.2f} minutes")
sys.stdout.flush()

# Save the model
print('save model')
sys.stdout.flush()
model_filename = os.path.join(modelfolder, 'rf_model_pca_.pkl')
joblib.dump(model, model_filename)

# Make predictions
print('predict test data and evaluate')
sys.stdout.flush()
y_pred = model.predict(X_test)

#print_save_stats(y_pred, 'y_pred','y_pred_test_set', 'Predicted Target Variable on Test Set')

rmse = root_mean_squared_error(y_test, y_pred)

print('y_test and y_pred are log transformed, retransform to original scale for rmse calculation')
sys.stdout.flush()
y_pred_retransform = np.expm1(y_pred)
y_test_retransform = np.expm1(y_test)

#print_save_stats(y_pred_retransform, 'y_pred_retransform','y_pred_test_set_retransformed', 'Predicted Target Variable on Test Set Retransformed')
#print_save_stats(y_test_retransform, 'y_test_retransform','y_test_set_retransformed', 'Test Set Target Variable Retransformed')
rmse_retransform = root_mean_squared_error(y_test_retransform, y_pred_retransform)
#save mse and rmse to a text file
# with open(os.path.join(figfolder, 'xgb_pca_model_performance_%s.txt'%train), 'w') as f:
#     f.write(f' RMSE: {rmse:.4f}\n')
# print(f'RMSE: {rmse:.4f}')
# sys.stdout.flush()
with open(os.path.join(figfolder, 'xgb_pca_model_performance_%s.txt'%train), 'w') as f:
    f.write(f' RMSE (log transformed): {rmse:.4f}\n')
    f.write(f' RMSE (retransformed): {rmse_retransform:.4f}\n')
print(f'RMSE (log transformed): {rmse:.4f}')
print(f'RMSE (retransformed): {rmse_retransform:.4f}')
sys.stdout.flush()

# Print feature importance
print('feature importance')
sys.stdout.flush()
importances = model.feature_importances_
feature_importance = pd.DataFrame({
        'Feature': xgb_inp_var,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
#save feature importance
feature_importance.to_csv(os.path.join(figfolder, 'xgb_pca_feature_importance.csv'), index=False)
print(feature_importance)
sys.stdout.flush()

# Plot feature importance
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
# Plot feature importance
feature_importance.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax[0])
if case == 'base_all':
    ax[0].set_title(f'RF feauture importance - base:all, rmse {rmse:.4f}', fontsize=14)
elif case == 'base_phys':
    ax[0].set_title(f'RF feauture importance - base:phys, rmse {rmse:.4f}', fontsize=14)
elif case == 'norm_all':
    ax[0].set_title(f'RF feauture importance - norm:all, rmse {rmse:.4f}', fontsize=14)
elif case == 'norm_phys':
    ax[0].set_title(f'RF feauture importance - norm:phys, rmse {rmse:.4f}', fontsize=14)
ax[0].set_ylabel('Importance')
ax[0].set_xlabel('')
ax[0].tick_params(axis='x', rotation=90)

# List feature importance results
ax[1].axis('off')  # Turn off the axis
for i, (importance, feature) in enumerate(zip(feature_importance['Importance'], feature_importance['Feature'])):
    ax[1].text(0.1, 0.9-(i * 0.05), f' {importance:.4f}: {feature}', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(figfolder, 'feature_pca_importance.png'), bbox_inches='tight', dpi=300)
plt.close()

print('save y_test and y_pred for further analysis')
sys.stdout.flush()
# # Predict recovery rate for the entire dataset
recRates = pd.DataFrame({
    "simulated_recovery_rate_log": y_test,
    "predicted_recovery_rate_log": y_pred,
    "simulated_recovery_rate_retrans": y_test_retransform,	
    "predicted_recovery_rate_retrans": y_pred_retransform
})
recRates.to_pickle(os.path.join(figfolder, 'y_test_y_pred.pkl'))

