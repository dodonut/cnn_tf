import numpy as np
import os
Join = os.path.join

LABELS = ['Bacillusscereus', 'Bacillussubtilis', 'Coryniumbacteriumlutaminum',
          'Enterobactearerogenes', 'Enterobactercloacal', 'Enterococcusfaecalis', 'Escheriachiacoli',
          'Klesbsialapneumonial', 'Micrococcusluteus', 'Proteusmirabilis', 'Pseudomonasaeoruginosas', 'Salmonellaspp',
          'Serratiamarcences', 'Staphylococcusaureus_6538', 'Staphylococcusaureus_25923', 'Staphylococcusepidemides']

COLORS_HEX = {
    'Bacillusscereus': '#ff1900',
    'Bacillussubtilis': '#c27c51',
    'Coryniumbacteriumlutaminum': '#7d5e20',
    'Enterobactearerogenes': '#dbcf5c',
    'Enterobactercloacal': '#9db031',
    'Enterococcusfaecalis': '#9dff00',
    'Escheriachiacoli': '#b58ad4',
    'Klesbsialapneumonial': '#f200ff',
    'Micrococcusluteus': '#6e9669',
    'Proteusmirabilis': '#11521d',
    'Pseudomonasaeoruginosas': '#85868c',
    'Salmonellaspp': '#17e68f',
    'Serratiamarcences': '#4ad9d9',
    'Staphylococcusaureus_6538': '#1aaeb0',
    'Staphylococcusaureus_25923': '#9117cf',
    'Staphylococcusepidemides': '#bf324b',
}

COLORS_RGB = {
    'Bacillusscereus': np.array([255, 25, 0])/255.0,
    'Bacillussubtilis': np.array([194, 124, 81])/255.0,
    'Coryniumbacteriumlutaminum': np.array([125, 94, 32])/255.0,
    'Enterobactearerogenes': np.array([219, 207, 92])/255.0,
    'Enterobactercloacal': np.array([157, 176, 49])/255.0,
    'Enterococcusfaecalis': np.array([157, 255, 0])/255.0,
    'Escheriachiacoli': np.array([181, 138, 212])/255.0,
    'Klesbsialapneumonial': np.array([242, 0, 255])/255.0,
    'Micrococcusluteus': np.array([110, 150, 105])/255.0,
    'Proteusmirabilis': np.array([17, 82, 29])/255.0,
    'Pseudomonasaeoruginosas': np.array([133, 134, 140])/255.0,
    'Salmonellaspp': np.array([23, 230, 143])/255.0,
    'Serratiamarcences': np.array([74, 217, 217])/255.0,
    'Staphylococcusaureus_6538': np.array([26, 74, 176])/255.0,
    'Staphylococcusaureus_25923': np.array([145, 23, 207])/255.0,
    'Staphylococcusepidemides': np.array([191, 50, 75])/255.0,
}

ROOT = 'D:\\TCC'
ORIGINAL_STORE = 'D:\\TCC\\Datasets\\bacterias_new'
PREPROCESS_STORE = 'D:\\TCC\\Datasets\\preprocess_bac_new'
TEST_PROPORTION = 0.3
N_BACS = 16
MODELS_PATH_V3_150 = os.path.join(ROOT, 'models_v3_150')
MODELS_PATH_V3_170 = os.path.join(ROOT, 'models_v3_170')
MODELS_PATH_V3_241 = os.path.join(ROOT, 'models_v3_241')
LABELS = ['Bacillusscereus', 'Bacillussubtilis', 'Coryniumbacteriumlutaminum',
          'Enterobactearerogenes', 'Enterobactercloacal', 'Enterococcusfaecalis', 'Escheriachiacoli',
          'Klesbsialapneumonial', 'Micrococcusluteus', 'Proteusmirabilis', 'Pseudomonasaeoruginosas', 'Salmonellaspp',
          'Serratiamarcences', 'Staphylococcusaureus_6538', 'Staphylococcusaureus_25923', 'Staphylococcusepidemides']
METRICS_PATH_V2_150 = Join(ROOT, 'metrics_v2_150spc')
METRICS_PATH_V2_170 = Join(ROOT, 'metrics_v2_170spc')
METRICS_PATH_V2_241 = Join(ROOT, 'metrics_v2_241spc')
MLP_241 = Join(METRICS_PATH_V2_241, 'mlp')
MLP_170 = Join(METRICS_PATH_V2_170, 'mlp')
MLP_150 = Join(METRICS_PATH_V2_150, 'mlp')
CNN_241 = Join(METRICS_PATH_V2_241, 'cnn')
CNN_170 = Join(METRICS_PATH_V2_170, 'cnn')
CNN_150 = Join(METRICS_PATH_V2_150, 'cnn')
