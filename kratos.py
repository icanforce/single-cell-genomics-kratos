import os, time, csv
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# random split
from sklearn.model_selection import train_test_split
import os, argparse, random, time, csv;

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC;
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from enum import Enum

import sklearn.metrics
from sklearn.metrics import  silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

import umap
from sklearn.manifold import TSNE
import seaborn as sns

def get_data(args):
    logger.info('input_dir={}'.format(args.input_dir)); assert os.path.exists(args.input_dir);

    cell_profiles = np.genfromtxt(os.path.join(args.input_dir, 'counts.txt'),  delimiter = ',')
    gene_ids = np.genfromtxt(os.path.join(args.input_dir, 'gene_ids.txt'), dtype=str)
    cell_ids = np.genfromtxt(os.path.join(args.input_dir, 'cell_ids.txt'), dtype=str)
    clust_ids = np.genfromtxt(os.path.join(args.input_dir, 'clust_ids.txt')).astype(int) + 1

    logger.info('cell_profiles={}'.format(cell_profiles.shape))
    logger.info('cell_ids={}'.format(len(cell_ids)))
    logger.info('gene_ids={}'.format(len(gene_ids)))
    logger.info('clust_ids={}'.format(len(clust_ids)))

    return cell_profiles, cell_ids, gene_ids, clust_ids

class our_model(object):
    # Build our classifier
    def __init__(self, input_dim,
                 clust_num,
                 batch_size=64,
                 learning_rate=.001,
                 restore_folder='',
                 save_folder='',
                 limit_gpu_fraction=.3,
                 use_gpu=True):

        self.input_dim = input_dim
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.iteration = 0
        self.learning_rate = learning_rate
        self.clust_num = clust_num;

        if restore_folder:
            self._restore(restore_folder)
            return

        self._build();
        self.init_session(use_gpu);

    def _build(self):
        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_shape=(self.input_dim,), name='Input'))
        self.model.add(Dense(64, activation='relu', name='Dense_1'))
        self.model.add(Dense(self.clust_num, activation='softmax', name='Classifier'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def init_session(self, use_gpu):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def train(self, cell_profiles, clust_ids, steps=100):
        BATCH_SIZE = self.batch_size
        EPOCHS = steps
        self.history = self.model.fit(cell_profiles, clust_ids,
                                      validation_split=0.1,
                                      epochs=EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      verbose=0)

    def save(self, save_folder):
        self.model.save(save_folder)

    def _restore(self, restore_folder):
        self.model = tf.keras.models.load_model(restore_folder)

    def get_layer_output(self, data, layer_name):
        func = tf.keras.backend.function([self.model.get_layer('Input').input],
                                         [self.model.get_layer(layer_name).output])
        layer_output = func(data)[0]
        return layer_output

def pos_sample_mapping(model, cell_profile, clust_ids, gene_ids, cell_ids):
    clust_pos_samples_map = {}
    samples_by_clust_list = []
    for clust_c_id in np.unique(clust_ids):
        samples_by_clust = cell_profile[clust_ids == clust_c_id, :]
        num_clust_c = samples_by_clust.shape[0]
        samples_by_clust_list.append(samples_by_clust)
        layer_name = 'Classifier'  # Name the classifier activation layer as 'clust_act'
        activations = model.get_layer_output(samples_by_clust, layer_name) # model.get_layer function add
        clust_pred = np.argmax(activations, 1) + 1
        clust_c_pos_samples = samples_by_clust[clust_pred == clust_c_id, :]
        clust_pos_samples_map[clust_c_id] = clust_c_pos_samples
    return clust_pos_samples_map

def get_classifier_act_gradients(model, input, clust_id):
  input_tensor = tf.convert_to_tensor(input, dtype=tf.float32)
  with tf.GradientTape() as t:
    t.watch(input_tensor)
    loss = model.model(input_tensor)[:, clust_id-1]
  gradients = t.gradient(loss, input_tensor)
  return gradients.numpy()

def calc_saliency(model, cell_profile, clust_ids, gene_ids, cell_ids, saliency_type=0):
  clustid_pos_sample_profiles_map = pos_sample_mapping(model, cell_profile, clust_ids, gene_ids, cell_ids)# from relavance preparation
  clustid_relevance_map = {};
  for clust_c_id in np.unique(clust_ids):
    x0 = np.copy(clustid_pos_sample_profiles_map[clust_c_id]);
    if saliency_type == 0:
      """vanilla gradient"""
      clust_grads = get_classifier_act_gradients(model, x0, clust_c_id);
    elif saliency_type == 1:
      """Smooth Gradient"""
      clust_grads = np.zeros_like(x0);
      repeat = 20;
      stdev = np.std(x0) * 0.2;
      for iter in range(repeat):
        noise = np.random.normal(0, stdev, x0.shape);
        clust_grads += get_classifier_act_gradients(model, x0 + noise, clust_c_id);
    elif saliency_type == 2:
      """Integrated Gradients"""
      clust_grads = np.zeros_like(x0);
      baseline = np.zeros_like(x0);
      steps = 20;
      for alpha in np.linspace(0, 1, steps):
        clust_grads += get_classifier_act_gradients(model,
                  alpha * x0, clust_c_id);
    elif saliency_type == 3:
      """SHAP: conda install -c conda-forge shap """
      import shap;
      e = shap.GradientExplainer((X, clust_c_Act_tensor), x0, session=model.sess,
                                  local_smoothing=0);
      clust_grads = e.shap_values(x0);
    else:
      assert False;
    clust_grads_mean = np.sum(clust_grads, axis=1);
    inf_or_nan_indices = np.where(np.isinf(clust_grads_mean) | np.isnan(clust_grads_mean))[0];
    logger.info('x0={}\tinf_or_nan_indices={}'.format(x0.shape, inf_or_nan_indices.shape));
    if len(inf_or_nan_indices) == len(x0): raise RuntimeError(
      'explanation failed with Inf/Nan values ');
    valid_indices = np.setdiff1d(list(range(len(x0))), inf_or_nan_indices);
    clustid_relevance_map[clust_c_id] = clust_grads[valid_indices];
    logger.info('clust_id={}\tclust_grads={}'.format(clust_c_id, clustid_relevance_map[clust_c_id].shape))

  return clustid_relevance_map;

def calc_perturbation_loss(model, input, clust_c_id, clust_ids, margin = 0):
  input_tensor = tf.convert_to_tensor(input, dtype=tf.float32)
  activations = model.model(input_tensor)
  clust_c_Act_tensor = activations[:, clust_c_id - 1]
  clust_k_Act_tensor_list = [];
  for clust_k_id in np.unique(clust_ids):
    if clust_c_id == clust_k_id: continue;
    clust_k_Act_tensor = activations[:, clust_k_id-1];
    clust_k_Act_tensor_list.append(clust_k_Act_tensor);
  clust_k_Act_tensor = clust_k_Act_tensor_list[0];
  for clust_k_idx in range(1, len(clust_k_Act_tensor_list)):
    clust_k_Act_tensor = tf.math.maximum(clust_k_Act_tensor, clust_k_Act_tensor_list[clust_k_idx]);
  clust_loss_tensor = tf.math.maximum(margin+clust_c_Act_tensor-clust_k_Act_tensor, 0);
  return clust_loss_tensor

def calc_carlini_wagner_one_vs_rest(model, cell_profile, clust_ids, gene_ids, cell_ids, lamda=1e2, max_iter=5000,
                                    lr=2e-3, margin=0):
    """explain each cluster """
    clustid_relevance_map = {};
    clustid_pos_sample_profiles_map = pos_sample_mapping(model, cell_profile, clust_ids, gene_ids,
                                                         cell_ids)  # from relavance preparation

    for clust_c_id in np.unique(clust_ids):
        """starting carlini_wagner optimization"""
        x0 = np.copy(clustid_pos_sample_profiles_map[clust_c_id]);
        curr_x = x0 + 1e-2;
        curr_x_tensor = tf.convert_to_tensor(curr_x, dtype=tf.float32)
        clust_loss = calc_perturbation_loss(model, x0, clust_c_id, clust_ids, margin)
        for iter in range(max_iter):
            with tf.GradientTape() as t:
                t.watch(curr_x_tensor)
                clust_loss_tensor = calc_perturbation_loss(model, curr_x_tensor,
                                                           clust_c_id, clust_ids, margin)
            clust_loss_grads = t.gradient(clust_loss_tensor, curr_x_tensor).numpy()

            if np.any([np.isinf(x) for x in clust_loss_grads]) or np.any([np.isnan(x) for x in clust_loss_grads]):
                clust_loss_grads_mean = np.sum(clust_loss_grads, axis=1);
                inf_or_nan_indices = np.where(np.isinf(clust_loss_grads_mean) | np.isnan(clust_loss_grads_mean))[0];
                logger.info('x0={}\tinf_or_nan_indices={}'.format(x0.shape, inf_or_nan_indices.shape));

                if len(inf_or_nan_indices) == len(x0): raise RuntimeError(
                    'explanation failed with Inf/Nan values ');
                valid_indices = np.setdiff1d(list(range(len(x0))), inf_or_nan_indices);
                x0 = x0[valid_indices];
                curr_x = x0 + 1e-10;
                continue;

            x_to_x0_sgn = np.asarray((curr_x - x0) >= 0, dtype=float);
            x_to_x0_sgn[x_to_x0_sgn <= 0] = -1;
            curr_x = curr_x - lr * (x_to_x0_sgn + lamda * clust_loss_grads);
            curr_x_tensor = tf.convert_to_tensor(curr_x, dtype=tf.float32)
            x_to_x0_loss = np.sum(np.fabs(curr_x - x0));
            clust_loss = calc_perturbation_loss(model, curr_x, clust_c_id, clust_ids, margin).numpy()
            clust_loss = np.sum(np.fabs(clust_loss))
            if (iter + 1) % 5 == 0: logger.info(
                'iter={}\tx_to_x0_loss={}\tclust_loss={}'.format(iter, x_to_x0_loss, clust_loss, ));

        clustid_relevance_map[clust_c_id] = (curr_x - x0);
    return clustid_relevance_map;

def preprocessing(cell_profiles, cell_ids, clust_ids, gene_ids):
  clust_num = len(np.unique(clust_ids))
  cell_num = cell_profiles.shape[0]
  clust_ids_int = [int(_) for _ in clust_ids]
  clust_ids_np = np.zeros((cell_num, clust_num))
  for i in range(0, cell_num):
    clust_ids_np[i,clust_ids_int[i]-1] = 1

  seed = 7   # fix the random seed
  test_size_ratio = 0.3   # ratio of test size compared to the whole data

  min_max_scaler = MinMaxScaler()
  norm_cells = min_max_scaler.fit_transform(cell_profiles)
  train_cells, test_cells, train_ids, test_ids = train_test_split(norm_cells,
            clust_ids_np, test_size=test_size_ratio, random_state=seed,
            stratify=clust_ids_np, shuffle=True)
  train_ids = np.int32(train_ids)
  test_ids = np.int32(test_ids)
  return train_cells, train_ids, test_cells, test_ids

def output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, use_abs=False, tag=''):
  """output the relevance for each clust_id"""
  for clust_id in clustid_relevance_map:
    logger.info('clust_id={}'.format(clust_id, ));
    if use_abs:
      relevance_arr = np.mean(np.fabs(clustid_relevance_map[clust_id]), axis=0);
      gene_relev_list = zip(relevance_arr, gene_ids);
      #gene_relev_list.sort(key=lambda x: -x[0]);
      gene_relev_list = sorted(gene_relev_list, key=lambda x: -x[0])
    else:
      relevance_arr = np.mean(clustid_relevance_map[clust_id], axis=0);
      gene_relev_list = zip(relevance_arr, gene_ids);
      #gene_relev_list.sort(key=lambda x: x[0]);
      gene_relev_list = sorted(gene_relev_list, key = lambda x: x[0])

    f = open(os.path.join(output_dir, "genes_{}_clustid{}_abs{}.csv".format(tag, clust_id, int(use_abs) )), "w");
    f.write("Rank\tGene\tScore\n");
    for idx in range(len(gene_relev_list)):
      score, gene = gene_relev_list[idx];
      f.write("{}\t{}\t{}\n".format(idx+1, gene, score));
    f.close();

def get_clust_prediction(model, cell_profiles, clust_ids):
    activations = model.get_layer_output(cell_profiles, 'Classifier')
    clust_pred = np.argmax(activations, 1) + 1
    return clust_pred

def clust_metrics(model, cell_profiles, clust_ids, output_dir):
  csv_dir = output_dir + '/validation'
  if not os.path.isdir(csv_dir): os.mkdir(csv_dir)
  csv_dir = csv_dir + '/metrics.csv'
  cell_embeddings = model.get_layer_output(cell_profiles, 'Dense_1')
  sil_score = silhouette_score(cell_embeddings, clust_ids, metric='euclidean')
  clust_pred = get_clust_prediction(model, cell_profiles, clust_ids)
  ARI_score = sklearn.metrics.adjusted_rand_score(clust_ids, clust_pred)
  AMI_score = sklearn.metrics.adjusted_mutual_info_score(clust_ids, clust_pred)
  with open(csv_dir, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['silhouette score', 'ARI', 'AMI'])
    csv_writer.writerow([sil_score, ARI_score, AMI_score])
  return sil_score, ARI_score, AMI_score

def umap_plot(cell_profiles, cell_embeddings, clust_ids, model_dir):
    plot_dir = os.path.join(model_dir, 'validation');
    latent_plot_dir = os.path.join(plot_dir, 'latent_umap.pdf')
    original_plot_dir = os.path.join(plot_dir, 'original_umap.pdf')

    # Latent Plot
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(cell_embeddings)

    figure = plt.figure(figsize=(12, 12));
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[clust_id] for clust_id in clust_ids])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Latent Embeddings', fontsize=24)
    pp = PdfPages(latent_plot_dir);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

    # Original Plot
    reducer = umap.UMAP(n_neighbors=80)
    embedding = reducer.fit_transform(cell_profiles)

    figure = plt.figure(figsize=(12, 12));
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[clust_id] for clust_id in clust_ids])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Original Embeddings', fontsize=24)
    pp = PdfPages(original_plot_dir);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

    return None

def tsne_plot(cell_profiles, cell_embeddings, clust_ids, model_dir):
    plot_dir = os.path.join(model_dir, 'validation');
    latent_plot_dir = os.path.join(plot_dir, 'latent_tsne.pdf')
    original_plot_dir = os.path.join(plot_dir, 'original_tsne.pdf')

    # Latent Plot
    embedding = TSNE(n_components=2, perplexity = 30).fit_transform(cell_embeddings)
    figure = plt.figure(figsize=(12, 12));
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[clust_id] for clust_id in clust_ids])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('tSNE projection of the Latent Embeddings', fontsize=24)
    pp = PdfPages(latent_plot_dir);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

    # Original Plot
    embedding = TSNE(n_components=2, perplexity=30).fit_transform(cell_profiles)

    figure = plt.figure(figsize=(12, 12));
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[clust_id] for clust_id in clust_ids])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('tSNE projection of the Original Embeddings', fontsize=24)
    pp = PdfPages(original_plot_dir);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

    return None

def main(args):
    cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args)
    input_dir = args.input_dir
    steps = args.steps
    label = args.label
    max_iter = args.max_iter
    lr = args.lr
    margin = args.margin
    lamda = args.lamda
    use_abs = args.use_abs
    input_dim = cell_profiles.shape[1]
    clust_num = len(np.unique(clust_ids))
    cell_num = cell_profiles.shape[0]

    train_cells, train_ids, test_cells, test_ids = preprocessing(cell_profiles,
                                               cell_ids, clust_ids, gene_ids)

    learning_rate = 0.001
    batch_size = np.int32(64)
    output_dir = './results' + '/' + label + '/'
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    para_dir = 'lr{}_steps{}_lamda{}'.format(lr, steps, lamda)
    model_dir = output_dir  + 'model_' + para_dir
    results_dir = output_dir + '/results_' + para_dir
    model = our_model(input_dim=input_dim, clust_num=clust_num,
                      batch_size=64, learning_rate=learning_rate)
    model.train(train_cells, train_ids, steps=steps)
    model.save(model_dir)

    pos_sample_map = pos_sample_mapping(model, cell_profiles, clust_ids, gene_ids, cell_ids)
    tag = 'vanila_onevsrest';
    relevance_url = results_dir + '/{}.npy'.format(tag)
    if os.path.isfile(relevance_url):
        clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        print('Vanila')
        clustid_relevance_map = calc_saliency(model, cell_profiles, clust_ids, gene_ids, cell_ids, saliency_type=0);
        if not os.path.isdir(results_dir): os.makedirs(results_dir)
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(results_dir, clustid_relevance_map, gene_ids, tag=tag);

    tag = 'smoothgrad_onevsrest';
    relevance_url = output_dir + '/results_' + para_dir + '/{}.npy'.format(tag)
    if os.path.isfile(relevance_url):
        clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        print('Smoothgrad')
        clustid_relevance_map = calc_saliency(model, cell_profiles, clust_ids, gene_ids, cell_ids, saliency_type=1);
        if not os.path.isdir(results_dir): os.makedirs(results_dir)
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(results_dir, clustid_relevance_map, gene_ids, tag=tag);

    tag = 'integrad_onevsrest';
    relevance_url = output_dir + '/results_' + para_dir + '/{}.npy'.format(tag)
    if os.path.isfile(relevance_url):
        clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        print('Integrate')
        clustid_relevance_map = calc_saliency(model, cell_profiles, clust_ids, gene_ids, cell_ids, saliency_type=2);
        if not os.path.isdir(results_dir): os.makedirs(results_dir)
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(results_dir, clustid_relevance_map, gene_ids, tag=tag);

    tag = 'cw_onevsrest_lamda{}_iter{}_lr{}'.format(lamda, max_iter, lr);
    relevance_url = output_dir + '/results_' + para_dir + '/{}.npy'.format(tag)
    if os.path.isfile(relevance_url):
        clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = calc_carlini_wagner_one_vs_rest(model, cell_profiles, clust_ids, gene_ids,
                                                                cell_ids, lamda=lamda, max_iter=max_iter, lr=lr,
                                                                margin=margin);
        if not os.path.isdir(results_dir): os.makedirs(results_dir)
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(results_dir, clustid_relevance_map, gene_ids, tag=tag);

    sil, ari, ami = clust_metrics(model, cell_profiles, clust_ids, results_dir)
    cell_embeddings = model.get_layer_output(cell_profiles, 'Dense_1')
    umap_plot(cell_profiles, cell_embeddings, clust_ids, results_dir)
    tsne_plot(cell_profiles, cell_embeddings, clust_ids, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--input_dir', type=str, help='input_dir')
    parser.add_argument('--steps', type=int, help='steps for nn traning', default=50)

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2)
    parser.add_argument('--lr', type=float, help='learning rate for perturbation', default=1e-3)
    parser.add_argument('--max_iter', type=int, help='max iteration', default=5000)
    parser.add_argument('--margin', type=float, help='margin', default=0)

    parser.add_argument('--label', type=str, help='dataset label', default='PBMC')
    parser.add_argument('--use_abs', type=int, help='use abs', default=1)

    args = parser.parse_args()
    main(args)

# --input_dir ./data/PBMC --steps 50 --lamda 100 --lr 0.001 --max_iter 500 --margin 0 --label PBMC --use_abs 1












