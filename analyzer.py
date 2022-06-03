import os, argparse, random, time, csv;
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC;
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from enum import Enum

class Type(Enum):
    Prediction = 1
    Correlation = 2

def get_feat_correlation(cell_profiles):
    corr_mat = 1.0 - pairwise_distances(cell_profiles.T, metric='correlation');
    # logger.info('corr_mat={}'.format(corr_mat.shape));
    # figure = plt.figure(figsize=(12, 12));
    # ax = figure.add_subplot(1, 1, 1);
    # ax.hist(corr_mat.ravel(), bins=200, );
    # ax.set_xlabel('Correlation', fontsize=25);
    # ax.set_ylabel('# of feature pair', fontsize=25);
    # plt.show();
    return np.mean(corr_mat), np.std(corr_mat);

def classify_cell_clust(cell_profiles, clust_ids, target_clustid, cv_fold=3, total_repeat=1, test_size_ratio=0.3):
  """
  clust_ids: 在SVM训练中应该是一个list 而不是 matrix. 要注意区别 NN network & SVM 的 label 输入
  total_repeat: 循环次数，同时作为random_seed输入，最后输出 auc_list 长度为 total_repeat
  cv_fold: cross validation 多少折
  """
  assert target_clustid in np.unique(clust_ids);
  # param_grid = {'C': np.power(5.0, np.arange(-5, 6))};
  # param_grid = {'C': np.power(3.0, np.arange(-7, 8))};
  param_random = {'C': stats.expon(scale=100), 'kernel': ['rbf'], 'gamma': stats.expon(scale=0.1)}

  true_indices = np.where(clust_ids == target_clustid)[0];
  false_indices = np.where(clust_ids != target_clustid)[0];
  logger.info('clustid={}\ttrue_indices={}\tfalse_indices={}'.format(target_clustid, len(true_indices), len(false_indices) ));
  cell_labels = np.zeros(len(clust_ids)); cell_labels[true_indices] = 1;

  auc_list = [];
  for repeat in range(total_repeat):   # 原来code中一个repeat会产生cv_fold个auc，现在一个repeat只会产生一个auc. 全样本重复k-fold感觉意义不大
    kf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=repeat);
    # cell_profiles_train, cell_profiles_test, cell_labels_train, cell_labels_test = train_test_split(cell_profiles.T, cell_labels, test_size=test_size_ratio, random_state=repeat, stratify=cell_labels, shuffle=True)
    # 到时候要看输入的 cell_profiles 需不需要转置
    # logger.info('train_positive={}/{}\ttest_positive={}/{}'.format(np.sum(cell_labels_train > 0), len(cell_labels_train), np.sum(cell_labels_test > 0), len(cell_labels_test),  ));
    for train_indices, test_indices in kf.split(cell_profiles, cell_labels):
      logger.info('train_positive={}/{}\ttest_positive={}/{}'.format(np.sum(cell_labels[train_indices] > 0), len(train_indices), np.sum(cell_labels[test_indices] > 0), len(test_indices),  ));

      cell_profiles_train = cell_profiles[train_indices]; cell_labels_train = cell_labels[train_indices];
      cell_profiles_test = cell_profiles[test_indices]; cell_labels_test = cell_labels[test_indices];

      svm = SVC(kernel='rbf', random_state=0, class_weight='balanced', probability=False);
      searchCV = RandomizedSearchCV(svm, param_random, scoring='roc_auc', cv=cv_fold, n_jobs=-1, n_iter=200);
      searchCV = searchCV.fit(cell_profiles_train, cell_labels_train);
      optC = searchCV.best_params_['C'];
      optGamma = searchCV.best_params_['gamma']
      # svm = SVC(kernel='rbf', random_state=0, C=searchCV.best_params_['C'], class_weight='balanced', probability=False);
      # svm.fit(cell_profiles_train, cell_labels_train);
      # 多此一举，默认就是 best found hyperparameters
      auc_score = roc_auc_score(y_true=cell_labels_test, y_score=searchCV.decision_function(cell_profiles_test));
      logger.info('auc_score={}\toptC={}\toptGamma={}'.format(auc_score, optC, optGamma));
      auc_list.append(auc_score);
  logger.info('auc_list={}'.format(auc_list ));
  return auc_list;

def process_cell_clust_by_feat_subset(gene_relevance_url, cell_profiles, gene_ids, clust_ids, target_clustid, output_url, process_type, use_abs=False ):
    gene_score_list = [];
    with open(gene_relevance_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            gene = row['Gene'];
            raw_score = float(row['Score']);
            abs_score = np.fabs(raw_score);
            gene_score_list.append((gene, raw_score, abs_score));

    if use_abs: gene_score_list.sort(key=lambda x: -x[2]);
    else: gene_score_list.sort(key=lambda x: x[1]);

    ranked_gene_list = [x[0] for x in gene_score_list]; #
    # assert len(gene_ids) == len(np.intersect1d(ranked_gene_list, gene_ids));
    gene_to_index_map = dict(zip(gene_ids, np.arange(len(gene_ids)) ));

    # percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 101, 2), dtype=float) )) / 100;
    # percent_arr = np.asarray(range(1, 101), dtype=float) / 100;
    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 40, 2), dtype=float), np.asarray(range(40, 101, 4), dtype=float), )) / 100;

    feat_subset_arr = np.asarray(np.round(percent_arr * len(gene_ids)), dtype=int);
    logger.info('feat_subset_arr={}'.format(feat_subset_arr));

    f = open(output_url, "w");
    if process_type == Type.Prediction:
        f.write("Selected_gene_percent\tSelected_gene_cnt\tAuc_scores\n");
    elif process_type == Type.Correlation:
        f.write("Selected_gene_percent\tSelected_gene_cnt\tCorr_mean\tCorr_std\n");
    else: assert False;

    for feat_subset_percent, feat_subset_cnt in zip(percent_arr, feat_subset_arr):
        logger.info('feat_subset_percent={}\tfeat_subset_cnt={}'.format(feat_subset_percent, feat_subset_cnt ));
        ranked_gene_subset = ranked_gene_list[:feat_subset_cnt];
        ranked_gene_indices = np.unique([gene_to_index_map[str(gene)] for gene in ranked_gene_subset]);

        if process_type == Type.Prediction:
            auc_list = classify_cell_clust(cell_profiles[:, ranked_gene_indices], clust_ids, target_clustid, );
            f.write("{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, ','.join(['{:.4f}'.format(x) for x in auc_list]), ));
        elif process_type == Type.Correlation:
            corr_mean, corr_std = get_feat_correlation(cell_profiles[:, ranked_gene_indices]);
            f.write("{}\t{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, corr_mean, corr_std, ));
        else: assert False;

    f.close();

def plot_subset_upsetplot(target_clustid, top_k=100):

    def get_topk_feat(gene_relevance_url, top_k, use_abs=False, ):
        gene_score_list = [];
        with open(gene_relevance_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                gene = row['Gene'];
                raw_score = float(row['Score']);
                abs_score = np.fabs(raw_score);
                gene_score_list.append((gene, raw_score, abs_score));

        if use_abs: gene_score_list.sort(key=lambda x: -x[2]);
        else: gene_score_list.sort(key=lambda x: x[1]);
        ranked_gene_list = [x[0] for x in gene_score_list];
        return ranked_gene_list[:top_k];

    parent_dir = '../data/PBMC/results_kmeans';

    label_gce = 'GCE'; use_abs_gce = True;
    relevance_url_gce = os.path.join(parent_dir, 'gce_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_gce = get_topk_feat(relevance_url_gce, top_k, use_abs=use_abs_gce, );

    label_grs = 'GRS'; use_abs_grs = True;
    relevance_url_grs = os.path.join(parent_dir, 'grs_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_grs = get_topk_feat(relevance_url_grs, top_k, use_abs=use_abs_grs, );

    label_jsd = 'JSD'; use_abs_jsd = False;
    relevance_url_jsd = os.path.join(parent_dir, 'jsd_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_jsd = get_topk_feat(relevance_url_jsd, top_k, use_abs=use_abs_jsd, );

    label_deseq2 = 'DESeq2'; use_abs_deseq2 = False;
    relevance_url_deseq2 = os.path.join(parent_dir, 'deseq2_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_deseq2 = get_topk_feat(relevance_url_deseq2, top_k, use_abs=use_abs_deseq2, );

    data_dir = os.path.join(parent_dir, 'results_pbmc_beta0.01_margin0_fd256_latent512,256,128,2/');

    label_smoothgrad = 'SmoothGrad'; use_abs_smoothgrad = True;
    relevance_url_smoothgrad = os.path.join(data_dir, 'genes_smoothgrad_onevsrest_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_smoothgrad = get_topk_feat(relevance_url_smoothgrad, top_k, use_abs=use_abs_smoothgrad, );

    label_shap = 'SHAP'; use_abs_shap = True;
    relevance_url_shap = os.path.join(data_dir, 'genes_shap_onevsrest_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_shap = get_topk_feat(relevance_url_shap, top_k, use_abs=use_abs_shap, );

    label_ace = 'ACE'; use_abs_ace = True;
    relevance_url_ace = os.path.join(data_dir, 'genes_cw_onevsrest_lamda100.0_iter5000_lr0.001_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_ace = get_topk_feat(relevance_url_ace, top_k, use_abs=use_abs_ace, );

    gene_list_all = np.unique(np.concatenate((gene_list_gce, gene_list_grs, gene_list_jsd, gene_list_deseq2, gene_list_smoothgrad, gene_list_shap, gene_list_ace )));
    logger.info('gene_list_all={}'.format(len(gene_list_all) ));
    label_genes_map = {label_gce:gene_list_gce, label_grs:gene_list_grs, label_jsd:gene_list_jsd, label_deseq2:gene_list_deseq2, label_smoothgrad: gene_list_smoothgrad, label_shap:gene_list_shap, label_ace:gene_list_ace };

    other_label_list = [label_gce, label_grs, label_jsd, label_deseq2, label_smoothgrad, label_shap];
    f = open(os.path.join(parent_dir, 'results_subset_uniqgenes_clustid{}_top{}.txt'.format(target_clustid, top_k )), "w");
    for gene in gene_list_ace:
        if not np.any([(gene in label_genes_map[label]) for label in other_label_list]):
            f.write("{}\n".format(gene));
    f.close();

    from upsetplot import UpSet, plot, from_contents
    figure = plt.figure(figsize=(20, 8));
    data_pd = from_contents(label_genes_map);
    plot(data_pd, show_counts='%d', sort_by='cardinality', subset_size='count', );

    plt.ylabel('Intersection size', fontsize=20);
    plt.yticks(fontsize=18)
    # plt.show();
    fig_url = os.path.join(parent_dir, 'results_subset_upsetplot_clustid{}_top{}.pdf'.format(target_clustid, top_k ));
    plt.savefig(fig_url);

def get_data(args, ):
    cell_profiles = np.genfromtxt(os.path.join(args.input_dir, 'counts.txt'), delimiter=',');
    cell_profiles = preprocessing(cell_profiles)
    gene_ids = np.genfromtxt(os.path.join(args.input_dir, 'gene_ids.txt'), dtype=None);
    cell_ids = np.genfromtxt(os.path.join(args.input_dir, 'cell_ids.txt'), dtype=None);
    clust_ids = np.genfromtxt(os.path.join(args.input_dir, 'clust_ids.txt')).astype(int) + 1;

    logger.info('cell_profiles={}'.format(cell_profiles.shape));
    logger.info('cell_ids={}'.format(len(cell_ids)));
    logger.info('gene_ids={}'.format(len(gene_ids)));
    logger.info('clust_ids={}'.format(len(clust_ids)));

    return cell_profiles, cell_ids, np.asarray(gene_ids, dtype=str), clust_ids;

def preprocessing(cell_profiles):
    min_max_scaler = MinMaxScaler()
    norm_cells = min_max_scaler.fit_transform(cell_profiles)
    return norm_cells

def main(args):
    assert os.path.isfile(args.gene_relevance_url);
    cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);

    ### subset feature classification
    output_url = args.gene_relevance_url.replace('.csv', '.subset_pred.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Prediction, use_abs=args.use_abs);

    ### subset feature correlation
    output_url = args.gene_relevance_url.replace('.csv', '.subset_corr.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Correlation, use_abs=args.use_abs);

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_dir', type=str, help='input_dir');
    parser.add_argument('--gene_relevance_url', type=str, help='gene_relevance_url', default='');

    parser.add_argument('--target_clustid', type=int, help='target_clustid');
    parser.add_argument('--use_abs', type=int, help='use_abs');

    args = parser.parse_args();
    main(args);


"""example command"""
# --input_dir ./data/PBMC --gene_relevance_url ./results/PBMC/results_lr0.001_steps50_lamda100.0/genes_cw_onevsrest_lamda100.0_iter500_lr0.001_clustid1_abs0.csv --target_clustid 1 --use_abs 1
