import os, argparse, random, time, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats

from enum import Enum

def plot_subset_pred(args):
    result_url = args.input_dir + '/analyze_pred.csv'
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'ACE_cw', 'ACE_sg', 'ACE_shap', 'Kratos_sg', 'Kratos_cw', 'Kratos_shap']
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f']
    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=int), np.asarray(range(10, 40, 2), dtype=int), np.asarray(range(40, 101, 4), dtype=int),));
    #percent_arr = np.linspace(0.001, 0.05, 21)
    label_percent_auc_map = {};
    for label in label_list:
        label_percent_auc_map[label] = {};
        for percent in percent_arr:
            percent = float(format(percent, '.5f'))
            label_percent_auc_map[label][percent] = np.asarray([], dtype=float);

    with open(result_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            percent = np.int32(100*float((row['Selected_gene_percent'])));
            clustid = int(row['clust_id']);
            # if clustid == 4: continue;

            for label in label_list:
                auc_arr = np.asarray(row[label].split(','), dtype=float);
                label_percent_auc_map[label][percent] = np.concatenate((label_percent_auc_map[label][percent], auc_arr ));

    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    for label, color in zip(label_list, color_list):
        mean_list = []; std_list = [];
        for percent in percent_arr:
            percent = float(format(percent, '.5f'))
            mean_list.append(np.mean(label_percent_auc_map[label][percent]));
            std_list.append(np.std(label_percent_auc_map[label][percent]));

        ax.errorbar(percent_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('AUROC', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=20);
    ax.yaxis.set_tick_params(labelsize=25);
    ax.set_xscale('log');

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{}%'.format((100*x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def plot_subset_corr(args):
    result_url = args.input_dir + '/analyze_corr.csv'
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'ACE_cw', 'ACE_sg', 'ACE_shap', 'Kratos_sg', 'Kratos_cw', 'Kratos_shap']
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f']

    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=int), np.asarray(range(10, 40, 2), dtype=int), np.asarray(range(40, 101, 4), dtype=int),));

    label_percent_mean_map = {}; label_percent_std_map = {};
    for label in label_list:
        label_percent_mean_map[label] = {};
        label_percent_std_map[label] = {};
        for percent in percent_arr:
            label_percent_mean_map[label][percent] = [];
            label_percent_std_map[label][percent] = [];

    with open(result_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            percent = int(float(row['Selected_gene_percent'])*100);
            clustid = int(row['clust_id']);

            for label in label_list:
                mean_val = float(row['{}_mean'.format(label)]);
                std_val = float(row['{}_std'.format(label)]);
                label_percent_mean_map[label][percent].append(mean_val);
                label_percent_std_map[label][percent].append(std_val);


    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    for label, color in zip(label_list, color_list):
        mean_list = []; std_list = [];
        for percent in percent_arr:
            mean_list.append(np.mean(label_percent_mean_map[label][percent]));
            # std_list.append(np.mean(label_percent_std_map[label][percent]));
            std_list.append(stats.sem(label_percent_std_map[label][percent], axis=None));

        ax.errorbar(percent_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('Pearson correlation', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=25);
    ax.yaxis.set_tick_params(labelsize=25);
    ax.set_xscale('log');

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:d}%'.format(int(x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def main(args):
    ### plotting
    plot_subset_pred(args);
    plot_subset_corr(args);

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_dir', type=str, help='input_dir');

    args = parser.parse_args();
    main(args);


# --input_dir ./results/PBMC/results_lr0.001_steps50_lamda100.0






