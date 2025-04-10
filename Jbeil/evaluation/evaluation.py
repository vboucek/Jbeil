import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score as ap_score, roc_auc_score, confusion_matrix, roc_curve, \
    auc as auc_value, precision_recall_fscore_support, average_precision_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def compute_metrics(ys, y_hats, scoress):
    attack_idxs = (ys == 1).nonzero()[0]

    overall_acc = (ys == y_hats).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        ys, y_hats, average="binary"
    )

    tp = y_hats[ys == 1].sum()
    fp = y_hats[ys == 0].sum()
    tpr = y_hats[ys == 1].mean()
    fpr = y_hats[ys == 0].mean()

    len_positives = len((ys == 1).nonzero()[0])
    len_negatives = len((ys == 0).nonzero()[0])

    if scoress is not None:
        try:
            auc = roc_auc_score(ys, scoress)
            auc_fpr, auc_tpr, _ = roc_curve(ys, scoress)
        except:
            auc = float("nan")
            auc_fpr = float("nan")
            auc_tpr = float("nan")

        try:
            ap = ap_score(ys, scoress)
        except:
            ap = float("nan")
    else:
        auc = float("nan")
        auc_fpr = float("nan")
        auc_tpr = float("nan")
        ap = float("nan")

    print(
        f"Evaluation: found {tp} attacks / {len(attack_idxs)} attack samples ({(tp / len(attack_idxs)) * 100:.5f}%)."
    )
    print(
        f"TPR: {tpr:.3f} | FPR: {fpr:.6f} | Overall acc: {overall_acc:.6f} | AP: {ap:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | recall: {recall:.3f} | precision: {precision:.3f}"
    )
    print(f"TP: {tp}/{len_positives} | FP: {fp}/{len_negatives}")

    return auc, precision, recall, tpr, fpr, f1, ap


def eval_edge_detection(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    all_labels, all_preds, all_logits = [], [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, _ = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                           negative_samples, timestamps_batch,
                                                           edge_idxs_batch, n_neighbors)

            pos_prob = pos_prob.cpu().numpy()
            true_label = data.labels[s_idx: e_idx]

            all_logits.append(pos_prob)
            all_labels.append(true_label)

        all_logits = np.concatenate(all_logits).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        def find_best_threshold_gmean(all_logits, ys):
            fpr, tpr, thresholds = roc_curve(ys, all_logits)
            # Compute geometric mean of sensitivity (TPR) and specificity (1 - FPR)
            gmeans = np.sqrt(tpr * (1 - fpr))
            optimal_idx = np.argmax(gmeans)
            optimal_threshold = thresholds[optimal_idx]
            return optimal_threshold

        all_logits = 1 - all_logits

        threshold_opt = find_best_threshold_gmean(all_logits, all_labels)
        all_preds = np.where(all_logits > threshold_opt, 1, 0)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        print(f"Average predicted score: {all_logits.mean():.3f}")
        print(f"Threshold: {threshold_opt:.3f}")
        print("")

        print(f"TP: {tp:.3f}/{all_labels.sum()} | FP: {fp:.3f} | FN: {fn:.3f} | TN: {tn:.3f}")

        compute_metrics(all_labels, all_preds, all_logits)
