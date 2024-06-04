import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score as ap_score, roc_auc_score, confusion_matrix, recall_score, precision_score, roc_curve, auc as auc_value, precision_recall_fscore_support


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_recall, val_precision, val_fp, val_fn, val_tp, val_tn = [], [], [], [], [], [], [], []
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

      new_pred_score = []
      #print("########################")
      #print("Pred_score")
      #print(pred_score.shape)
      #print(np.squeeze(pred_score).shape)
      #print(pred_score[:30])
      
      #print("########################")
      #print(pred_score[:30])

      #print("########################")
      #print("true_label")
      #print(true_label.shape)
      #print(true_label[:30])

      # val_ap.append(average_precision_score(true_label, pred_score))
      # val_auc.append(roc_auc_score(true_label, pred_score))
      
      ### Joseph: Added roc_cuve to get the optial threshold ####
      fpr, tpr, thresholds = roc_curve(true_label, pred_score)
      
      gmean = np.sqrt(tpr * (1 - fpr))

      # Find the optimal threshold
      index = np.argmax(gmean)
      #### Use the  "thresholdOpt" for the best threshold
      thresholdOpt = round(thresholds[index], ndigits = 4)
      gmeanOpt = round(gmean[index], ndigits = 4)
      
      # print("########################")
      # print("thresholdOpt ", thresholdOpt)
      # print("gmeanOpt ", gmeanOpt)
      # print("our threshold 0.50")
      # print("########################")

      for i in range(len(pred_score)):
        ## Modified byb Joseph
        # if pred_score[i] <= 0.50:
        if pred_score[i] < thresholdOpt:
          pred_score[i] = 0
        else:
          pred_score[i] = 1
      
      val_ap.append(ap_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))  
      
      #print(recall_score(true_label, pred_score))
      val_recall.append(recall_score(true_label, pred_score))
      val_precision.append(precision_score(true_label, pred_score))

      

      # conf_mtrx = confusion_matrix(true_label, pred_score)
      # val_fp.append(conf_mtrx.sum(axis=0) - np.diag(conf_mtrx))
      # val_fn.append(conf_mtrx.sum(axis=1) - np.diag(conf_mtrx))
      # val_tp.append(np.diag(conf_mtrx))
      # val_tn.append(conf_mtrx.values.sum() - (FP + FN + TP))
      
      tn, fp, fn, tp = confusion_matrix(true_label, pred_score).ravel()
      val_fp.append(fp)
      val_fn.append(fn)
      val_tp.append(tp)
      val_tn.append(tn)

  #return np.mean(val_ap), np.mean(val_auc), np.mean(val_fp), np.mean(val_fn), np.mean(val_tp), np.mean(val_tn)
  # return np.mean(val_ap), np.mean(val_auc), np.mean(val_recall), np.mean(val_precision)
  return np.mean(val_ap), np.mean(val_auc), np.mean(val_recall), np.mean(val_precision), np.mean(val_fp), np.mean(val_fn), np.mean(val_tp), np.mean(val_tn), thresholdOpt

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
      f"Evaluation: found {tp} attacks / {len(attack_idxs)} attack samples ({(tp/len(attack_idxs))*100:.5f}%)."
  )
  print(
      f"TPR: {tpr:.3f} | FPR: {fpr:.6f} | Overall acc: {overall_acc:.6f} | AP: {ap:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | recall: {recall:.3f} | precision: {precision:.3f}"
  )
  print(f"TP: {tp}/{len_positives} | FP: {fp}/{len_negatives}")

  return auc, precision, recall, tpr, fpr, f1, ap

def eval_edge_detection(model, negative_edge_sampler, data, n_neighbors, thresholdOpt, batch_size=200):
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

    def find_best_threshold_auc(all_logits, ys):
      fpr, tpr, thresholds = roc_curve(ys, all_logits)
      roc_auc = auc_value(fpr, tpr)

      # Filter out the points where TPR is less than 0.84
      valid_indices = np.where(tpr >= 0.84)[0]
      fpr_valid = fpr[valid_indices]
      thresholds_valid = thresholds[valid_indices]

      # Find the threshold corresponding to the lowest FPR among valid points
      optimal_idx = np.argmin(fpr_valid)
      optimal_threshold = thresholds_valid[optimal_idx]

      return optimal_threshold

    all_logits = 1 - all_logits
    
    thresholdOpt = find_best_threshold_auc(all_logits, all_labels)
    all_preds = np.where(all_logits > thresholdOpt, 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Average predicted score: {all_logits.mean():.3f}")
    print(f"Threshold: {thresholdOpt:.3f}")
    print("")

    print(f"TP: {tp:.3f}/{all_labels.sum()} | FP: {fp:.3f} | FN: {fn:.3f} | TN: {tn:.3f}")
    
    compute_metrics(all_labels, all_preds, all_logits)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
