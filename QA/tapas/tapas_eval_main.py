"""
inference only for tapas model,
for evaluation, use official evaluator of each dataset
"""

from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd
import os
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from QA.data_structure import TableQADataset
from QA.metric import DenotationAccuracy
from tqdm import tqdm


class TapasDataset(torch.utils.data.Dataset, TableQADataset):
    def __init__(self, data_dir, source, tokenizer):
        super(TapasDataset, self).__init__(
            data_dir, split='test', source=source)

        self.tokenizer = tokenizer
        assert source in ["web", "textract"], f"Unknown source: {source}"
        self._attr_str = "table_retrieval_file_" + source

    def __getitem__(self, idx):
        this_sample = self.qa_examples[idx]
        table = self.table_index[this_sample.__getattribute__(self._attr_str)]
        # print(table)
        encoding = self.tokenizer(
            table=table,
            queries=this_sample.question,
            padding="max_length",
            truncation="drop_rows_to_fit",
            return_tensors="pt",
        )
        return encoding

    def __len__(self):
        return len(self.qa_examples)

    def __iter__(self):
        for i in range(len(self.qa_examples)):
            this_table = self.table_index[
                self.qa_examples[i].__getattribute__(self._attr_str)]
            try:
                yield self.qa_examples[i], this_table, self.__getitem__(i)
            except Exception as e:
                print("Skipping sample due to ", e)
                continue


def parse_result(
        table, predicted_answer_coordinates,
        predicted_aggregation_indices):
    cells = []
    assert len(predicted_answer_coordinates) == 1, \
        'violating single-sample per batch assumption'
    for coordinates in predicted_answer_coordinates[0]:
        cells.append(table.iat[coordinates])

    # due to single example batch:
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregator = id2aggregation[predicted_aggregation_indices[0]]
    if aggregator == 'NONE':
        return (aggregator, cells), cells
    if aggregator == "SUM":
        try:
            return (aggregator, cells), [sum(float(x) for x in cells)]
        except Exception as e:
            return (aggregator, cells), cells
    if aggregator == 'AVERAGE':
        try:
            return (aggregator, cells), \
                   [sum(float(x) for x in cells) / len(cells)]
        except Exception as e:
            return (aggregator, cells), cells
    if aggregator == "COUNT":
        return (aggregator, cells), [len(cells)]
    raise AttributeError("Unreachable Error")


def write_in_wtq_format(answers, raw_data, fp):
    ans_str = '\t'.join([str(x) for x in answers])
    sample_id = raw_data.sample_id
    fp.write(f"{sample_id}\t{ans_str}\n")


def inference(args):
    # GPU version throws over 4000 cuda side errors for 4154 samples
    device = torch.device('cuda')
    model = TapasForQuestionAnswering.from_pretrained(
        args.model_name).to(device)
    tokenizer = TapasTokenizer.from_pretrained(args.model_name)

    train_dataset = TapasDataset(args.data_dir, args.table_source, tokenizer)
    print("Table Dataset created", len(train_dataset))
    metric = DenotationAccuracy()
    logdir = f"{Path(__file__).parent}/{args.table_source}/{args.model_name.split('/')[-1]}-out/"
    os.makedirs(logdir, exist_ok=True)
    output_file = open(f"{logdir}/prediction.tsv", "w")
    count_failures = 0
    count = 0
    for raw_data, table_df, model_input in tqdm(train_dataset):
        try:
            model_input = {k: v.to(device) for k, v in model_input.items()}
            outputs = model(**model_input)
            res = tokenizer.convert_logits_to_predictions(
                {k: v.cpu() for k, v in model_input.items()},
                outputs.logits.cpu().detach(),
                outputs.logits_aggregation.cpu().detach())
            predicted_coords, predicted_aggregation_indices = res

            (agg_pred, cell_pred), answer_pred = parse_result(
                table_df,
                predicted_coords,
                predicted_aggregation_indices)
            correct = metric.update_state([raw_data.answer], [answer_pred])[0]
            log_item = {"id": raw_data.sample_id, "subset": raw_data.subset,
                        "target": raw_data.answer, "pred": answer_pred,
                        "pred_agg": agg_pred, "pred_cells": cell_pred,
                        "evaluate": correct}
            output_file.write(json.dumps(log_item) + "\n")
            count += 1
        except Exception as e:
            print(e)
            count_failures += 1
    print(f"Failed to inference on {count_failures} samples")
    smy_file = open(f"{logdir}/final.txt", "w")
    smy_file.write("=============Summary===============\n")
    smy_file.write(f"Samples: {count}\n")
    smy_file.write(f"Overall Accuracy: {metric.accuracy}\n")
    smy_file.write(f"Single Answer Accuracy: {metric.single_answer_accuracy}\n")
    smy_file.write(f"Multiple Answe Accuracy: {metric.multi_answer_accuracy}")
    smy_file.close()
    output_file.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', default='google/tapas-base-finetuned-wtq',
                      type=str,
                      help='pretrain model to load:\n '
                           'available models:    \n'
                           '# large models\n'
                           'google/tapas-large,\n'
                           'google/tapas-large-finetuned-sqa,\n'
                           'google/tapas-large-finetuned-wtq",\n'
                           'google/tapas-large-finetuned-wikisql-supervised,\n'
                           'google/tapas-large-finetuned-tabfact\n'
                           '# base models\n'
                           'google/tapas-base,\n'
                           'google/tapas-base-finetuned-sqa,\n'
                           'google/tapas-base-finetuned-wtq,\n'
                           'google/tapas-base-finetuned-wikisql-supervised,\n'
                           'google/tapas-base-finetuned-tabfact, \n'
                           '# small models\n'
                           'google/tapas-small,\n '
                           'google/tapas-small-finetuned-sqa, \n'
                           'google/tapas-small-finetuned-wtq, \n'
                           'google/tapas-small-finetuned-wikisql-supervised, \n'
                           'google/tapas-small-finetuned-tabfact, \n'
                           '# mini models\n'
                           'google/tapas-mini,\n '
                           'google/tapas-mini-finetuned-sqa,\n '
                           'google/tapas-mini-finetuned-wtq, \n'
                           'google/tapas-mini-finetuned-wikisql-supervised,\n '
                           'google/tapas-mini-finetuned-tabfact, \n'
                           '# tiny models'
                           'google/tapas-tiny, \n'
                           'google/tapas-tiny-finetuned-sqa, \n'
                           'google/tapas-tiny-finetuned-wtq, \n'
                           'google/tapas-tiny-finetuned-wikisql-supervised, \n'
                           'google/tapas-tiny-finetuned-tabfact\n')
    args.add_argument('--data_dir',
                      default='/home/shiki/hdd/TQA_Data/', type=str,
                      help='base directory of TableQA dataset')
    args.add_argument('--table_source', default='textract', 
                      type=str,
                      help='if the tables are from ground-truth (web) or from'
                           ' recognition result (textract)')
    args.add_argument('--output_file', default=None,
                      type=str, help='file to dump inference results')
    args.add_argument('--annotation_file', default=None,
                      type=str,
                      help='file to dump cell selection and'
                           ' aggregator prediction')
    inference(args.parse_args())
