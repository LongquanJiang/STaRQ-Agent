# -*- coding: utf-8 -*-
from core.utils import *
from core.chat_manager import ChatManager
from core.const import SYSTEM_NAME
from tqdm import tqdm
import time
import argparse
import sys
import os
import json
import traceback


def init_message(idx: int, item: dict) -> dict:
    kg_id, question, gt, entities = item['kb'], item['question'], item['query'], item["entities"]
    user_message = {
        "idx": idx,
        "kg_id": kg_id,
        "question": question,
        "extracted_schema": {},
        "entities": entities,
        "ground_truth": gt,
        "send_to": SYSTEM_NAME
    }
    return user_message


def run_batch(kg_id, input_file, output_file, ontology_json_path, start_pos=0, log_file=None,
              dataset_mode='dev', use_gold_schema=False, without_selector=False):
    chat_manager = ChatManager(ontology_json_path=ontology_json_path,
                               log_path=log_file,
                               kg_id=kg_id,
                               model_name='gpt-4',
                               lazy=True,
                               without_selector=without_selector)
    # load dataset
    batch = load_json_file(input_file)
    # resume from last checkpoint
    finished_ids = set()
    if os.path.exists(output_file):
        output_data_lst = load_jsonl_file(output_file)
        for o in output_data_lst:
            finished_ids.add(o['idx'])
    unfinished_ids = [item["idx"] for item in batch if item["idx"] not in finished_ids]
    print(f"len(unfinished_data) = {len(unfinished_ids)}")

    # add question_id if needed
    for k, item in enumerate(batch):
        if 'idx' not in item:
            item['idx'] = k

    new_batch = []
    for k, item in enumerate(batch):
        q_id = item['idx']
        if q_id not in unfinished_ids:
            continue
        new_batch.append(item)

    time.sleep(2)
    batch = new_batch

    # generate SQL one by one, and save result one by one
    with open(output_file, 'a+', encoding='utf-8') as fp:
        total_num = len(batch)
        for cur_idx, item in tqdm(enumerate(batch), total=total_num):
            idx = item['idx']
            kg_id = item['kb']
            if idx not in unfinished_ids: continue
            user_message = init_message(idx, item)  # imitate user send a question to system
            try:
                chat_manager.start(user_message)
                try:
                    del user_message['desc_str']
                    del user_message['send_to']
                    del user_message['chosen_db_schem_dict']["entities"]
                except:
                    pass
                print(json.dumps(user_message, ensure_ascii=False), file=fp, flush=True)
            except Exception as e:
                # for debug
                traceback.print_exc()
                print(f"Exception: {e}, sleep 20 seconds.", flush=True)
                time.sleep(20)
                # raise Exception(str(e))
            print(f"\n\ndeal {cur_idx + 1}/{total_num} done!\n\n")
        print(f"Result dump into {output_file}", file=sys.stdout, flush=True)

    # export evaluation results
    out_dir = os.path.dirname(output_file)

    evaluation_file_path = f"{out_dir}/pred_{dataset_mode}.txt"
    sparql_lst = []
    output_json_lst = load_jsonl_file(output_file)
    for output_json in output_json_lst:
        pred_sparql = output_json['pred']
        pred_sparql = replace_multiple_spaces(pred_sparql)
        sparql_lst.append(pred_sparql.strip() + '\n')
    save_file(evaluation_file_path, sparql_lst)


def check_all_paths(args):
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} not found")
    if not os.path.exists(args.ontology_json_path):
        raise FileNotFoundError(f"Ontology json path {args.tables_json_path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_id', type=str, help='kg identifier')
    parser.add_argument('--dataset_mode', type=str, default='dev', choices=['train', 'dev', 'test'],
                        help='dataset mode')
    parser.add_argument('--input_file', type=str, required=True, help='path to dataset input') ## 问题集
    parser.add_argument('--ontology_json_path', type=str, default=None, help='path to ontology.json')
    parser.add_argument('--output_file', type=str, required=True, help='path to predicted output')
    parser.add_argument('--log_file', type=str, default='', help='path to log file if needed')
    parser.add_argument('--start_pos', type=int, default=0, help='start position of a batch')
    parser.add_argument('--use_gold_schema', action='store_true', default=False)
    parser.add_argument('--without_selector', action='store_true', default=False)
    parser.add_argument('--without_refiner', action='store_true', default=False)
    args = parser.parse_args()
    # 打印args中的键值对
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    check_all_paths(args)

    # pretty print args json
    args_json_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
    print(f"args:\n{args_json_str}")
    time.sleep(3)

    run_batch(
        kg_id=args.kg_id,
        dataset_mode=args.dataset_mode,
        input_file=args.input_file,
        output_file=args.output_file,
        ontology_json_path=args.ontology_json_path,
        log_file=args.log_file,
        start_pos=args.start_pos,
        use_gold_schema=args.use_gold_schema,
        without_selector=args.without_selector
    )