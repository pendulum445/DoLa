# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
import argparse
import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from dola import DoLa

transformers.logging.set_verbosity(40)

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"


def load_csv(file_path: str):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    """
    Data format:

    ,full_prefix,doc_id,completion,contradiction_0,contradiction_1,contradiction_2,longest_completions,turncated_prefixes
    0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. ",0,Whether or not it gets a second season of The Witcher is another question.,Whether or not it gets a second season of Stranger Things is another question.,Whether or not it gets a fifth season of The Witcher is another question.,Whether or not it gets a second season of Black Mirror is another question.,15.0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. "

    """
    list_data_dict: list[dict] = []
    df = pd.read_csv(file_path)
    if 'news' in file_path:
        prefix_type: str = 'full_prefix'
    else:
        prefix_type: str = 'turncated_prefixes'
    for idx in range(len(df)):
        item: dict = dict(
            prefix=df[prefix_type][idx],
            completion=df['completion'][idx],
            contradiction_0=df['contradiction_0'][idx],
            contradiction_1=df['contradiction_1'][idx],
            contradiction_2=df['contradiction_2'][idx],
        )
        list_data_dict.append(item)
    return list_data_dict


def get_parser_args() -> argparse.Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--model-name",
                        type=str,
                        default="/data/lyj/hf_models/bloom-560m")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device",
                        type=str,
                        choices=["cuda", "cpu"],
                        default="cpu")
    parser.add_argument("--data-path", type=str, default="./wiki_factor.csv")
    parser.add_argument("--output-path",
                        type=str,
                        default="./wiki_result.json")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers",
                        type=str,
                        default="0,2,4,6,8,10,12,14,16,18,20,22,24")
    # noinspection DuplicatedCode
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    # noinspection DuplicatedCode
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--adj_layer_jsd", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    return parser.parse_args()


def get_substring(s: str) -> str:
    last_slash_index: int = s.rindex('/')
    return s[last_slash_index + 1:]


if __name__ == "__main__":
    args: argparse.Namespace = get_parser_args()
    model_name: str = args.model_name
    num_gpus: int = args.num_gpus
    device: str = args.device
    # Get test file
    fp: str = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")
    list_data_dict: list[dict] = load_csv(fp)
    # noinspection DuplicatedCode
    if args.parallel:
        chunk_size: int = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id *
                                        chunk_size:(args.shard_id + 1) *
                                        chunk_size]
    if args.debug:
        list_data_dict = list_data_dict[:10]
    llm: DoLa = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers: list[int] = [
        int(x) for x in args.early_exit_layers.split(',')
    ]
    # noinspection DuplicatedCode
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode: str = "baseline"
        mature_layer: int | None = None
        premature_layer: int | None = None
        candidate_premature_layers: list[int] | None = None
    elif len(early_exit_layers) == 2:
        print(
            f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}"
        )
        mode: str = "dola-static"
        mature_layer: int | None = early_exit_layers[1]
        premature_layer: int | None = early_exit_layers[0]
        candidate_premature_layers: list[int] | None = None
    else:
        # noinspection DuplicatedCode
        print(
            f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}"
        )
        mode: str = "dola"
        mature_layer: int | None = early_exit_layers[-1]
        premature_layer: int | None = None
        candidate_premature_layers: list[int] | None = early_exit_layers[:-1]
        premature_layer_dist: dict = {
            it: 0
            for it in candidate_premature_layers
        }
    answers: list[bool] = []
    result_dict: dict = {
        'is_correct': [],
        'model_answer': [],
        'model_completion': [],
        'full_input_text': []
    }
    for sample in tqdm(list_data_dict):
        context: str = sample['prefix']
        answer_true: str = ' ' + sample['completion']
        answers_false: list[str] = []
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        generate_kwargs: dict = dict(
            mode=mode,
            mature_layer=mature_layer,
            premature_layer=premature_layer,
            candidate_premature_layers=candidate_premature_layers,
            relative_top=args.relative_top,
            relative_top_value=args.relative_top_value,
            use_adj_layer_jsd=args.adj_layer_jsd)
        answer_true_log_prob, c_dist = llm.lm_score(context, answer_true,
                                                    **generate_kwargs)
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        answer_false_log_probs: list[float] = []
        for answer_false in answers_false:
            answer_false_log_prob, c_dist = llm.lm_score(
                context, answer_false, **generate_kwargs)
            if mode == "dola":
                for k, v in c_dist.items():
                    premature_layer_dist[k] += v
            answer_false_log_probs.append(answer_false_log_prob)
        if args.debug:
            print(f'log prob of answers: {answer_true_log_prob}', end=' ')
            for answer_false_log_prob in answer_false_log_probs:
                print(f'{answer_false_log_prob}', end=' ')
            print()
        is_cor: bool = True
        for answer_false_log_prob in answer_false_log_probs:
            if answer_true_log_prob < answer_false_log_prob:
                is_cor = False
                break
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] +
                                               answer_false_log_probs)
    print(f'model name: {model_name}')
    print(f'dataset: {get_substring(args.data_path)}')
    print(f'mode: {mode}')
    print(f'early exit layers: {early_exit_layers if mode == "dola" else "-"}')
    print(f'result: Num of total question: {len(answers)}, '
          f'correct num: {sum(answers)}, '
          f'correct rate: {float(sum(answers)) / len(answers)}.')
    # noinspection DuplicatedCode
    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for it in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(
                    it, premature_layer_dist[it],
                    round(premature_layer_dist[it] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split(
        '/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (
        args.output_path + "_" + str(args.shard_id) + ".json")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
