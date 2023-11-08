import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import (LLamaQaStoppingCriteria,
                                                       StoppingCriteriaList)

from utils import get_relative_top_filter, plot_colored_table


class DoLa:

    def __init__(self,
                 model_name: str,
                 device: str,
                 num_gpus: int,
                 max_gpu_memory: int = 27) -> None:
        self.stop_words = None
        self.stopping_criteria = None
        self.model_name: str = model_name
        self.device: str = device
        self.num_gpus: int = num_gpus
        self.max_gpu_memory: int = max_gpu_memory
        self.model, self.tokenizer = self.load_model(model_name)

    def __llm_score_baseline(self, input_ids: torch.Tensor,
                             prefix_ids: torch.Tensor,
                             continue_ids: torch.Tensor) -> tuple[float, None]:
        outputs: torch.Tensor = self.model(input_ids)[0].squeeze(
            0).log_softmax(-1)
        # skip tokens in the prompt -- we only care about the answer
        outputs = outputs[prefix_ids.shape[-1] - 1:-1, :]
        # get log probs for each token in the answer
        return outputs[range(outputs.shape[0]),
                       continue_ids].sum().item(), None

    def __llm_score_dola_static(
            self,
            input_ids: torch.Tensor,
            prefix_ids: torch.Tensor,
            continue_ids: torch.Tensor,
            mature_layer=None,
            premature_layer=None,
            relative_top: float = 0.1,
            relative_top_value: float = -1000.0,
            post_softmax: bool = True) -> tuple[float, None]:
        dict_outputs, outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            early_exit_layers=[premature_layer, mature_layer],
        )
        assert premature_layer is not None
        base_logits: torch.Tensor = dict_outputs[premature_layer][
            0, prefix_ids.shape[-1] - 1:-1, :].log_softmax(dim=-1)
        final_logits: torch.Tensor = dict_outputs[mature_layer][
            0, prefix_ids.shape[-1] - 1:-1, :].log_softmax(dim=-1)
        diff_logits: torch.Tensor = final_logits - base_logits
        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)
        if relative_top > 0.0:
            relative_top_mask: torch.Tensor = get_relative_top_filter(
                final_logits, relative_top)
            diff_logits = torch.where(relative_top_mask, relative_top_value,
                                      diff_logits)
        return diff_logits[range(diff_logits.shape[0]),
                           continue_ids].sum().item(), None

    def __llm_score_dola(self,
                         input_ids: torch.Tensor,
                         prefix_ids: torch.Tensor,
                         continue_ids: torch.Tensor,
                         mature_layer=None,
                         candidate_premature_layers=None,
                         relative_top: float = 0.1,
                         relative_top_value: float = -1000.0,
                         post_softmax: bool = True,
                         draw_jsd_table: bool = True) -> tuple[float, dict]:
        premature_layer_dist: dict = {
            it: 0
            for it in candidate_premature_layers
        }
        premature_layers: list[int] = []
        dict_outputs, outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            early_exit_layers=candidate_premature_layers + [mature_layer],
        )
        col_labels: list[str] = []
        row_labels: list[str] = [str(i)
                                 for i in candidate_premature_layers][::-1]
        jsd_list: list[np.ndarray] = []
        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
            stacked_premature_layers: torch.Tensor = torch.stack([
                dict_outputs[i][:, seq_i, :]
                for i in candidate_premature_layers
            ],
                                                                 dim=0)
            softmax_mature_layer: torch.Tensor = F.softmax(
                dict_outputs[mature_layer][:, seq_i, :], dim=-1)
            softmax_premature_layers: torch.Tensor = F.softmax(
                stacked_premature_layers, dim=-1)
            M: torch.Tensor = 0.5 * (softmax_mature_layer[None, :, :] +
                                     softmax_premature_layers)
            log_softmax_mature_layer: torch.Tensor = F.log_softmax(
                dict_outputs[mature_layer][:, seq_i, :], dim=-1)
            log_softmax_premature_layers: torch.Tensor = F.log_softmax(
                stacked_premature_layers, dim=-1)
            kl1: torch.Tensor = F.kl_div(log_softmax_mature_layer[None, :, :],
                                         M,
                                         reduction='none').mean(-1)
            kl2: torch.Tensor = F.kl_div(log_softmax_premature_layers,
                                         M,
                                         reduction='none').mean(-1)
            js_divs: torch.Tensor = 0.5 * (kl1 + kl2).mean(-1) * 1e5
            premature_layer: int = candidate_premature_layers[int(
                js_divs.argmax().cpu().item())]
            premature_layer_dist[premature_layer] += 1
            premature_layers.append(premature_layer)
            col_labels.append(self.tokenizer.convert_ids_to_tokens(seq_i))
            jsd_list.append(js_divs.cpu().numpy())
        if draw_jsd_table:
            plot_colored_table(
                np.vstack(jsd_list).T[::-1, :], row_labels, col_labels)
        base_logits: torch.Tensor = torch.zeros_like(
            dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
        for i, layer in enumerate(premature_layers):
            base_logits[i] = dict_outputs[layer][0,
                                                 prefix_ids.shape[-1] - 1 + i]
        final_logits: torch.Tensor = F.log_softmax(
            dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1], dim=-1)
        # noinspection DuplicatedCode
        base_logits = base_logits.log_softmax(dim=-1)
        diff_logits: torch.Tensor = final_logits - base_logits
        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)
        if relative_top > 0.0:
            relative_top_mask: torch.Tensor = get_relative_top_filter(
                final_logits, relative_top)
            diff_logits = torch.where(relative_top_mask, relative_top_value,
                                      diff_logits)
        log_probs: float = diff_logits[range(diff_logits.shape[0]),
                                       continue_ids].sum().item()
        return log_probs, premature_layer_dist

    def __llm_score_dola_adj_layer_jsd(
            self,
            input_ids: torch.Tensor,
            prefix_ids: torch.Tensor,
            continue_ids: torch.Tensor,
            mature_layer=None,
            candidate_premature_layers=None,
            relative_top: float = 0.1,
            relative_top_value: float = -1000.0,
            post_softmax: bool = True,
            draw_adj_layer_jsd: bool = False) -> tuple[float, dict]:
        early_exit_layers: list[int] = candidate_premature_layers + [
            mature_layer
        ]
        dict_outputs, outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            early_exit_layers=early_exit_layers,
        )
        premature_layer_dist: dict = {
            it: 0
            for it in candidate_premature_layers
        }
        premature_layers: list[int] = []
        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
            adj_layer_jsd_list: list[torch.Tensor] = []
            for i in range(len(early_exit_layers) - 1):
                softmax_p: torch.Tensor = F.softmax(
                    dict_outputs[early_exit_layers[i]][:, seq_i, :], dim=-1)
                softmax_q: torch.Tensor = F.softmax(
                    dict_outputs[early_exit_layers[i + 1]][:, seq_i, :],
                    dim=-1)
                m: torch.Tensor = 0.5 * (softmax_p + softmax_q)
                log_softmax_p: torch.Tensor = F.log_softmax(
                    dict_outputs[early_exit_layers[i]][:, seq_i, :], dim=-1)
                log_softmax_q: torch.Tensor = F.log_softmax(
                    dict_outputs[early_exit_layers[i + 1]][:, seq_i, :],
                    dim=-1)
                kl_p_m: torch.Tensor = F.kl_div(log_softmax_p[None, :, :],
                                                m,
                                                reduction='none').mean(-1)
                kl_q_m: torch.Tensor = F.kl_div(log_softmax_q[None, :, :],
                                                m,
                                                reduction='none').mean(-1)
                adj_layer_jsd: torch.Tensor = 0.5 * (kl_p_m + kl_q_m).mean(-1)
                adj_layer_jsd_list.append(adj_layer_jsd)
            adj_layer_jsd_stack: torch.Tensor = torch.stack(adj_layer_jsd_list,
                                                            dim=0)
            premature_layer: int = early_exit_layers[int(
                adj_layer_jsd_stack.argmax().cpu().item())]
            premature_layers.append(premature_layer)
        base_logits: torch.Tensor = torch.zeros_like(
            dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
        for i, layer in enumerate(premature_layers):
            base_logits[i] = dict_outputs[layer][0,
                                                 prefix_ids.shape[-1] - 1 + i]
        final_logits: torch.Tensor = dict_outputs[mature_layer][
            0, prefix_ids.shape[-1] - 1:-1].log_softmax(dim=-1)
        # noinspection DuplicatedCode
        base_logits = base_logits.log_softmax(dim=-1)
        diff_logits: torch.Tensor = final_logits - base_logits
        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)
        if relative_top > 0.0:
            relative_top_mask: torch.Tensor = get_relative_top_filter(
                final_logits, relative_top)
            diff_logits = torch.where(relative_top_mask, relative_top_value,
                                      diff_logits)
        log_probs: float = diff_logits[range(diff_logits.shape[0]),
                                       continue_ids].sum().item()
        return log_probs, premature_layer_dist

    def load_model(
            self,
            model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.device == "cuda":
            kwargs: dict = {
                "torch_dtype": torch.float16,
                "offload_folder": f"{model_name}/offload"
            }
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {
                            i: f"{self.max_gpu_memory}GiB"
                            for i in range(self.num_gpus)
                        },
                    })
        elif self.device == "cpu":
            kwargs: dict = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs)
        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        return model, tokenizer

    def set_stop_words(self, stop_words: list[str]) -> None:
        self.stop_words: list[str] = stop_words
        self.stopping_criteria: StoppingCriteriaList = StoppingCriteriaList()
        list_stop_word_ids: list[list[int]] = []
        for stop_word in self.stop_words:
            stop_word_ids: list[int] = self.tokenizer.encode('\n' +
                                                             stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ",
                  stop_word,
                  'with the ids',
                  stop_word_ids,
                  flush=True)
        self.stopping_criteria.append(
            LLamaQaStoppingCriteria(list_stop_word_ids))

    @torch.no_grad()
    def generate(self,
                 input_text: str,
                 max_new_tokens: int = 256,
                 top_p: float = 0.95,
                 top_k: int = 0,
                 temperature: float = 0.8,
                 mature_layer=None,
                 premature_layer=None,
                 candidate_premature_layers=None,
                 mode: str = 'baseline',
                 verbose: bool = True,
                 remove_stop_words: bool = False,
                 relative_top: float = 0.1,
                 **kwargs) -> tuple[str, dict | None]:
        if candidate_premature_layers is None:
            candidate_premature_layers: list[int] = []
        input_ids: torch.Tensor = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to(self.device)
        max_len: int = input_ids.shape[-1] + max_new_tokens
        if mode == 'baseline':
            outputs: torch.Tensor = self.model.generate(
                input_ids,
                max_length=max_len,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                dola_decoding=False,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                stopping_criteria=self.stopping_criteria,
                **kwargs)
        elif mode == 'dola-static':
            assert mature_layer is not None, "mature_layer must be specified"
            assert premature_layer is not None, "premature_layer must be specified"
            outputs = self.model.generate(
                input_ids,
                max_length=max_len,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                dola_decoding=True,
                mature_layer=mature_layer,
                premature_layer=premature_layer,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                stopping_criteria=self.stopping_criteria,
                relative_top=relative_top,
                **kwargs)
        elif mode == 'dola':
            assert mature_layer is not None, "mature_layer must be specified"
            assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
            outputs = self.model.generate(
                input_ids,
                max_length=max_len,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                dola_decoding=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                stopping_criteria=self.stopping_criteria,
                relative_top=relative_top,
                mature_layer=mature_layer,
                premature_layer=None,
                candidate_premature_layers=candidate_premature_layers,
                **kwargs,
            )
            premature_layer_dist = outputs.premature_layer_dist
        # skip the tokens in the input prompt
        gen_sequences = outputs.sequences[:, input_ids.shape[-1]:][0, :]
        output_str: str = self.tokenizer.decode(gen_sequences,
                                                skip_special_tokens=True)
        if verbose:
            print('MODEL OUTPUT: \n{0}'.format(output_str))
        if remove_stop_words:
            for stop_word in self.stop_words:
                length_to_remove: int = len(stop_word)
                if output_str[-length_to_remove:] == stop_word:
                    output_str = output_str[:-length_to_remove]
            output_str = output_str.strip()
        if self.device:
            torch.cuda.empty_cache()
        return output_str, (premature_layer_dist if mode == 'dola' else None)

    @torch.no_grad()
    def lm_score(
            self,
            input_text1: str,
            input_text2: str,
            mature_layer=None,
            premature_layer=None,
            candidate_premature_layers=None,
            mode: str = 'baseline',
            relative_top: float = 0.1,
            relative_top_value: float = -1000.0,
            post_softmax: bool = True,
            use_adj_layer_jsd: bool = False,
            draw_adj_layer_jsd: bool = False) -> tuple[float, None | dict]:
        input_text: str = input_text1 + input_text2
        input_ids: torch.Tensor = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to(self.device)
        prefix_ids: torch.Tensor = self.tokenizer(
            input_text1, return_tensors="pt").input_ids.to(self.device)
        continue_ids: torch.Tensor = input_ids[0, prefix_ids.shape[-1]:]
        if mode == 'baseline':
            return self.__llm_score_baseline(input_ids, prefix_ids,
                                             continue_ids)
        elif mode == 'dola-static':
            return self.__llm_score_dola_static(input_ids, prefix_ids,
                                                continue_ids, mature_layer,
                                                premature_layer, relative_top,
                                                relative_top_value,
                                                post_softmax)
        elif mode == 'dola':
            return self.__llm_score_dola_adj_layer_jsd(
                input_ids, prefix_ids, continue_ids, mature_layer,
                candidate_premature_layers, relative_top, relative_top_value,
                post_softmax, draw_adj_layer_jsd
            ) if use_adj_layer_jsd else self.__llm_score_dola(
                input_ids, prefix_ids, continue_ids, mature_layer,
                candidate_premature_layers, relative_top, relative_top_value,
                post_softmax)
