from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers import DDIMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from torch.optim.adam import Adam
from PIL import Image


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        # 查询当前类型，self QKV都来自同一个输入，即当前时间步的隐藏状态 hidden_states。
        # 在交叉注意力中，查询（query）来自当前时间步的隐藏状态 hidden_states，而键（key）
        # 和值（value）来自编码器的隐藏状态 encoder_hidden_states。这通常用于编码器-解码器结构中，例如 Transformer 模型中的解码器部分
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change: 调用控制器调整注意力概率
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def create_controller(
    prompts: List[str], cross_attention_kwargs: Dict, num_inference_steps: int, tokenizer, device, attn_res
) -> AttentionControl:
    edit_type = cross_attention_kwargs.get("edit_type", None)
    local_blend_words = cross_attention_kwargs.get("local_blend_words", None)
    equalizer_words = cross_attention_kwargs.get("equalizer_words", None)
    equalizer_strengths = cross_attention_kwargs.get("equalizer_strengths", None)
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)
    inds_replace = cross_attention_kwargs.get("inds_replace", None)
    threshold = cross_attention_kwargs.get("threshold", 0.3)

    if edit_type == "keep":
        if local_blend_words is None:
            return AttentionKeep(
            prompts, 
            num_inference_steps, 
            n_cross_replace, 
            n_self_replace, 
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            inds_replace=inds_replace
        ) 
        else:
            lb = LocalBlend(prompts, 
                        local_blend_words, 
                        tokenizer=tokenizer, 
                        device=device, 
                        attn_res=attn_res,
                        threshold=threshold)
            return AttentionReplace(
            prompts, 
            num_inference_steps, 
            n_cross_replace, 
            n_self_replace, 
            lb, 
            tokenizer=tokenizer, 
            device=device, 
            attn_res=attn_res,
            inds_replace=inds_replace
        )
    
    # only replace
    if edit_type == "replace" and local_blend_words is None:
        return AttentionReplace(
            prompts, 
            num_inference_steps, 
            n_cross_replace, 
            n_self_replace, 
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            inds_replace=inds_replace
        )

    # replace + localblend
    if edit_type == "replace" and local_blend_words is not None:
        lb = LocalBlend(prompts, 
                        local_blend_words, 
                        tokenizer=tokenizer, 
                        device=device, 
                        attn_res=attn_res,
                        threshold=threshold)
        return AttentionReplace(
            prompts, 
            num_inference_steps, 
            n_cross_replace, 
            n_self_replace, 
            lb, 
            tokenizer=tokenizer, 
            device=device, 
            attn_res=attn_res,
            inds_replace=inds_replace
        )

    # only refine
    if edit_type == "refine" and local_blend_words is None:
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # refine + localblend
    if edit_type == "refine" and local_blend_words is not None:
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, lb, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # only reweight
    if edit_type == "reweight" and local_blend_words is None:
        assert (
            equalizer_words is not None and equalizer_strengths is not None
        ), "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(
            equalizer_strengths
        ), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
        )

    # reweight and localblend
    if edit_type == "reweight" and local_blend_words:
        assert (
            equalizer_words is not None and equalizer_strengths is not None
        ), "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(
            equalizer_strengths
        ), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
            local_blend=lb,
        )

    if edit_type == "revision":
        return AttentionStore(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
            local_blend=lb,
        )
    
    raise ValueError(f"Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.")


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, attn_res=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.attn_res = attn_res


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=None):
        super(AttentionStore, self).__init__(attn_res)
        self.step_store = self.get_empty_store()
        self.attention_store = {}



class LocalBlend:
    def __call__(self, x_t, attention_store):
        # note that this code works on the latent level!
        k = 1
        # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]  # These are the numbers because we want to take layers that are 256 x 256, I think this can be changed to something smarter...like, get all attentions where thesecond dim is self.attn_res[0] * self.attn_res[1] in up and down cross.
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] +  attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, self.attn_res[0], self.attn_res[1], self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1) # since alpha_layers is all 0s except where we edit, the product zeroes out all but what we change. Then, the sum adds the values of the original and what we edit. Then, we average across dim=1, which is the number of layers.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)

        mask = mask[:1] + mask[1:]
        mask = mask.to(torch.float16)

        x_t = x_t[:1] + mask * (x_t - x_t[:1]) # x_t[:1] is the original image. mask*(x_t - x_t[:1]) zeroes out the original image and removes the difference between the original and each image we are generating (mostly just one). Then, it applies the mask on the image. That is, it's only keeping the cells we want to generate.
        return x_t

    def __init__(
        self, prompts: List[str], words: [List[List[str]]], tokenizer, device, threshold=0.3, attn_res=None
    ):
        self.max_num_words = 77
        self.attn_res = attn_res

        # 创建 alpha_layers 张量，用于存储每个提示词中需要进行局部混合的词的位置
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        # 遍历 prompts 和 local_blend_words，
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                # 为每个提示词中的每个词找到其在编码后的位置
                ind = get_word_inds(prompt, word, tokenizer)
                print(f'LocalBlend words {word}, ind = {ind}')
                # 在 alpha_layers 中相应的位置设置为 1
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device) # a one-hot vector where the 1s are the words we modify (source and target)
        self.threshold = threshold
        print(f'threshod:{threshold}')


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= self.attn_res[0]**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            att_replace.shape[2]
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    

    #####################################################################################
    ################################### 执行注意力注入 ###################################
    #####################################################################################

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)

        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            
            # 从batch_size拆分注意力
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]  
            
            if is_cross:
                # 查看在这个时间步需要具体怎么操作
                # 在初始化时通过从 get_time_words_attention_alpha 计算了self.cross_replace_alpha
                # 用于储存所有时间步的独热编码
                alpha_words = self.cross_replace_alpha[self.cur_step]  
                
                # 根据type进行选择替换方式
                # 根据alpha_words选择替换的word位置
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words
                    + (1 - alpha_words) * attn_replace
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)  # 替换自注意力
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device,
        attn_res=None,
    ):
        super(AttentionControlEdit, self).__init__(attn_res=attn_res)
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)

        # 生成跨注意力替换的 one_hot 矩阵
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, self.tokenizer
        ).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        # 自注意步数范围,eg:self_replace_steps 0,0.4 → num_self_replace 0,20
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionKeep(AttentionControlEdit):
    # 通过将 attn_base 和 self.mapper 相乘来替换交叉注意力
    def replace_cross_attention(self, attn_base, att_replace):
        # hpw 表示 attn_base 的维度。(heads, patches, width)
        # bwn 表示 self.mapper 的维度。
        # bhpn 表示输出的维度。
        # 对齐w维度进行求和
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        inds_replace:List=None
    ):
        super(AttentionKeep, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.mapper = get_replacement_mapper(prompts, self.tokenizer, inds_replace).to(self.device)


class AttentionReplace(AttentionControlEdit):
    # 通过将 attn_base 和 self.mapper 相乘来替换交叉注意力
    def replace_cross_attention(self, attn_base, att_replace):
        # hpw 表示 attn_base 的维度。(heads, patches, width)
        # bwn 表示 self.mapper 的维度。
        # bhpn 表示输出的维度。
        # 这个操作使用爱因斯坦求和约定，将两个张量相乘并输出指定的维度
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        inds_replace:List=None
    ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.mapper = get_replacement_mapper(prompts, self.tokenizer, inds_replace).to(self.device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # 使用 self.mapper 重新排列 attn_base 的维度
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        # 计算新的 attn_replace，将 attn_base_replace 和 att_replace 按比例 self.alphas 混合
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None
    ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        # 计算mapper和混合比例alphas
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # 如果 self.prev_controller 不为空，则调用其 replace_cross_attention 方法处理 attn_base
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        # 使用 self.equalizer 对 attn_base 进行重新加权
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
    ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller


### util functions for all Edits

# 根据提供的时间步范围和词语索引更新注意力权重
def update_alpha_time_word(
    alpha, 
    bounds: Union[float, Tuple[float, float]], 
    prompt_ind: int, 
    word_inds: Optional[torch.Tensor] = None
):
    # 如果 bounds 是浮点数，将其转换为元组 (0, bounds)
    if isinstance(bounds, float):
        bounds = 0, bounds
    
    # 计算开始和结束的时间步索引
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])

    # 如果没有提供词语索引，则使用所有词语的索引
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])

    # 在开始之前和结束之后的时间步将 alpha 设置为0
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


### 生成一个包含所有时间步和提示的注意力权重张量
# cross_replace_steps["default_"] = 1.0 表示在整个生成过程中，对所有提示词都应用完全的注意力替换。这意味着在生成每个时间步的图像时，
# 模型会始终使用prompt[0]的注意力权重
# cross_replace_steps["specific_word"] = 0.4，表示在生成过程中，只在前 40% 的时间步内应用注意力替换。这意味着在生成早期，
# 模型会使用prompt[0]的注意力权重，而在后期则逐渐减少对这些词语的替换，转而使用prompt[1]的注意力权重。
def get_time_words_attention_alpha(
    prompts, num_steps, cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]], tokenizer, max_num_words=77
):
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)

    # 创建一个全零张量 alpha_time_words [时间步, prompt数, 编码数77] 存放权重
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)

    # 第1次更新，使用cross_replace_steps["default_"]为所有提示词设置一个默认的注意力替换范围
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            # 获取cross_replace_steps[key]在每个提示中指定词语的索引
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            # 第2次更新，对制定的word设置cross_replace_steps[word]的注意力替换范围
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words




### util functions for LocalBlend and ReplacementEdit
# 该函数用于获取指定词语在words_encode中的索引位置
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.replace(",", "").split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


### util functions for ReplacementEdit
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77, inds_replace:List=None):
    words_x = x.split(" ")
    words_y = y.split(" ")
    if len(words_x) != len(words_y):
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length"
            f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    if inds_replace is None:
        inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    print(f'inds_replace={inds_replace}')
    
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    # return torch.from_numpy(mapper).float()
    return torch.from_numpy(mapper).to(torch.float16)


def get_replacement_mapper(prompts, tokenizer, inds_replace=None, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len, inds_replace)
        mappers.append(mapper)
    return torch.stack(mappers)


### util functions for ReweightEdit
def get_equalizer(
    text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer
):
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = torch.FloatTensor(values[i])
    return equalizer


### util functions for RefinementEdit
class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap  # 间隙得分
        self.match = match  # 匹配得分
        self.mismatch = mismatch  # 错配得分

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch  # 返回错配得分
        else:
            return self.match  # 返回匹配得分


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap  # 初始化第一行
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap  # 初始化第一列
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1  # 初始化第一行
    matrix[1:, 0] = 2  # 初始化第一列
    matrix[0, 0] = 4  # 设置起始点
    return matrix


# 执行全局比对，生成得分矩阵和回溯矩阵
def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)  # 初始化得分矩阵
    trace_back = get_traceback_matrix(len(x), len(y))  # 初始化回溯矩阵
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap  # 左方得分
            up = matrix[i - 1, j] + score.gap  # 上方得分
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])  # 对角线得分
            matrix[i, j] = max(left, up, diag)  # 选择最大得分
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


# 根据回溯矩阵生成对齐的序列和映射
def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    # 将输入字符串x和y编码为token序列
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)

    # 定义比对得分参数：匹配得分为1，错配得分为-1，间隙得分为-1
    score = ScoreParams(0, 1, -1)

    # 进行全局比对，生成比对矩阵和回溯矩阵
    matrix, trace_back = global_align(x_seq, y_seq, score)

    # 根据回溯矩阵生成比对后的序列映射
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]

    # 初始化权重alphas，默认为1
    alphas = torch.ones(max_len)

    # 根据映射结果调整权重，如果比对结果中存在错配，权重设为 0
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()

    # 初始化映射矩阵mapper，默认为0
    mapper = torch.zeros(max_len, dtype=torch.int64)

    # 根据比对结果生成映射矩阵，将映射关系填入 mapper 中
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    
    # 对于 mapper 剩余的部分，填充为序列 y 的长度加上递增索引
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))

    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    # 获取第一个提示序列
    x_seq = prompts[0]

    # 初始化映射矩阵和权重矩阵列表
    mappers, alphas = [], []

    # 遍历其余的提示序列
    for i in range(1, len(prompts)):
        # 生成当前提示序列和第一个提示序列的映射矩阵和权重矩阵
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)

    # 返回堆叠后的映射矩阵和权重矩阵
    return torch.stack(mappers), torch.stack(alphas)