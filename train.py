#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

# 该函数为新增的 tokens 做预处理
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 在 BERT 分词表上增加新的特殊词(special tokens): EOS, PAD, BOS, UNK
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 增加特殊词后，词表的大小也要相应的调整
    model.resize_token_embeddings(len(tokenizer))

    # 判断是否又新的 token 加入，如果又,则为新的 token 分配一个合理的嵌入向量, 
    # 避免新添加的符号的嵌入向量被随机初始化，导致模型性能下降或不稳定。
    if num_new_tokens > 0:
        # 获取模型的输入和输出 embedding 层的权重矩阵，分别存储在model 的 input_embeddings和output_embeddings中。
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        # 计算原有输入和输出embedding层的平均值。
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        # 将新添加的token的嵌入向量设置为平均值，即用平均值替换权重矩阵中最后几行。
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 将字符串序列分词。
# 输入字符串序列和分词器，输出分词结果。
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # 用分词器对序列中的每一个字符串进行token 化
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    # 获取 token 的 ids
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # 获取 ids 的有效长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    # 把获取到的 ids, label, len 组合成一个字典返回
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# 调用上面的 _tokenized_fn 函数对数据进行预处理和分词
# 输入dataset中的 instruct、input、output、分词器，输出分词结果
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # 将source文本和target文本拼接成一个完整的sample，其中source文本包含datasets任务说明(instruct)和输入（input），目标文本包含输出（output）
    examples = [s + t for s, t in zip(sources, targets)]
    # 对每个sample和source文本使用_tokenize_fn函数进行分词，得到输入id、标签、长度
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    # 将标签复制一份，并将source文本部分的标签设置为忽略索引，因为模型只需要预测target文本部分的单词
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

# Step 1: 构建训练集。继承 `Dataset` 这个类
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    # 定义初始化函数，输入数据路径和分词器进行初始化
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # 从数据路径中加载数据，每个sample包含任务说明（instruction）、输入(input)和输出(output)。
        # 数据结构详见 alpaca 的 repo
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # 构建 source 文本和 target 文本，以便调用 preprocess 函数进行预处理
        # 根据输入(input)是否为空，选择不同的提示模板，将任务说明（instruction）和输入(input)格式化为源文本，将输出(output)加上结束符作为目标文本。
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        # 调用preprocess函数对source文本和target文本进行预处理和分词，得到输入id和标签，并将它们作为类的属性
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# Step 2:  data_collector。将多个 sample 整合成一个批次，并进行填充和掩码处理。
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # 定义一个可调用方法，接收一个sample的序列作为参数，并返回一个批次的张量。
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从每个数据实例中提取输入id和标签，并将它们组合成元组
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 对输入id和标签进行填充，使它们具有相同的长度，并使用分词器中的填充符和忽略索引作为填充值。
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# 1. 构建训练集；2. 如果要评估，则构建评估数据集；3. 构建 data_collector
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # 使用分词器和对话数据文本，创建一个pytorch可用的训练集实例
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # 创建一个 datacollector的实例，在 train 过程中调用
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# ------------------------ 以上是数据构造部分 -------------------------------------------------------
# ------------------------ 以下是训练代码 -----------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载预训练的模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 加载预训练模型对应的分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # 定义特殊 token
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 定义特殊 token 之后，要对词表做相应的调整
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # 构建训练集
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 开始训练：
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
