import json
import os
from dataclasses import dataclass

import transformers
from transformers import Trainer

from dataset import MultiModalDataset, DataCollatorForMultiModalDataset
from model import LVLMConfig, LVLMForConditionalGeneration


@dataclass
class ModelArguments:
    cache_dir_hf: str = None

    llm_name_or_path: str = None
    llm_max_length: int = None
    llm_padding_side: str = "right"
    llm_attn_implementation: str = "eager"

    encoder_image_name_or_path: str = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    data_type: str = None
    data_path: str = None
    image_dir: str = ""
    image3d_dir: str = ""

    tune_type_llm: str = "frozen"
    tune_type_encoder_image: str = "frozen"
    tune_type_connector_image: str = "frozen"
    tune_type_encoder_image3d: str = "frozen"
    tune_type_connector_image3d: str = "frozen"


def tune_type_setting(training_arguments, model):
    if training_arguments.tune_type_llm == "full":
        model.llm.requires_grad_(True)

    if model.encoder_image is not None:
        if training_arguments.tune_type_encoder_image == "full":
            model.encoder_image.requires_grad_(True)

        if training_arguments.tune_type_connector_image == "full":
            for p in model.connector_image.parameters():
                p.requires_grad = True

    if model.encoder_image3d is not None:
        if training_arguments.tune_type_encoder_image3d == "full":
            model.encoder_image3d.requires_grad_(True)

        if training_arguments.tune_type_connector_image3d == "full":
            for p in model.connector_image3d.parameters():
                p.requires_grad = True

    return model


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_arguments, training_arguments, = parser.parse_args_into_dataclasses()

    if training_arguments.resume_from_checkpoint is None:
        config = LVLMConfig(
            cache_dir_hf=model_arguments.cache_dir_hf,
            llm_name_or_path=model_arguments.llm_name_or_path,
            llm_max_length=model_arguments.llm_max_length,
            llm_padding_side=model_arguments.llm_padding_side,
            llm_attn_implementation=model_arguments.llm_attn_implementation,
            encoder_image_name_or_path=model_arguments.encoder_image_name_or_path,
        )
        model = LVLMForConditionalGeneration(config)
    else:
        model = LVLMForConditionalGeneration.from_pretrained(training_arguments.resume_from_checkpoint)
    model = tune_type_setting(training_arguments, model)

    with open(training_arguments.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = MultiModalDataset(
        args=training_arguments,
        mode="train",
        data=data,
        tokenizer=model.tokenizer,
        processor_image=model.processor_image,
        processor_image3d=model.processor_image3d,
    )
    data_collator = DataCollatorForMultiModalDataset(
        mode="train",
        tokenizer=model.tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # save model: trainer会先聚合不同GPU上的模型，再调用model.save_pretrained保存
    trainer.save_model(training_arguments.output_dir)

    # save tokenizer: output_dir/tokenizer/ 只在主进程中进行
    if training_arguments.local_rank == 0:
        os.makedirs(os.path.join(training_arguments.output_dir, "tokenizer"), exist_ok=True)
        model.tokenizer.save_pretrained(os.path.join(training_arguments.output_dir, "tokenizer"))
