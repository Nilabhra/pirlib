import argparse

import pytorch_lightning as pl
from train_utils import LoggingCallback, T5FineTuner

from pirlib.iotypes import DirectoryPath
from pirlib.pipeline import pipeline
from pirlib.task import task


@task
def t5_tune(data: DirectoryPath) -> DirectoryPath:
    args_dict = dict(
        data_dir="",  # path for data files
        output_dir="",  # path to save the checkpoints
        model_name_or_path="t5-base",
        tokenizer_name_or_path="t5-base",
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=8,
        num_train_epochs=2,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args_dict.update(
        {"data_dir": "aclImdb", "output_dir": "t5_imdb_sentiment", "num_train_epochs": 2}
    )
    args = argparse.Namespace(**args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    output_dir = task.context().output
    model.model.save_pretrained(f"{output_dir}/t5_base_imdb_sentiment")


def t5_tuning_pipeline(tuning_dataset: DirectoryPath) -> DirectoryPath:
    model_dir = t5_tune(tuning_dataset)
    return model_dir
