from run_qg import run_qg

args_dict = {
    "model_name_or_path": "facebook/bart-base",
    "model_type": "bart",
    "tokenizer_name_or_path": "bart_tokenizer",
    "output_dir": "bart_model",
    "train_file_path": "data/train_data_bart.pt", #####
    "valid_file_path": "data/valid_data_bart.pt", #####
    "per_device_train_batch_size": 4, #------
    "per_device_eval_batch_size": 4, #-------
    "gradient_accumulation_steps": 1, #------
    "learning_rate": 1e-4, #------
    "num_train_epochs": 10, #------
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}

# start training
run_qg(args_dict)
