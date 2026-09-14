from domainbed.lib import reporting
from domainbed import model_selection


def load_hparams_from_dg(args, hparams):
    if args.dg_exps_dir == None: 
        raise ValueError(f"You should define a path to DG experiments to select good global hparams")
        
    # Select hparams of each methods found during DG experiments  
    if args.algorithm == "ERM":
        DG_global_hparams = ["batch_size", "lr", "wav2vec2_dropout", "weight_decay"]

    elif args.algorithm == "DANN":
        DG_global_hparams = [
            "batch_size", "lr", "wav2vec2_dropout", "weight_decay",
            "lr_g", "lr_d", "weight_decay_g", "weight_decay_d",
            "d_steps_per_g_step", "grad_penalty", "beta1",
            "mlp_width", "mlp_depth", "mlp_dropout"
        ]

    elif args.algorithm in ["MMD", "CORAL"]:
        DG_global_hparams = [
            "batch_size", "lr", "wav2vec2_dropout",
            "weight_decay", "mmd_gamma"
        ]

    else:
        raise NotImplementedError
        
    if args.selection_method_DA == "IIDAccuracy":
        selection_method = model_selection.IIDAccuracySelectionMethod
    elif args.selection_method_DA == "LeaveOneOut":
        selection_method = model_selection.LeaveOneOutSelectionMethod
    elif args.selection_method_DA == "Oracle":
        selection_method = model_selection.OracleSelectionMethod
    elif args.selection_method_DA == "IID_AutoLR_Accuracy":
        selection_method = model_selection.IIDAutoLRAccuracySelectionMethod
    else: 
        raise NotImplementedError

    # Load and group records
    records = reporting.load_records(args.dg_exps_dir)
    grouped_records = reporting.get_grouped_records(records)
    grouped_records = (
        grouped_records
        .map(lambda group: {
            **group,
            "hparams_accs": selection_method.hparams_accs(group["records"])
        })
        .filter(lambda g: g["hparams_accs"] is not None)  # remove groups with no valid runs
    )

    # keep only groups with same trial_seed
    grouped_records = grouped_records.filter(
        lambda g: g["trial_seed"] == args.trial_seed
    )

    if len(grouped_records) == 0:
        raise ValueError(
            f"No DG runs found for trial_seed={args.trial_seed}"
        )

    best_hparams_per_test_env = grouped_records.map(lambda g: {
        "trial_seed": g["trial_seed"],
        "test_env": g["test_env"],
        "val_acc": g["hparams_accs"][0][0]["val_acc"],        # validation accuracy of best run
        "hparams": g["hparams_accs"][0][1][0]["hparams"]      # hyperparameters of best run
    })

    if len(args.test_envs) > 1: 
        raise NotImplementedError # Not verified for multiple test_envs implementation
        # mono-source exps
        # all_envs = [0,1,2,3]
        # envs = [x for x in all_envs if x not in args.test_envs]
    
    # Build a dictionary of best runs per test_env
    best_hparams_dict = {
        env: max(
            best_hparams_per_test_env.filter(lambda r: r["test_env"] == env),
            key=lambda r: r["val_acc"]
        )
        for env in args.test_envs
        # for env in envs
    }

    print(best_hparams_dict)
    

    for env in args.test_envs: 
    # for env in envs: # monosource exps
        from_dg_hparams = best_hparams_dict[env]
        # Update only the keys listed in DG_global_hparams
        for h in DG_global_hparams:
            hparams[h] = from_dg_hparams["hparams"][h]

    return hparams


def load_finetuned_checkpoint(args, hparams):
    if args.finetuned_exps_dir is None:
        raise ValueError(
            "You should define a path to experiments to select the best checkpoint"
        )

    # ---------------------------------------------------------
    # Selection method
    # ---------------------------------------------------------
    if args.selection_method_DA == "IIDAccuracy":
        selection_method = model_selection.IIDAccuracySelectionMethod
    elif args.selection_method_DA == "LeaveOneOut":
        selection_method = model_selection.LeaveOneOutSelectionMethod
    elif args.selection_method_DA == "Oracle":
        selection_method = model_selection.OracleSelectionMethod
    elif args.selection_method_DA == "IID_AutoLR_Accuracy":
        selection_method = model_selection.IIDAutoLRAccuracySelectionMethod
    else:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Load records
    # ---------------------------------------------------------
    records = reporting.load_records(args.finetuned_exps_dir)
    grouped_records = reporting.get_grouped_records(records)

    # ---------------------------------------------------------
    # Compute validation accuracy
    # ---------------------------------------------------------
    grouped_records = (
        grouped_records
        .map(lambda group: {
            **group,
            "hparams_accs": selection_method.hparams_accs(
                group["records"]
            )
        })
        .filter(lambda g: g["hparams_accs"] is not None)
    )

    # ---------------------------------------------------------
    # Keep only desired trial seed
    # ---------------------------------------------------------
    grouped_records = grouped_records.filter(
        lambda g: g["trial_seed"] == args.trial_seed
    )

    # ---------------------------------------------------------
    # Keep ONLY desired test environment
    # ---------------------------------------------------------
    target_test_env = args.test_envs[0]

    grouped_records = grouped_records.filter(
        lambda g: g["test_env"] == target_test_env
    )

    if len(grouped_records) == 0:
        raise ValueError(
            f"No finetuned runs found for "
            f"trial_seed={args.trial_seed}, "
            f"test_env={target_test_env}"
        )

    # ---------------------------------------------------------
    # Get best run
    # ---------------------------------------------------------
    best_runs = grouped_records.map(lambda g: {
        "trial_seed": g["trial_seed"],
        "test_env": g["test_env"],
        "val_acc": g["hparams_accs"][0][0]["val_acc"],
        "record": g["hparams_accs"][0][1][0],
    })

    best_run = max(
        best_runs,
        key=lambda r: r["val_acc"]
    )

    best_record = best_run["record"]

    return best_record['args']['output_dir']