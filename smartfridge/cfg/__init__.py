from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smartfridge import __version__
from smartfridge.utils import (
    DEFAULT_CFG_DICT,
    FLOAT_OR_INT,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    STR_OR_PATH,
    TESTS_RUNNING,
    YAML,
    IterableSimpleNamespace,
    colorstr,
    deprecation_warn,
)


# Define keys for arg type checks
CFG_FLOAT_KEYS = frozenset(
    {  # integer or float arguments, i.e. x=2 and x=2.0
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "degrees",
        "shear",
        "time",
        "workspace",
        "batch",
    }
)
CFG_FRACTION_KEYS = frozenset(
    {  # fractional float arguments with 0.0<=values<=1.0
        "dropout",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "conf",
        "iou",
        "fraction",
        "multi_scale",
    }
)
CFG_INT_KEYS = frozenset(
    {  # integer-only arguments
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
    }
)
CFG_BOOL_KEYS = frozenset(
    {  # boolean-only arguments
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "half",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "end2end",
    }
)


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    """Convert a configuration object to a dictionary.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted. Can be a file path, a string, a
            dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace:
    """Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from smartfridge.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = str(cfg.get("model", "")).partition(".")[0]
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg: dict, hard: bool = True) -> None:
    """Check configuration argument types and values.

    This function validates the types and values of configuration arguments, ensuring correctness and converting them if
    necessary. It checks for specific key types defined in global variables such as `CFG_FLOAT_KEYS`,
    `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`.

    Args:
        cfg (dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    """Return the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task', 'mode',
            and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name' or the
            'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        runs/detect/my_project/train
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from smartfridge.utils.files import increment_path

        project = args.project or ""
        if not Path(project).is_absolute():
            project = (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task / project
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir).resolve()  # resolve to display full path in console


def _handle_deprecation(custom: dict) -> dict:
    """Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (dict): Configuration dictionary potentially containing deprecated keys.

    Returns:
        (dict): Updated configuration dictionary with deprecated keys replaced.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom


def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None:
    """Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (dict): The base configuration dictionary containing valid keys.
        custom (dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.
        allowed_custom_keys (set | None): Optional set of additional keys that are allowed in the custom dictionary.

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    # Allow 'augmentations' as a valid custom parameter for custom Albumentations transforms
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations", "save_dir"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e