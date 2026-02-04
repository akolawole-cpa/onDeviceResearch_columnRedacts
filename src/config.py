"""
Configuration Loader Module

Handles loading YAML configuration and resolving paths with run_id.
This allows multiple users to run the pipeline simultaneously without conflicts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. Defaults to configs/data_paths.yaml

    Returns
    -------
    dict
        Raw configuration dictionary
    """
    if config_path is None:
        # Default to configs/data_paths.yaml relative to project root
        config_path = Path(__file__).parent.parent / "configs" / "data_paths.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_output_path(
    config: Dict[str, Any],
    key: str,
    run_id: Optional[str] = None,
) -> str:
    """
    Get an output file path with run_id substituted.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from load_config)
    key : str
        Key in output_files section (e.g., 'user_info_df', 'test_results_df')
    run_id : str, optional
        Run identifier to use. If None, uses config['run_id']

    Returns
    -------
    str
        Resolved file path with {run_id} replaced

    Example
    -------
    >>> config = load_config()
    >>> path = get_output_path(config, 'user_info_df', run_id='dan')
    >>> print(path)
    'dbfs:/FileStore/misc/dan/user_info_df_pullcomplete.parquet'
    """
    if run_id is None:
        run_id = config.get("run_id", "default")

    path_template = config["output_files"][key]
    return path_template.format(run_id=run_id)


def get_all_output_paths(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get all output file paths with run_id substituted.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    run_id : str, optional
        Run identifier to use. If None, uses config['run_id']

    Returns
    -------
    dict
        Dictionary of key -> resolved path
    """
    if run_id is None:
        run_id = config.get("run_id", "default")

    return {
        key: path.format(run_id=run_id)
        for key, path in config["output_files"].items()
    }


class PipelineConfig:
    """
    Configuration manager for the wonky study pipeline.

    Handles loading config and resolving paths with run_id namespacing.

    Usage
    -----
    # In your notebook, at the top:
    from src.config import PipelineConfig

    # Option 1: Use run_id from config file
    cfg = PipelineConfig()

    # Option 2: Override run_id for your session
    cfg = PipelineConfig(run_id="dan")

    # Access paths
    silver_path = cfg.silver_path
    output_path = cfg.get_output_path("user_info_df")

    # Or get all output paths at once
    output_paths = cfg.output_paths
    """

    def __init__(
        self,
        config_path: str = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize configuration.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML config file
        run_id : str, optional
            Override run_id from config file. Use your name/identifier
            to avoid conflicts with other users.
        """
        self.config = load_config(config_path)
        self._run_id = run_id if run_id is not None else self.config.get("run_id", "default")

    @property
    def run_id(self) -> str:
        """Current run identifier."""
        return self._run_id

    @run_id.setter
    def run_id(self, value: str):
        """Update run identifier."""
        self._run_id = value

    # Base paths (read-only, shared)
    @property
    def base_path(self) -> str:
        return self.config["base_path"]

    @property
    def bronze_path(self) -> str:
        return self.config["bronze_path"]

    @property
    def silver_path(self) -> str:
        return self.config["silver_path"]

    @property
    def gold_path(self) -> str:
        return self.config["gold_path"]

    @property
    def checkpoint_path(self) -> str:
        return self.config["checkpoint_path"]

    @property
    def project_repository_path(self) -> str:
        return self.config["project_repository_path"]

    # Tables
    @property
    def tables(self) -> Dict[str, str]:
        return self.config["tables"]

    # Filters
    @property
    def filters(self) -> Dict[str, Any]:
        return self.config.get("filters", {})

    # Output paths (namespaced by run_id)
    def get_output_path(self, key: str) -> str:
        """
        Get a single output path with run_id substituted.

        Parameters
        ----------
        key : str
            Key in output_files (e.g., 'user_info_df')

        Returns
        -------
        str
            Resolved file path
        """
        return get_output_path(self.config, key, self._run_id)

    @property
    def output_paths(self) -> Dict[str, str]:
        """Get all output paths with run_id substituted."""
        return get_all_output_paths(self.config, self._run_id)

    def __repr__(self) -> str:
        return f"PipelineConfig(run_id='{self._run_id}')"
