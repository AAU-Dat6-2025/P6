import argparse
import json
import os
from logging import getLogger
from typing import Optional, Any, Dict, List

import torch.multiprocessing as mp
import torch.distributed as dist
import torch

from RecBole.recbole.config.configurator import Config
from RecBole.recbole.data.utils import create_dataset, data_preparation
from RecBole.recbole.model.general_recommender.bpr_zipf import BPRZipf
from RecBole.recbole.model.general_recommender.bpr_mmr import BPRMMR
from RecBole.recbole.utils import get_trainer, set_color
from RecBole.recbole.utils.utils import init_seed, get_model, get_environment
from RecBole.recbole.model.general_recommender.lightgcn_zipf import LightGCNZipf
from RecBole.recbole.model.general_recommender.multivae_zipf import MultiVAEZipf
from RecBole.recbole.model.general_recommender.ngcf_zipf import NGCFZipf
from RecBole.recbole.model.general_recommender.bpr_zipf import BPRZipf

from sampler import InteractionSampler

##################################
######### Configurations #########
##################################
model_folder = "./saved_models/"
metrics_results_folder = "./metrics_results/"

methods = {"BPR": None, "LightGCN": None, "NGCF": None, "MultiVAE": None, "Random": None, "BPRZipf": BPRZipf, "BPRMMR": BPRMMR, "LightGCNZipf": LightGCNZipf, "NGCFZipf": NGCFZipf, "MultiVAEZipf": MultiVAEZipf }

datasets = ["ml-100k", "ml-1m", "ml-20m", "gowalla-merged", "steam-merged"]
config_dictionary = {
    "metrics": ["Recall", "MRR", "NDCG", "Precision", "Hit", "Exposure", "ShannonEntropy", "Novelty", "RecommendedGraph"]
}
config_file = ["config.yaml"]
eval_config_file = ["eval_config.yaml"]


class RecboleRunner:
    def __init__(self, model_name: str, dataset_name: str, config_file_list: List[str] = None, config_dict: Dict[str, Any] = None, retrain: bool = False, over_sample_ratio: float = 0.0, under_sample_ratio: float = 0.0, save_model_as: str = None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_dict = config_dict if config_dict is not None else {}
        self.config_file_list = config_file_list if config_file_list is not None else []
        self.save_model_as = save_model_as
        self.config = self.create_config()
        self.gpus = self.get_available_cuda_gpus()
        self.retrain = retrain
        self.over_sample_ratio = over_sample_ratio
        self.under_sample_ratio = under_sample_ratio

        # Configuration for distributed training
        self.config_dict["offset"] = 0
        self.config_dict["ip"] = "127.0.0.1"
        self.config_dict["port"] = str(find_available_port(5670, 5680))
        self.config_dict["checkpoint_dir"] = model_folder + self.dataset_name

    def get_trained_model_path(self) -> Optional[str]:
        """
        Returns the path of a model if the model has been trained on the specified dataset
        :return: ``str`` | ``None`` Name of the saved model if it exists
        """
        if os.path.isdir(self.config_dict["checkpoint_dir"]):
            saved_models = os.listdir(self.config_dict["checkpoint_dir"])
            for saved_model in saved_models:
                if self.save_model_as is not None:
                    if saved_model == f"{self.save_model_as}.pth":
                        return self.config_dict["checkpoint_dir"] + "/" + saved_model
                elif saved_model == f"{self.model_name}.pth":
                    return self.config_dict["checkpoint_dir"] + "/" + saved_model
        return None

    def get_model_name_or_class(self) -> str | object:
        """
        Get the model name or class if custom model
        :return: ``str | object`` The model name or class
        """
        if methods[self.model_name] is None:
            return self.model_name
        return methods[self.model_name]

    def create_config(self, model=None, config_dict: Dict[str, Any] = None,
                      config_file_list: List[str] = None) -> Config:
        """
        Creates a configuration for the model
        :param model: ``str`` The model name
        :param config_dict: ``Dict[str, Any]`` The configuration dictionary
        :param config_file_list: ``List[str]`` The list of configuration files
        :return: ``Config`` The configuration
        """
        model = model if model is not None else self.get_model_name_or_class()
        config_dict = config_dict if config_dict is not None else self.config_dict
        config_file_list = config_file_list if config_file_list is not None else self.config_file_list
        config = Config(model=model, dataset=self.dataset_name, config_dict=config_dict, config_file_list=config_file_list)
        config["save_model_as"] = self.save_model_as if self.save_model_as is not None else f"{self.model_name}"
        return config

    def run_recbole(self, rank: int = None, queue: mp.SimpleQueue = None) -> dict[str, Any]:
        """
        Runs recbole, based on the run function from RecBole in "recbole.quick_start.quick_start"
        Changed to work with custom models and removed evaluation of the model after training
        :param rank: ``int`` The rank of the process
        :param queue: ``mp.SimpleQueue`` The queue for multiprocessing
        :return: ``dict[str, Any]`` The training results
        """
        logger = getLogger()
        config_dict = self.config_dict
        if rank is not None:
            config_dict["local_rank"] = rank
            logger.info(f"Process with rank {rank} started")

        config, model, dataset, train_data, valid_data, test_data = self.get_model_and_dataset(config_dict=config_dict)
        logger.info(config)

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config["show_progress"]
        )

        environment_tb = get_environment(config)
        logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )

        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

        result = {"best_valid_score": best_valid_score, "best_valid_result": best_valid_result}

        if rank is not None:
            if rank == 0:
                queue.put(result)  # for multiprocessing, e.g., mp.spawn
            dist.destroy_process_group()
        return result

    def run_recbole_multi_gpu(self) -> Dict[str, Any]:
        """
        Run and train the model on the specified dataset using multiple GPUs on a single node.
        Based on run function from RecBole in "recbole.quick_start.quick_start"
        """
        queue = mp.get_context("spawn").SimpleQueue()
        mp.spawn(self.run_recbole, args=(queue,), nprocs=self.config_dict["nproc"], join=True)

        result = None if queue.empty() else queue.get()
        return result

    def run_and_evaluate_model(self) -> Dict[str, Any]:
        """
        Run and evaluate the model on the specified dataset.
        :return: ``Dict[str, Any]`` The evaluation results.
        """
        if self.model_name == "Random":
            return self.evaluate_pre_trained_model("")

        # Skip pre-trained model check if retrain is True
        if not self.retrain:
            trained_model = self.get_trained_model_path()
            if trained_model is not None:
                print(f"Model {self.model_name} has been trained on dataset {self.dataset_name}. Skipping training.")
                return self.evaluate_pre_trained_model(trained_model)

        if len(self.gpus) == 1:
            return self.run_recbole()
        return self.run_recbole_multi_gpu()

    def get_available_cuda_gpus(self, max_gpus: int = None) -> List[str]:
        """
        Get all available CUDA GPUs
        :return: ``List[str]`` The list of available GPUs
        """
        if torch.cuda.is_available():
            gpus = [f"{torch.cuda.get_device_name(i)} - {i}" for i in range(torch.cuda.device_count())]
            print(f"GPU(s) available({len(gpus)}): {gpus}")
            if max_gpus is not None and len(gpus) > max_gpus:
                gpus = gpus[:max_gpus]
                print(f"Using only {max_gpus} GPU(s): {gpus}")
            self.config_dict["nproc"] = len(gpus)
            self.config_dict["world_size"] = self.config_dict["nproc"]
        else:
            print("No GPU available. Exiting.")
            exit(0)
        return gpus

    def model_supports_metrics(self, model_metrics: List) -> bool:
        """
        Check if the model supports the selected metrics
        :param model_metrics: ``List`` The metrics supported by the model
        :return: ``bool`` True if the model supports the selected metrics, False otherwise
        """
        metrics = [metric for metric in self.config_dict["metrics"] if metric not in model_metrics]
        if len(metrics) > 0:
            print(f"Model doesn't support some selected metrics: {metrics}")
            return False
        return True

    def get_model_and_dataset(self, model=None, config_dict: Dict[str, Any] = None, config_file_list: List[str] = None):
        """
        Get and initialise the Random model
        :return: ``Tuple[Config, torch.nn.Module, Dataset, Data, Data, Data]`` The configuration, model, dataset, train data, validation data and test data
        """
        config = self.create_config(model, config_dict, config_file_list)
        init_seed(config["seed"], config["reproducibility"])

        dataset = create_dataset(config)

        if self.over_sample_ratio > 0 or self.under_sample_ratio > 0:
            logger = getLogger()
            logger.info(set_color(f"Applying sampling using oversample ratio: {self.over_sample_ratio} and undersample ratio: {self.under_sample_ratio}", "blue"))
            original_shape = dataset.inter_feat.shape
            dataset = InteractionSampler(dataset).sample(self.under_sample_ratio, self.over_sample_ratio)
            logger.info(f"Dataset before sampling: {original_shape}, after sampling: {dataset.inter_feat.shape}")

        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        if methods[config["model"]] is None:
            model_class = get_model(config["model"])
        else:
            model_class = methods[config["model"]]
        model_class = model_class(config, train_data._dataset).to(config["device"])

        return config, model_class, dataset, train_data, valid_data, test_data

    def evaluate_pre_trained_model(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a pre-trained model
        :param model_path: ``str`` The path to the pre-trained model
        :return: ``Dict[str, Any]`` The evaluation results
        """
        logger = getLogger()

        config, model, dataset, train_data, valid_data, test_data = self.get_model_and_dataset()

        is_not_random = config["model"] != "Random"

        logger.info(f"Adjusted config: \n{config}")
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        if is_not_random:
            trainer.resume_checkpoint(model_path)

        # @TODO: Find better solution for this
        # Some data can only be accessed after the model has been fitted
        # This is a workaround to allow all metrics to be calculated
        # It will not train as the number of epochs is set to 0
        trainer.fit(train_data, valid_data, saved=False, show_progress=True)

        test_result = trainer.evaluate(test_data, load_best_model=is_not_random, show_progress=config["show_progress"])
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        return {"test_result": test_result}

    def save_metrics_results(self, results: Dict[str, Any]) -> None:
        """
        Save the evaluation results
        :param results: ``Dict[str,Any]`` The evaluation results
        :return: ``None``
        """
        path = f"{metrics_results_folder}{self.dataset_name}/{self.model_name}.json"
        os.makedirs(os.path.dirname(f"{metrics_results_folder}{self.dataset_name}"), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as file:
                all_results = json.load(file)
        else:
            all_results = {}

        if self.dataset_name not in all_results:
            all_results[self.dataset_name] = {}

        index = self.model_name if self.save_model_as is None else self.save_model_as

        # Merge old results with new results, where new results take precedence
        if index in all_results[self.dataset_name]:
            old_results = all_results[self.dataset_name].pop(index)
            results = old_results | results

        all_results[self.dataset_name][index] = results

        print(f"Saving results to {path}, results: {results}")
        with open(path, "w") as file:
            json.dump(all_results, file)

    def grid_search_zipf_alpha(self, alpha_values: List[float]) -> Dict[str, Any]:
        """
        Perform grid search to find the optimal zipf_alpha.
        :param alpha_values: ``List[float]`` List of zipf_alpha values to evaluate.
        :return: ``Dict[str, Any]`` Results for each zipf_alpha value.
        """
        results = {}
        original_alpha = self.config_dict["zipf_alpha"] if "zipf_alpha" in self.config_dict else 0.94  # Save original value

        for alpha in alpha_values:
            print(f"Evaluating zipf_alpha={alpha}")
            self.config_dict["zipf_alpha"] = alpha
            # Force retraining by setting retrain to True
            self.retrain = True
            result = self.run_and_evaluate_model()
            results[alpha] = result

        # Restore original zipf_alpha
        self.config_dict["zipf_alpha"] = original_alpha
        return results


def find_available_port(start: int, end: int) -> int:
    """
    Find an available port in the specified range
    :param start: ``int`` The start of the port range
    :param end: ``int`` The end of the port range
    """
    import socket
    for port in range(start, end + 1):
        # Try to connect to the port, if it fails (doesn't return 0), the port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                print(f"Binding to 127.0.0.1:{port} for multi-GPU training")
                return port
    raise RuntimeError("No available ports in the specified range")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate RecBole models")
    parser.add_argument("-d", "--dataset", type=str, help=f"Dataset to use: {datasets}")
    parser.add_argument("-m", "--method", type=str, help=f"Method to use: {methods.keys()}")
    parser.add_argument("-r", "--retrain", type=bool, help=f"Ignore pre-trained model and retrain", default=False)
    parser.add_argument("-e", "--evaluate", type=bool, help=f"Evaluate the selected model", default=False)
    parser.add_argument("-o", "--oversample", type=float, help=f"Ratio for oversampling", default=0.0)
    parser.add_argument("-u", "--undersample", type=float, help=f"Ratio for undersampling", default=0.0)
    parser.add_argument("-s", "--save_model_as", type=str, help=f"Name to save model as", default=None)
    parser.add_argument("-a", "--alpha_values", type=str, help="Comma-separated list of zipf_alpha values for grid search")

    args = parser.parse_args()

    if not args.dataset:
        print("Specify a dataset using flag -d or --dataset")
        exit(1)
    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not supported. Supported datasets: {datasets}")
        exit(1)
    if not args.method:
        print("Specify a method using flag -m or --method")
        exit(1)
    if args.method not in methods.keys():
        print(f"Method {args.method} not supported. Supported methods: {methods.keys()}")
        exit(1)

    if args.evaluate:
        config_file = eval_config_file

    if args.dataset == "steam-merged":
        config_file.append("config_steam.yaml")

    # Fixing compatibility issues
    import numpy as np

    np.float = np.float64
    np.complex = np.complex128
    np.unicode = np.str_

    print(f"\n------------- Running Recbole -------------\nArguments given: {args}\n")
    runner = RecboleRunner(model_name=args.method, dataset_name=args.dataset, config_file_list=config_file, config_dict=config_dictionary, retrain=args.retrain, over_sample_ratio=args.oversample, under_sample_ratio=args.undersample, save_model_as=args.save_model_as)

    if args.alpha_values:
        alpha_values = [float(a) for a in args.alpha_values.split(",")]
        results = runner.grid_search_zipf_alpha(alpha_values)
        print("Grid search results:", results)
    else:
        results = runner.run_and_evaluate_model()
    runner.save_metrics_results(results)
