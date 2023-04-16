# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from ..dataset import DATASETS
from ..model import MODELS
from ..network import NETWORKS


@configclass
class ForecastConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    model: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(input_size=trainset.num_variables, max_length=trainset.max_seq_len)
    model = config.model.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=trainset.num_classes,
    )
    model.fit(trainset, validset, testset)
    model.predict(trainset, "train")
    model.predict(validset, "valid")
    model.predict(testset, "test")
    return


if __name__ == "__main__":
    _config = ForecastConfig.fromcli()
    run_train(_config)
