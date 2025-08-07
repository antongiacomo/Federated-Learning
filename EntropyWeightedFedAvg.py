import json
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
from flwr.common import FitRes, Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class EntropyWeightedFedAvg(FedAvg):
    """Custom FedAvg strategy that uses entropy for weighted averaging.

    Lower entropy clients get higher weights in the aggregation.
    """

    def __init__(self, file_name: str, num_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name
        self.num_rounds = num_rounds
        self.loss_list = []
        self.metrics_list = []


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], ) -> Tuple[
        Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using entropy-based weighting."""
        if not results:
            return None, {}

        # Extract parameters and entropy values from fit results
        parameters_and_entropies = []
        total_examples = 0

        for client_proxy, fit_res in results:
            # Get entropy from the metrics returned by the client's fit method
            if "entropy" not in fit_res.metrics:
                print(f"Warning: No entropy metric received from client {client_proxy}")
            entropy = fit_res.metrics.get("entropy", 1.0)

            weight = max(entropy, 1e-8)

            parameters_and_entropies.append((
                fit_res.parameters,
                fit_res.num_examples,
                weight,
                entropy
            ))
            total_examples += fit_res.num_examples

            print(f"Client entropy: {entropy:.4f}, weight: {weight:.4f}, examples: {fit_res.num_examples}")

        # Aggregate parameters using entropy-based weights
        def aggregate_parameters(params_list):
            """Aggregate parameters with entropy-based weighting."""
            if not params_list:
                return None

            # Convert parameters to arrays
            arrays_list = []
            weights_list = []

            for params, num_examples, entropy_weight, entropy in params_list:
                arrays = parameters_to_ndarrays(params)
                # Combine entropy weight with number of examples
                final_weight = entropy_weight * num_examples
                arrays_list.append(arrays)
                weights_list.append(final_weight)

            # Normalize weights
            total_weight = sum(weights_list)
            normalized_weights = [w / total_weight for w in weights_list]

            # Weighted average of parameters
            if not arrays_list:
                return None

            # Initialize with zeros
            num_layers = len(arrays_list[0])
            aggregated_arrays = [np.zeros_like(arrays_list[0][i]) for i in range(num_layers)]

            # Compute weighted average
            for arrays, weight in zip(arrays_list, normalized_weights):
                for i in range(num_layers):
                    aggregated_arrays[i] += arrays[i] * weight

            return ndarrays_to_parameters(aggregated_arrays)

        # Aggregate using entropy-weighted averaging
        aggregated_parameters = aggregate_parameters(parameters_and_entropies)

        # Compute aggregated metrics
        entropies = [entropy for _, _, _, entropy in parameters_and_entropies]
        weights = [weight for _, _, weight, _ in parameters_and_entropies]

        aggregated_metrics = {
            "avg_entropy": float(np.mean(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "entropy_std": float(np.std(entropies)),
            "avg_weight": float(np.mean(weights)),
            "total_examples": total_examples,
        }

        print(f"Round {server_round} - Entropy-weighted aggregation:")
        print(f"  Average entropy: {aggregated_metrics['avg_entropy']:.4f}")
        print(f"  Entropy range: [{aggregated_metrics['min_entropy']:.4f}, {aggregated_metrics['max_entropy']:.4f}]")
        print(f"  Average weight: {aggregated_metrics['avg_weight']:.4f}")

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters):

        """Evaluate model parameters using an evaluation function."""
        loss, metrics = super().evaluate(server_round, parameters)

        if loss is not None and metrics is not None:
            # Record results
            self.loss_list.append(loss)
            self.metrics_list.append(metrics)

            print(f"Round {server_round} - Entropy-Weighted FedAvg:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")

            # If last round, save results only (no plot generation)
            if server_round == self.num_rounds:
                # Save to JSON
                with open(f"{self.file_name}.json", "w") as f:
                    json.dump({"loss": self.loss_list, "metrics": self.metrics_list}, f)

