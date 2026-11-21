"""
parameter_calculator.py
========================
Calculates the theoretical and architectural parameter counts and multiply-accumulate
(MAC) operations for feedforward neural networks (specifically MLP architectures of the
family 5 -> w -> w -> 1 and arbitrary hidden configurations).

Formulas:
---------
For a fully-connected (Linear) layer mapping from $n_{in}$ inputs to $n_{out}$ outputs:
  - Weight parameters:   $W = n_{in} \times n_{out}$
  - Bias parameters:     $B = n_{out}$ (assuming bias enabled)
  - Total parameters:    $P = W + B = n_{out} \times (n_{in} + 1)$
  - MAC operations:      $MAC = n_{in} \times n_{out}$

For an MLP with architecture 5 -> w -> w -> 1:
  - Layer 1 (Linear 5 -> w):
      $W_1 = 5w$
      $B_1 = w$
      $P_1 = 6w$
      $MAC_1 = 5w$
  - Layer 2 (Linear w -> w):
      $W_2 = w^2$
      $B_2 = w$
      $P_2 = w^2 + w$
      $MAC_2 = w^2$
  - Output Layer (Linear w -> 1):
      $W_3 = w$
      $B_3 = 1$
      $P_3 = w + 1$
      $MAC_3 = w$

  - Totals:
      $W_{total}   = W_1 + W_2 + W_3 = 5w + w^2 + w = w^2 + 6w$
      $B_{total}   = B_1 + B_2 + B_3 = w + w + 1 = 2w + 1$
      $P_{total}   = W_{total} + B_{total} = w^2 + 8w + 1$
      $MAC_{total} = MAC_1 + MAC_2 + MAC_3 = w^2 + 6w$

Author: Nandish (Mission 4 - Edge Device & Memory Analysis)
"""

from typing import Dict, Any, List, Tuple
import pandas as pd


def count_linear_layer(in_features: int, out_features: int, bias: bool = True) -> Dict[str, int]:
    """
    Calculate parameters and MACs for a single Linear (dense) layer.
    
    Args:
        in_features: Number of input dimensions.
        out_features: Number of output dimensions.
        bias: Whether bias parameters are included (default True).
        
    Returns:
        Dict with keys:
            'in_features', 'out_features', 'weights', 'biases', 'total_params', 'macs'
    """
    if not isinstance(in_features, int) or in_features <= 0:
        raise ValueError(f"in_features must be a positive integer, got {in_features}")
    if not isinstance(out_features, int) or out_features <= 0:
        raise ValueError(f"out_features must be a positive integer, got {out_features}")

    weights = in_features * out_features
    biases = out_features if bias else 0
    total_params = weights + biases
    macs = weights  # 1 MAC per weight connection in standard dense layer dot product

    return {
        "in_features": in_features,
        "out_features": out_features,
        "weights": weights,
        "biases": biases,
        "total_params": total_params,
        "macs": macs,
    }


def calculate_mlp_architecture(
    input_dim: int = 5,
    hidden_widths: Tuple[int, ...] = (32, 32),
    output_dim: int = 1,
    bias: bool = True,
) -> Dict[str, Any]:
    """
    Calculate layer-by-layer breakdown and total counts for an arbitrary MLP.
    
    Args:
        input_dim: Number of input features (default 5).
        hidden_widths: Tuple of hidden layer widths (default (32, 32)).
        output_dim: Number of output features (default 1).
        bias: Whether linear layers use bias (default True).
        
    Returns:
        Dict containing:
            - 'architecture': string description
            - 'layers': list of per-layer dicts
            - 'total_weights': sum of all weight parameters
            - 'total_biases': sum of all bias parameters
            - 'total_params': total trainable parameters
            - 'total_macs': total MAC operations per inference
    """
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError(f"input_dim must be positive int, got {input_dim}")
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(f"output_dim must be positive int, got {output_dim}")
    if not isinstance(hidden_widths, (tuple, list)) or len(hidden_widths) == 0:
        raise ValueError("hidden_widths must be a non-empty sequence of positive integers")
    for w in hidden_widths:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"hidden layer width must be a positive integer, got {w}")

    layers = []
    prev_dim = input_dim

    # Hidden layers
    for idx, width in enumerate(hidden_widths):
        layer_info = count_linear_layer(prev_dim, width, bias=bias)
        layer_info["name"] = f"linear{idx + 1}"
        layer_info["layer_type"] = "hidden"
        layers.append(layer_info)
        prev_dim = width

    # Output layer
    out_layer = count_linear_layer(prev_dim, output_dim, bias=bias)
    out_layer["name"] = f"linear{len(hidden_widths) + 1}"
    out_layer["layer_type"] = "output"
    layers.append(out_layer)

    total_weights = sum(layer["weights"] for layer in layers)
    total_biases = sum(layer["biases"] for layer in layers)
    total_params = total_weights + total_biases
    total_macs = sum(layer["macs"] for layer in layers)

    arch_str = f"{input_dim} -> " + " -> ".join(str(w) for w in hidden_widths) + f" -> {output_dim}"

    return {
        "architecture": arch_str,
        "input_dim": input_dim,
        "hidden_widths": tuple(hidden_widths),
        "output_dim": output_dim,
        "layers": layers,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_params": total_params,
        "total_macs": total_macs,
    }


def calculate_two_hidden_mlp_formulas(w: int, input_dim: int = 5, output_dim: int = 1) -> Dict[str, int]:
    """
    Direct evaluation of closed-form formulas for 5 -> w -> w -> 1 architecture.
    
    Formula:
        W_total = 5w + w^2 + w = w^2 + 6w
        B_total = w + w + 1 = 2w + 1
        P_total = w^2 + 8w + 1
        MAC_total = w^2 + 6w
    """
    if not isinstance(w, int) or w <= 0:
        raise ValueError(f"Width w must be a positive integer, got {w}")

    w1 = input_dim * w
    b1 = w
    p1 = w1 + b1
    mac1 = w1

    w2 = w * w
    b2 = w
    p2 = w2 + b2
    mac2 = w2

    w3 = w * output_dim
    b3 = output_dim
    p3 = w3 + b3
    mac3 = w3

    total_weights = w2 + (input_dim + output_dim) * w
    total_biases = 2 * w + output_dim
    total_params = total_weights + total_biases
    total_macs = total_weights

    return {
        "w": w,
        "l1_weights": w1,
        "l1_biases": b1,
        "l1_params": p1,
        "l1_macs": mac1,
        "l2_weights": w2,
        "l2_biases": b2,
        "l2_params": p2,
        "l2_macs": mac2,
        "out_weights": w3,
        "out_biases": b3,
        "out_params": p3,
        "out_macs": mac3,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_params": total_params,
        "total_macs": total_macs,
    }


def compare_five_widths(widths: List[int] = [32, 128, 256, 512, 2048]) -> pd.DataFrame:
    """
    Generate parameter and MAC comparison across the specified 5 benchmark widths.
    
    Args:
        widths: List of hidden widths to evaluate.
        
    Returns:
        pandas.DataFrame with detailed parameter and MAC breakdown.
    """
    records = []
    for w in widths:
        res = calculate_two_hidden_mlp_formulas(w)
        records.append({
            "Width (w)": w,
            "Architecture": f"5 -> {w} -> {w} -> 1",
            "L1 Weights (5*w)": res["l1_weights"],
            "L1 Biases (w)": res["l1_biases"],
            "L1 MACs": res["l1_macs"],
            "L2 Weights (w^2)": res["l2_weights"],
            "L2 Biases (w)": res["l2_biases"],
            "L2 MACs": res["l2_macs"],
            "Out Weights (w*1)": res["out_weights"],
            "Out Biases (1)": res["out_biases"],
            "Out MACs": res["out_macs"],
            "Total Weights (w^2+6w)": res["total_weights"],
            "Total Biases (2w+1)": res["total_biases"],
            "Total Trainable Params (w^2+8w+1)": res["total_params"],
            "Total MACs (w^2+6w)": res["total_macs"],
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = compare_five_widths()
    print("=== PARAMETER & MAC COMPARISON ACROSS 5 WIDTHS ===")
    print(df.to_string(index=False))

