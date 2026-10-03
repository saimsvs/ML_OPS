import logging
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFusion(nn.Module):
    """
    Author's LoMix WeightedFusion implementation.

    Computes per-pixel attention weights for each prediction
    and fuses them using a weighted sum.
    """

    def __init__(self, num_stages, num_classes):
        super(WeightedFusion, self).__init__()

        self.weight_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    num_classes,
                    1,
                    kernel_size=1
                )
                for _ in range(num_stages)
            ]
        )

    def forward(self, predictions):

        weight_maps = [
            conv(pred)
            for pred, conv in zip(
                predictions,
                self.weight_convs
            )
        ]

        weights = torch.stack(
            weight_maps,
            dim=1
        )

        weights = F.softmax(
            weights,
            dim=1
        )

        preds = torch.stack(
            predictions,
            dim=1
        )

        fused = torch.sum(
            weights * preds,
            dim=1
        )

        return fused


class CombinatorialMutationsLossModule(nn.Module):
    """
    LoMix implementation adapted for BUSI binary segmentation.

    LoMix combines decoder predictions using:

        add
        mul
        weighted fusion
        concat

    Each original prediction and synthesized prediction
    has a learnable weight transformed using Softplus.

    BUSI adaptation:
    BCEWithLogitsLoss requires floating-point targets,
    so label_batch.float() is used for BCE.

    For binary segmentation with one output channel,
    DiceLoss uses sigmoid rather than softmax.
    """

    def __init__(
        self,
        original_num_maps,
        num_classes,
        selecetd_num_maps,
        operations=None,
        use_learnable_weights=True,
        supervision='mutation',
        lc1=0.3,
        lc2=0.7
    ):

        super().__init__()

        if operations is None:
            operations = [
                'add',
                'sub',
                'mul',
                'concat',
                'weighted_fusion',
                'avg',
                'max'
            ]

        self.num_maps = selecetd_num_maps
        self.num_classes = num_classes
        self.operations = operations
        self.use_learnable_weights = use_learnable_weights
        self.supervision = supervision
        self.lc1 = lc1
        self.lc2 = lc2

        # --------------------------------------------------
        # Learnable weights for original predictions
        # --------------------------------------------------

        if use_learnable_weights:

            self.original_weights = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1))
                    for _ in range(original_num_maps)
                ]
            )

        # --------------------------------------------------
        # Store all combinations
        # --------------------------------------------------

        self.combination_indices = {}

        if use_learnable_weights:
            self.synthesized_weights = nn.ModuleDict()

        for op in self.operations:

            comb_list = []

            for k in range(
                2,
                selecetd_num_maps + 1
            ):

                comb_list.extend(
                    list(
                        combinations(
                            range(selecetd_num_maps),
                            k
                        )
                    )
                )

            self.combination_indices[op] = comb_list

            if use_learnable_weights:

                self.synthesized_weights[op] = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(1))
                        for _ in range(len(comb_list))
                    ]
                )

        # --------------------------------------------------
        # Weighted fusion modules
        # --------------------------------------------------

        self.weighted_fusion_modules = nn.ModuleDict(
            {
                str(k): WeightedFusion(
                    k,
                    num_classes
                )
                for k in range(
                    2,
                    selecetd_num_maps + 1
                )
            }
        )

        # --------------------------------------------------
        # Create concat convolution layers once here.
        # --------------------------------------------------

        self.concat_modules = nn.ModuleDict(
            {
                str(k): nn.Conv2d(
                    num_classes * k,
                    num_classes,
                    kernel_size=1
                )
                for k in range(
                    2,
                    selecetd_num_maps + 1
                )
            }
        )

    # ------------------------------------------------------
    # Calculate positive LoMix weights
    # ------------------------------------------------------

    def _compute_all_weights(self):

        if not self.use_learnable_weights:
            return None, None

        # Softplus makes weights positive
        orig_vals = [
            F.softplus(w)
            for w in self.original_weights
        ]

        synth_vals = {}

        for op, plist in self.synthesized_weights.items():

            synth_vals[op] = [
                F.softplus(p)
                for p in plist
            ]

        return orig_vals, synth_vals

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(
        self,
        output_maps,
        label_batch=None,
        ce_loss=None,
        dice_loss=None
    ):

        device = output_maps[0].device

        deep_supervision_loss = 0.0
        mutation_loss = 0.0

        fused_logits = []

        # --------------------------------------------------
        # Get weights
        # --------------------------------------------------

        if not self.use_learnable_weights:

            orig_weights = [
                torch.ones(
                    1,
                    device=device
                )
                for _ in output_maps
            ]

            syn_weights = {}

            for op, combos in self.combination_indices.items():

                syn_weights[op] = [
                    torch.ones(
                        1,
                        device=device
                    )
                    for _ in combos
                ]

        else:

            orig_vals, synth_vals = (
                self._compute_all_weights()
            )

            orig_weights = orig_vals
            syn_weights = synth_vals

        # ==================================================
        # 1. Original decoder predictions
        # ==================================================

        for i, fmap in enumerate(output_maps):

            w_i = orig_weights[i]

            if (
                label_batch is None
                and ce_loss is None
                and dice_loss is None
            ):
                continue

            # BUSI binary BCE requires FLOAT targets
            loss_ce = ce_loss(
                fmap,
                label_batch.float()
            )

            # FIX:
            # Binary segmentation has one output channel,
            # so DiceLoss must use sigmoid, not softmax.
            loss_dice = dice_loss(
                fmap,
                label_batch,
                softmax=False
            )

            combined_loss = (
                self.lc1 * loss_ce
                + self.lc2 * loss_dice
            )

            deep_supervision_loss += (
                w_i * combined_loss
            )

        # ==================================================
        # 2. Combinatorial mutations
        # ==================================================

        for op in self.operations:

            combos = self.combination_indices[op]

            if self.use_learnable_weights:

                w_list = syn_weights[op]

            else:

                w_list = [
                    torch.ones(
                        1,
                        device=device
                    )
                    for _ in combos
                ]

            for idx, comb in enumerate(combos):

                # ------------------------------------------
                # Addition
                # ------------------------------------------

                if op == 'add':

                    mutated = sum(
                        output_maps[c]
                        for c in comb
                    )

                # ------------------------------------------
                # Average
                # ------------------------------------------

                elif op == 'avg':

                    mutated = (
                        sum(
                            output_maps[c]
                            for c in comb
                        )
                        / len(comb)
                    )

                # ------------------------------------------
                # Subtraction
                # ------------------------------------------

                elif op == 'sub':

                    mutated = (
                        output_maps[comb[0]]
                        - sum(
                            output_maps[c]
                            for c in comb[1:]
                        )
                    )

                # ------------------------------------------
                # Multiplication
                # ------------------------------------------

                elif op == 'mul':

                    mutated = output_maps[comb[0]]

                    for c in comb[1:]:

                        mutated = (
                            mutated
                            * output_maps[c]
                        )

                # ------------------------------------------
                # Concatenation
                # ------------------------------------------

                elif op == 'concat':

                    cat = torch.cat(
                        [
                            output_maps[c]
                            for c in comb
                        ],
                        dim=1
                    )

                    conv = self.concat_modules[
                        str(len(comb))
                    ]

                    mutated = conv(cat)

                # ------------------------------------------
                # Weighted Fusion
                # ------------------------------------------

                elif op in [
                    'weighted_fusion',
                    'wf'
                ]:

                    mod = self.weighted_fusion_modules[
                        str(len(comb))
                    ]

                    mutated = mod(
                        [
                            output_maps[c]
                            for c in comb
                        ]
                    )

                # ------------------------------------------
                # Maximum
                # ------------------------------------------

                elif op == 'max':

                    mutated = torch.stack(
                        [
                            output_maps[c]
                            for c in comb
                        ],
                        dim=0
                    ).max(
                        dim=0
                    ).values

                else:

                    raise ValueError(
                        f"Unsupported op: {op}"
                    )

                fused_logits.append(mutated)

                # ------------------------------------------
                # Mutation loss
                # ------------------------------------------

                if (
                    label_batch is None
                    and ce_loss is None
                    and dice_loss is None
                ):
                    continue

                # BUSI binary BCE requires FLOAT targets
                loss_ce = ce_loss(
                    mutated,
                    label_batch.float()
                )

                # FIX:
                # Binary segmentation has one output channel,
                # so DiceLoss must use sigmoid, not softmax.
                loss_dice = dice_loss(
                    mutated,
                    label_batch,
                    softmax=False
                )

                combined_loss = (
                    self.lc1 * loss_ce
                    + self.lc2 * loss_dice
                )

                mutation_loss += (
                    w_list[idx]
                    * combined_loss
                )

        # ==================================================
        # Return predictions when no labels are provided
        # ==================================================

        if (
            label_batch is None
            and ce_loss is None
            and dice_loss is None
        ):

            return fused_logits

        # ==================================================
        # Final LoMix loss
        # ==================================================

        if self.supervision in [
            'mutation',
            'lomix'
        ]:

            final_loss = (
                deep_supervision_loss
                + mutation_loss
            )

        else:

            final_loss = deep_supervision_loss

        return (
            final_loss,
            deep_supervision_loss,
            mutation_loss
        )

    # ======================================================
    # Print learned weights
    # ======================================================

    def print_weights(self):

        if not self.use_learnable_weights:

            logging.info(
                "No learnable weights. Using uniform weighting."
            )

            return

        raw_orig = [
            p.item()
            for p in self.original_weights
        ]

        orig_vals, synth_vals = (
            self._compute_all_weights()
        )

        logging.info(
            "Original Weights (softplus): %s",
            " ".join(
                f"{v.item():.4f}"
                for v in orig_vals
            )
        )

        for op in self.operations:

            if op not in synth_vals:
                continue

            svals = synth_vals[op]

            logging.info(
                "Synthesized Weights for '%s': %s",
                op,
                " ".join(
                    f"{sv.item():.4f}"
                    for sv in svals
                )
            )

    # ======================================================
    # Save weights
    # ======================================================

    def save_weights(self, save_path):

        torch.save(
            self.state_dict(),
            save_path
        )

        logging.info(
            f"Saved parameters to {save_path}"
        )

    # ======================================================
    # Load weights
    # ======================================================

    def load_weights(self, load_path):

        self.load_state_dict(
            torch.load(load_path)
        )

        logging.info(
            f"Loaded parameters from {load_path}"
        )


# ==========================================================
# Simple test
# ==========================================================

if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # Four U-Net decoder predictions
    predictions = [
        torch.randn(
            1,
            1,
            256,
            256,
            device=device
        )
        for _ in range(4)
    ]

    lomix = CombinatorialMutationsLossModule(
        original_num_maps=4,
        num_classes=1,
        selecetd_num_maps=4,
        operations=[
            'add',
            'mul',
            'wf',
            'concat'
        ],
        use_learnable_weights=True,
        supervision='lomix'
    ).to(device)

    # Forward pass without labels
    outputs = lomix(predictions)

    print(
        "Number of LoMix mutation outputs:",
        len(outputs)
    )

    print(
        "First output shape:",
        outputs[0].shape
    )

    print(
        "LoMix module test completed successfully."
    )
