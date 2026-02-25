
# Inspired from https://github.com/hiverge/cifar10-speedrun/blob/main/cifar10_speedrun.py

#############################################
#                  Setup                    #
#############################################

import os
import logging

import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
from torch.optim import Adam

from model import CifarNet
from loader import CifarLoader

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    @torch.compile()
    def _get_tta_logits(model, images_batch, pad):
        batch_size = images_batch.shape[0]
        padded_inputs = F.pad(images_batch, (pad,) * 4, "reflect")
        crop_tl = padded_inputs[:, :, 0:32, 0:32]
        crop_br = padded_inputs[:, :, 2:34, 2:34]
        base_views = torch.cat([images_batch, crop_tl, crop_br], dim=0)
        flipped_views = base_views.flip(-1)
        combined_inputs = torch.cat([base_views, flipped_views], dim=0)
        combined_logits = model(combined_inputs)
        num_views = combined_inputs.shape[0] // batch_size
        reshaped_logits = combined_logits.view(num_views, batch_size, -1)
        averaged_logits = reshaped_logits.mean(dim=0)
        return averaged_logits

    @torch.compile()
    def tta(model, test_images) -> torch.Tensor:
        with torch.no_grad():
            model.eval()
            device = test_images.device
            B = 2000
            pad = 1
            n = test_images.shape[0]
            all_logits_list = []
            for inputs_batch in test_images.split(B):
                inputs_batch = inputs_batch.contiguous(
                    memory_format=torch.channels_last
                )
                all_logits_list.append(model(inputs_batch).clone())
            initial_logits = torch.cat(all_logits_list, dim=0)
            probs = F.softmax(initial_logits, dim=1)
            confidences, _ = probs.max(dim=1)
            UNCERTAIN_QUANTILE = 0.25
            k_uncertain = int(n * UNCERTAIN_QUANTILE)
            _, uncertain_indices = torch.topk(
                confidences, k_uncertain, largest=False, sorted=False
            )

            tta_logits_parts = []
            tta_batch_size = 2000
            for i in range(0, k_uncertain, tta_batch_size):
                cur_batch_size = min(tta_batch_size, k_uncertain - i)
                batch_indices = uncertain_indices[i : i + cur_batch_size]
                images_batch = test_images[batch_indices]
                logits_batch = _get_tta_logits(
                    model,
                    images_batch.contiguous(memory_format=torch.channels_last),
                    pad,
                )
                tta_logits_parts.append(logits_batch)

            if tta_logits_parts:
                all_tta_logits_for_uncertain = torch.cat(tta_logits_parts, dim=0)
                final_logits = initial_logits.clone()
                final_logits[uncertain_indices] = all_tta_logits_for_uncertain
                return final_logits
            return initial_logits

    test_images = loader.normalize(loader.images)
    if tta_level < 2:
        model.eval()
        infer_fn = [infer_basic, infer_mirror, None][tta_level]
        with torch.no_grad():
            return torch.cat(
                [infer_fn(inputs, model) for inputs in test_images.split(2000)]
            )
    else:  # tta_level == 2
        return tta(model, test_images)

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model):
    training_batch_size = 1536

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=training_batch_size,
        aug={
            "flip": True,
            "translate": 2,
            "color_jitter": {
                "enabled": True,
                "brightness_range": 0.1399,
                "contrast_range": 0.1308,
            },
        },
    )
    if run == "warmup":
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
        train_loader.images = torch.randn_like(
            train_loader.images, device=train_loader.images.device
        )
        test_loader.labels = torch.randint(
            0, 10, size=(len(test_loader.labels),), device=test_loader.labels.device
        )
        test_loader.images = torch.randn_like(
            test_loader.images, device=test_loader.images.device
        )
    total_train_steps = ceil(7.65 * len(train_loader))
    whiten_bias_train_steps = ceil(0.2 * len(train_loader))
    model.reset()

    step = 0
    with torch.no_grad():
        train_images = train_loader.normalize(train_loader.images[:960])
        model.init_whiten(train_images)


    # Compile the forward pass function with reduced overhead
    @torch.compile()
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")
        return loss
    
    optimizer = Adam(model.parameters(), lr=3*1e-4)

    n_epochs = ceil(total_train_steps / len(train_loader))
    print("Number of epochs")
    print(n_epochs)

    for epoch in range(n_epochs):
        ####################
        #     Training     #
        ####################

        print(f"Epoch: {epoch}")
        model.train()
        for inputs, labels in train_loader:
            # Determine if we should train whiten bias
            whiten_bias_grad = step < whiten_bias_train_steps
            
            # Execute training step
            loss = forward_step(inputs, labels, whiten_bias_grad)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step >= total_train_steps:
                break
        if step >= total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    epoch = "eval"
    train_acc = evaluate(model, train_loader, tta_level=0)
    val_acc = evaluate(model, test_loader, tta_level=0)
    # print_training_details(locals(), is_final_entry=True)
    return (val_acc, tta_val_acc)

if __name__ == "__main__":
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    print(f"number of parameters: {sum([p.numel() for p in model.parameters()])}")
    # model.compile(mode="max-autotune")
    print_columns(logging_columns_list, is_head=True)
    val_acc, tta_val_acc = main("warmup", model)
    print(val_acc, tta_val_acc)