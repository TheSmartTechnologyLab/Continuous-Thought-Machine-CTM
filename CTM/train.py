import torch
from tqdm import tqdm
import os

# local utils
from CTM.utils import (
    get_loss,
    calculate_accuracy,
    save_checkpoint_local,
    save_checkpoint_to_hf,
    load_checkpoint,
    get_latest_checkpoint,
)


def train_ctm(model, trainloader, testloader, iterations, device, test_every=100, lr=1e-4, checkpoint_dir=None, hf_repo=None, hf_token=None, resume=False, args=None):
    """Train loop for CTM. Supports optional checkpoint saving, HF upload and resume.

    This extends the original signature with checkpointing options so the pipeline
    can delegate full training to this function.
    """
    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=lr, eps=1e-8)
    model.train()

    start_iteration = 0
    history = {'train_losses': [], 'test_losses': [], 'train_accuracies_most_certain': [], 'test_accuracies_most_certain': []}

    # Resume if requested
    if resume and checkpoint_dir is not None:
        ck = get_latest_checkpoint(checkpoint_dir)
        if ck:
            print(f"Resuming from checkpoint: {ck}")
            checkpoint = load_checkpoint(ck, device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
            start_iteration = checkpoint.get('training_iteration', 0)
            history.update({k: checkpoint.get(k, []) for k in history.keys()})

    pbar = tqdm(total=iterations)
    trainloader_iter = iter(trainloader)

    test_loss = None
    test_accuracy = None

    for stepi in range(start_iteration, iterations):
        # --- batch ---
        try:
            inputs, targets = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            inputs, targets = next(trainloader_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # --- forward/backward ---
        predictions, certainties, _ = model(inputs, track=False)
        train_loss, where_most_certain = get_loss(predictions, certainties, targets)
        train_accuracy = calculate_accuracy(predictions, targets, where_most_certain)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # record training history
        history['train_losses'].append(train_loss.item())
        history['train_accuracies_most_certain'].append(train_accuracy)

        # --- periodic evaluation ---
        if stepi % test_every == 0:
            model.eval()
            with torch.inference_mode():
                all_test_predictions = []
                all_test_targets = []
                all_test_where_most_certain = []
                all_test_losses = []

                for inputs_t, targets_t in testloader:
                    inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
                    predictions_t, certainties_t, _ = model(inputs_t, track=False)
                    t_loss, where_most_certain_t = get_loss(predictions_t, certainties_t, targets_t)
                    all_test_losses.append(t_loss.item())
                    all_test_predictions.append(predictions_t)
                    all_test_targets.append(targets_t)
                    all_test_where_most_certain.append(where_most_certain_t)

                all_test_predictions = torch.cat(all_test_predictions, dim=0)
                all_test_targets = torch.cat(all_test_targets, dim=0)
                all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                test_accuracy = calculate_accuracy(all_test_predictions, all_test_targets, all_test_where_most_certain)
                test_loss = sum(all_test_losses) / len(all_test_losses)

            history['test_losses'].append(test_loss)
            history['test_accuracies_most_certain'].append(test_accuracy)
            model.train()

        # --- progress ---
        pbar.set_description(f'Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.3f} Test Loss: {test_loss} Test Acc: {test_accuracy}')
        pbar.update(1)

        # Periodic checkpointing every test_every steps (or at the end)
        if checkpoint_dir is not None and (stepi % test_every == 0 or stepi == iterations - 1):
            ck_path = os.path.join(checkpoint_dir, f'checkpoint_{stepi}.pt')
            checkpoint = {
                'training_iteration': stepi,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args) if args is not None else {},
                'train_losses': history['train_losses'],
                'test_losses': history['test_losses'],
                'train_accuracies_most_certain': history['train_accuracies_most_certain'],
                'test_accuracies_most_certain': history['test_accuracies_most_certain'],
            }
            save_checkpoint_local(checkpoint, ck_path)
            print(f"Saved checkpoint to {ck_path}")
            if hf_repo:
                try:
                    save_checkpoint_to_hf(ck_path, hf_repo, token=hf_token, commit_message=f'Checkpoint {stepi}')
                    print(f"Uploaded checkpoint {ck_path} to HF repo {hf_repo}")
                except Exception as e:
                    print(f"Failed to upload checkpoint to HF: {e}")

    pbar.close()
    return model