def train_ctm(model, trainloader, testloader, iterations, device, test_every=100):
    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=0.0001, eps=1e-8)
    model.train()

    with tqdm(total=iterations, initial=0, dynamic_ncols=True) as pbar:
        test_loss = None
        test_accuracy = None
        for stepi in range(iterations):
            try:
                inputs, targets = next(iter(trainloader))
            except StopIteration:
                # Reset the trainloader iterator if it's exhausted
                trainloader_iterator = iter(trainloader)
                inputs, targets = next(trainloader_iterator)

            inputs, targets = inputs.to(device), targets.to(device)
            predictions, certainties, _ = model(inputs, track=False)
            train_loss, where_most_certain = get_loss(predictions, certainties, targets)
            train_accuracy = calculate_accuracy(predictions, targets, where_most_certain)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if stepi % test_every == 0:
                model.eval()
                with torch.inference_mode():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in testloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        predictions, certainties, _ = model(inputs, track=False)
                        test_loss, where_most_certain = get_loss(predictions, certainties, targets)
                        all_test_losses.append(test_loss.item())

                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_accuracy = calculate_accuracy(all_test_predictions, all_test_targets, all_test_where_most_certain)
                    test_loss = sum(all_test_losses) / len(all_test_losses)
                model.train()

            pbar.set_description(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')
            pbar.update(1)

    return model