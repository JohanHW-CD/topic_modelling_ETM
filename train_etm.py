import torch
import numpy as np

from measures import create_bag_of_words

def train_topic_model(topic_model, training_data, num_epochs=2):
    """
    Train the Embedded Topic Model (ETM).
    """
    training_tokens = training_data['tokens']
    training_counts = training_data['counts']

    # Display initial topics before training
    topic_model.print_top_words_per_topic(label="Topics before training")

    for epoch in range(num_epochs):
        topic_model.train()
        batch_count = 0
        total_reconstruction_loss = 0
        total_epoch_loss = 0

        # Generate randomized document batches
        shuffled_doc_indices = torch.randperm(len(training_tokens))
        batched_doc_indices = torch.split(shuffled_doc_indices, topic_model.batch_size)

        for batch_indices in batched_doc_indices:
            topic_model.optimizer.zero_grad()
            batch_data = create_bag_of_words(training_tokens, training_counts, batch_indices, topic_model.vocab_size)
            batch_data_normalized = batch_data / batch_data.sum(1, keepdim=True)

            loss, reconstruction_loss = topic_model.forward(batch_data, batch_data_normalized)
            loss.backward()
            topic_model.optimizer.step()

            total_reconstruction_loss += reconstruction_loss.item()
            total_epoch_loss += loss.item()
            batch_count += 1

        average_reconstruction_loss = round(total_reconstruction_loss / batch_count, 3)
        average_loss = round(total_epoch_loss / batch_count, 3)

        topic_model.elbo_history.append(average_loss)
        topic_model.reconstruction_loss_history.append(average_reconstruction_loss)
        print(f'Epoch {epoch} - Reconstruction loss: {average_reconstruction_loss} - NELBO: {average_loss}')

    # Display final topics after training
    topic_model.print_top_words_per_topic(label="Topics after training")
    return topic_model
