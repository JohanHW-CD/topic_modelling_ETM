import torch
import numpy as np
import math

############
# Perplexity
############

def compute_perplexity(topic_model, test_dataset):
    """
    Compute document completion perplexity on test dataset.
    """
    topic_model.eval()
    with torch.no_grad():
        topic_word_distribution = topic_model.get_topic_word_distribution()
        test_tokens1 = test_dataset['test1']['tokens']
        test_counts1 = test_dataset['test1']['counts']
        test_tokens2 = test_dataset['test2']['tokens']
        test_counts2 = test_dataset['test2']['counts']

        total_negative_log_likelihood = 0.0
        total_word_count = 0.0

        # Create document indices
        all_doc_indices = torch.arange(len(test_tokens1))
        batched_doc_indices = torch.split(all_doc_indices, topic_model.batch_size)

        for batch_indices in batched_doc_indices:
            batch1 = create_bag_of_words(test_tokens1, test_counts1, batch_indices, topic_model.vocab_size)
            batch1_normalized = batch1 / (batch1.sum(1, keepdim=True) + 1e-6)
            topic_distribution, _ = topic_model.get_document_topic_distribution(batch1_normalized)

            batch2 = create_bag_of_words(test_tokens2, test_counts2, batch_indices, topic_model.vocab_size)
            reconstructed_log_probs = torch.log(torch.mm(topic_distribution, topic_word_distribution) + 1e-6)

            reconstruction_loss = -(reconstructed_log_probs * batch2).sum(dim=1)
            total_negative_log_likelihood += reconstruction_loss.sum().item()
            total_word_count += batch2.sum().item()

        raw_perplexity = math.exp(total_negative_log_likelihood / total_word_count) if total_word_count > 0 else float('inf')
        return raw_perplexity

############
# Coherence
############

def prepare_documents_for_coherence(dataset):
    """
    Convert dataset into a list of sets, where each set contains unique word indices from a document.
    """
    return [set(document) for document in dataset['tokens']]

def compute_document_frequency(document_sets, word_index, second_word_index=None):
    """
    Count how many documents contain the given word(s).
    """
    if second_word_index is None:
        return sum(word_index in document for document in document_sets)
    else:
        doc_count_word_j = sum(second_word_index in document for document in document_sets)
        doc_count_both_words = sum((word_index in document and second_word_index in document) for document in document_sets)
        return (doc_count_word_j, doc_count_both_words)

def compute_topic_coherence(topic_model, document_sets, top_n_words=10):
    """
    Compute topic coherence for a trained topic model.
    """
    topic_word_distribution = topic_model.get_topic_word_distribution().detach().cpu().numpy()
    num_topics = topic_word_distribution.shape[0]
    num_documents = len(document_sets)

    topic_coherence_scores = []
    total_word_pairs = 0.0

    for topic_idx in range(num_topics):
        top_word_indices = topic_word_distribution[topic_idx].argsort()[-top_n_words:][::-1]
        topic_coherence = 0.0

        for i, word_i in enumerate(top_word_indices):
            doc_count_word_i = compute_document_frequency(document_sets, word_i)
            for word_j in top_word_indices[i + 1:]:
                doc_count_word_j, doc_count_both = compute_document_frequency(document_sets, word_i, word_j)
                if doc_count_both <= 0 or num_documents <= 0:
                    pairwise_score = 0.0
                else:
                    try:
                        pairwise_score = -1 + (
                            np.log(doc_count_word_i) + np.log(doc_count_word_j) - 2.0 * np.log(num_documents)
                        ) / (
                            np.log(doc_count_both) - np.log(num_documents)
                        )
                    except (ValueError, ZeroDivisionError):
                        pairwise_score = 0.0
                topic_coherence += pairwise_score
                total_word_pairs += 1

        topic_coherence_scores.append(topic_coherence)

    if total_word_pairs == 0:
        return 0.0
    return np.mean(topic_coherence_scores) / total_word_pairs

############
# Utility
############

def create_bag_of_words(document_words, word_counts, indices, vocabulary_size):
    """
    Generate a dense bag-of-words matrix for the given batch of documents.
    """
    bag_of_words_matrix = np.zeros((len(indices), vocabulary_size), dtype=np.float32)
    for row_index, doc_index in enumerate(indices):
        if doc_index < 0:
            continue
        words = document_words[doc_index]
        counts = word_counts[doc_index]
        np.put(bag_of_words_matrix[row_index], words, counts)
    return torch.from_numpy(bag_of_words_matrix)
