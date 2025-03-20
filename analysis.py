"""
Produces plots to analyse ETM model.
This file uses data_preprocessing.py, define_etm.py, train_etm.py, and measures.py. It saves to local machine in CSV form.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from data_preprocessing import data_preprocessing
from define_etm import EmbeddedTopicModel
from train_etm import train_topic_model
from measures import (
    compute_perplexity,
    compute_topic_coherence,
    prepare_documents_for_coherence
)

def analyze_topic_model_performance(
    min_df_values,
    cleaned_documents,
    stopwords,
    pretrained_embeddings,
    num_epochs=2,
    max_df_threshold=0.70
):
    """
    Train an Embedded Topic Model (ETM) for each min_df in min_df_values,
    compute perplexity, topic coherence, and visualize the results.
    """
    vocabulary_sizes = []
    perplexity_scores = []
    coherence_scores = []
    coherence_normalized_perplexity = []
    topic_diversity_scores = []

    for min_df in min_df_values:
        print(f"\n===== min_df = {min_df} =====")

        # Preprocess data
        vocabulary, training_data, testing_data = data_preprocessing(
            cleaned_documents, min_df=min_df, max_df=max_df_threshold, stops=stopwords
        )
        vocab_size = len(vocabulary)
        print(f"Vocabulary size: {vocab_size}")

        # Initialize the Topic Model
        topic_model = EmbeddedTopicModel(
            vocabulary,
            num_topics=20,
            hidden_layer_size=800,
            embedding_dim=300,
            pretrained_embeddings=pretrained_embeddings,
            train_embeddings=False
        )

        # Train the Model
        train_topic_model(topic_model, training_data, num_epochs=num_epochs)

        # Compute Perplexity
        perplexity = compute_perplexity(topic_model, testing_data)

        # Compute Topic Coherence
        document_sets = prepare_documents_for_coherence(training_data)
        coherence = compute_topic_coherence(topic_model, document_sets, top_n_words=10)

        # Compute Topic Diversity (If needed do it here, its defined and ready)
        # from measures import compute_topic_diversity
        # beta_matrix = topic_model.get_topic_word_distribution().detach().cpu().numpy()
        # topic_diversity = compute_topic_diversity(beta_matrix, topk=10)
        topic_diversity = 0.0  # Placeholder if not implemented

        # Compute Coherence-Normalized Perplexity (CNP)
        cnp_score = np.nan if coherence <= 0 else (perplexity / vocab_size) / coherence

        # Store results
        vocabulary_sizes.append(vocab_size)
        perplexity_scores.append(perplexity)
        coherence_scores.append(coherence)
        coherence_normalized_perplexity.append(cnp_score)
        topic_diversity_scores.append(topic_diversity)

        print(f"  Perplexity = {perplexity:.2f},  Coherence = {coherence:.4f},  CNP = {cnp_score:.4f}")

    # Save results to CSV
    with open('topic_model_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vocabulary Size', 'Perplexity', 'Topic Coherence', 'CNP'])
        writer.writerows(zip(vocabulary_sizes, perplexity_scores, coherence_scores, coherence_normalized_perplexity))

    # Plot results
    plt.figure(figsize=(10, 10))

    # Perplexity Plot
    plt.subplot(4, 1, 1)
    plt.plot(vocabulary_sizes, perplexity_scores, marker='o', color='red')
    plt.title("Perplexity vs. Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Perplexity")

    # Topic Coherence Plot
    plt.subplot(4, 1, 2)
    plt.plot(vocabulary_sizes, coherence_scores, marker='o', color='green')
    plt.title("Topic Coherence vs. Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Coherence")

    # CNP Plot
    plt.subplot(4, 1, 3)
    plt.plot(vocabulary_sizes, coherence_normalized_perplexity, marker='o', color='blue')
    plt.title("Coherence-Normalized Perplexity vs. Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("CNP")

    # Topic Diversity Plot
    plt.subplot(4, 1, 4)
    plt.plot(vocabulary_sizes, topic_diversity_scores, marker='o', color='purple')
    plt.title("Topic Diversity vs. Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Diversity")

    plt.tight_layout()
    plt.show()
