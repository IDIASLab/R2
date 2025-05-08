import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.base import BaseEstimator
import random
import re




class TripleEvaluator:
    """
    This class evaluates triples (head, relation, tail) in a knowledge graph using entity and predicate embeddings.
    It supports multiple embedding models (TransE, DistMult, ComplEx, RotatE).
    """

    def __init__(self, entity_embeddings, predicate_embeddings, train_df, test_df, model, k=10,
                 precomputed_top_k_tail=None, precomputed_top_k_head=None,
                 head_relation_answers=None, tail_relation_answers=None):
        """
        Initialize the TripleEvaluator with entity, predicate embeddings, and model type.

        Parameters:
        ----------
        entity_embeddings : dict
            Dictionary containing the embeddings of entities.
        predicate_embeddings : dict
            Dictionary containing the embeddings of predicates.
        train_df : pd.DataFrame
            DataFrame containing triples in the training set with columns 'head', 'relation', 'tail'.
        test_df : pd.DataFrame
            DataFrame containing triples in the test set with columns 'head', 'relation', 'tail'.
        model : str
            The embedding model to use ('TransE', 'DistMult', 'ComplEx', 'RotatE').
        k : int
            The number of top candidates to precompute for head and tail predictions.
        precomputed_top_k_tail : dict, optional
            Precomputed top-k tail predictions. If not provided, they will be computed.
        precomputed_top_k_head : dict, optional
            Precomputed top-k head predictions. If not provided, they will be computed.
        head_relation_answers : dict, optional
            Precomputed head relation answers. If not provided, they will be computed.
        tail_relation_answers : dict, optional
            Precomputed tail relation answers. If not provided, they will be computed.
        """

        self.train_df = train_df
        self.test_df = test_df
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.train_triples_set = set(zip(train_df['head'], train_df['relation'], train_df['tail']))
        self.model = model  # Store the specified model type
        self.k = k

        # Precompute a single list of entity IDs and one stacked matrix of all entity embeddings
        self._entity_list = list(self.entity_embeddings.keys())
        self._all_entity_embeddings = np.vstack([self.entity_embeddings[e] 
                                                 for e in self._entity_list])

        # Precomputed top-k predictions
        self.precomputed_top_k_tail = precomputed_top_k_tail if precomputed_top_k_tail is not None else {}
        self.precomputed_top_k_head = precomputed_top_k_head if precomputed_top_k_head is not None else {}
        if not precomputed_top_k_tail or not precomputed_top_k_head:
            self._precompute_top_k_predictions(test_df)

        # Precomputed answers for evaluation
        self.head_relation_answers = head_relation_answers if head_relation_answers is not None else {}
        self.tail_relation_answers = tail_relation_answers if tail_relation_answers is not None else {}
        if not head_relation_answers or not tail_relation_answers:
            self._precompute_answers(test_df)


    def extract_state(self):
        """
        Extract the current state of the TripleEvaluator object as a dictionary.

        Returns:
        -------
        dict : A dictionary containing all the current attributes necessary to reinitialize the object.
        """
        return {
            'entity_embeddings': self.entity_embeddings,
            'predicate_embeddings': self.predicate_embeddings,
            'train_df' : self.train_df,
            'test_df' : self.test_df,
            'k': self.k,
            'precomputed_top_k_tail': self.precomputed_top_k_tail,
            'precomputed_top_k_head': self.precomputed_top_k_head,
            'head_relation_answers': self.head_relation_answers,
            'tail_relation_answers': self.tail_relation_answers
        }

    def _precompute_answers(self, test_df):
      """
      Precompute answer dictionaries based on the unique (head, relation) and (tail, relation) pairs in the test dataset.

      Parameters:
      ----------
      test_df : pd.DataFrame
          DataFrame containing triples with columns 'head', 'relation', 'tail'.
      """

      # Populate the dictionaries by iterating over each row in the test DataFrame
      for row in test_df.itertuples(index=False):
          head, relation, tail = row.head, row.relation, row.tail

          # Add tail as answer for the (head, relation) key
          if (head, relation) not in self.head_relation_answers:
              self.head_relation_answers[(head, relation)] = set()
          self.head_relation_answers[(head, relation)].add(tail)

          # Add head as answer for the (tail, relation) key
          if (tail, relation) not in self.tail_relation_answers:
              self.tail_relation_answers[(tail, relation)] = set()
          self.tail_relation_answers[(tail, relation)].add(head)

      # Convert sets to lists for consistency in accessing answers
      self.head_relation_answers = {key: list(values) for key, values in self.head_relation_answers.items()}
      self.tail_relation_answers = {key: list(values) for key, values in self.tail_relation_answers.items()}    

    def _precompute_top_k_predictions(self, test_df):
        """
        Precompute top-k predictions for head and tail for each triple in the test set, using filtering.

        Parameters:
        ----------
        test_df : pd.DataFrame
            Test set containing triples with columns 'head', 'relation', 'tail'.
        """
        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="Precomputing top-k predictions"):
            head, relation, tail = row.head, row.relation, row.tail

            # Check if embeddings are available for the head, relation, and tail
            if head not in self.entity_embeddings or tail not in self.entity_embeddings or relation not in self.predicate_embeddings:
                continue  # Skip if any embedding is missing

            # Precompute top-k tail predictions using filtered method
            top_k_tail = self.find_top_k_candidates_filtered(head, relation, tail, self.k, find_for="tail")
            self.precomputed_top_k_tail[(head, relation)] = top_k_tail

            # Precompute top-k head predictions using filtered method
            top_k_head = self.find_top_k_candidates_filtered(head, relation, tail, self.k, find_for="head")
            self.precomputed_top_k_head[(relation, tail)] = top_k_head



    def get_embedding(self, triple):
        """
        Retrieve the concatenated embeddings for a given triple (head, relation, tail).

        Parameters:
        ----------
        triple : dict
            A dictionary containing 'head', 'relation', and 'tail' as keys.

        Returns:
        -------
        np.array : A concatenated numpy array containing the embeddings for head, relation, and tail.

        Raises:
        ------
        KeyError : If the head, relation, or tail is missing from the embeddings.
        """
        # Ensure that all components of the triple (head, relation, tail) are in the embeddings
        if triple['head'] not in self.entity_embeddings:
            raise KeyError(f"Head entity {triple['head']} is missing from entity embeddings.")
        if triple['relation'] not in self.predicate_embeddings:
            raise KeyError(f"Relation {triple['relation']} is missing from predicate embeddings.")
        if triple['tail'] not in self.entity_embeddings:
            raise KeyError(f"Tail entity {triple['tail']} is missing from entity embeddings.")

        # Retrieve the corresponding embeddings
        head_emb = self.entity_embeddings[triple['head']]
        rel_emb = self.predicate_embeddings[triple['relation']]
        tail_emb = self.entity_embeddings[triple['tail']]

        # Concatenate head, relation, and tail embeddings into a single vector
        triple_embedding = np.concatenate([head_emb, rel_emb, tail_emb])

        return triple_embedding
    
    
    def compute_scores(self, head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for):
        """
        Compute scores based on the specified embedding model.
        """
        if self.model == "TransE":
            target_emb = head_emb + rel_emb if find_for == "tail" else tail_emb - rel_emb
            return np.linalg.norm(all_entity_embeddings - target_emb, axis=1)  # Distance (lower is better)
        elif self.model == "DistMult":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb * rel_emb
            return np.dot(all_entity_embeddings, target_emb)  # Dot product (higher is better)
        elif self.model == "ComplEx":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb * np.conj(rel_emb)
            return np.real(np.dot(all_entity_embeddings, np.conj(target_emb)))  # Real part of dot product
        elif self.model == "RotatE":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb / rel_emb
            return np.linalg.norm(all_entity_embeddings - target_emb, axis=1)  # Distance (lower is better)
        else:
            raise ValueError("Unknown model type")
        

    def find_top_k_candidates_filtered(self, head, relation, tail, k, find_for="tail"):
        """
        Find the top K candidate entities for a given entity and relation using the embedding model.
        """
        if head not in self.entity_embeddings or relation not in self.predicate_embeddings:
            raise ValueError(f"Entity or relation is missing from the embeddings: {head}, {relation}")
        if tail not in self.entity_embeddings:
            raise ValueError(f"Tail entity is missing from the embeddings: {tail}")

        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]

        # Use the pre-constructed matrix and entity list instead of rebuilding and re-keying
        all_entity_embeddings = self._all_entity_embeddings
  
        scores = self.compute_scores(head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for)

        # Full sort (as in original code)
        top_k_indices = np.argsort(-scores if self.model in ["DistMult", "ComplEx"] else scores)

        filtered_top_k_candidates = []
        for idx in top_k_indices:
            # Retrieve the entity string using the stored list
            candidate_entity = self._entity_list[idx]

            if find_for == "tail":
                candidate_triple = (head, relation, candidate_entity)
                original_triple = (head, relation, tail)
            else:  # find_for == "head"
                candidate_triple = (candidate_entity, relation, tail)
                original_triple = (head, relation, tail)

            if candidate_triple in self.train_triples_set:
                continue

            filtered_top_k_candidates.append((candidate_entity, scores[idx]))
            if len(filtered_top_k_candidates) >= k:
                break

        return filtered_top_k_candidates

    def score_triple(self, head, relation, tail, model):
        """
        Score the validity of a given (head, relation, tail) triple using a trained model.

        Parameters:
        ----------
        head : str
            The head entity of the triple.
        relation : str
            The relation (predicate) of the triple.
        tail : str
            The tail entity of the triple.
        model : BaseEstimator
            The trained model (e.g., RandomForestClassifier) used for scoring the triple.

        Returns:
        -------
        float : The score representing the validity of the triple (0-1).

        Raises:
        ------
        ValueError : If the head, relation, or tail is missing from the embeddings.
        """
        # Ensure that the head, relation, and tail are present in the embeddings
        if head not in self.entity_embeddings or tail not in self.entity_embeddings or relation not in self.predicate_embeddings:
            raise ValueError(f"One of the entities or the relation is missing from the embeddings: {head}, {relation}, {tail}")

        # Retrieve the embeddings for head, relation, and tail
        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]

        # Concatenate the embeddings to form a single vector
        triple_embedding = np.concatenate([head_emb, rel_emb, tail_emb])

        # Reshape the vector to match the model's input (1 sample with the concatenated triple embedding)
        triple_embedding_reshaped = triple_embedding.reshape(1, -1)

        # Predict the score using the trained model
        score = model.predict(triple_embedding_reshaped)[0]

        return score

    def create_predictions(self, head, relation, tail, models, predict_for):
        """
        Create predictions from multiple models for a single triple.

        Parameters:
        ----------
        head : str
            The head entity of the triple.
        relation : str
            The relation (predicate) of the triple.
        tail : str
            The tail entity of the triple.
        models : list
            List of trained models to use for predictions.
        predict_for : str
            Specify whether to predict for 'head' or 'tail'.

        Returns:
        -------
        list : A list of predictions from all models for the given triple.
        """
        predictions = []
        for model in models:
            # If model is a scikit-learn estimator, use score_triple; otherwise, use the model's own scoring method
            if isinstance(model, BaseEstimator):
                prediction = self.score_triple(head, relation, tail, model)
            else:
                prediction = model.score_triple(head, relation, tail, predict_for)
            predictions.append(prediction)
        return predictions

    
    # !! Handles multiple possible true entities. Looks for the any match from the test triple
    # Loop over unique patterns
    def evaluate_hitk_original_2_on_patterns(self, k, max_elements, evaluate_for="tail", verbose=False):
        """
        Evaluate the model's predictions on the test set by retrieving precomputed top K candidates and scoring them.

        Parameters:
        ----------
        test_df : pd.DataFrame
            Test set containing triples with columns 'head', 'relation', 'tail'.
        k : int
            The number of top candidates to consider for evaluation.
        evaluate_for : str
            Specify whether to evaluate for 'head' or 'tail'. Default is 'tail'.

        Returns:
        -------
        tuple : (total_count, hit_count) - Number of evaluated triples and number of hits.
        """
        # Initialize counters
        hit_count = 0
        total_count = 0

        # Select precomputed data
        selected_iterator = (
            self.precomputed_top_k_tail if evaluate_for == "tail" else self.precomputed_top_k_head
        )
        if not selected_iterator:
            raise ValueError(f"Invalid value for `evaluate_for`: {evaluate_for}")

        keys = list(selected_iterator.keys())
        if max_elements is not None:
            keys = keys[:max_elements]

        iterator = (
            tqdm(keys, total=len(keys), desc=f"Evaluating {evaluate_for.capitalize()} Predictions")
            if verbose else keys
        )

        for key in iterator:
            # Set the true entities based on evaluation type and retrieve from precomputed dictionaries
            if evaluate_for == "tail":
                  head = key[0]
                  relation = key[1]
                  true_entities = self.head_relation_answers.get((head, relation), [])  # Predicting tail
            elif evaluate_for == "head":
                  tail = key[1]
                  relation = key[0]
                  true_entities = self.tail_relation_answers.get((tail, relation), [])  # Predicting head

            # Step 1: Find top K candidate entities for the evaluation type
            if evaluate_for == "tail":
                top_k_candidates = self.precomputed_top_k_tail.get((head, relation), [])[:k]  # Get only top k
            elif evaluate_for == "head":
                top_k_candidates = self.precomputed_top_k_head.get((relation, tail), [])[:k]  # Get only top k

            predicted_entities = {entity for entity, _ in top_k_candidates}

            # Check if any true entity is in the top-k predictions
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1

            total_count += 1

        return total_count, hit_count


    
    def rerank(self, k, model, threshold=10, evaluate_for="tail", verbose=False):
        """
        Re-rank candidates using a trained model and evaluate top-k predictions in batches.

        Parameters:
        -----------
        k : int
            The number of top-ranked candidates to evaluate.
        model : object
            A trained model with a `predict_proba` method for scoring candidates.
        threshold : int, optional
            The maximum number of candidates to consider for re-ranking. Default is 10.
        evaluate_for : str, optional
            Specifies whether to evaluate predictions for 'tail' or 'head'. Default is "tail".
        verbose : bool, optional
            If True, displays a progress bar using `tqdm`. Default is True.

        Returns:
        --------
        total_count : int
            Total number of keys (patterns) evaluated.
        hit_count : int
            Number of keys where at least one true entity is in the top-k predictions.

        Raises:
        -------
        ValueError
            If `evaluate_for` is not 'tail' or 'head'.
        """
        # Initialize counters for hits and total evaluations
        hit_count = 0
        total_count = 0

        # Select the appropriate precomputed dictionary based on evaluation type
        selected_iterator = (
            self.precomputed_top_k_tail if evaluate_for == "tail" else self.precomputed_top_k_head
        )
        if selected_iterator is None:
            raise ValueError("Invalid value for `evaluate_for`. Use 'tail' or 'head'.")

        # Determine whether to use tqdm based on the verbose parameter
        iterator = (
            tqdm(
                selected_iterator.keys(),
                total=len(selected_iterator),
                desc=f"Evaluating {evaluate_for.capitalize()} Predictions"
            ) if verbose else selected_iterator.keys()
        )

        # Iterate over each unique key (pattern)
        for key in iterator:
            # Extract relevant entities and relationships from the key
            if evaluate_for == "tail":
                head, relation = key
                true_entities = self.head_relation_answers.get((head, relation), [])
            elif evaluate_for == "head":
                relation, tail = key
                true_entities = self.tail_relation_answers.get((tail, relation), [])

            # Retrieve top candidates up to the specified threshold
            all_candidates = selected_iterator.get(key, [])[:threshold]

            # Collect embeddings and construct feature vectors
            if evaluate_for == "tail":
                head_emb = self.entity_embeddings[head]
                rel_emb = self.predicate_embeddings[relation]
                features = [
                    np.concatenate([head_emb, rel_emb, self.entity_embeddings[entity]])
                    for entity, _ in all_candidates
                ]
            elif evaluate_for == "head":
                tail_emb = self.entity_embeddings[tail]
                rel_emb = self.predicate_embeddings[relation]
                features = [
                    np.concatenate([self.entity_embeddings[entity], rel_emb, tail_emb])
                    for entity, _ in all_candidates
                ]

            # Extract the list of candidate entities
            candidate_entities = [entity for entity, _ in all_candidates]

            # Predict probabilities for each candidate using the trained model
            features_array = np.array(features)
            probabilities = model.predict_proba(features_array)[:, 1]

            # Combine candidates with their predicted scores and sort in descending order
            reranked_candidates = sorted(
                zip(candidate_entities, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            # Extract the top-k predicted entities
            predicted_entities = {entity for entity, _ in reranked_candidates[:k]}

            # Check if any true entity is among the top-k predictions
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1

            # Increment the total evaluation count
            total_count += 1

        return total_count, hit_count
    
    



class Training:
    def __init__(self, entity_embeddings, predicate_embeddings, train_df, test_df, model, factor=100):
        """
        Initialize the Training class with entity, predicate embeddings, and model type.

        Parameters:
        ----------
        entity_embeddings : dict
            Dictionary containing the embeddings of entities.
        predicate_embeddings : dict
            Dictionary containing the embeddings of predicates.
        train_df : pd.DataFrame
            DataFrame containing triples in the training set with columns 'head', 'relation', 'tail'.
        test_df : pd.DataFrame
            DataFrame containing triples in the test set with columns 'head', 'relation', 'tail'.
        model : str
            The embedding model to use ('TransE', 'DistMult', 'ComplEx', 'RotatE').
        """

        # Import torch modules here
        global torch, nn, optim
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Check device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_df = train_df
        self.test_df = test_df
        self.entity_embeddings = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in entity_embeddings.items()}
        self.predicate_embeddings = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in predicate_embeddings.items()}
        self.model = model
        self.factor = factor

        # Precompute useful structures
        self.entity_list = np.array(list(self.entity_embeddings.keys()))
        self.all_entity_embeddings = torch.stack(list(self.entity_embeddings.values())).to(self.device)  # Precomputed matrix
        self.existing_triples = set(zip(train_df['head'], train_df['relation'], train_df['tail'])) | set(zip(test_df['head'], test_df['relation'], test_df['tail']))  # Combined lookup set

    def get_embedding(self, triple):
        """
        Retrieve the concatenated embeddings for a given triple (head, relation, tail).
        """
        head_emb = self.entity_embeddings[triple['head']]
        rel_emb = self.predicate_embeddings[triple['relation']]
        tail_emb = self.entity_embeddings[triple['tail']]
        triple_embedding = torch.cat([head_emb, rel_emb, tail_emb])
        return triple_embedding

    def compute_scores(self, head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for):
        """
        Compute scores based on the specified embedding model.
        """
        if self.model == "TransE":
            target_emb = head_emb + rel_emb if find_for == "tail" else tail_emb - rel_emb
            return torch.sum((all_entity_embeddings - target_emb) ** 2, dim=1)  # Squared distance
        elif self.model == "DistMult":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb * rel_emb
            return torch.matmul(all_entity_embeddings, target_emb)  # Dot product
        elif self.model == "ComplEx":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb * torch.conj(rel_emb)
            return torch.real(torch.matmul(all_entity_embeddings, torch.conj(target_emb)))  # Real part of dot product
        elif self.model == "RotatE":
            target_emb = head_emb * rel_emb if find_for == "tail" else tail_emb / rel_emb
            return torch.norm(all_entity_embeddings - target_emb, dim=1)  # Distance
        else:
            raise ValueError("Unknown model type")

    def training_find_top_k_candidates_filtered(self, head, relation, tail, n, find_for="tail"):
        """
        Find top-k candidates for training with filtering.
        """
        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]

        # Compute scores
        scores = self.compute_scores(head_emb, rel_emb, tail_emb, self.all_entity_embeddings, find_for)

        # Get top-k indices
        if self.model in ["DistMult", "ComplEx"]:
            _, topk_indices = torch.topk(scores, n * self.factor, largest=True)
        else:
            _, topk_indices = torch.topk(scores, n * self.factor, largest=False)

        # Filter top-k candidates
        filtered_top_k_candidates = []
        for idx in topk_indices:
            candidate_entity = self.entity_list[idx.item()]

            if (find_for == "tail" and candidate_entity == head) or (find_for == "head" and candidate_entity == tail):
                continue

            candidate_triple = (head, relation, candidate_entity) if find_for == "tail" else (candidate_entity, relation, tail)

            if candidate_triple in self.existing_triples:
                continue

            filtered_top_k_candidates.append((candidate_entity, scores[idx].item()))

            if len(filtered_top_k_candidates) >= n:
                break

        return filtered_top_k_candidates

    def create_training_data_filtered(self, n=1, creating_for="head"):
        """
        Create training data by filtering top-k candidates.
        """
        unique_triples = set()
        training_data = []

        for head, relation, tail in tqdm(self.train_df.itertuples(index=False), total=len(self.train_df), desc="Processing triples", unit="triple"):
            top_k_candidates = self.training_find_top_k_candidates_filtered(head, relation, tail, n, find_for=creating_for)

            if creating_for == "head":
                new_triples = {(candidate_head, relation, tail) for candidate_head, _ in top_k_candidates}
            else:
                new_triples = {(head, relation, candidate_tail) for candidate_tail, _ in top_k_candidates}

            for triple in new_triples:
                if triple not in unique_triples:
                    unique_triples.add(triple)
                    training_data.append({'head': triple[0], 'relation': triple[1], 'tail': triple[2], 'label': 0})

        return pd.DataFrame(training_data)

    def create_random_negative_samples(self, n=1, creating_for="head", seed=42):
        """
        Create negative samples by randomly selecting entities instead of finding top-k candidates.
        """
        if seed is not None:
            random.seed(seed)

        training_data = []
        entity_list = self.entity_list.tolist()

        for head, relation, tail in self.train_df.itertuples(index=False):
            new_triples = set()

            while len(new_triples) < n:
                random_entity = random.choice(entity_list)

                if creating_for == "head":
                    if random_entity == head or (random_entity, relation, tail) in self.existing_triples:
                        continue
                    new_triples.add((random_entity, relation, tail))
                else:
                    if random_entity == tail or (head, relation, random_entity) in self.existing_triples:
                        continue
                    new_triples.add((head, relation, random_entity))

            training_data.extend({'head': h, 'relation': r, 'tail': t, 'label': 0} for h, r, t in new_triples)

        return pd.DataFrame(training_data)