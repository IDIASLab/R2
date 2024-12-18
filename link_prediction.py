import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.base import BaseEstimator
import re

class TripleEvaluator:
    """
    A class for evaluating and working with (head, relation, tail) triples in a knowledge graph using various embedding models.

    This class can:
    - Evaluate triples in a knowledge graph context with embeddings produced by embedding models (TransE, DistMult, ComplEx, RotatE).
    - Precompute and store top-k candidate predictions for both head and tail given a triple pattern.
    - Precompute and store valid answers (entities) for given (head, relation) and (tail, relation) pairs.
    - Generate negative samples for training by filtering out known triples.
    - Evaluate model performance by checking how often the true entity is found in the top-k ranked candidates.
    - Aggregate and re-rank predictions from multiple models.

    Attributes
    ----------
    train_set_df : pd.DataFrame
        DataFrame containing training triples with columns ['head', 'relation', 'tail'].
    test_set_df : pd.DataFrame
        DataFrame containing test triples with columns ['head', 'relation', 'tail'].
    entity_embeddings : dict
        Dictionary mapping each entity to its embedding vector (np.ndarray).
    predicate_embeddings : dict
        Dictionary mapping each relation (predicate) to its embedding vector (np.ndarray).
    train_triples_set : set
        A set of (head, relation, tail) tuples representing the training triples for fast lookup.
    test_triples_set : set
        A set of (head, relation, tail) tuples representing the test triples for fast lookup.
    model : str
        The specified embedding model type: 'TransE', 'DistMult', 'ComplEx', or 'RotatE'.
    k : int
        The number of top candidates to store for precomputed predictions.
    precomputed_top_k_tail : dict
        A dictionary mapping (head, relation) to a list of top-k tail candidates.
    precomputed_top_k_head : dict
        A dictionary mapping (relation, tail) to a list of top-k head candidates.
    head_relation_answers : dict
        A dictionary mapping (head, relation) to the list of valid tail entities.
    tail_relation_answers : dict
        A dictionary mapping (tail, relation) to the list of valid head entities.
    """

    def __init__(self, entity_embeddings, predicate_embeddings, train_df, test_df, model, k=10,
                 precomputed_top_k_tail=None, precomputed_top_k_head=None,
                 head_relation_answers=None, tail_relation_answers=None):
        """
        Initialize the TripleEvaluator with provided embeddings, dataframes, and configurations.

        Parameters
        ----------
        entity_embeddings : dict
            Dictionary mapping each entity to a numpy embedding vector.
        predicate_embeddings : dict
            Dictionary mapping each relation (predicate) to a numpy embedding vector.
        train_df : pd.DataFrame
            DataFrame of training triples with columns ['head', 'relation', 'tail'].
        test_df : pd.DataFrame
            DataFrame of test triples with columns ['head', 'relation', 'tail'].
        model : str
            The embedding model to use: 'TransE', 'DistMult', 'ComplEx', or 'RotatE'.
        k : int
            The number of top candidates to be precomputed for head and tail.
        precomputed_top_k_tail : dict, optional
            Precomputed top-k tails for (head, relation) pairs. If None, it will be computed.
        precomputed_top_k_head : dict, optional
            Precomputed top-k heads for (relation, tail) pairs. If None, it will be computed.
        head_relation_answers : dict, optional
            Precomputed answers mapping (head, relation) to a set/list of valid tails. If None, it will be computed.
        tail_relation_answers : dict, optional
            Precomputed answers mapping (tail, relation) to a set/list of valid heads. If None, it will be computed.
        """
        # Store the dataframes and embeddings
        self.train_set_df = train_df
        self.test_set_df = test_df
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        # Create sets of triples for O(1) lookup during filtering
        self.train_triples_set = self.create_set(train_df)
        self.test_triples_set = self.create_set(test_df)

        # Store model type and top-k parameter
        self.model = model
        self.k = k

        # Handle precomputed top-k predictions for tail and head
        self.precomputed_top_k_tail = precomputed_top_k_tail if precomputed_top_k_tail is not None else {}
        self.precomputed_top_k_head = precomputed_top_k_head if precomputed_top_k_head is not None else {}

        # If top-k were not provided, compute them for the test set
        if not precomputed_top_k_tail or not precomputed_top_k_head:
            self._precompute_top_k_predictions(test_df)

        # Handle precomputed answer dictionaries
        self.head_relation_answers = head_relation_answers if head_relation_answers is not None else {}
        self.tail_relation_answers = tail_relation_answers if tail_relation_answers is not None else {}

        # If answers not provided, compute them from test set
        if not head_relation_answers or not tail_relation_answers:
            self._precompute_answers(test_df)

    def extract_state(self):
        """
        Extract the current state of the evaluator as a dictionary.

        Returns
        -------
        dict
            Dictionary containing all necessary data to re-initialize a new TripleEvaluator with identical state.
        """
        return {
            'entity_embeddings': self.entity_embeddings,
            'predicate_embeddings': self.predicate_embeddings,
            'train_set_df': pd.DataFrame(list(self.train_triples_set), columns=['head', 'relation', 'tail']),
            'test_set_df': pd.DataFrame(list(self.test_triples_set), columns=['head', 'relation', 'tail']),
            'k': self.k,
            'precomputed_top_k_tail': self.precomputed_top_k_tail,
            'precomputed_top_k_head': self.precomputed_top_k_head,
            'head_relation_answers': self.head_relation_answers,
            'tail_relation_answers': self.tail_relation_answers
        }

    def _precompute_answers(self, test_df):
        """
        Precompute answer sets for (head, relation) and (tail, relation) pairs from the test triples.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame of test triples with columns ['head', 'relation', 'tail'].

        Notes
        -----
        This method populates:
        - self.head_relation_answers: (head, relation) -> set of valid tail entities
        - self.tail_relation_answers: (tail, relation) -> set of valid head entities
        """
        # Iterate over test triples to populate answer dictionaries
        for row in test_df.itertuples(index=False):
            head, relation, tail = row.head, row.relation, row.tail

            # Populate head_relation_answers
            if (head, relation) not in self.head_relation_answers:
                self.head_relation_answers[(head, relation)] = set()
            self.head_relation_answers[(head, relation)].add(tail)

            # Populate tail_relation_answers
            if (tail, relation) not in self.tail_relation_answers:
                self.tail_relation_answers[(tail, relation)] = set()
            self.tail_relation_answers[(tail, relation)].add(head)

        # Convert sets to lists for consistency
        self.head_relation_answers = {key: list(values) for key, values in self.head_relation_answers.items()}
        self.tail_relation_answers = {key: list(values) for key, values in self.tail_relation_answers.items()}    

    def _precompute_top_k_predictions(self, test_df):
        """
        Precompute top-k candidate entities for heads and tails for each test triple pattern.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame of test triples with columns ['head', 'relation', 'tail'].

        Notes
        -----
        For each (head, relation, tail) in the test set:
        - Find top-k tail candidates given (head, relation) and store in self.precomputed_top_k_tail.
        - Find top-k head candidates given (relation, tail) and store in self.precomputed_top_k_head.
        """
        # Use tqdm for progress visualization
        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="Precomputing top-k predictions"):
            head, relation, tail = row.head, row.relation, row.tail

            # Skip if embeddings are missing for any component
            if head not in self.entity_embeddings or tail not in self.entity_embeddings or relation not in self.predicate_embeddings:
                continue

            # Precompute top-k tails for (head, relation)
            top_k_tail = self.find_top_k_candidates_filtered(head, relation, tail, self.k, find_for="tail")
            self.precomputed_top_k_tail[(head, relation)] = top_k_tail

            # Precompute top-k heads for (relation, tail)
            top_k_head = self.find_top_k_candidates_filtered(head, relation, tail, self.k, find_for="head")
            self.precomputed_top_k_head[(relation, tail)] = top_k_head

    def _normalize_embeddings(self, embeddings_dict):
        """
        Normalize embeddings by their L2 norm.

        Parameters
        ----------
        embeddings_dict : dict
            Dictionary of embeddings {key: np.ndarray}.

        Returns
        -------
        dict
            Dictionary of L2-normalized embeddings.
        """
        # Perform L2 normalization for each embedding vector
        normalized_embeddings = {
            key: emb / np.linalg.norm(emb) for key, emb in embeddings_dict.items()
        }
        return normalized_embeddings

    def create_set(self, triples_df):
        """
        Convert a DataFrame of triples into a set of (head, relation, tail) tuples.

        Parameters
        ----------
        triples_df : pd.DataFrame
            DataFrame with columns ['head', 'relation', 'tail'].

        Returns
        -------
        set
            A set of (head, relation, tail) tuples for O(1) membership checking.
        """
        return set(zip(triples_df['head'], triples_df['relation'], triples_df['tail']))

    def get_embedding(self, triple):
        """
        Get the concatenated embedding vector for a given triple.

        Parameters
        ----------
        triple : dict
            Dictionary with keys: {'head', 'relation', 'tail'}.

        Returns
        -------
        np.ndarray
            Concatenated embedding of [head_emb, relation_emb, tail_emb].

        Raises
        ------
        KeyError
            If any of head, relation, or tail is missing in embeddings.
        """
        # Check existence of embeddings
        if triple['head'] not in self.entity_embeddings:
            raise KeyError(f"Head entity {triple['head']} is missing from entity embeddings.")
        if triple['relation'] not in self.predicate_embeddings:
            raise KeyError(f"Relation {triple['relation']} is missing from predicate embeddings.")
        if triple['tail'] not in self.entity_embeddings:
            raise KeyError(f"Tail entity {triple['tail']} is missing from entity embeddings.")

        # Retrieve embeddings and concatenate
        head_emb = self.entity_embeddings[triple['head']]
        rel_emb = self.predicate_embeddings[triple['relation']]
        tail_emb = self.entity_embeddings[triple['tail']]
        triple_embedding = np.concatenate([head_emb, rel_emb, tail_emb])

        return triple_embedding
    

    def compute_scores(self, head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for):
        """
        Compute scores for candidate entities given the model type.

        Parameters
        ----------
        head_emb : np.ndarray
            Embedding of the head entity.
        rel_emb : np.ndarray
            Embedding of the relation.
        tail_emb : np.ndarray
            Embedding of the tail entity.
        all_entity_embeddings : np.ndarray
            Stacked array of all entity embeddings.
        find_for : str
            'tail' if predicting tail entities given (head, relation),
            'head' if predicting head entities given (relation, tail).

        Returns
        -------
        np.ndarray
            Array of scores, one for each candidate entity. Higher or lower is better depending on the model.

        Raises
        ------
        ValueError
            If the model type is not recognized.
        """
        # Compute target embeddings depending on the model and what we are predicting
        if self.model == "TransE":
            # TransE: distance-based model, lower is better
            target_emb = (head_emb + rel_emb) if find_for == "tail" else (tail_emb - rel_emb)
            return np.linalg.norm(all_entity_embeddings - target_emb, axis=1)
        elif self.model == "DistMult":
            # DistMult: similarity-based model, higher is better
            target_emb = (head_emb * rel_emb) if find_for == "tail" else (tail_emb * rel_emb)
            return np.dot(all_entity_embeddings, target_emb)
        elif self.model == "ComplEx":
            # ComplEx: complex embeddings, we take the real part of the dot product
            target_emb = (head_emb * rel_emb) if find_for == "tail" else (tail_emb * np.conj(rel_emb))
            return np.real(np.dot(all_entity_embeddings, np.conj(target_emb)))
        elif self.model == "RotatE":
            # RotatE: distance-based model in a complex space
            target_emb = (head_emb * rel_emb) if find_for == "tail" else (tail_emb / rel_emb)
            return np.linalg.norm(all_entity_embeddings - target_emb, axis=1)
        else:
            raise ValueError("Unknown model type")

    def training_find_top_k_candidates_filtered(self, head, relation, tail, n, find_for="tail"):
        """
        Find top-k negative candidate entities for training, filtering known triples to avoid contamination.

        Parameters
        ----------
        head : str
            Head entity.
        relation : str
            Relation.
        tail : str
            Tail entity.
        n : int
            Number of candidates to generate.
        find_for : str, optional
            Either 'head' or 'tail', to determine which part of the triple to replace.

        Returns
        -------
        list
            A list of (candidate_entity, score) tuples representing top-k negative examples.

        Raises
        ------
        ValueError
            If embeddings are missing or find_for is invalid.
        """
        # Check existence of embeddings
        if head not in self.entity_embeddings or relation not in self.predicate_embeddings:
            raise ValueError(f"Entity or relation is missing from the embeddings: {head}, {relation}")
        if tail not in self.entity_embeddings:
            raise ValueError(f"Tail entity is missing from the embeddings: {tail}")

        # Fetch embeddings
        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]

        # Compute scores for all entities
        all_entity_embeddings = np.vstack(list(self.entity_embeddings.values()))
        scores = self.compute_scores(head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for)

        # Sort indices by scores. For DistMult/ComplEx: higher is better (sort descending), else ascending
        top_k_indices = np.argsort(-scores if self.model in ["DistMult", "ComplEx"] else scores)

        filtered_top_k_candidates = []
        keys_list = list(self.entity_embeddings.keys())

        # Filter out known triples and reflexive matches
        for idx in top_k_indices:
            candidate_entity = keys_list[idx]
            if find_for == "tail":
                if candidate_entity == head:
                    continue
                candidate_triple = (head, relation, candidate_entity)
            elif find_for == "head":
                if candidate_entity == tail:
                    continue
                candidate_triple = (candidate_entity, relation, tail)
            else:
                raise ValueError("find_for must be either 'head' or 'tail'")

            # Skip if the candidate triple exists in train/test set
            if candidate_triple in self.train_triples_set or candidate_triple in self.test_triples_set:
                continue

            filtered_top_k_candidates.append((candidate_entity, scores[idx]))
            if len(filtered_top_k_candidates) >= n:
                break

        return filtered_top_k_candidates
    
    def create_training_data_filtered(self, n=10, creating_for="head"):
        """
        Create filtered negative training samples by generating top-k candidates that do not form known triples.

        Parameters
        ----------
        n : int, optional
            Number of negative candidates per triple, by default 10.
        creating_for : str, optional
            'head' or 'tail', indicating which part to replace to create negatives.

        Returns
        -------
        pd.DataFrame
            DataFrame of generated negative samples with columns: ['head', 'relation', 'tail', 'label'].
            The label is 0 for negatives.
        """
        unique_triples = set()
        training_data = []

        # Iterate over each triple in the training set
        for row in tqdm(self.train_set_df.itertuples(index=False), total=len(self.train_set_df), desc="Processing triples", unit="triple"):
            head, relation, tail = row.head, row.relation, row.tail

            # Skip if embeddings are missing
            if head not in self.entity_embeddings or tail not in self.entity_embeddings or relation not in self.predicate_embeddings:
                continue

            # Generate negative candidates depending on whether we're replacing head or tail
            if creating_for == "head":
                top_k_candidates = self.training_find_top_k_candidates_filtered(head, relation, tail, n, find_for="head")
                for candidate_head, _ in top_k_candidates:
                    candidate_triple = (candidate_head, relation, tail)
                    if candidate_triple not in unique_triples:
                        unique_triples.add(candidate_triple)
                        training_data.append({
                            'head': candidate_head,
                            'relation': relation,
                            'tail': tail,
                            'label': 0
                        })
            elif creating_for == "tail":
                top_k_candidates = self.training_find_top_k_candidates_filtered(head, relation, tail, n, find_for="tail")
                for candidate_tail, _ in top_k_candidates:
                    candidate_triple = (head, relation, candidate_tail)
                    if candidate_triple not in unique_triples:
                        unique_triples.add(candidate_triple)
                        training_data.append({
                            'head': head,
                            'relation': relation,
                            'tail': candidate_tail,
                            'label': 0
                        })

        return pd.DataFrame(training_data)
    

    def find_top_k_candidates_filtered(self, head, relation, tail, k, find_for="tail"):
        """
        Retrieve top-k candidate entities for a given triple pattern, filtering out known triples.

        Parameters
        ----------
        head : str
            Head entity.
        relation : str
            Relation (predicate).
        tail : str
            Tail entity.
        k : int
            Number of top candidates to return.
        find_for : str, optional
            'tail' or 'head', determining which entity to predict.

        Returns
        -------
        list
            List of (candidate_entity, score) tuples for the top-k candidates.

        Raises
        ------
        ValueError
            If embeddings are missing or find_for is invalid.
        """
        # Check for embedding availability
        if head not in self.entity_embeddings or relation not in self.predicate_embeddings:
            raise ValueError(f"Entity or relation is missing from the embeddings: {head}, {relation}")
        if tail not in self.entity_embeddings:
            raise ValueError(f"Tail entity is missing from the embeddings: {tail}")

        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]

        # Stack all entity embeddings
        all_entity_embeddings = np.vstack(list(self.entity_embeddings.values()))
        # Compute scores based on model
        scores = self.compute_scores(head_emb, rel_emb, tail_emb, all_entity_embeddings, find_for)

        # Determine sorting order
        top_k_indices = np.argsort(-scores if self.model in ["DistMult", "ComplEx"] else scores)
        keys_list = list(self.entity_embeddings.keys())
        filtered_top_k_candidates = []

        # Filter out known triples
        for idx in top_k_indices:
            candidate_entity = keys_list[idx]
            if find_for == "tail":
                candidate_triple = (head, relation, candidate_entity)
                original_triple = (head, relation, tail)
            elif find_for == "head":
                candidate_triple = (candidate_entity, relation, tail)
                original_triple = (head, relation, tail)
            else:
                raise ValueError("find_for must be either 'head' or 'tail'")

            if candidate_triple in self.train_triples_set:
                continue

            filtered_top_k_candidates.append((candidate_entity, scores[idx]))
            if len(filtered_top_k_candidates) >= k:
                break

        return filtered_top_k_candidates

    def score_triple(self, head, relation, tail, model):
        """
        Score a given triple using a trained machine learning model (e.g., RandomForest).

        Parameters
        ----------
        head : str
            Head entity.
        relation : str
            Relation.
        tail : str
            Tail entity.
        model : BaseEstimator
            A trained scikit-learn style model with a predict method.

        Returns
        -------
        float
            The model-predicted score (e.g., probability or class label).

        Raises
        ------
        ValueError
            If any embeddings are missing for the triple components.
        """
        # Check embeddings
        if head not in self.entity_embeddings or tail not in self.entity_embeddings or relation not in self.predicate_embeddings:
            raise ValueError(f"One of the entities or the relation is missing from the embeddings: {head}, {relation}, {tail}")

        # Concatenate embeddings
        head_emb = self.entity_embeddings[head]
        rel_emb = self.predicate_embeddings[relation]
        tail_emb = self.entity_embeddings[tail]
        triple_embedding = np.concatenate([head_emb, rel_emb, tail_emb]).reshape(1, -1)

        # Use the model to predict score
        score = model.predict(triple_embedding)[0]
        return score

    def create_predictions(self, head, relation, tail, models, predict_for):
        """
        Create predictions from multiple models for a single triple.

        Parameters
        ----------
        head : str
            Head entity.
        relation : str
            Relation.
        tail : str
            Tail entity.
        models : list
            A list of models to use for predictions.
        predict_for : str
            'head' or 'tail', indicating what is being predicted.

        Returns
        -------
        list
            List of predictions from all provided models.
        """
        predictions = []
        for model in models:
            # If model is sklearn-like, use score_triple; otherwise call model's own method
            if isinstance(model, BaseEstimator):
                prediction = self.score_triple(head, relation, tail, model)
            else:
                prediction = model.score_triple(head, relation, tail, predict_for)
            predictions.append(prediction)
        return predictions

    def aggregate_predictions(self, predictions, method="majority"):
        """
        Aggregate multiple model predictions into a single prediction.

        Parameters
        ----------
        predictions : list
            List of binary predictions (0 or 1) from multiple models.
        method : str, optional
            Method of aggregation: 'majority', 'min_vote', or 'max_vote'.

        Returns
        -------
        float
            Aggregated prediction (0 or 1).
        """
        prediction_counts = Counter(predictions)

        if method == "majority":
            # Return the most common prediction
            return prediction_counts.most_common(1)[0][0]
        elif method == "min_vote":
            # Return the least common prediction
            return prediction_counts.most_common()[-1][0]
        elif method == "max_vote":
            # Return 1 if any model predicted 1, else 0
            return 1 if 1 in predictions else 0


    def evaluate_hitk_original(self, test_df, k, evaluate_for="tail"):
        """
        Evaluate hits@k on the test set using precomputed top-k candidates without additional classification.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test set of triples with columns ['head', 'relation', 'tail'].
        k : int
            Number of top candidates to consider.
        evaluate_for : str, optional
            'tail' or 'head', specifying which side to predict.

        Returns
        -------
        tuple
            (total_count, hit_count) where:
            - total_count: number of evaluated triples
            - hit_count: how many times the true entity is in the top-k predictions.
        """
        hit_count = 0
        total_count = 0

        # Iterate over test triples
        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if row.head not in self.entity_embeddings or row.tail not in self.entity_embeddings or row.relation not in self.predicate_embeddings:
                continue

            # Determine which entity to predict
            if evaluate_for == "tail":
                head = row.head
                relation = row.relation
                true_entity = row.tail
                top_k_candidates = self.precomputed_top_k_tail.get((head, relation), [])[:k]
            else:
                true_entity = row.head
                relation = row.relation
                tail = row.tail
                top_k_candidates = self.precomputed_top_k_head.get((relation, tail), [])[:k]

            predicted_entities = [entity for entity, _ in top_k_candidates]
            if true_entity in predicted_entities:
                hit_count += 1

            total_count += 1

        return total_count, hit_count


    def evaluate_hitk_original_2(self, test_df, k, evaluate_for="tail"):
        """
        Evaluate hits@k when multiple true entities may exist for a given pattern.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test set of triples.
        k : int
            Number of top candidates to consider.
        evaluate_for : str, optional
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count) as above, but now we check if any true entity (from a set) is in the top-k.
        """
        hit_count = 0
        total_count = 0

        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if row.head not in self.entity_embeddings or row.tail not in self.entity_embeddings or row.relation not in self.predicate_embeddings:
                continue

            # Retrieve all true entities for the pattern
            if evaluate_for == "tail":
                head = row.head
                relation = row.relation
                true_entities = self.head_relation_answers.get((head, relation), [])
                top_k_candidates = self.precomputed_top_k_tail.get((head, relation), [])[:k]
            else:
                tail = row.tail
                relation = row.relation
                true_entities = self.tail_relation_answers.get((tail, relation), [])
                top_k_candidates = self.precomputed_top_k_head.get((relation, tail), [])[:k]

            predicted_entities = {entity for entity, _ in top_k_candidates}
            # Check if any of the true entities is present
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1
            total_count += 1

        return total_count, hit_count


    def evaluate_hitk_original_2_on_patterns(self, k, evaluate_for="tail"):
        """
        Evaluate hits@k over patterns rather than individual triples, considering multiple possible true entities.

        Parameters
        ----------
        k : int
            Number of top candidates.
        evaluate_for : str, optional
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count) counting patterns instead of individual triples.
        """
        hit_count = 0
        total_count = 0

        selected_iterator = self.precomputed_top_k_tail if evaluate_for == 'tail' else self.precomputed_top_k_head

        # Iterate over patterns
        for key in tqdm(selected_iterator.keys(), total=len(selected_iterator), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if evaluate_for == "tail":
                head = key[0]
                relation = key[1]
                true_entities = self.head_relation_answers.get((head, relation), [])
                top_k_candidates = self.precomputed_top_k_tail.get((head, relation), [])[:k]
            else:
                relation = key[0]
                tail = key[1]
                true_entities = self.tail_relation_answers.get((tail, relation), [])
                top_k_candidates = self.precomputed_top_k_head.get((relation, tail), [])[:k]

            predicted_entities = {entity for entity, _ in top_k_candidates}
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1
            total_count += 1

        return total_count, hit_count


    def evaluate_hitk(self, test_df, k, models, ensemble_method='majority', evaluate_for="tail"):
        """
        Evaluate hits@k using multiple models to classify top-k candidates.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test DataFrame of triples.
        k : int
            Number of top candidates to consider.
        models : list
            List of trained models to predict candidate validity.
        ensemble_method : str, optional
            Aggregation method for multiple model predictions.
        evaluate_for : str, optional
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count) after filtering candidates by predicted scores.
        """
        hit_count = 0
        total_count = 0

        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if row.head not in self.entity_embeddings or row.tail not in self.entity_embeddings or row.relation not in self.predicate_embeddings:
                continue

            # Determine true entity and candidates
            if evaluate_for == "tail":
                head, relation, true_entity = row.head, row.relation, row.tail
                top_k_candidates = self.precomputed_top_k_tail.get((head, relation), [])[:k]
            else:
                true_entity = row.head
                relation = row.relation
                tail = row.tail
                top_k_candidates = self.precomputed_top_k_head.get((relation, tail), [])[:k]

            true_candidates = []
            # Classify each candidate using the ensemble of models
            for entity, _ in top_k_candidates:
                if evaluate_for == "tail":
                    predictions = self.create_predictions(head, relation, entity, models, evaluate_for)
                else:
                    predictions = self.create_predictions(entity, relation, tail, models, evaluate_for)

                score = self.aggregate_predictions(predictions, method=ensemble_method)
                # Keep only candidates predicted as true
                if score > 0.5:
                    true_candidates.append((entity, score))

            predicted_entities = [entity for entity, _ in true_candidates]
            if true_entity in predicted_entities:
                hit_count += 1
            total_count += 1

        return total_count, hit_count


    def evaluate_true_hitk(self, test_df, k, models, ensemble_method='majority', evaluate_for="tail"):
        """
        Evaluate hits@k by searching through candidates until k true ones are found or none left.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test DataFrame of triples.
        k : int
            Number of true candidates to collect before stopping.
        models : list
            Models for predicting candidate validity.
        ensemble_method : str
            Aggregation method of model predictions.
        evaluate_for : str
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count, incomplete_count) indicating how many patterns were evaluated,
            how many had a hit@k, and how many had fewer than k true candidates.
        """
        hit_count = 0
        total_count = 0
        incomplete_count = 0

        # Iterate over each test triple
        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if row.head not in self.entity_embeddings or row.tail not in self.entity_embeddings or row.relation not in self.predicate_embeddings:
                continue

            # Get all candidates
            if evaluate_for == "tail":
                head, relation, true_entity = row.head, row.relation, row.tail
                all_candidates = self.precomputed_top_k_tail.get((head, relation), [])
            else:
                true_entity = row.head
                relation = row.relation
                tail = row.tail
                all_candidates = self.precomputed_top_k_head.get((relation, tail), [])

            true_candidates = []
            # Classify candidates until we find k true ones or exhaust the list
            for entity, _ in all_candidates:
                if evaluate_for == "tail":
                    predictions = self.create_predictions(head, relation, entity, models, evaluate_for)
                else:
                    predictions = self.create_predictions(entity, relation, tail, models, evaluate_for)
                score = self.aggregate_predictions(predictions, method=ensemble_method)
                if score > 0.5:
                    true_candidates.append((entity, score))
                if len(true_candidates) >= k:
                    break

            # Check if we got fewer than k true candidates
            if len(true_candidates) < k:
                incomplete_count += 1

            predicted_entities = [entity for entity, _ in true_candidates]
            if true_entity in predicted_entities:
                hit_count += 1
            total_count += 1

        return total_count, hit_count, incomplete_count


    def evaluate_true_hitk_2(self, test_df, k, models, ensemble_method='majority', evaluate_for="tail"):
        """
        Similar to evaluate_true_hitk, but handles multiple possible true entities for each pattern.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test DataFrame.
        k : int
            Number of true candidates to find.
        models : list
            Models for predictions.
        ensemble_method : str
            Method of aggregating predictions.
        evaluate_for : str
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count, incomplete_count) as above, now considering multiple possible true entities.
        """
        hit_count = 0
        total_count = 0
        incomplete_count = 0

        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if row.head not in self.entity_embeddings or row.tail not in self.entity_embeddings or row.relation not in self.predicate_embeddings:
                continue

            # Retrieve all true entities for the pattern
            if evaluate_for == "tail":
                head, relation = row.head, row.relation
                true_entities = self.head_relation_answers.get((head, relation), [])
                all_candidates = self.precomputed_top_k_tail.get((head, relation), [])
            else:
                tail, relation = row.tail, row.relation
                true_entities = self.tail_relation_answers.get((tail, relation), [])
                all_candidates = self.precomputed_top_k_head.get((relation, tail), [])

            true_candidates = []
            # Collect up to k true candidates
            for entity, _ in all_candidates:
                if evaluate_for == "tail":
                    predictions = self.create_predictions(head, relation, entity, models, evaluate_for)
                else:
                    predictions = self.create_predictions(entity, relation, tail, models, evaluate_for)
                score = self.aggregate_predictions(predictions, method=ensemble_method)
                if score > 0.5:
                    true_candidates.append((entity, score))
                if len(true_candidates) >= k:
                    break

            if len(true_candidates) < k:
                incomplete_count += 1

            predicted_entities = {entity for entity, _ in true_candidates}
            # Check if any of the known true entities is predicted
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1
            total_count += 1

        return total_count, hit_count, incomplete_count


    def evaluate_true_hitk_2_on_patterns(self, k, models, ensemble_method='majority', evaluate_for="tail"):
        """
        Evaluate patterns rather than individual triples, considering multiple true entities and up to k true candidates.

        Parameters
        ----------
        k : int
            Number of true candidates required.
        models : list
            Models for prediction.
        ensemble_method : str
            Aggregation method.
        evaluate_for : str
            'tail' or 'head'.

        Returns
        -------
        tuple
            (total_count, hit_count, incomplete_count) for patterns.
        """
        hit_count = 0
        total_count = 0
        incomplete_count = 0

        selected_iterator = self.precomputed_top_k_tail if evaluate_for == 'tail' else self.precomputed_top_k_head

        # Iterate over unique patterns
        for key in tqdm(selected_iterator.keys(), total=len(selected_iterator), desc=f"Evaluating {evaluate_for.capitalize()} Predictions"):
            if evaluate_for == "tail":
                head = key[0]
                relation = key[1]
                true_entities = self.head_relation_answers.get((head, relation), [])
                all_candidates = self.precomputed_top_k_tail.get((head, relation), [])
            else:
                relation = key[0]
                tail = key[1]
                true_entities = self.tail_relation_answers.get((tail, relation), [])
                all_candidates = self.precomputed_top_k_head.get((relation, tail), [])

            true_candidates = []
            # Classify candidates to find up to k true ones
            for entity, _ in all_candidates:
                if evaluate_for == "tail":
                    predictions = self.create_predictions(head, relation, entity, models, evaluate_for)
                else:
                    predictions = self.create_predictions(entity, relation, tail, models, evaluate_for)
                score = self.aggregate_predictions(predictions, method=ensemble_method)
                if score > 0.5:
                    true_candidates.append((entity, score))
                if len(true_candidates) >= k:
                    break

            if len(true_candidates) < k:
                incomplete_count += 1

            predicted_entities = {entity for entity, _ in true_candidates}
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1
            total_count += 1

        return total_count, hit_count, incomplete_count
    
    def rerank(self, k, model, threshold=10, evaluate_for="tail", verbose=False):
        """
        Re-rank candidates using a trained classifier model and evaluate hits@k.

        Parameters
        ----------
        k : int
            Number of top-ranked candidates to consider after re-ranking.
        model : object
            A trained model with `predict_proba` method (e.g., a classifier).
        threshold : int, optional
            Maximum number of candidates to consider before selecting top-k, by default 10.
        evaluate_for : str, optional
            'tail' or 'head', indicating which entity we are predicting.
        verbose : bool, optional
            If True, show a progress bar, otherwise silent.

        Returns
        -------
        tuple
            (total_count, hit_count) after re-ranking and evaluating hits@k.
        """
        hit_count = 0
        total_count = 0

        # Choose the appropriate precomputed dictionary
        selected_iterator = self.precomputed_top_k_tail if evaluate_for == "tail" else self.precomputed_top_k_head
        if selected_iterator is None:
            raise ValueError("Invalid value for `evaluate_for`. Use 'tail' or 'head'.")

        # Optionally use tqdm for iteration
        iterator = (
            tqdm(selected_iterator.keys(), total=len(selected_iterator), desc=f"Evaluating {evaluate_for.capitalize()} Predictions")
            if verbose else selected_iterator.keys()
        )

        # Iterate over each pattern
        for key in iterator:
            if evaluate_for == "tail":
                head, relation = key
                true_entities = self.head_relation_answers.get((head, relation), [])
                all_candidates = selected_iterator.get((head, relation), [])[:threshold]

                # Prepare features for model prediction
                head_emb = self.entity_embeddings[head]
                rel_emb = self.predicate_embeddings[relation]
                features = [
                    np.concatenate([head_emb, rel_emb, self.entity_embeddings[entity]])
                    for entity, _ in all_candidates
                ]
            else:
                relation, tail = key
                true_entities = self.tail_relation_answers.get((tail, relation), [])
                all_candidates = selected_iterator.get((relation, tail), [])[:threshold]

                tail_emb = self.entity_embeddings[tail]
                rel_emb = self.predicate_embeddings[relation]
                features = [
                    np.concatenate([self.entity_embeddings[entity], rel_emb, tail_emb])
                    for entity, _ in all_candidates
                ]

            candidate_entities = [entity for entity, _ in all_candidates]

            if len(features) == 0:
                # No candidates available
                total_count += 1
                continue

            features_array = np.array(features)
            # Use the classifier model to predict probabilities
            probabilities = model.predict_proba(features_array)[:, 1]

            # Re-rank candidates by predicted probability
            reranked_candidates = sorted(
                zip(candidate_entities, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            # Take top-k after re-ranking
            predicted_entities = {entity for entity, _ in reranked_candidates[:k]}

            # Check if any true entity is among the top-k
            if any(entity in predicted_entities for entity in true_entities):
                hit_count += 1

            total_count += 1

        return total_count, hit_count