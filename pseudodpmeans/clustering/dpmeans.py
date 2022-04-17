# Copyright 2021 Bloomberg L.P.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Edited 2022 Raphael Du Sablon
# DASC 6040 Final Project
# East Carolina University

import logging
import random
import time
import uuid

from typing import Any, Dict, FrozenSet, List, Tuple

import numpy as np
import scipy.spatial

from pseudodpmeans.clustering.clusteringcomponents import Cluster, ClusteringModel

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


class FanaticClusterModel(ClusteringModel):
    def consume_config(self, config: Dict[str, Any]) -> None:
        """Consume the configuration with the hyperparameters.

        Args:
            config: configuration containing the hyperparameters.

        Returns:
            (nothing)
        """
        self.clustering_lambda = config["clustering_lambda"]
        self.token_probability_threshold = config["token_probability_threshold"]
        self.max_num_clusters = config["max_num_clusters"]
        self.distance_metric = config["distance_metric"]
        self.merge_close_clusters_max_iterations = config[
            "merge_close_clusters_max_iterations"
        ]
        self.merge_close_clusters_lambda_fraction = config[
            "merge_close_clusters_lambda_fraction"
        ]
        self.batch_size = config["batch_size"]
        self.min_cluster_size = config["min_cluster_size"]
        self.max_clustering_time = config["max_clustering_time"]
        
    def initialize_clustering(self) -> Tuple[np.ndarray, List]:
        """
        Initialize the clustering algorithm.
    
        Args:
            (nothing)
    
        Returns:
            cluster_vectors: keeps the centroid of each cluster
            document_keys: the frozenset key associated with each document
        """
    
        # initialize all documents to belong to same cluster
        self.clusters = []
        cluster_id = uuid.uuid4().hex
        documents = list(self.documents.values())
        cluster = Cluster(cluster_id, documents)
        cluster.calculate_center()
        self.clusters.append(cluster)
    
        cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)
    
        # randomly shuffle document order
        document_keys = list(self.documents.keys())
        random.shuffle(document_keys)
        
        
        #find the average distance   
        i = 1
        adists = 0
        for doc_i, document_key in enumerate(document_keys):
            document = self.documents[document_key]

            # find closest cluster
            adists = adists + scipy.spatial.distance.cdist(
                cluster_vectors, [document.vector], metric=self.distance_metric
            ).flatten()
            i = i + 1
        adist = adists[0]/i
        print("Average Distance from centroid: ")
        print(adist)
        
        
        # clear out document lists in clusters
        cluster.documents.clear()
                
        return cluster_vectors, document_keys, adist
    
    
    # cluster
    
    
    def cluster(self, seed: int) -> Dict[str, Any]:
        """MAIN driver that performs a FANATIC clustering against a set of input documents.
        See Silburt et al. (2021) published in EMNLP for more details.
    
        Args:
            seed: random seed to shuffle the document order.
    
        Returns:
            stats: stats object containing timing of the clustering.
        """
    
        # initialize
        start_time = time.time()
        logger.info(f"using random seed={seed}")
        random.seed(seed)
        cluster_vectors, document_keys, averagedistance = self.initialize_clustering()
        
    
        # MAIN LOOP: Perform pseudodpmeans clustering until convergence or time limit reached
        while True:
            # check for time limit
            if (time.time() - start_time) > self.max_clustering_time:
                logger.info("Reached time limit! Terminating")
                break
    
            logger.info("Number of clusters: {}".format(len(self.clusters)))
            
            # clear out document lists in clusters
            for cluster in self.clusters:
                cluster.documents.clear()
            loopiterations = 0
                
            # document loop
            for doc_i, document_key in enumerate(document_keys):
                document = self.documents[document_key]
    
                loopiterations = loopiterations + 1
    
                # find closest cluster
                dists = scipy.spatial.distance.cdist(
                    cluster_vectors, [document.vector], metric=self.distance_metric
                ).flatten()
                (filter_idx,) = np.where(
                    dists < (averagedistance * (self.clustering_lambda + 1)) #this is a modification of the lambda argument to improve clustering
                )  # filter by lambda
                all_idx = filter_idx[np.argsort(dists[filter_idx])]
                try:
                    # filter by token probability
                    idx = next(
                        (
                            i
                            for i in all_idx
                            if sum(
                                self.clusters[i].token_probability.get(t, 0)
                                for t in document.tokens
                            )
                            / float(len(document.tokens))
                            >= self.token_probability_threshold
                        )
                    )
                    cluster = self.clusters[idx]
                    cluster.documents.append(document)
                except StopIteration:
                    # create new cluster containing this document if min distance exceeds clustering threshold
                    # or token probability was too low and there are less than max_num_clusters

                    loopiterations = 0
                    cluster_id = uuid.uuid4().hex
                    documents = [document]
                    cluster = Cluster(cluster_id, documents)
                    cluster.calculate_center()
                    self.clusters.append(cluster)
                    cluster_vectors = np.vstack(
                        (cluster_vectors, cluster.center))
                    
                 
            # randomize order of looking at documents
            #random.shuffle(document_keys)
    
            # check if all documents have been assigned to a cluster
            if loopiterations == len(document_keys):
                logger.info(
                    f"Clustering metric hasnt improved by at least {self._convergence_improvement_threshold} "
                    f"in {self._patience} iterations. Terminating."
                )
                
                break
            
        
        # filter out empty clusters
        self.clusters = [
            cluster for cluster in self.clusters if len(cluster.documents) > 0
        ]
    
        # clustering is over, perform final assignment of documents to clusters
        for cluster in self.clusters:
            cluster.documents.clear()
    
        self.assign_documents_to_fixed_clusters(
            document_keys=document_keys,
            cluster_vectors=cluster_vectors,
            clustering_lambda=self.clustering_lambda,
            token_probability_threshold=self.token_probability_threshold,
            distance_metric=self.distance_metric,
            flag_update_documents=True,
            batch_size=self.batch_size,
        )
        
        
        # filter out empty clusters
        self.clusters = [
            cluster for cluster in self.clusters if len(cluster.documents) > 0
        ]
    
    
    
        # cluster stats object
        self.stats["cluster_time"] = time.time() - start_time
        logger.info(
            f"FANATIC clustering took {self.stats['cluster_time']} seconds")
    
        return self.stats

    def assign_documents_to_fixed_clusters(
        self,
        document_keys: List[FrozenSet],
        cluster_vectors: np.ndarray,
        clustering_lambda: float,
        token_probability_threshold: float,
        distance_metric: str,
        flag_update_documents: bool,
        batch_size: int = 150000,
    ) -> None:
        """
        Assigns (the remaining) documents to static clusters, i.e. max_num_clusters has been reached and
        no new clusters can be made. Optimized to be faster than the single-document-per-loop way.

        Args:
            document_keys: The (remaining) document keys that will be assigned to the fixed clusters
            cluster_vectors: array of cluster centers
            clustering_lambda: "lambda" parameter that determines cluster size.
            token_probability_threshold: The token probability required to add a document to existing cluster.
            distance_metric: metric used for calculating distance.
            flag_update_documents: if True, update document objects too (only set to True for final document assignment)
            batch_size: How many documents to cdist at a time (more docs = more memory needed)

        Returns:
            (nothing)
        """
        # init
        cluster_idx = np.arange(len(cluster_vectors))
        n_documents = len(document_keys)
        if batch_size is None:
            batch_size = n_documents

        # go through documents in batches
        for i in range(0, n_documents, batch_size):
            document_keys_batch = document_keys[i: i + batch_size]
            document_vectors_batch = [
                self.documents[document_key].vector
                for document_key in document_keys_batch
            ]

            # find distances of all documents to clusters in batch
            dists_batch = scipy.spatial.distance.cdist(
                document_vectors_batch, cluster_vectors, metric=distance_metric
            )
            filter_idx_batch = (
                dists_batch < clustering_lambda
            )  # boolean 2D array filtering out < clustering_lambda
            for j, document_key in enumerate(document_keys_batch):
                dists_below_lamda = dists_batch[j][
                    filter_idx_batch[j]
                ]  # keep only dists < lambda
                cluster_idx_below_lamda = cluster_idx[
                    filter_idx_batch[j]
                ]  # and get corresponding cluster indices
                sorted_dummy_idx = np.argsort(
                    dists_below_lamda.flatten()
                )  # sort indices by distance, yields "dummy" indices
                all_idx = cluster_idx_below_lamda[
                    sorted_dummy_idx
                ]  # map dummy to original cluster idx again
                document = self.documents[document_key]
                try:
                    idx = next(
                        (
                            k
                            for k in all_idx
                            if sum(
                                self.clusters[k].token_probability.get(t, 0)
                                for t in document.tokens
                            )
                            / float(len(document.tokens))
                            >= token_probability_threshold
                        )
                    )
                    cluster = self.clusters[idx]
                    cluster.documents.append(document)
                    if flag_update_documents is True:
                        min_dist = dists_batch[j, idx]
                        document.cluster = cluster
                        document.cluster_id = cluster.cluster_id
                        document.dist_to_cluster_center = min_dist
                except StopIteration:
                    if flag_update_documents is True:
                        document.cluster = None
                        document.cluster_id = None
                        document.dist_to_cluster_center = -1
