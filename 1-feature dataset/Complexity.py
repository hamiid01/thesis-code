import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

np.random.seed(0)

class Complexity:
    def __init__(self, X, y, distance_func="euclidean"):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)
        self.classes = np.unique(self.y)
        assert len(self.classes) == 2, "Dataset must be a 2-class dataset."
        self.dist_matrix = self.__calculate_distance_matrix(self.X, distance_func)
        self.class_count = self.__count_class_instances()
        self.class_inxs = self.__get_class_inxs()

    def __calculate_distance_matrix(self, X, distance_func="euclidean"):
        if distance_func == "euclidean":
            dist_matrix = squareform(pdist(X, metric=distance_func))
        else:
            raise ValueError("Unsupported distance function")
        return dist_matrix

    def __count_class_instances(self):
        return np.bincount(self.y)

    def __get_class_inxs(self):
        class_inxs = []
        for cls in self.classes:
            class_inxs.append(np.where(self.y == cls)[0])
        return class_inxs

    def F2(self, epsilon=1e-6):
        # Identify majority and minority classes
        majority_class_idx = np.argmax(self.class_count)
        minority_class_idx = np.argmin(self.class_count)

        # Get samples for each class
        sample_majority = self.X[self.class_inxs[majority_class_idx]]
        sample_minority = self.X[self.class_inxs[minority_class_idx]]

        # Calculate max and min values across features for both classes
        max_majority = np.max(sample_majority, axis=0)
        max_minority = np.max(sample_minority, axis=0)
        min_majority = np.min(sample_majority, axis=0)
        min_minority = np.min(sample_minority, axis=0)

        # Compute intermediate vectors for F2 calculation
        maxmax = np.maximum(max_majority, max_minority)
        maxmin = np.maximum(min_majority, min_minority)
        minmin = np.minimum(min_majority, min_minority)
        minmax = np.minimum(max_majority, max_minority)

        # Calculate F2 measure for majority class
        numer_majority = np.maximum(0.0, minmax - maxmin) + epsilon
        denom = maxmax - minmin
        denom = np.where(denom == 0, epsilon, denom)  # Avoid division by zero
        f2_majority = np.prod(numer_majority / denom)

        # Calculate F2 measure for minority class
        numer_minority = np.maximum(0.0, minmax - maxmin) + epsilon
        f2_minority = np.prod(numer_minority / denom)

        return f2_majority, f2_minority

    def N2(self):
        nemo = 0
        deno = 0
        for i in range(len(self.X)):
            same_class_distances = [self.dist_matrix[i, j] for j in range(len(self.X)) if self.y[i] == self.y[j] and i != j]
            different_class_distances = [self.dist_matrix[i, j] for j in range(len(self.X)) if self.y[i] != self.y[j]]

            if same_class_distances:
                nemo += min(same_class_distances)
            if different_class_distances:
                deno += min(different_class_distances)

        if deno == 0:
            return float('inf')  # Avoid division by zero if no different class neighbors exist

        return nemo / deno

    def N2_adapted(self):
        n2_adapted = {}
        for cls in self.classes:
            nemo2 = 0
            deno2 = 0
            class_indices = np.where(self.y == cls)[0]
            for i in class_indices:
                same_class_distances = [self.dist_matrix[i, j] for j in range(len(self.X)) if self.y[i] == self.y[j] and i != j]
                different_class_distances = [self.dist_matrix[i, j] for j in range(len(self.X)) if self.y[i] != self.y[j]]

                if same_class_distances:
                    nemo2 += min(same_class_distances)
                if different_class_distances:
                    deno2 += min(different_class_distances)

            if deno2 == 0:
                n2_adapted[cls] = float('inf')  # Avoid division by zero if no different class neighbors exist
            else:
                n2_adapted[cls] = nemo2 / deno2

        return n2_adapted

    def N1(self):
        # Build the MST
        mst = minimum_spanning_tree(self.dist_matrix).toarray()
        n1_sum = 0
        for i in range(len(mst)):
            for j in range(len(mst)):
                if mst[i, j] != 0 and self.y[i] != self.y[j]:
                    n1_sum += 1
        return n1_sum / len(self.X)

    def N1_adapted(self):
        mst = minimum_spanning_tree(self.dist_matrix).toarray()
        n1_adapted = {}
        for cls in self.classes:
            n1_sum_class = 0
            class_indices = np.where(self.y == cls)[0]
            for i in class_indices:
                for j in range(len(mst)):
                    if mst[i, j] != 0 and self.y[j] != cls:
                        n1_sum_class += 1
            n1_adapted[cls] = n1_sum_class / len(class_indices)
        return n1_adapted

    def N3(self):
        n3_sum = 0
        for i in range(len(self.X)):
            distances = self.dist_matrix[i]
            nearest_neighbor_index = np.argmin([distances[j] if i != j else np.inf for j in range(len(distances))])
            if self.y[i] != self.y[nearest_neighbor_index]:
                n3_sum += 1
        return n3_sum / len(self.X)

    def N3_adapted(self):
        n3_adapted = {}
        for cls in self.classes:
            n3_sum_class = 0
            class_indices = np.where(self.y == cls)[0]
            for i in class_indices:
                distances = self.dist_matrix[i]
                nearest_neighbor_index = np.argmin([distances[j] if i != j else np.inf for j in range(len(distances))])
                if self.y[nearest_neighbor_index] != cls:
                    n3_sum_class += 1
            n3_adapted[cls] = n3_sum_class / len(class_indices)
        return n3_adapted

    def CM(self, k=5):
        cm_sum = 0
        for i in range(len(self.X)):
            distances = self.dist_matrix[i]
            k_nearest_neighbors = np.argsort(distances)[:k]
            different_class_count = np.sum(self.y[k_nearest_neighbors] != self.y[i])
            if different_class_count > k / 2:
                cm_sum += 1
        return cm_sum / len(self.X)

    def CM_adapted(self, k=5):
        cm_adapted = {}
        for cls in self.classes:
            cm_sum_class = 0
            class_indices = np.where(self.y == cls)[0]
            for i in class_indices:
                distances = self.dist_matrix[i]
                k_nearest_neighbors = np.argsort(distances)[:k]
                different_class_count = np.sum(self.y[k_nearest_neighbors] != cls)
                if different_class_count > 0:
                    cm_sum_class += 1
            cm_adapted[cls] = cm_sum_class / len(class_indices)
        return cm_adapted

    def __find_nearest_opposite_class(self, i):
        distances = self.dist_matrix[i]
        nearest_enemy_index = np.argmin([distances[j] if self.y[i] != self.y[j] else np.inf for j in range(len(distances))])
        return nearest_enemy_index, distances[nearest_enemy_index]

    def __get_hypersphere_counts(self):
        radius = np.zeros(len(self.X))
        for i in range(len(self.X)):
            _, nearest_enemy_distance = self.__find_nearest_opposite_class(i)
            radius[i] = nearest_enemy_distance

        return radius

    def __remove_overlapping_hyperspheres(self, radius):
        sorted_indices = np.argsort(radius)
        sphere_counts = np.ones(len(self.X), dtype=int)

        for i, idx1 in enumerate(sorted_indices[:-1]):
            for idx2 in sorted_indices[i+1:]:
                if np.linalg.norm(self.X[idx1] - self.X[idx2]) < radius[idx2] - radius[idx1]:
                    sphere_counts[idx1] = 0
                    break

        return sphere_counts

    def T1(self):
        radius = self.__get_hypersphere_counts()
        sphere_counts = self.__remove_overlapping_hyperspheres(radius)
        t1_value = np.sum(sphere_counts) / len(self.X)
        return t1_value

    def T1_adapted(self):
        t1_adapted = {}
        for cls in self.classes:
            class_indices = np.where(self.y == cls)[0]
            radius = np.zeros(len(class_indices))
            for i, idx in enumerate(class_indices):
                _, nearest_enemy_distance = self.__find_nearest_opposite_class(idx)
                radius[i] = nearest_enemy_distance

            sphere_counts = self.__remove_overlapping_hyperspheres(radius)
            t1_adapted[cls] = np.sum(sphere_counts) / len(class_indices)
        return t1_adapted


