import random
import time
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from collections import defaultdict

#step1

# Generate a dataset of spatial objects with random locations and contextual keywords
def generate_dataset(num_objects=200, num_keywords=10, area_size=(100, 100)):
    """
    Generates a dataset of spatial objects with random locations and contextual keywords.
    """
    dataset = []
    keywords = [f"kw_{i}" for i in range(num_keywords)]
    
    for i in range(num_objects):
        obj_id = f"p_{i}"
        location = (random.uniform(0, area_size[0]), random.uniform(0, area_size[1]))
        context = set(random.sample(keywords, random.randint(1, 5)))  # Random set of keywords
        dataset.append((obj_id, location, context))
    
    return dataset

# Generate dataset
dataset = generate_dataset(num_objects=100, num_keywords=20)
df = pd.DataFrame(dataset, columns=["ObjectID", "Location", "Context"])
#print(df.head())  # Preview dataset

#step2
def compute_jaccard_similarity(set1, set2):
    """
    Computes Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def micro_set_jaccard_hashing(dataset):
    """
    Computes pairwise Jaccard similarity using an optimized micro set hashing technique.
    """
    msht = defaultdict(set)
    similarities = {}

    # Step 1: Build the inverted index (hash table)
    for obj_id, _, context in dataset:
        for keyword in context:
            msht[keyword].add(obj_id)

    # Step 2: Compute Jaccard similarity using inverted lists
    for keyword, objects in msht.items():
        object_list = list(objects)
        for obj1, obj2 in itertools.combinations(object_list, 2):
            key = tuple(sorted((obj1, obj2)))
            if key not in similarities:
                context1 = next(ctx for oid, _, ctx in dataset if oid == obj1)
                context2 = next(ctx for oid, _, ctx in dataset if oid == obj2)
                similarities[key] = compute_jaccard_similarity(context1, context2)

    return similarities

# Compute Jaccard similarities
jaccard_similarities = micro_set_jaccard_hashing(dataset)
#print(list(jaccard_similarities.items())[:5])  # Preview some similarities


#step3
def create_grid(dataset, grid_size=(10, 10), area_size=(100, 100)):
    """
    Creates a grid structure and assigns objects to corresponding grid cells.
    """
    grid = defaultdict(list)
    cell_width = area_size[0] / grid_size[0]
    cell_height = area_size[1] / grid_size[1]

    for obj_id, location, _ in dataset:
        x_idx = int(location[0] // cell_width)
        y_idx = int(location[1] // cell_height)
        grid[(x_idx, y_idx)].append((obj_id, location))

    return grid, (cell_width, cell_height)

def compute_spatial_similarity(grid):
    """
    Computes spatial similarity based on Ptolemy’s measure within a grid structure.
    """
    spatial_similarities = {}

    grid_cells = list(grid.keys())
    for (x1, y1), (x2, y2) in itertools.combinations(grid_cells, 2):
        center1 = ((x1 + 0.5) * cell_width, (y1 + 0.5) * cell_height)
        center2 = ((x2 + 0.5) * cell_width, (y2 + 0.5) * cell_height)

        for obj1, loc1 in grid[(x1, y1)]:
            for obj2, loc2 in grid[(x2, y2)]:
                dist_obj = euclidean(loc1, loc2)
                dist_center = euclidean(center1, center2)

                similarity = 1 - (dist_obj / (dist_center + 1e-5))
                spatial_similarities[(obj1, obj2)] = max(0, similarity)

    return spatial_similarities

# Create grid and compute spatial similarities
grid, (cell_width, cell_height) = create_grid(dataset, grid_size=(10, 10))
spatial_similarities = compute_spatial_similarity(grid)
#print(list(spatial_similarities.items())[:5])  # Preview some similarities

#step4
def greedy_IAdU(dataset, jaccard_similarities, spatial_similarities, k=10):
    """
    Implements the Incremental Additive Update (IAdU) greedy selection algorithm.
    """
    selected_set = []
    remaining_set = {obj[0] for obj in dataset}

    while len(selected_set) < k:
        best_object = None
        best_score = float('-inf')

        for obj in remaining_set:
            contribution = sum(
                jaccard_similarities.get((obj, sel), 0) + spatial_similarities.get((obj, sel), 0)
                for sel in selected_set
            )

            if contribution > best_score:
                best_score = contribution
                best_object = obj

        if best_object:
            selected_set.append(best_object)
            remaining_set.remove(best_object)

    return selected_set

def greedy_ABP(dataset, jaccard_similarities, spatial_similarities, k=10):
    """
    Implements the Adaptive Bi-Partitioning (ABP) greedy selection algorithm.
    """
    selected_set = set()
    remaining_set = {obj[0] for obj in dataset}

    while len(selected_set) < k:
        best_pair = None
        best_score = float('-inf')

        for obj1, obj2 in itertools.combinations(remaining_set, 2):
            score = jaccard_similarities.get((obj1, obj2), 0) + spatial_similarities.get((obj1, obj2), 0)
            if score > best_score:
                best_score = score
                best_pair = (obj1, obj2)

        if best_pair:
            selected_set.update(best_pair)
            remaining_set.difference_update(best_pair)

    return list(selected_set)[:k]


selected_IAdU = greedy_IAdU(dataset, jaccard_similarities, spatial_similarities, k=10)
selected_ABP = greedy_ABP(dataset, jaccard_similarities, spatial_similarities, k=10)

#print("Selected objects (IAdU):", selected_IAdU)
#print("Selected objects (ABP):", selected_ABP)

#step5
import time

# Measure execution time
start_time = time.time()
greedy_IAdU(dataset, jaccard_similarities, spatial_similarities, k=10)
IAdU_time = time.time() - start_time

start_time = time.time()
greedy_ABP(dataset, jaccard_similarities, spatial_similarities, k=10)
ABP_time = time.time() - start_time

#print(f"IAdU Execution Time: {IAdU_time:.4f} seconds")
#print(f"ABP Execution Time: {ABP_time:.4f} seconds")



#baseline implemention

#Baseline 1: Contextual Proportionality (Brute-force Jaccard Similarity)
def baseline_jaccard_similarity(dataset):
    """
    Computes pairwise Jaccard similarity using a brute-force method.
    """
    similarities = {}

    for obj1, obj2 in itertools.combinations(dataset, 2):
        key = tuple(sorted((obj1[0], obj2[0])))
        context1, context2 = obj1[2], obj2[2]
        similarities[key] = compute_jaccard_similarity(context1, context2)

    return similarities

# Compute baseline Jaccard similarities
baseline_jaccard_similarities = baseline_jaccard_similarity(dataset)
#print(list(baseline_jaccard_similarities.items())[:5])  # Preview some results

#Baseline 2: Spatial Proportionality (Brute-force Euclidean Distance)
def baseline_spatial_similarity(dataset):
    """
    Computes spatial similarity for all object pairs using a brute-force Euclidean distance.
    """
    spatial_similarities = {}

    for obj1, obj2 in itertools.combinations(dataset, 2):
        key = tuple(sorted((obj1[0], obj2[0])))
        loc1, loc2 = obj1[1], obj2[1]
        dist = euclidean(loc1, loc2)
        spatial_similarities[key] = 1 / (1 + dist)  # Normalizing similarity

    return spatial_similarities

# Compute baseline spatial similarities
baseline_spatial_similarities = baseline_spatial_similarity(dataset)
#print(list(baseline_spatial_similarities.items())[:5])  # Preview some results

#Baseline 3: Random Selection (Naive Approach)
def baseline_random_selection(dataset, k=10):
    """
    Selects k random objects from the dataset.
    """
    return random.sample([obj[0] for obj in dataset], k)

# Run baseline selection
baseline_random_selection_result = baseline_random_selection(dataset, k=10)
#print("Baseline Random Selection:", baseline_random_selection_result)


#Baseline 4: Top-k Selection (Based on Relevance Only)
def baseline_top_k_selection(dataset, jaccard_similarities, spatial_similarities, k=10):
    """
    Selects the top-k objects based on the sum of contextual and spatial relevance.
    """
    relevance_scores = {}

    for obj in dataset:
        obj_id = obj[0]
        relevance_scores[obj_id] = sum(jaccard_similarities.get((obj_id, other), 0) +
                                       spatial_similarities.get((obj_id, other), 0)
                                       for other, _, _ in dataset if obj_id != other)

    # Select top-k based on the highest scores
    sorted_objects = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    return [obj[0] for obj in sorted_objects[:k]]

# Run baseline top-k selection
baseline_top_k_result = baseline_top_k_selection(dataset, baseline_jaccard_similarities, baseline_spatial_similarities, k=10)
#print("Baseline Top-k Selection:", baseline_top_k_result)

#Compare Execution Time of Baseline vs Optimized
# Measure execution time
start_time = time.time()
baseline_jaccard_similarity(dataset)
baseline_time_jaccard = time.time() - start_time

start_time = time.time()
baseline_spatial_similarity(dataset)
baseline_time_spatial = time.time() - start_time

start_time = time.time()
baseline_top_k_selection(dataset, baseline_jaccard_similarities, baseline_spatial_similarities, k=10)
baseline_time_topk = time.time() - start_time

start_time = time.time()
greedy_IAdU(dataset, jaccard_similarities, spatial_similarities, k=10)
IAdU_time = time.time() - start_time

start_time = time.time()
greedy_ABP(dataset, jaccard_similarities, spatial_similarities, k=10)
ABP_time = time.time() - start_time

# Display execution time comparison
execution_times = pd.DataFrame({
    "Algorithm": ["Baseline Jaccard", "Baseline Spatial", "Baseline Top-k", "IAdU", "ABP"],
    "Execution Time (seconds)": [baseline_time_jaccard, baseline_time_spatial, baseline_time_topk, IAdU_time, ABP_time]
})

print(execution_times)


#visualization
def plot_selected_objects(dataset, selected_sets, labels):
    """
    Plots selected spatial objects on a 2D coordinate map for comparison.
    """
    plt.figure(figsize=(10, 8))

    # Plot all dataset points in light gray
    for obj_id, location, _ in dataset:
        plt.scatter(location[0], location[1], color="lightgray", s=50, alpha=0.5)

    # Define colors for different selection methods
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (selected_set, label) in enumerate(zip(selected_sets, labels)):
        selected_locations = [loc for obj_id, loc, _ in dataset if obj_id in selected_set]
        xs, ys = zip(*selected_locations)  # Extract x and y coordinates
        plt.scatter(xs, ys, color=colors[i % len(colors)], s=100, label=label, edgecolors='black', alpha=0.7)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Comparison of Selected Spatial Objects")
    plt.legend()
    plt.grid(True)
    plt.show()


# Compute selected sets
selected_IAdU = greedy_IAdU(dataset, jaccard_similarities, spatial_similarities, k=10)
selected_ABP = greedy_ABP(dataset, jaccard_similarities, spatial_similarities, k=10)
baseline_random_selection_result = baseline_random_selection(dataset, k=10)
baseline_top_k_result = baseline_top_k_selection(dataset, jaccard_similarities, spatial_similarities, k=10)

# Plot the selected results
plot_selected_objects(
    dataset,
    [set(selected_IAdU), set(selected_ABP), set(baseline_random_selection_result), set(baseline_top_k_result)],
    ["IAdU Selection", "ABP Selection", "Random Selection", "Top-k Selection"]
)
