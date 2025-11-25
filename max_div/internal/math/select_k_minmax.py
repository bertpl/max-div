import numba
import numpy as np
from numpy.typing import NDArray


# =================================================================================================
#  select_k_min
# =================================================================================================
@numba.njit("int32[:](float32[:], int32)", fastmath=True, inline="always")
def select_k_min(arr: NDArray[np.float32], k: np.int32) -> NDArray[np.int32]:
    """
    Find indices of k smallest elements in a float32 array using Numba.

    This implementation uses a max-heap approach with O(n log k) complexity,
    which is efficient when k << n. The heap maintains the k smallest elements
    seen so far, with the largest of these at the root.

    Parameters:
    -----------
    arr : NDArray[np.float32]
        Input array with n elements (typically 1000-10000)
    k : int
        Number of smallest elements to find

    Returns:
    --------
    indices : NDArray[np.int32]
        Array of k indices pointing to the smallest elements.
        Indices are returned in arbitrary order (not sorted by value).

    Performance:
    ------------
    - 2-8x faster than np.argpartition for small to moderate k (k ~ 10-100)
    - Best when k << n (e.g., k=100, n=10000)
    - Uses fastmath=True for additional SIMD optimizations
    """
    n = len(arr)
    heap_idx = np.empty(k, dtype=np.int32)  # indices (into arr) of elements in the heap
    heap_values = np.empty(k, dtype=np.float32)  # values of elements in the heap; largest at heap_values[0]

    # Build initial heap with first k elements
    for i in range(k):
        heap_idx[i] = i
        heap_values[i] = arr[i]

    # Heapify: Convert initial k elements into a max-heap
    # -----------------------------------------------------------------------------------
    #
    # assuming we want to represent values v0 >= v1 >= v2 >= v3 >= v4 >= v5 >= v6 into a heap:
    #
    #            v0
    #          /    \
    #        v1      v2
    #       / \     / \
    #     v3  v4  v5  v6
    #
    # Invariant relations:
    #   - parents >= leaves   (i.e. v0 >= v1,v2; v1 >= v3,v4; v2 >= v5,v6)
    #   - leaves of same parent are not necessarily sorted (!)  (i.e. the tree could swap branches v1 & v2)
    #   - if a parent is at index i, then its children are at indices 2*i+1 and 2*i+2
    # -----------------------------------------------------------------------------------
    # Start from last non-leaf node and sift down
    for i in range(k // 2 - 1, -1, -1):
        i_parent = i
        value = heap_values[i_parent]
        idx = heap_idx[i_parent]

        # Sift down: move element down until heap property is restored
        while True:
            i_child_left = 2 * i_parent + 1
            i_child_right = i_child_left + 1
            i_child_largest = -1

            # Find the largest child
            if i_child_left < k:
                if i_child_right < k:
                    i_child_largest = (
                        i_child_left if heap_values[i_child_left] > heap_values[i_child_right] else i_child_right
                    )
                else:
                    i_child_largest = i_child_left

            # If no children or value is larger than the largest child, we're done
            if i_child_largest == -1 or value >= heap_values[i_child_largest]:
                heap_values[i_parent] = value
                heap_idx[i_parent] = idx
                break

            # Otherwise, move the larger child up and continue
            heap_values[i_parent] = heap_values[i_child_largest]
            heap_idx[i_parent] = heap_idx[i_child_largest]
            i_parent = i_child_largest

    # Process remaining elements
    # For each element, if it's smaller than heap maximum, replace and sift down
    for i in range(k, n):
        value = arr[i]
        if value < heap_values[0]:  # heap_values[0] is the maximum of k smallest
            i_parent = 0

            # Sift down from root
            while True:
                i_child_left = 2 * i_parent + 1
                i_child_right = i_child_left + 1
                i_child_largest = -1

                # Find the largest child
                if i_child_left < k:
                    if i_child_right < k:
                        i_child_largest = (
                            i_child_left if heap_values[i_child_left] > heap_values[i_child_right] else i_child_right
                        )
                    else:
                        i_child_largest = i_child_left

                # If no children or val is larger than largest child, we're done
                if i_child_largest == -1 or value >= heap_values[i_child_largest]:
                    heap_values[i_parent] = value
                    heap_idx[i_parent] = i
                    break

                # Otherwise, move the larger child up and continue
                heap_values[i_parent] = heap_values[i_child_largest]
                heap_idx[i_parent] = heap_idx[i_child_largest]
                i_parent = i_child_largest

    return heap_idx


# =================================================================================================
#  select_k_max
# =================================================================================================
@numba.njit("int32[:](float32[:], int32)", fastmath=True, inline="always")
def select_k_max(arr: NDArray[np.float32], k: np.int32) -> NDArray[np.int32]:
    """
    Find indices of k largest elements in a float32 array using Numba.

    This implementation uses a min-heap approach with O(n log k) complexity,
    which is efficient when k << n. The heap maintains the k largest elements
    seen so far, with the smallest of these at the root.

    Parameters:
    -----------
    arr : NDArray[np.float32]
        Input array with n elements (typically 1000-10000)
    k : int
        Number of largest elements to find

    Returns:
    --------
    indices : NDArray[np.int32]
        Array of k indices pointing to the largest elements.
        Indices are returned in arbitrary order (not sorted by value).

    Performance:
    ------------
    - 2-8x faster than np.argpartition for small to moderate k (k ~ 10-100)
    - Best when k << n (e.g., k=100, n=10000)
    - Uses fastmath=True for additional SIMD optimizations
    """
    n = len(arr)
    heap_idx = np.empty(k, dtype=np.int32)  # indices (into arr) of elements in the heap
    heap_values = np.empty(k, dtype=np.float32)  # values of elements in the heap; smallest at heap_values[0]

    # Build initial heap with first k elements
    for i in range(k):
        heap_idx[i] = i
        heap_values[i] = arr[i]

    # Heapify: Convert initial k elements into a min-heap
    # -----------------------------------------------------------------------------------
    #
    # assuming we want to represent values v0 <= v1 <= v2 <= v3 <= v4 <= v5 <= v6 into a heap:
    #
    #            v0
    #          /    \
    #        v1      v2
    #       / \     / \
    #     v3  v4  v5  v6
    #
    # Invariant relations:
    #   - parents <= leaves   (i.e. v0 <= v1,v2; v1 <= v3,v4; v2 <= v5,v6)
    #   - leaves of same parent are not necessarily sorted (!)  (i.e. the tree could swap branches v1 & v2)
    #   - if a parent is at index i, then its children are at indices 2*i+1 and 2*i+2
    # -----------------------------------------------------------------------------------
    # Start from last non-leaf node and sift down
    for i in range(k // 2 - 1, -1, -1):
        i_parent = i
        value = heap_values[i_parent]
        idx = heap_idx[i_parent]

        # Sift down: move element down until heap property is restored
        while True:
            i_child_left = 2 * i_parent + 1
            i_child_right = i_child_left + 1
            i_child_smallest = -1

            # Find the smallest child
            if i_child_left < k:
                if i_child_right < k:
                    i_child_smallest = (
                        i_child_left if heap_values[i_child_left] < heap_values[i_child_right] else i_child_right
                    )
                else:
                    i_child_smallest = i_child_left

            # If no children or value is smaller than the smallest child, we're done
            if i_child_smallest == -1 or value <= heap_values[i_child_smallest]:
                heap_values[i_parent] = value
                heap_idx[i_parent] = idx
                break

            # Otherwise, move the smaller child up and continue
            heap_values[i_parent] = heap_values[i_child_smallest]
            heap_idx[i_parent] = heap_idx[i_child_smallest]
            i_parent = i_child_smallest

    # Process remaining elements
    # For each element, if it's larger than heap minimum, replace and sift down
    for i in range(k, n):
        value = arr[i]
        if value > heap_values[0]:  # heap_values[0] is the minimum of k largest
            i_parent = 0

            # Sift down from root
            while True:
                i_child_left = 2 * i_parent + 1
                i_child_right = i_child_left + 1
                i_child_smallest = -1

                # Find the smallest child
                if i_child_left < k:
                    if i_child_right < k:
                        i_child_smallest = (
                            i_child_left if heap_values[i_child_left] < heap_values[i_child_right] else i_child_right
                        )
                    else:
                        i_child_smallest = i_child_left

                # If no children or val is smaller than smallest child, we're done
                if i_child_smallest == -1 or value <= heap_values[i_child_smallest]:
                    heap_values[i_parent] = value
                    heap_idx[i_parent] = i
                    break

                # Otherwise, move the smaller child up and continue
                heap_values[i_parent] = heap_values[i_child_smallest]
                heap_idx[i_parent] = heap_idx[i_child_smallest]
                i_parent = i_child_smallest

    return heap_idx
