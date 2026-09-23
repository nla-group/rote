# Copyright (c) 2021, nla group, manchester
# All rights reserved. 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import Counter
import math

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Damerau-Levenshtein distance between two strings.
    Operations: insertion, deletion, substitution, and transposition of adjacent characters.
    Each operation has a cost of 1.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : int
        The minimum number of edit operations (insertions, deletions, substitutions, or transpositions) required.
   
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    # Create a matrix of size (len(s1) + 1) x (len(s2) + 1)
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize first row and column
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )
            
            # Check for transposition (only if i >= 2, j >= 2 and characters are swapped)
            if (i > 1 and j > 1 and
                s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1)  # Transposition
    
    return dp[len1][len2]




def jaro_winkler_distance(s1: str, s2: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """
    Compute the Jaro-Winkler similarity between two strings.
    The similarity is based on Jaro similarity with a prefix adjustment.
    Distance can be computed as 1 - similarity.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    p : float, optional
        Prefix scaling factor (default is 0.1).
    max_prefix : int, optional
        Maximum prefix length to consider (default is 4).
    
    Returns
    -------
    similarity : float
        The Jaro-Winkler similarity score, ranging from 0 (no similarity) to 1 (identical strings).
    
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    
    # Maximum distance to consider a character match
    match_bound = max(len1, len2) // 2 - 1
    
    # Find matching characters
    matches = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    for i in range(len1):
        start = max(0, i - match_bound)
        end = min(len2, i + match_bound + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    transpositions = 0
    j = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[j]:
                j += 1
            if s1[i] != s2[j]:
                transpositions += 1
            j += 1
    
    transpositions //= 2
    
    # Jaro similarity
    jaro = (1/3) * (
        matches / len1 +
        matches / len2 +
        (matches - transpositions) / matches
    )
    
    # Find common prefix length (up to max_prefix)
    prefix_len = 0
    for i in range(min(len1, len2, max_prefix)):
        if s1[i] != s2[i]:
            break
        prefix_len += 1
    
    # Jaro-Winkler similarity
    jaro_winkler = jaro + prefix_len * p * (1 - jaro)
    
    # Ensure the result is within [0, 1]
    return min(1.0, max(0.0, jaro_winkler))



def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : int
        The minimum number of edit operations (insertions, deletions, or substitutions) required.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )
    
    return dp[len1][len2]



def hamming_distance(s1: str, s2: str) -> int:
    """Calculate the Hamming distance between two strings of equal length.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : int
        The number of positions where the strings differ.
    
    Raises
    ------
    ValueError
        If the strings have different lengths.
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length for Hamming distance")
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))



def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate the Jaro similarity between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The Jaro similarity score, ranging from 0 (no similarity) to 1 (identical strings).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    match_bound = max(len1, len2) // 2 - 1
    
    matches = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    for i in range(len1):
        start = max(0, i - match_bound)
        end = min(len2, i + match_bound + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    transpositions = 0
    j = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[j]:
                j += 1
            if s1[i] != s2[j]:
                transpositions += 1
            j += 1
    
    transpositions //= 2
    
    return (1/3) * (
        matches / len1 +
        matches / len2 +
        (matches - transpositions) / matches
    )



def cosine_similarity(s1: str, s2: str) -> float:
    """Calculate the cosine similarity between two strings using word vectors.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The cosine similarity score, ranging from 0 (no similarity) to 1 (identical word sets).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    # Split into words
    words1 = s1.split()
    words2 = s2.split()
    
    if not words1 or not words2:
        return 0.0
    
    # Create word frequency vectors
    vec1 = Counter(words1)
    vec2 = Counter(words2)
    
    # Compute dot product
    dot_product = sum(vec1[word] * vec2[word] for word in set(vec1) & set(vec2))
    
    # Compute magnitudes
    mag1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
    mag2 = math.sqrt(sum(count ** 2 for count in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def cosine_bigram_similarity(s1: str, s2: str) -> float:
    """Calculate the cosine similarity between two strings using bigram vectors.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The cosine similarity score, ranging from 0 (no similarity) to 1 (identical bigram sets).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    # Extract bigrams
    def get_bigrams(s):
        return [s[i:i+2] for i in range(len(s)-1)] if len(s) > 1 else []
    
    bigrams1 = Counter(get_bigrams(s1))
    bigrams2 = Counter(get_bigrams(s2))
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    # Compute dot product
    dot_product = sum(bigrams1[bg] * bigrams2[bg] for bg in set(bigrams1) & set(bigrams2))
    
    # Compute magnitudes
    mag1 = math.sqrt(sum(count ** 2 for count in bigrams1.values()))
    mag2 = math.sqrt(sum(count ** 2 for count in bigrams2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)



def lcs_distance(s1: str, s2: str) -> int:
    """Calculate the LCS-based distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : int
        The LCS distance, defined as len(s1) + len(s2) - 2 * len(LCS).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[len1][len2]
    return len1 + len2 - 2 * lcs_length



def dice_coefficient(s1: str, s2: str) -> float:
    """Calculate Dice's coefficient between two strings using bigrams.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        Dice's coefficient, ranging from 0 (no shared bigrams) to 1 (identical bigram sets).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    
    bigrams1 = get_bigrams(s1)
    bigrams2 = get_bigrams(s2)
    
    common = len(bigrams1 & bigrams2)
    total = len(bigrams1) + len(bigrams2)
    
    if total == 0:
        return 0.0
    
    return (2 * common) / total


def smith_waterman_distance(s1: str, s2: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -1) -> int:
    """Calculate the Smith-Waterman distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    match_score : int, optional
        Score for matching characters (default is 2).
    mismatch_score : int, optional
        Score for mismatching characters (default is -1).
    gap_score : int, optional
        Score for gaps (default is -1).
    
    Returns
    -------
    distance : int
        The inverse of the maximum alignment score (lower scores indicate greater distance).
    """
    if s1 == s2:
        return 0
    if not s1 or not s2:
        return 0  # No alignment possible, score = 0, distance = -0
    
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    max_score = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match = dp[i-1][j-1] + (match_score if s1[i-1] == s2[j-1] else mismatch_score)
            delete = dp[i-1][j] + gap_score
            insert = dp[i][j-1] + gap_score
            dp[i][j] = max(0, match, delete, insert)
            max_score = max(max_score, dp[i][j])
    
    return -max_score




# 1. Normalized Levenshtein Distance
def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    """Calculate the normalized Levenshtein distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : float
        The normalized Levenshtein distance, ranging from 0 to 1.
    """
    def levenshtein_distance(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Deletion
                    dp[i][j-1] + 1,      # Insertion
                    dp[i-1][j-1] + cost  # Substitution
                )
        
        return dp[len1][len2]
    
    raw_distance = levenshtein_distance(s1, s2)
    if not s1 and not s2:
        return 0.0
    max_length = max(len(s1), len(s2))
    return raw_distance / max_length if max_length > 0 else 0.0



# 2. Normalized Hamming Distance
def normalized_hamming_distance(s1: str, s2: str) -> float:
    """Calculate the normalized Hamming distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : float
        The normalized Hamming distance, ranging from 0 to 1. Raises ValueError for unequal lengths.
    """
    def hamming_distance(s1: str, s2: str) -> int:
        if len(s1) != len(s2):
            raise ValueError("Strings must have equal length for Hamming distance")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    if s1 == s2:
        return 0.0
    if not s1 and not s2:
        return 0.0
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length for Hamming distance")
    
    raw_distance = hamming_distance(s1, s2)
    max_length = len(s1)  # Since lengths are equal
    return raw_distance / max_length if max_length > 0 else 0.0



# 3. Jaro Similarity (already normalized, retained for completeness)
def normalized_jaro_similarity(s1: str, s2: str) -> float:
    """Calculate the Jaro similarity between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The Jaro similarity, ranging from 0 to 1.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    matches = 0
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2
    
    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                matches += 1
                hash_s1[i] = 1
                hash_s2[j] = 1
                break
    
    if matches == 0:
        return 0.0
    
    transpositions = 0
    j = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[j] == 0:
                j += 1
            if s1[i] != s2[j]:
                transpositions += 1
            j += 1
    
    transpositions = transpositions // 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3
    return jaro



# 4. Jaro-Winkler Distance (already normalized, retained for completeness)
def normalized_jaro_winkler_similarity(s1: str, s2: str, p: float =  0.1) -> float:
    """Calculate the Jaro-Winkler similarity between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    p : float, optional
        Prefix scaling factor (default is 0.1).
    
    Returns
    -------
    similarity : float
        The Jaro-Winkler similarity, ranging from 0 to 1.
    """
    jaro = normalized_jaro_similarity(s1, s2)
    if jaro == 0:
        return 0.0
    
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * p * (1 - jaro)


def normalized_jaro_winkler_distance(s1: str, s2: str, p: float =  0.1) -> float:
    """Calculate the Jaro-Winkler distance between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    p : float, optional
        Prefix scaling factor (default is 0.1).
    
    Returns
    -------
    similarity : float
        The Jaro-Winkler similarity, ranging from 0 to 1.
    """
        
    return 1 - normalized_jaro_winkler_similarity(s1, s2, p)
    
    

# 5. Cosine Similarity (Word-Based) (already normalized, retained for completeness)
def normalized_cosine_similarity(s1: str, s2: str) -> float:
    """Calculate the word-based cosine similarity between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The cosine similarity, ranging from 0 to 1.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    words1 = s1.split()
    words2 = s2.split()
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    
    all_words = set(counter1.keys()) | set(counter2.keys())
    dot_product = sum(counter1.get(word, 0) * counter2.get(word, 0) for word in all_words)
    
    mag1 = math.sqrt(sum(count ** 2 for count in counter1.values()))
    mag2 = math.sqrt(sum(count ** 2 for count in counter2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

# 6. Cosine Bigram Similarity (already normalized, retained for completeness)
def normalized_cosine_bigram_similarity(s1: str, s2: str) -> float:
    """Calculate the bigram-based cosine similarity between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The cosine bigram similarity, ranging from 0 to 1.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    def get_bigrams(s):
        return [s[i:i+2] for i in range(len(s)-1)] if len(s) > 1 else []
    
    bigrams1 = Counter(get_bigrams(s1))
    bigrams2 = Counter(get_bigrams(s2))
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    dot_product = sum(bigrams1[bg] * bigrams2[bg] for bg in set(bigrams1) & set(bigrams2))
    
    mag1 = math.sqrt(sum(count ** 2 for count in bigrams1.values()))
    mag2 = math.sqrt(sum(count ** 2 for count in bigrams2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

# 7. Normalized LCS Distance
def normalized_lcs_distance(s1: str, s2: str) -> float:
    """Calculate the normalized LCS distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : float
        The normalized LCS distance, ranging from 0 to 1.
    """
    def lcs_distance(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[len1][len2]
        return len1 + len2 - 2 * lcs_length
    
    raw_distance = lcs_distance(s1, s2)
    if not s1 and not s2:
        return 0.0
    max_length = max(len(s1), len(s2))
    return raw_distance / max_length if max_length > 0 else 0.0

# 8. Dice’s Coefficient (already normalized, retained for completeness)
def normalized_dice_coefficient(s1: str, s2: str) -> float:
    """Calculate the Dice’s coefficient between two strings (inherently normalized).
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    similarity : float
        The Dice’s coefficient, ranging from 0 to 1.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    def get_bigrams(s):
        return [s[i:i+2] for i in range(len(s)-1)] if len(s) > 1 else []
    
    bigrams1 = set(get_bigrams(s1))
    bigrams2 = set(get_bigrams(s2))
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    common_bigrams = len(bigrams1 & bigrams2)
    total_bigrams = len(bigrams1) + len(bigrams2)
    
    return 2 * common_bigrams / total_bigrams

# 9. Normalized Smith-Waterman Distance
def normalized_smith_waterman_distance(s1: str, s2: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -1) -> float:
    """Calculate the normalized Smith-Waterman distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    match_score : int, optional
        Score for matching characters (default is 2).
    mismatch_score : int, optional
        Score for mismatching characters (default is -1).
    gap_score : int, optional
        Score for gaps (default is -1).
    
    Returns
    -------
    distance : float
        The normalized Smith-Waterman distance, ranging from 0 to 1.
    """
    def smith_waterman_distance(s1: str, s2: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -1) -> int:
        if s1 == s2:
            return 0
        if not s1 or not s2:
            return 0  # No alignment possible
        
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        max_score = 0
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                match = dp[i-1][j-1] + (match_score if s1[i-1] == s2[j-1] else mismatch_score)
                delete = dp[i-1][j] + gap_score
                insert = dp[i][j-1] + gap_score
                dp[i][j] = max(0, match, delete, insert)
                max_score = max(max_score, dp[i][j])
        
        return -max_score
    
    raw_distance = smith_waterman_distance(s1, s2, match_score, mismatch_score, gap_score)
    if not s1 and not s2:
        return 0.0
    # Normalize by maximum possible score (match_score * min(len(s1), len(s2)))
    max_score = match_score * min(len(s1), len(s2))
    if max_score == 0:
        return 0.0
    # Convert negative distance to positive and normalize
    normalized_distance = -raw_distance / max_score
    return max(0.0, min(1.0, normalized_distance))  # Ensure [0, 1]


# 10. Normalized Damerau-Levenshtein Distance
def normalized_damerau_levenshtein_distance(s1: str, s2: str) -> float:
    """Calculate the normalized Damerau-Levenshtein distance between two strings.
    
    Parameters
    ----------
    s1 : str
        The first input string.
    s2 : str
        The second input string.
    
    Returns
    -------
    distance : float
        The normalized Damerau-Levenshtein distance, ranging from 0 to 1.
    """
    raw_distance = damerau_levenshtein_distance(s1, s2)
    if not s1 and not s2:
        return 0.0
    max_length = max(len(s1), len(s2))
    return raw_distance / max_length if max_length > 0 else 0.0
