"""Sample dataset: 6 users x 5 movies. NaN = missing rating."""
import numpy as np
nan = np.nan
USERS = ["Asha", "Ben", "Chen", "Divya", "Eli", "Farah"]
MOVIES = ["Inception", "Titanic", "Avengers", "Notebook", "Interstellar"]
RATINGS = np.array([
    [5, nan, 4, 1, 5],
    [4, 2, nan, 1, 5],
    [nan, 5, 2, 5, 1],
    [2, 5, 1, nan, 2],
    [5, 1, 5, 2, nan],
    [1, 4, nan, 5, 2],
], dtype=float)
