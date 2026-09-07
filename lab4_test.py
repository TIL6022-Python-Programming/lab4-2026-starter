"""Test code for TIL Python programming jupyter notebook Lab 4 - Data Analysis"""
import pytest
import os.path
from testbook import testbook

@pytest.fixture(scope='module')
def tb():
    with testbook('lab4.ipynb', execute=True) as tb:
        yield tb

# Q1
def test_int_read_1_1(tb):
    assert tb.ref('int_read')(os.path.join('data', 'addition.txt')) == [2, 3, 5, 10, 20], 'integers not read correctly'

def test_int_read_1_2(tb):
    assert tb.ref('int_sum')([2, 3, 5, 10, 20]) == 40, 'sum of integers not correct'

def test_int_read_1_3():
    assert os.path.isfile(os.path.join('data', 'addition2.txt')), 'addition2.txt does not exist inside data folder'

def test_int_read_1_4(tb):
    assert tb.ref('int_read')(os.path.join('data', 'addition2.txt')) == [2, 3, 5, 10, 20, 40], 'extra integers not written correctly'

# Q2
def test_pandas_2_1(tb):
    df = tb.ref('df')
    assert df.shape[0] == 584, 'Netflix dataframe does not have the right amount of movies'
    assert df.loc[2, 'Premiere'] == 'December 26, 2019', 'Incorrect data point in dataframe'
    assert abs(df.loc[333, 'IMDB Score'] - 6.5) < 1e-6, 'Incorrect data point in dataframe'
    assert df.loc[579, 'Title'] == 'Taylor Swift: Reputation Stadium Tour', 'Incorrect data point in dataframe'

def test_pandas_2_2(tb):
    movie100 = tb.ref('movie100')
    assert len(movie100) == 2, 'Incorrect number of columns '
    assert movie100['Title'] == 'Game Over, Man!'
    assert movie100['Genre'] == 'Action/Comedy'

def test_pandas_2_3(tb):
    df2 = tb.ref('df2')
    assert abs(df2.loc['Searching for Sheela', 'IMDB Score'] - 4.1) < 1e-6
    assert df2.loc['Whipped', 'Premiere'] == 'September 18, 2020'
    assert df2.loc['All Because of You', 'Language'] == 'Malay'

def test_pandas_2_4(tb):
    assert tb.ref('number_of_movies')(tb.ref('df')) == 584

def test_pandas_2_5(tb):
    assert abs(tb.ref('highest_score')(tb.ref('df')) - 9.0) < 1e-6

def test_pandas_2_6(tb):
    assert tb.ref('best_scoring_movie')(tb.ref('df')) == 'David Attenborough: A Life on Our Planet'

def test_pandas_2_7(tb):
    df_english = tb.ref('above_average_english')(tb.ref('df'))
    assert df_english.shape[0] == 210
    assert df_english.iloc[2, df_english.columns.get_loc('Premiere')] == 'March 30, 2018'
    assert df_english.iloc[206, df_english.columns.get_loc('Title')] == 'Ben Platt: Live from Radio City Music Hall'

def test_pandas_2_8(tb):
    assert tb.ref('number_of_languages')(tb.ref('df')) == 38

def test_pandas_2_9():
    assert os.path.isfile(os.path.join('data', 'action_movies.csv')), 'action_movies.csv does not exist inside data folder'

def test_pandas_2_10(tb):
    assert tb.ref('best_genre')(tb.ref('df')) == 'Animation/Christmas/Comedy/Adventure'

def test_least_squares_3(tb):
    ab = tb.ref('ab')
    assert abs(ab[0] - 28.71725425316547) < 1e-3 and abs(ab[1] - 6.780870328313551) < 1e-3
    for x0 in range(0, 2):
        x0f = float(x0)
        for x1 in range(5, 7):
            x1f = float(x1)
            for y0 in range(0, 2):
                y0f = float(y0)
                for y1 in range(0, 2):
                    if y0 == y1:
                        continue # formula cannot deal with horizontal line as sum_dv = 0
                    y1f = float(y1)
                    cd = tb.value(f'cf_fit(np.array([{x0f}, {x1f}]), np.array([{y0f}, {y1f}]))')
                    b = (y1 - y0) / (x1 - x0)
                    a = y0 - b*x0
                    assert abs(cd[0] - a) < 1e-3 and abs(cd[1] - b) < 1e-3