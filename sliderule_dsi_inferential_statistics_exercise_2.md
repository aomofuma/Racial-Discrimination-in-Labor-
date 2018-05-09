
# Examining Racial Discrimination in the US Job Market

### Background
Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.

### Data
In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.

Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
```


```python
data = pd.io.stata.read_stata('data/us_job_market_discrimination.dta')
```


```python
# number of callbacks for black-sounding names
sum(data[data.race=='b'].call)
```




    157.0




```python
# number of callbacks for white-sounding names
sum(data[data.race=='w'].call)
```




    235.0




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ad</th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>...</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
      <th>ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>316</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Nonprofit</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



# 1. What test is appropriate for this problem? Does CLT apply?


```python
print('There are', len(data), 'observations in this sample')
```

    There are 4870 observations in this sample
    

Due to the size of the data sample we can apply the Central limit Theorem. Since we are testing the mean success rate of two different samples, a two-sample t-test is most appropriate.

# 2. What are the null and alternate hypotheses?

The null hypothesis we are testing is that the number of callbackss for black-sounding applicants equals the number of callbacks for white-sounding applicants. So the alternate hypothesis is the scenario where the number of callbacks for both races are not equal.
#### Ho: Pb - Pw = 0
#### H1: Pb - Pw != 0


```python
w = data[data.race=='w'].call #callbacks for white-sounding names
b = data[data.race=='b'].call #callbacks for black-sounding names
```

# 3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches. 


```python
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

w_np = w.values
b_np = b.values
```


```python
# bootstrap replicate function to generate replicate datasets
def bootstrap_replicate_1d(data, func, seed=1):
    np.random.seed(seed)
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size=1, seed=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func, seed+i)

    return bs_replicates
```


```python
empirical_diff_of_means = diff_of_means(w_np, b_np)
race_concat = np.concatenate((w_np, b_np))
bs_replicates = np.empty(10000)
```


```python
# Find the mean of all callbacks
mean_callback = np.mean(race_concat)

# Generate shifted arrays
w_np_shifted = w_np - np.mean(w_np) + mean_callback
b_np_shifted = b_np - np.mean(b_np) + mean_callback

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_w = draw_bs_reps(w_np_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(b_np_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_w - bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_of_means) / len(bs_replicates)
print('p-value =', p)
```

    p-value = 0.0
    


```python
# Compute 95% confidence intervals
conf_int_w = np.percentile(bs_replicates_w, [2.5, 97.5])
conf_int_b = np.percentile(bs_replicates_b, [2.5, 97.5])
```


```python
print('The average callback rate for white-sounding names is ', conf_int_w, 'with 95% certainty')
print('The average callback rate for black-sounding names is ', conf_int_b, 'with 95% certainty')
```

    The average callback rate for white-sounding names is  [0.06899384 0.09199178] with 95% certainty
    The average callback rate for black-sounding names is  [0.07104725 0.0903491 ] with 95% certainty
    


```python
data_b = data[data['race'] == 'b']
data_w = data[data['race'] == 'w']

# using frequentist statistical approaches
#import stats module
import statsmodels.stats.api as sms
two_sample = stats.ttest_ind(data_w['call'], data_b['call'])
cm = sms.CompareMeans(sms.DescrStatsW(data_w['call']), sms.DescrStatsW(data_b['call']))

print('The t-statistic and p-value are given as', two_sample)
```

    The t-statistic and p-value are given as Ttest_indResult(statistic=4.114705290861751, pvalue=3.940802103128886e-05)
    


```python
print('The 95% confidence interval about the mean difference is ({:.3f}, {:.3f}).'.format(cm.tconfint_diff(usevar='unequal')[0],
                                                                                          cm.tconfint_diff(usevar='unequal')[1]))
```

    The 95% confidence interval about the mean difference is (0.017, 0.047).
    


```python
print('The margin of error is {:.3f}.'.format((data_b['call'].mean() - data_w['call'].mean()) 
                                              - cm.tconfint_diff(usevar='unequal')[0]))
```

    The margin of error is -0.049.
    

# 4. Write a story describing the statistical significance in the context or the original problem.

From the analysis, black-sounding names have a 6.4% callback rate as opposed to white sounding names. This difference gives us reason to be suspicious of hiring patterns. Given the p-value, test-statistic and confidence interval; we can prove that there is a statistically significant difference between the number of black-sounding names that recieved callbacks and white-sounding names that received callbacks. We are 95% confident that white-sounding names receive between 0.017 to 0.047 callbacks more than black-sounding names. Thus, from a statistical standpoint racial discriminationin the U.S Labor Market is still a major obstacle.  

# 5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

Although from the analysis we can infer that race plays somewhat of a role in the rate of callback success for job applicants, It is not enough to indicate that this is th emost important factor in callback success. There are other factors such as education, experience etc that contribute to the rate of callback success. We can go further and do a regression analysis to determine the strength of the relationship between race and callback success. 
