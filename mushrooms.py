import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr, stats
import statsmodels.api as sm
import plotly.express as px

mushrooms = pd.read_csv('/Users/gabecrain/Desktop/Mushroom_Project/mushrooms.csv')
print(mushrooms.info())

mushrooms.columns = mushrooms.columns.str.replace('-', '_')
mushrooms['class'].replace({'e': 'edible', 'p': 'poisonous'}, inplace=True)
mushrooms['cap_shape'].replace({'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'}, inplace=True)
mushrooms['cap_surface'].replace({'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'}, inplace=True)
mushrooms['cap_color'].replace({'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['bruises'].replace({'t': 'True', 'f': 'False'}, inplace=True)
mushrooms['odor'].replace({'a': 'almond', 'l': 'anise', 'c': 'creosot', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'}, inplace=True)
mushrooms['gill_attachment'].replace({'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'}, inplace=True)
mushrooms['gill_spacing'].replace({'c': 'close', 'w': 'crowded', 'd': 'distant'}, inplace=True)
mushrooms['gill_size'].replace({'b': 'broad', 'n': 'narrow'}, inplace=True)
mushrooms['gill_color'].replace({'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['stalk_shape'].replace({'e': 'enlarging', 't': 'tapering'}, inplace=True)
mushrooms['stalk_root'].replace({'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'}, inplace=True)
mushrooms['stalk_surface_above_ring'].replace({'f': 'fibrous', 'y': 'scaly', 's': 'smooth', 'k': 'silky'}, inplace=True)
mushrooms['stalk_surface_below_ring'].replace({'f': 'fibrous', 'y': 'scaly', 's': 'smooth', 'k': 'silky'}, inplace=True)
mushrooms['stalk_color_above_ring'].replace({'n': 'brown','b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['stalk_color_below_ring'].replace({'n': 'brown','b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['veil_type'].replace({'u': 'universal', 'p': 'partial'}, inplace=True)
mushrooms['veil_color'].replace({'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['ring_number'].replace({'n': 'none', 'o': 'one', 't': 'two'}, inplace=True)
mushrooms['ring_type'].replace({'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's':' sheathing', 'z': 'zone'}, inplace=True)
mushrooms['spore_print_color'].replace({'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'}, inplace=True)
mushrooms['population'].replace({'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'}, inplace=True)
mushrooms['habitat'].replace({'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}, inplace=True)

print(mushrooms.info())
print('list of all column names:\n', mushrooms.columns.tolist())

#plot some initial charts using seaborn

#mushroom classification
sns.countplot(x='class', data=mushrooms)
plt.xlabel('Mushroom Classification')
plt.ylabel('Frequency')
plt.title('Frequency of Poisonous vs Non Poisonous Mushrooms')
# plt.show()
plt.clf()

#cap surface
sns.countplot(x='cap_surface', data=mushrooms)
plt.xlabel('Cap Surface')
plt.ylabel('Frequency')
plt.title('Frequency of Cap Surface')
# plt.show()
plt.clf()

#odor
sns.countplot(x='odor', hue='odor', data=mushrooms, order=mushrooms['odor'].value_counts(ascending=True).index)
plt.xticks(rotation=-30)
plt.xlabel('Odor')
plt.ylabel('Frequency')
plt.title('Frequency of Odor')
# plt.show()
plt.clf()

#ring number
sns.countplot(x='ring_number', hue='ring_number', data=mushrooms, order=['none', 'one', 'two'])
plt.xlabel('Ring Number')
plt.ylabel('Frequency')
plt.title('Frequency of Ring Number')
# plt.show()
plt.clf()

#population
population_counts = mushrooms['population'].value_counts()
plt.pie(population_counts, labels=population_counts.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Mushroom Population Distribution')
# plt.show()
plt.clf()

#habitat
habitat_counts = mushrooms['habitat'].value_counts()
plt.pie(habitat_counts, labels=habitat_counts.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Mushroom Habitat Counts Distribution')
# plt.show()
plt.clf()

print('mushroom habitat value counts:\n', habitat_counts)