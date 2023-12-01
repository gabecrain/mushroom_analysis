import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr, stats
import statsmodels.api as sm
import plotly.express as px

mushrooms = pd.read_csv('/Users/gabecrain/Desktop/Documents/Data Science/Mushroom_Project/mushrooms.csv')
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

#find the proportion of mushrooms that bruise vs those that dont
bruising_frequency = np.sum(mushrooms.bruises == 'True')
print('amount of mushrooms that bruise:', bruising_frequency)

bruising_proportion = np.mean(mushrooms.bruises == 'True')
print('proportion of mushrooms that bruise', bruising_proportion)
#approximately 42% of the mushrooms in this population bruise, meaning the other 58% do not bruise.

print('\n')

#find proportion of mushrooms that have a solitary population vs those that do not
solitary_habitat_frequency = np.sum(mushrooms.population == 'solitary')
print('frequency of mushrooms in a solitary population:', solitary_habitat_frequency)

solitary_habitat_proportion = np.mean(mushrooms.population == 'solitary')
print('proportion of mushrooms in a solitary population:', solitary_habitat_proportion)
#approximately 21% of mushrooms in this dataset have a solitary population

print('\n')

#find proportion of mushrooms that are edible vs poisonous
mushroom_class_frequency = np.sum(mushrooms['class'] == 'edible')
print('frequency of edible mushrooms:', mushroom_class_frequency)

mushroom_class_proportion = np.mean(mushrooms['class'] == 'edible')
print('proportion of edible mushrooms', mushroom_class_proportion)
#approximately 52% of mushrooms in this dataset are classified as edible, meaning the other 48% are classified as poisonous.

print('\n')

#find median and central tendency of ring_number from mushrooms dataset
ring_number_order = ['none', 'one', 'two']

mushrooms['ring_number'] = pd.Categorical(mushrooms['ring_number'], ring_number_order, ordered=True)

ring_number_values = mushrooms.ring_number.value_counts()
print('mushroom ring number value counts:\n', ring_number_values)

median_ring_number_index = np.median(mushrooms.ring_number.cat.codes)
print('median ring number index:', median_ring_number_index)

median_ring_number_category = ring_number_order[int(median_ring_number_index)]
print('median ring number category:', median_ring_number_category)

mean_ring_number_category = np.mean(mushrooms.ring_number.cat.codes)
print('mean ring number category:', mean_ring_number_category)
#this calculation can be interpreted as falling between one ring number and two, being much closer to one ring.
