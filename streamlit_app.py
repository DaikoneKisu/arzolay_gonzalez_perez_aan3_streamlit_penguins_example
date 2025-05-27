import streamlit
import matplotlib.pyplot as plt
import seaborn
import numpy
import pandas
from sklearn.model_selection import train_test_split

color = "#83c9ff"

@streamlit.cache_data
def get_penguins():
    return seaborn.load_dataset("penguins")

@streamlit.cache_resource
def get_linear_regression(x_train, y_train):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    return model.fit(x_train, y_train)

dataset = get_penguins()

dataset = dataset.dropna()

feature_columns = dataset.columns.difference(pandas.Index(["body_mass_g", "island", "sex", "species"]))

x = dataset[feature_columns]
y = dataset[["body_mass_g"]]
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = get_linear_regression(x_train, y_train)

y_predict = model.predict(x_test)

streamlit.title("Ejemplo con Streamlit: Dataset de pingüinos")

streamlit.header("Vista general del dataset")

streamlit.subheader("Primeros 10 registros del dataset")
streamlit.dataframe(dataset.head(10))

streamlit.subheader("Muestra de valores perdidos")
streamlit.dataframe(
    dataset.isna().sum(), 
    column_config={
        0: "Columnas del dataset",
        1: "Cantidad de valores perdidos"
    }
)

seaborn.set_theme(style="ticks")
rcParams = {
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white', 
    'axes.titlecolor': 'white',
    'axes.edgecolor': 'white',
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False
}
plt.rcParams.update(rcParams)

hist_figure, hist_axes = plt.subplots(dataset.columns.size, 1, figsize=(10, 30))

for i, column in enumerate(dataset.columns):
    seaborn.histplot(dataset[column], ax=hist_axes[i], kde=True, bins="auto", color=color)

hist_figure.tight_layout()
streamlit.subheader("Histogramas de las variables numéricas")
hist_figure

dataset_only_numeric = dataset.select_dtypes(include=[numpy.float64])

correlation = dataset_only_numeric.corr()

mask = numpy.triu(numpy.ones_like(correlation, dtype=bool))

correlation_figure, _ = plt.subplots(figsize=(11, 9))

cmap = seaborn.color_palette("crest", as_cmap=True)

ax = seaborn.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", linecolor="none")
ax.spines.bottom.set_visible(True)
ax.spines.left.set_visible(True)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_color("white")
ax.spines.left.set_color("white")

streamlit.subheader("Mapa de calor de la correlación entre variables numéricas")
correlation_figure

streamlit.subheader("Gráfico de pares de las variables numéricas")
streamlit.pyplot(seaborn.pairplot(dataset, hue="species"))

specie = streamlit.selectbox(
    "Selecciona una especie de pingüino para ver su distribución",
    options=dataset["species"].unique(),
    index=0
)

streamlit.subheader(f"Distribución de la especie {specie}")
streamlit.pyplot(seaborn.pairplot(dataset[dataset["species"] == specie], hue="species"))

fig, axes = plt.subplots(len(x_test.columns.values), 1, sharey=True, constrained_layout=True, figsize=(30,15))
fig.legend()

for i, e in enumerate(x_test.columns):
  axes[i].set_title("Mejor linea de ajuste")
  axes[i].set_xlabel(str(e))
  axes[i].set_ylabel('exam_score')
  axes[i].scatter(x_test[e].values[:,numpy.newaxis], y_test, color='blue', label='Real')
  axes[i].scatter(x_test[e].values[:,numpy.newaxis], y_predict, color='red', label='Predicción')

fig