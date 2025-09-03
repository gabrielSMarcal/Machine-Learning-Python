import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import utils as u

# Como é a relação entre área construida e o preço do imovel?

plt.scatter(u.dados['area_primeiro_andar'], u.dados['preco_de_venda'])
plt.axline(xy1 = (66, 250000), xy2 = (190, 1800000), color='red')

plt.title('Relação entre preço e área')
plt.xlabel('Área do primeiro andar (m²)')
plt.ylabel('Preço de venda (R$)')

# Qual a reta que melhor se adequa a relação?
fig = px.scatter(u.dados, x='area_primeiro_andar', y='preco_de_venda', trendline_color_override='red', trendline='ols')
fig.show()