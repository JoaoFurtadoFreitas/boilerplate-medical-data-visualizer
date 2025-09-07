import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importa o dado
df = pd.read_csv("medical_examination.csv")

# 2. Adiciona a coluna "overweight"
# BMI = weight / (height/100)^2
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
# Remove coluna auxiliar BMI
df = df.drop(columns=['BMI'])


# 3. Normaliza os dados 
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Desenha o gráfico de barras
def draw_cat_plot():
    # 5. Cria o DataFrame para o gráfico de barras usando `pd.melt`
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Agrupa e conta os dados para cada combinação de `cardio`, `variable` e `value`
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])\
                   .size().reset_index(name='total')

    # 7. Desenha o gráfico de barras
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        data=df_cat
    ).fig

    # 8. Salva a figura
    fig.savefig('catplot.png')
    return fig


# 9. Desenha o heatmap
def draw_heat_map():
    # 10. Limpa os dados
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calcula a correlação
    corr = df_heat.corr()

    # 12. Gera a máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Configura o tamanho da figura
    fig, ax = plt.subplots(figsize=(12, 12))

    # 14. Desenha o heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        vmax=0.3,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # 15. Salva a figura
    fig.savefig('heatmap.png')
    return fig
