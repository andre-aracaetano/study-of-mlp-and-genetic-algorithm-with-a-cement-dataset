# Predição da Resistência à Compressão de Concretos com MLP, XGBoost, SHAP e Algoritmo Genético

Pipeline completo de aprendizado de máquina aplicado à engenharia de materiais: treino e comparação de modelos preditivos, interpretabilidade e otimização de composição guiada por modelo.

> Este projeto nasceu como trabalho final da disciplina de Aprendizado de Máquina (CMCC, UFABC), mas foi construído como uma **pipeline genérica**: qualquer dataset tabular de regressão, com um espaço de variáveis contínuas passível de otimização, pode ser encaixado nas mesmas etapas com poucas adaptações.

## Sobre o projeto

Dado um conjunto de variáveis de composição de uma mistura (neste caso, concreto), o projeto responde a três perguntas em sequência:

1. **Qual modelo prevê melhor a propriedade de interesse?** Comparação entre uma rede neural (MLP, PyTorch) e um modelo de Gradient Boosting (XGBoost), sob o mesmo protocolo de validação cruzada e conjunto de teste isolado.
2. **O que o modelo vencedor realmente aprendeu?** Interpretabilidade via SHAP, indo além da métrica de erro para verificar se os padrões aprendidos são fisicamente/domínio-coerentes.
3. **Dado o modelo, qual é a melhor composição possível?** Um Algoritmo Genético usa o modelo treinado como função de avaliação (*surrogate-assisted optimization*) para buscar a composição que maximiza a propriedade prevista, com e sem restrições de domínio.

O ponto central do trabalho não é a originalidade de cada técnica isoladamente (MLP, XGBoost, SHAP e GA são todos bem estabelecidos, inclusive combinados entre si na literatura), mas o percurso de diagnóstico: a etapa de otimização mostra explicitamente o que acontece quando um algoritmo genético é guiado por um modelo estatístico **sem** restrições de domínio, como isso é detectado a partir de indicadores derivados do próprio dataset, e como isso muda dependendo do modelo usado como avaliador (XGBoost vs. MLP). Esse padrão de diagnóstico é reutilizável em qualquer problema de otimização guiada por modelo.

## Resultados principais

| Modelo | RMSE (MPa) | MAE (MPa) | R² |
|---|---|---|---|
| MLP | 4.909 | 3.347 | 0.914 |
| XGBoost | 4.124 | 2.713 | 0.939 |

| Origem da mistura | Resistência prevista/observada (MPa) |
|---|---|
| Melhor amostra do dataset | 82.60 |
| GA livre, avaliado pelo XGBoost | 89.34 |
| GA restrito, avaliado pelo XGBoost | 84.26 |
| GA livre, avaliado pela MLP | 145.58 |

A busca livre do GA converge para composições estatisticamente incomuns e fisicamente pouco plausíveis (extrapolação), um problema mais severo quando o modelo avaliador é a MLP do que quando é o XGBoost. Impondo restrições de domínio (limites de material cimentício total e de razão água/material cimentício, derivados do próprio dataset), o GA converge para uma composição muito próxima da melhor mistura real observada.

O relatório completo, com a análise detalhada de cada etapa, está em [`report/relatorio.pdf`](report/relatorio.pdf).

## Estrutura do repositório

```
.
├── notebooks/
│   └── concreto_mlp.ipynb      # Pipeline completo, do carregamento dos dados à otimização
├── report/
│   └── relatorio.pdf            # Relatório final em formato SBC
├── requirements.txt
└── README.md
```

*(ajuste os nomes/caminhos acima para bater exatamente com o que está no seu repositório)*

## Pipeline

1. **Dados**: carregamento, EDA (distribuição do alvo, correlações), split treino/teste (85/15), normalização ajustada só no treino.
2. **Modelagem**: busca de hiperparâmetros via K-Fold (5 folds) para MLP (PyTorch) e XGBoost, com amostragem aleatória do espaço de configurações. Avaliação final única no conjunto de teste.
3. **Interpretabilidade**: SHAP sobre o modelo de melhor desempenho — importância global, gráficos de dependência e explicação de previsões individuais.
4. **Otimização**: Algoritmo Genético (implementação própria, sem dependência externa) usando o modelo treinado como função de avaliação, nas versões livre e restrita por regras de domínio.

## Como reproduzir

```bash
git clone https://github.com/andrearacaetano/study-of-mlp-and-genetic-algorithm-with-a-cement-dataset.git
cd study-of-mlp-and-genetic-algorithm-with-a-cement-dataset

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Baixe o dataset [Concrete Compressive Strength](https://www.kaggle.com/datasets/vinayakshanawad/cement-manufacturing-concrete-dataset) e coloque em `data/concrete.csv`, depois abra `notebooks/concreto_mlp.ipynb`.

**Dataset original**: Yeh, I-C. (2007). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. [10.24432/C5PK67](https://doi.org/10.24432/C5PK67)

## Adaptando para outro dataset

A pipeline foi escrita para ser portável entre datasets tabulares de regressão com composição/proporções como entrada. Para reaproveitar em outro contexto, os pontos de ajuste são:

- **Colunas de entrada e alvo**: trocar a lista de features e o nome da coluna-alvo.
- **Restrições de domínio do GA**: os limites usados aqui (total de material cimentício, razão água/binder) são específicos de concreto; em outro domínio, substituir pelas relações de coerência física/técnica relevantes ao novo problema.
- **Variável fixada na otimização** (aqui, a idade de cura): identificar se existe uma variável análoga que não deve ser otimizada junto das demais.

O restante (split, K-Fold, busca de hiperparâmetros, SHAP, estrutura do GA) é agnóstico ao domínio.

## Tecnologias

Python · PyTorch · XGBoost · SHAP · scikit-learn · pandas · matplotlib

## Autor

André de A. Caetano — Centro de Matemática, Computação e Cognição (CMCC), Universidade Federal do ABC (UFABC)
Orientação: Ronaldo C. Prati

## Licença

Este projeto está sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais detalhes.

## Declaração de uso de IA

Ferramentas de inteligência artificial foram utilizadas de forma pontual e auxiliar neste projeto, restritas a: apoio na identificação e correção de erros básicos de implementação em trechos de código, revisão de erros de português no texto do relatório, e geração do esquema ilustrativo presente no relatório. As decisões metodológicas, a análise crítica dos resultados e as conclusões são de responsabilidade do autor.****
