# Detecção de Pneumonia com Deep Learning

## Descrição do Projeto

Este projeto consiste em uma aplicação de **Deep Learning** desenvolvida para a detecção de pneumonia em imagens de raio-X. O modelo utiliza a arquitetura **DenseNet121** pré-treinada, aprimorada com camadas personalizadas, e é treinado com um conjunto de dados dividido em imagens de treino, validação e teste. 

Além disso, o projeto adota técnicas de aumento de dados para melhorar a robustez e a generalização do modelo.

---

## Estrutura do Código

1. **Preparação do Ambiente:**
   - Importação das bibliotecas necessárias, incluindo `TensorFlow`, `Keras` e `scikit-learn`.
   - Definição dos caminhos para os conjuntos de dados (treino, validação e teste).

2. **Aumento e Processamento de Dados:**
   - Utilização de `ImageDataGenerator` para aplicar transformações como rotação, zoom e flip horizontal, com objetivo de melhorar a generalização.
   - Redimensionamento das imagens para `(224, 224)` e normalização dos valores de pixels entre 0 e 1.

3. **Arquitetura do Modelo:**
   - Modelo base: **DenseNet121** com pesos pré-treinados no ImageNet.
   - Camada de pooling global (`GlobalAveragePooling2D`) e uma camada densa final com ativação sigmoide para classificação binária.

4. **Treinamento:**
   - Otimizador: `Adam` com taxa de aprendizado de `0.0001`.
   - Função de perda: `binary_crossentropy`.
   - Métrica: `accuracy`.
   - Callback: `EarlyStopping` para evitar overfitting, restaurando os melhores pesos do modelo baseado na menor perda de validação.

5. **Avaliação:**
   - Predição das classes para o conjunto de teste.
   - Geração de relatório de classificação e matriz de confusão usando `classification_report` e `confusion_matrix`.

6. **Armazenamento:**
   - Salvamento do modelo treinado no arquivo `modelo_pneumonia.h5`.

---

## Tecnologias Utilizadas

- **TensorFlow/Keras**: Framework para construção, treinamento e avaliação do modelo.
- **scikit-learn**: Utilizado para geração de métricas e análise de desempenho.
- **Python**: Linguagem principal para desenvolvimento.

---

## Pré-requisitos

- Python 3.8 ou superior.
- Bibliotecas necessárias:
  - TensorFlow
  - scikit-learn
  - NumPy

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install tensorflow scikit-learn numpy
```
## Como Executar o Projeto

```bash
python app.py
```

Salvar o modelo treinado: O modelo será salvo automaticamente como modelo_pneumonia.h5.

Estrutura de Saída
Relatório de Classificação: Exibe métricas de desempenho (precisão, recall, F1-Score) por classe.
Matriz de Confusão: Mostra os acertos e erros do modelo para cada classe.
Modelo Treinado: Salvo como modelo_pneumonia.h5 para uso futuro.
