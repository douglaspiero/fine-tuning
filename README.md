# Fine-tuning — Classificador de Mensagens de Clientes

Projeto de fine-tuning de um modelo de linguagem em português para classificar automaticamente mensagens de clientes de uma rede varejista em duas categorias: **venda** e **suporte**.

---

## 📋 Descrição do Problema

Uma grande rede varejista deseja automatizar a triagem das mensagens recebidas em seu Bot de Atendimento. O modelo treinado deve identificar a intenção por trás de cada mensagem:

| Categoria | Descrição                         | Exemplo                                 |
| --------- | --------------------------------- | --------------------------------------- |
| `venda`   | Intenção de compra de um produto  | _"Gostaria de adquirir o novo produto"_ |
| `suporte` | Dúvida ou problema com um produto | _"Como funciona a TV Smart x0912?"_     |

---

## 🤖 Modelo Utilizado

**[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)** — `neuralmind/bert-base-portuguese-cased`

Versão do BERT pré-treinada em português brasileiro, com desempenho superior ao BERT multilingual para textos em PT-BR.

---

## 📁 Estrutura do Projeto

```
.
├── fine_tuning_classificador.ipynb   # Notebook principal (Google Colab)
├── fine_tuning_classificador.py      # Script Python equivalente
├── treino.jsonl                      # Dataset de treino (baixado automaticamente)
├── teste.jsonl                       # Dataset de teste (baixado automaticamente)
├── resultados/                       # Checkpoints gerados durante o treino
└── modelo_classificador/             # Modelo final salvo após o treino
```

---

## 🗂️ Formato dos Dados

Os datasets estão no formato **JSONL** (uma linha por exemplo):

```json
{"prompt": "Olá, gostaria de fazer a aquisição do novo produto", "completion": "venda"}
{"prompt": "tudo bom, queria verificar como funciona a TV Smart x0912", "completion": "suporte"}
```

> Os dados foram gerados sinteticamente para fins didáticos e não representam interações reais de clientes.

---

## ⚙️ Hiperparâmetros de Treinamento

| Parâmetro              | Valor      |
| ---------------------- | ---------- |
| Épocas                 | 5          |
| Batch size (treino)    | 16         |
| Batch size (avaliação) | 32         |
| Learning rate          | 2e-5       |
| Weight decay           | 0.01       |
| Warmup ratio           | 0.1        |
| Max tokens             | 128        |
| Precisão               | FP16 (GPU) |
| Melhor modelo por      | F1-score   |

---

## 🚀 Como Executar

### Opção 1 — Google Colab (recomendado)

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → selecione `fine_tuning_classificador.ipynb`
3. Ative a GPU: **Runtime → Change runtime type → T4 GPU**
4. Execute tudo: **Runtime → Run all**

O notebook já baixa os datasets automaticamente do Notion.

### Opção 2 — Ambiente local

```bash
# Instalar dependências
pip install transformers datasets accelerate scikit-learn torch

# Executar o script
python fine_tuning_classificador.py
```

---

## 📊 Saídas Esperadas

Ao final da execução, o notebook exibe:

- **Métricas por época** (accuracy e F1 no conjunto de teste)
- **Relatório de classificação** com precision, recall e F1 por classe
- **Predições para frases novas** com nível de confiança

Exemplo de saída da inferência:

```
Mensagem                                                Classe     Confiança
--------------------------------------------------------------------------------
Quero comprar uma geladeira duplex                      venda      98.7%
Minha TV não está ligando, o que eu faço?               suporte    97.3%
Vocês têm o micro-ondas 32L em promoção?                venda      96.1%
O produto chegou com a embalagem danificada             suporte    99.2%
```

---

## 💾 Salvando o Modelo

Ao final do notebook, o modelo treinado é salvo em `./modelo_classificador/` e um arquivo `.zip` é gerado para download.

Para carregar o modelo salvo posteriormente:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("./modelo_classificador")
model     = AutoModelForSequenceClassification.from_pretrained("./modelo_classificador")

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(clf("Quero comprar um fogão 5 bocas"))
# [{'label': 'venda', 'score': 0.987}]
```

---

## 🛠️ Dependências

```
transformers
datasets
accelerate
scikit-learn
torch
pandas
numpy
```

---

## 📌 Observações

- Recomenda-se GPU para treinamento (T4 gratuita no Google Colab).
- Sem GPU, o treinamento funciona em CPU, porém será significativamente mais lento.
- O modelo final pode ser publicado no [Hugging Face Hub](https://huggingface.co) para uso em produção.
