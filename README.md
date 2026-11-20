# Detector de gestos associados a facções em fotos

Projeto da disciplina **Robótica e Visão Computacional** (Ciência da Computação – Unochapecó, 2026).
Professor: Wagner Titon.

**Integrantes:** Eduardo V. Bee, Felipe A. Weirich, Julia Cacciatori, Paulo A. Henrich, Vinicius M. Stuani.

> 🎥 Vídeo de apresentação: _(colocar o link do YouTube aqui)_

---

## 1. O problema

Gestos comuns em fotos, como o **"V" com dois dedos** (paz e amor / vitória) e o **"três" com os dedos**,
foram apropriados por facções criminosas como sinais de identificação:

| Gesto | Significado atribuído |
|---|---|
| Dois dedos ("V") | "Tudo dois", associado ao Comando Vermelho (CV) |
| Três dedos | "Tudo três", associado ao PCC e grupos aliados |

Pessoas sem nenhum vínculo com o crime publicam fotos com esses gestos nas redes sociais (Instagram,
Facebook, WhatsApp, TikTok) e acabam confundidas com membros de facções rivais. Há casos registrados de
ameaças, sequestros e mortes em vários estados (BA, CE, MT, RS), muitos de turistas e jovens que não
conheciam o significado do gesto. Hoje a única prevenção é a própria pessoa saber do risco.

A descrição completa do problema está no documento da Etapa 1.

## 2. A solução

Um protótipo que **analisa a foto antes da publicação**, encontra as mãos, identifica quais dedos
estão esticados e **alerta o usuário** se houver um gesto de risco. O usuário então decide:
publicar mesmo assim, **publicar uma versão com a mão borrada** (gerada automaticamente) ou cancelar.

### Pipeline

```
Foto / vídeo / webcam
        │
        ▼
 ┌──────────────┐   1. YOLO11n (COCO) localiza as PESSOAS na imagem.
 │    YOLO      │      Cada pessoa é recortada e ampliada → mãos pequenas
 └──────┬───────┘      em fotos de grupo ficam detectáveis.
        ▼
 ┌──────────────┐   2. MediaPipe Hand Landmarker encontra as MÃOS (na imagem
 │  MediaPipe   │      inteira e em cada recorte) e devolve 21 pontos 3D por mão.
 └──────┬───────┘      Mãos repetidas são unidas por IoU.
        ▼
 ┌──────────────┐   3. Regras geométricas: ângulos das articulações de cada dedo
 │   Regras     │      → dedo esticado ou dobrado → padrão do gesto
 └──────┬───────┘      (V, três dedos ou neutro).
        ▼
 ┌──────────────┐   4. OpenCV desenha o esqueleto da mão, a caixa do gesto,
 │   OpenCV     │      a faixa de alerta e gera a versão com o gesto pixelado.
 └──────────────┘
```

### Por que regras em vez de treinar uma rede de gestos?

- Os gestos de interesse são **poucos e bem definidos** (combinações de dedos esticados), conclusão da Etapa 1.
- Usar ângulos entre articulações em coordenadas 3D torna a regra **independente de rotação, escala e
  posição** da mão, sem precisar de um dataset grande rotulado.
- A decisão é **explicável**: o sistema mostra quais dedos considerou esticados.

### Gestos reconhecidos

| Dedos esticados | Classificação |
|---|---|
| indicador + médio | **V (dois dedos)** ⚠️ |
| indicador + médio + anelar | **Três dedos** ⚠️ |
| polegar + indicador + médio | **Três dedos** ⚠️ (variação comum no Brasil) |
| qualquer outro (mão aberta, punho, joinha, apontar...) | Neutro ✅ |

## 3. Tecnologias e modelos

| Tecnologia | Uso |
|---|---|
| Python 3.10+ | linguagem |
| [OpenCV](https://opencv.org/) | leitura de imagens/vídeo/webcam, recortes, redimensionamento, desenho, pixelização |
| [Ultralytics YOLO11n](https://docs.ultralytics.com/) | detecção de pessoas (modelo pré-treinado no COCO, `models/yolo11n.pt`) |
| [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | 21 pontos 3D de cada mão (`models/hand_landmarker.task`) |
| NumPy | cálculo dos ângulos entre articulações |
| Streamlit | interface web que simula a tela de postagem |
| pytest | testes das regras de classificação |

Os dois modelos já estão na pasta `models/`. Se forem apagados, são baixados automaticamente na primeira execução.

## 4. Estrutura do projeto

```
├── app.py              # interface web (simulação de postagem)
├── detectar.py         # linha de comando: foto, pasta, vídeo ou webcam
├── avaliar.py          # métricas em um conjunto de fotos rotuladas
├── src/
│   ├── detector.py     # pipeline YOLO → MediaPipe → regras → OpenCV
│   └── gestos.py       # regras de dedos esticados e classificação
├── tests/test_gestos.py
├── models/             # yolo11n.pt e hand_landmarker.task
├── exemplos/           # fotos de exemplo
└── requirements.txt
```

## 5. Instalação

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Execução

**Interface web (recomendado para demonstração):**

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`. Envie uma foto (ou ative a câmera), veja a análise e as opções de publicação.

**Linha de comando:**

```bash
python detectar.py --imagem exemplos/victory.jpg --mostrar   # uma foto
python detectar.py --pasta exemplos --borrar                 # várias fotos (+ versão borrada)
python detectar.py --video meu_video.mp4                     # vídeo (salva resultados/<nome>_analise.mp4)
python detectar.py --webcam 0                                # tempo real ([q] sair, [s] salvar, [b] borrar)
```

As imagens anotadas são salvas em `resultados/`. Use `--sem-yolo` para rodar mais rápido em fotos com uma pessoa perto da câmera.

**Avaliação com fotos rotuladas:**

```
dataset/
├── V/        fotos com "V"
├── TRES/     fotos com três dedos
└── NEUTRO/   fotos sem gesto de risco
```

```bash
python avaliar.py --dataset dataset
```

Mostra matriz de confusão, acurácia, precisão/recall do alerta e tempo médio por foto, e salva `resultados/avaliacao.csv`.

**Testes:**

```bash
python -m pytest tests
```

## 7. Resultados

_(preencher com os números do `avaliar.py` no dataset do grupo)_

| Métrica | Valor |
|---|---|
| Fotos avaliadas | |
| Acurácia (V / três / neutro) | |
| Precisão do alerta | |
| Recall do alerta | |
| Tempo médio por foto (CPU) | |

## 8. Limitações e melhorias futuras

- **Mãos muito pequenas, borradas ou parcialmente cobertas** podem não ser detectadas; o recorte por pessoa com YOLO reduz, mas não elimina o problema.
- **Contexto não é considerado**: o sistema alerta sobre o gesto, não sobre a intenção. Um "V" inocente também gera alerta — por isso a decisão final é sempre do usuário.
- **Variações regionais** de sinais (outros gestos, tatuagens, símbolos em roupas, pichações) não são cobertos.
- Limiares de ângulo foram ajustados manualmente; poderiam ser calibrados com um dataset maior.
- Melhorias: treinar um classificador (ex.: YOLO ou MLP sobre os 21 pontos) com o dataset HaGRID; app mobile; integração como filtro antes do upload.

## 9. Referências

Ver lista completa no documento da Etapa 1 (A Gazeta do Acre, A Tarde, Band, BNews, Correio, SBT News, TNH1).
Imagens de `exemplos/`: amostras públicas da documentação do MediaPipe e da Ultralytics.
