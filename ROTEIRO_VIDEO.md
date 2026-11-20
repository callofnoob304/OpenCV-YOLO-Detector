# Roteiro do vídeo (máx. 10 min)

Sugestão de divisão para os 5 integrantes falarem. Ajustem os nomes conforme o grupo preferir.
Lembrete: apresentação **só com slides não é aceita** — a parte 3 precisa mostrar o sistema rodando.

---

## 1. Problema (até 2 min) — Eduardo + Julia

**Eduardo (≈1 min)**
- Apresenta todos os integrantes.
- "Gestos comuns em fotos, como o V de paz e amor e o três com os dedos, passaram a ser usados por
  facções como sinais: o V como 'tudo dois' (CV) e o três como 'tudo três' (PCC)."
- Onde ocorre: redes sociais (Instagram, Facebook, WhatsApp, TikTok); a foto pode ser vista de qualquer lugar.

**Julia (≈1 min)**
- Quem é impactado: qualquer pessoa que publica fotos — principalmente jovens, turistas, recém-chegados.
- Casos: Jericoacoara (CE, 2024), Camaçari (BA, 2024), Sorriso (MT, 2025), Porto Alegre (RS, 2026).
- Como identificamos: notícias e relatos (Etapa 1). Hoje a prevenção depende da pessoa já saber do risco.
- Importância: um alerta no momento da postagem permite decidir de forma informada.

## 2. Solução (até 2 min) — Felipe

- Proposta: verificar a foto **antes de publicar** e alertar.
- Tecnologias: Python, **OpenCV**, **YOLO11** (Ultralytics), **MediaPipe Hand Landmarker**, Streamlit.
- Funcionamento (mostrar o diagrama do README):
  1. YOLO acha as pessoas e recorta/amplia cada uma → mãos pequenas ficam visíveis.
  2. MediaPipe devolve 21 pontos 3D de cada mão.
  3. Calculamos os ângulos das articulações → cada dedo esticado ou dobrado.
  4. Padrão de dedos → V / três / neutro.
  5. OpenCV desenha o resultado e gera a versão com a mão pixelada.
- Etapas de desenvolvimento: estudo do problema → escolha do pipeline → regras de dedos →
  testes → interface → avaliação.

## 3. Demonstração (até 4 min) — Paulo + Vinicius

**Paulo — interface web (≈2 min)** — `streamlit run app.py`
1. Enviar uma foto **sem** gesto (ex.: joinha) → faixa verde, "pode publicar".
2. Enviar uma foto com **V** → alerta vermelho, explicação, dedos detectados.
3. Mostrar a **versão borrada** e os botões "publicar borrada / publicar mesmo assim".
4. Foto de **grupo** com alguém fazendo o gesto → ligar "Mostrar caixas das pessoas" para mostrar o YOLO.

**Vinicius — tempo real e resultados (≈2 min)**
1. `python detectar.py --webcam 0` → fazer V, três (indicador+médio+anelar), três (polegar+indicador+médio),
   mão aberta, punho. Apertar **b** para mostrar o borrão em tempo real.
2. `python avaliar.py --dataset dataset` → mostrar matriz de confusão, precisão, recall e tempo.
3. Explicar: o sistema resolve o problema avisando **antes** da publicação; a decisão continua com o usuário.

## 4. Resultados e considerações (até 2 min) — todos, um tópico cada

- **Resultados** (Eduardo): números da avaliação; tempo por foto na CPU.
- **Dificuldades** (Julia): mãos pequenas em fotos de grupo (resolvido com recorte via YOLO); polegar
  ambíguo; ajustar os limiares de ângulo; incompatibilidades de versões das bibliotecas.
- **Limitações** (Felipe): não entende contexto/intenção; mãos cobertas ou borradas; apenas dois gestos.
- **Melhorias** (Paulo): treinar classificador com dataset HaGRID; mais sinais (tatuagens, roupas);
  app mobile / integração com a câmera do celular.
- **Conclusão** (Vinicius): problema real e grave; visão computacional permite um alerta simples e
  explicável; a pessoa publica de forma consciente.

---

## Checklist antes de gravar

- [ ] Montar a pasta `dataset/` com fotos do grupo (V, três, neutro — idealmente 20+ de cada) e rodar `avaliar.py`.
- [ ] Preencher a tabela de **Resultados** do README com os números.
- [ ] Separar uma foto de grupo com gesto para a demo.
- [ ] Testar webcam e Streamlit no computador que vai gravar.

## Checklist de entrega

- [ ] Repositório **público** no GitHub com código, README, requirements e modelos.
- [ ] Vídeo no YouTube **público ou não listado** (não privado), ≤ 10 min, todos falando.
- [ ] Link do vídeo no README.
- [ ] Abrir os dois links numa janela anônima para confirmar que estão acessíveis.
