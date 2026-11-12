# PyModels Studio 2.0

Editor 3D desktop em Python, com interface CustomTkinter e viewport OpenGL.

## Instalação (Windows)

1. Instale Python 3.10 ou superior.
2. Abra o Prompt de Comando na pasta do projeto.
3. Instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Inicie o programa:

   ```bash
   python PyModels.py
   ```

## Recursos

- Primitivas: cubo, esfera, cilindro, cone, plano, pirâmide e toro.
- Ferramentas de mover, rotacionar e escalar pelo gizmo.
- Seleção múltipla com Shift + clique.
- Hierarquia com grupos, duplicação, renomeação, visibilidade e bloqueio.
- Inspector com posição, rotação, escala e cor RGB.
- Snap configurável, grid, eixos XYZ e modo wireframe.
- Vistas de câmera frontal, traseira, laterais, superior, inferior e isométrica.
- Desfazer/refazer, salvamento automático e recuperação de sessão.
- Importação de malhas OBJ e exportação de cena em OBJ + MTL.
- Compatibilidade de leitura com arquivos JSON da versão anterior do PyModels.

## Atalhos

| Atalho | Ação |
|---|---|
| Ctrl + N | Novo projeto |
| Ctrl + O | Abrir projeto |
| Ctrl + S | Salvar |
| Ctrl + Shift + S | Salvar como |
| Ctrl + E | Exportar OBJ |
| Ctrl + Z / Ctrl + Y | Desfazer / refazer |
| Ctrl + D | Duplicar |
| Ctrl + G | Agrupar seleção |
| Ctrl + Shift + G | Desagrupar |
| F2 | Renomear |
| Delete | Excluir |
| G / R / S | Mover / rotacionar / escalar |
| F / Home | Enquadrar selecionado / cena inteira |
| 1–4 | Criar cubo / esfera / cilindro / cone |

## Controles do viewport

- Botão esquerdo ou direito + arrastar: orbitar a câmera.
- Botão do meio + arrastar: deslocar a câmera.
- Scroll: zoom.
- Arraste as alças do gizmo para transformar o objeto.

> Observação: o importador OBJ atual lê vértices e faces poligonais. Materiais, texturas, UVs e rigs de outros programas ainda não são importados.
